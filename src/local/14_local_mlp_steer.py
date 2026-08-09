"""
Local steering faithfulness for the MLP SAE probe (DeepSHAP).

Parallel to local/9_local_steer.py, but:
  - Attributions are per-sentence DeepSHAP φ(x) on the MLP filtered feature set
  - Global rankings come from outputs/13_mlp_ranking/ (as in global_mlp/14_mlp_steer.py)
  - Candidates are per-example: top-k by |φ(x)| ∪ top-k by |Φ| (same recipe as
    local_shap_faithfulness.py) so local and global compete on an equal footing
  - Intervention effects Δ(x, i) are kept per example (never dataset-averaged
    before correlating)
  - Reports mean Spearman ρ of local φ vs Δ, and of global Φ / probe / IG / GA
    vs the same per-example Δ — separating local SHAP soundness from errors
    introduced by averaging into Φ
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from sae_lens import SAE
from scipy.stats import spearmanr
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
from transformer_lens import HookedTransformer

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils import load_model, load_splits


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_shap12 = _load("mlp_shap12", _SRC / "global_mlp" / "12_mlp_shap.py")
_steer14 = _load("mlp_steer14", _SRC / "global_mlp" / "14_mlp_steer.py")

LAYER = 7
N_EXAMPLES = 100
N_BACKGROUND = 100
K_LOCAL = 20
K_GLOBAL = 20
SEED = 0
MODE = "ablation"  # "ablation" | "steering"
STEERING_ALPHA = 0.6723
CHECKPOINT = "checkpoints/mlp_probe_layer_7.joblib"
RANKINGS_DIR = Path("outputs/13_mlp_ranking")
OUT_DIR = Path("outputs/14_local_mlp_steer")


def spearman_pair(
    scores: np.ndarray, deltas: np.ndarray, *, importance: bool
) -> tuple[float, float]:
    if len(scores) < 3:
        return float("nan"), float("nan")
    a = np.abs(scores) if importance else scores
    b = np.abs(deltas) if importance else deltas
    rho, p = spearmanr(a, b)
    return float(rho), float(p)


def candidate_indices_for_example(
    local_phi: np.ndarray,
    global_phi_filtered: np.ndarray,
    k_local: int,
    k_global: int,
) -> np.ndarray:
    """Union of top-|local| and top-|global| feature indices (into filtered axis)."""
    k_local = min(k_local, len(local_phi))
    k_global = min(k_global, len(global_phi_filtered))
    top_local = np.argsort(np.abs(local_phi))[::-1][:k_local]
    top_global = np.argsort(np.abs(global_phi_filtered))[::-1][:k_global]
    return np.unique(np.concatenate([top_local, top_global]))


def build_per_example_candidates(
    local_shap: np.ndarray,
    global_phi_filtered: np.ndarray,
    k_local: int,
    k_global: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Returns
    -------
    cands_per_example : list of length n; each is filtered-axis indices for that x
    union_local : sorted unique filtered indices across all examples (intervention set)
    """
    cands: list[np.ndarray] = []
    for e in range(local_shap.shape[0]):
        cands.append(
            candidate_indices_for_example(
                local_shap[e], global_phi_filtered, k_local, k_global
            )
        )
    union_local = np.unique(np.concatenate(cands))
    return cands, union_local


def get_per_example_intervention_effects(
    model: HookedTransformer,
    sae: SAE,
    all_tokens: torch.Tensor,
    candidate_global: np.ndarray,
    layer: int = LAYER,
    mode: str = "steering",
    steering_alpha: float = STEERING_ALPHA,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One-at-a-time SAE interventions; keep per-example Δ (no mean over dataset).

    Returns
    -------
    delta_signed : (n_examples, n_candidates)
    delta_abs    : (n_examples, n_candidates)
    """
    if mode not in ("ablation", "steering"):
        raise ValueError(f"Unknown mode={mode!r}")

    n = all_tokens.shape[0]
    k = len(candidate_global)
    intervene_point = f"blocks.{layer}.hook_resid_post"
    pad_id = _steer14._pad_token_id(model)
    pos_id, neg_id = _steer14.sentiment_token_ids(
        model, _steer14.POS_TOKEN, _steer14.NEG_TOKEN
    )

    def recon_hook(resid_post, hook):
        return sae.decode(sae.encode(resid_post))

    def score_batch(tokens: torch.Tensor, fwd_hooks: list) -> np.ndarray:
        with torch.no_grad():
            with model.hooks(fwd_hooks=fwd_hooks):
                logits = model(tokens)
            return _steer14.last_non_pad_logit_diff(
                logits, tokens, pad_id, pos_id, neg_id
            )

    def score_all(fwd_hooks: list, desc: str) -> np.ndarray:
        chunks = []
        for start in tqdm(range(0, n, batch_size), desc=desc, leave=False):
            batch = all_tokens[start : start + batch_size].to(model.cfg.device)
            chunks.append(score_batch(batch, fwd_hooks))
        return np.concatenate(chunks, axis=0)

    baseline = score_all([(intervene_point, recon_hook)], "Baselines")

    delta_signed = np.zeros((n, k), dtype=np.float64)
    loop_desc = "Ablating features" if mode == "ablation" else "Steering features"
    for j, orig_idx in enumerate(tqdm(candidate_global, desc=loop_desc)):
        idx = int(orig_idx)

        def intervention_hook(
            resid_post,
            hook,
            feature_idx=idx,
            _mode=mode,
            _alpha=steering_alpha,
        ):
            acts = _steer14._apply_feature_intervention(
                sae.encode(resid_post), feature_idx, _mode, _alpha
            )
            return sae.decode(acts)

        intervened = score_all(
            [(intervene_point, intervention_hook)], f"feature {idx}"
        )
        delta_signed[:, j] = intervened - baseline

    return delta_signed, np.abs(delta_signed)


def per_example_faithfulness(
    method_scores_filtered: dict[str, np.ndarray],
    local_shap: np.ndarray,
    cands_per_example: list[np.ndarray],
    union_local: np.ndarray,
    delta_signed: np.ndarray,
) -> dict[str, dict]:
    """
    method_scores_filtered[name]: (n_filtered,) global scores on filtered axis
    local_shap: (n_examples, n_filtered)
    cands_per_example[e]: filtered-axis indices for example e
    union_local: (n_union,) filtered indices aligning delta_signed columns
    delta_signed: (n_examples, n_union)
    """
    n = local_shap.shape[0]
    col_of = {int(f): j for j, f in enumerate(union_local)}
    out: dict[str, dict] = {}

    rho_imp, rho_dir = [], []
    for e in range(n):
        cand = cands_per_example[e]
        cols = np.asarray([col_of[int(c)] for c in cand], dtype=np.int64)
        s = local_shap[e, cand]
        d = delta_signed[e, cols]
        ri, _ = spearman_pair(s, d, importance=True)
        rd, _ = spearman_pair(s, d, importance=False)
        rho_imp.append(ri)
        rho_dir.append(rd)
    out["local_SHAP"] = {
        "importance_rho": np.asarray(rho_imp, dtype=np.float64),
        "directional_rho": np.asarray(rho_dir, dtype=np.float64),
    }

    for name, scores in method_scores_filtered.items():
        rho_imp, rho_dir = [], []
        for e in range(n):
            cand = cands_per_example[e]
            cols = np.asarray([col_of[int(c)] for c in cand], dtype=np.int64)
            s_cand = scores[cand]
            d = delta_signed[e, cols]
            ri, _ = spearman_pair(s_cand, d, importance=True)
            rd, _ = spearman_pair(s_cand, d, importance=False)
            rho_imp.append(ri)
            rho_dir.append(rd)
        out[name] = {
            "importance_rho": np.asarray(rho_imp, dtype=np.float64),
            "directional_rho": np.asarray(rho_dir, dtype=np.float64),
        }
    return out


def summarize_rhos(faith: dict[str, dict]) -> dict[str, dict]:
    summary = {}
    for name, block in faith.items():
        for kind in ("importance_rho", "directional_rho"):
            arr = block[kind]
            key = f"{name}:{kind}"
            summary[key] = {
                "mean": float(np.nanmean(arr)),
                "std": float(np.nanstd(arr)),
                "n_finite": int(np.isfinite(arr).sum()),
            }
    local_imp = faith["local_SHAP"]["importance_rho"]
    global_imp = faith["global_SHAP"]["importance_rho"]
    finite = np.isfinite(local_imp) & np.isfinite(global_imp)
    summary["frac_local_gt_global_importance"] = float(
        np.mean(local_imp[finite] > global_imp[finite]) if finite.any() else float("nan")
    )
    return summary


def print_summary(
    summary: dict, mode: str, k_local: int, k_global: int, n: int, n_union: int
) -> str:
    lines = [
        "Local steering faithfulness (MLP probe / DeepSHAP)",
        "=" * 50,
        f"mode={mode}  candidates=top-{k_local} local ∪ top-{k_global} global  "
        f"N={n}  n_union={n_union}",
        "",
        "Mean Spearman ρ across examples (± std)",
        "Importance (|attr| vs |Δ|):",
    ]
    order = [
        "local_SHAP",
        "global_SHAP",
        "Probe saliency",
        "IG",
        "GA",
    ]
    for name in order:
        key = f"{name}:importance_rho"
        if key not in summary:
            continue
        s = summary[key]
        lines.append(f"  {name:14s}  ρ={s['mean']:+.3f} ± {s['std']:.3f}")
    lines.append("Directional (attr vs Δ):")
    for name in order:
        key = f"{name}:directional_rho"
        if key not in summary:
            continue
        s = summary[key]
        lines.append(f"  {name:14s}  ρ={s['mean']:+.3f} ± {s['std']:.3f}")
    lines.append("")
    frac = summary.get("frac_local_gt_global_importance", float("nan"))
    lines.append(
        f"Fraction examples with ρ_local_SHAP > ρ_global_SHAP (importance): {frac:.3f}"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    payload = joblib.load(CHECKPOINT)
    probe: MLPClassifier = payload["probe"]
    feature_indices = np.asarray(payload["feature_indices"])
    ranking_indices = np.load(RANKINGS_DIR / "feature_indices.npy")
    if not np.array_equal(feature_indices, ranking_indices):
        raise ValueError(
            "Checkpoint feature_indices != outputs/13_mlp_ranking/feature_indices.npy"
        )

    probe_f = np.load(RANKINGS_DIR / "probe_scores.npy").astype(np.float64)
    ig_f = np.load(RANKINGS_DIR / "ig_scores.npy").astype(np.float64)
    ga_f = np.load(RANKINGS_DIR / "ga_scores.npy").astype(np.float64)
    shap_global_f = np.load(RANKINGS_DIR / "shap_scores.npy").astype(np.float64)

    n_filt = len(feature_indices)
    for name, arr in [
        ("probe", probe_f),
        ("ig", ig_f),
        ("ga", ga_f),
        ("shap", shap_global_f),
    ]:
        if arr.shape != (n_filt,):
            raise ValueError(
                f"{name}_scores shape {arr.shape} != ({n_filt},) filtered width"
            )

    print(
        f"MLP probe: n_filtered={n_filt}, hidden={payload.get('hidden_dim')}, "
        f"mode={MODE}, alpha={STEERING_ALPHA}"
    )

    rng = np.random.default_rng(SEED)
    shap_acts = np.load(f"activations/shap/layer_{LAYER}/activations.npy")
    train_acts = np.load(f"activations/probe_train/layer_{LAYER}/activations.npy")
    pick = rng.choice(len(shap_acts), size=N_EXAMPLES, replace=False)
    bg_idx = rng.choice(len(train_acts), size=N_BACKGROUND, replace=False)
    shap_eval = shap_acts[pick]
    background = train_acts[bg_idx]

    print(
        f"Computing local DeepSHAP on {N_EXAMPLES} examples × "
        f"{len(feature_indices)} features…"
    )
    local_shap = _shap12.run_deepshap(probe, shap_eval, background, feature_indices)
    print(f"  local_shap shape={local_shap.shape}")

    cands_per_example, union_local = build_per_example_candidates(
        local_shap, shap_global_f, K_LOCAL, K_GLOBAL
    )
    union_global = feature_indices[union_local]
    n_cands = np.asarray([len(c) for c in cands_per_example], dtype=np.int64)
    print(
        f"Per-example candidates: top-{K_LOCAL} |φ(x)| ∪ top-{K_GLOBAL} |Φ|  "
        f"(mean |cand|={n_cands.mean():.1f}, union={len(union_local)})"
    )

    print("Loading model + SAE…")
    model = load_model()
    sae = SAE.from_pretrained("gpt2-small-resid-post-v5-32k", "blocks.7.hook_resid_post")
    sae = sae.to(model.cfg.device)

    _, _, shap_ds = load_splits()
    sentences = [shap_ds[int(i)]["sentence"] for i in pick]
    all_tokens = model.to_tokens(sentences)

    pos_id, neg_id = _steer14.sentiment_token_ids(model)
    print(
        f"Logit-diff readout: {_steer14.POS_TOKEN!r}({pos_id}) - "
        f"{_steer14.NEG_TOKEN!r}({neg_id}); N={N_EXAMPLES}, mode={MODE}"
    )

    delta_signed, delta_abs = get_per_example_intervention_effects(
        model,
        sae,
        all_tokens,
        union_global,
        layer=LAYER,
        mode=MODE,
        steering_alpha=STEERING_ALPHA,
    )

    method_scores_f = {
        "global_SHAP": shap_global_f,
        "Probe saliency": probe_f,
        "IG": ig_f,
        "GA": ga_f,
    }
    faith = per_example_faithfulness(
        method_scores_f, local_shap, cands_per_example, union_local, delta_signed
    )
    summary = summarize_rhos(faith)
    text = print_summary(
        summary, MODE, K_LOCAL, K_GLOBAL, N_EXAMPLES, len(union_local)
    )
    print("\n" + text)

    mean_abs = delta_abs.mean(axis=0)
    mean_signed = delta_signed.mean(axis=0)
    print("Top union features by mean_|Δ| (display only):")
    for j in np.argsort(mean_abs)[::-1][:10]:
        print(
            f"  feature {int(union_global[j])}: |Δ̄|={mean_abs[j]:.4f}  "
            f"Δ̄={mean_signed[j]:+.4f}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "example_indices.npy", pick)
    np.save(OUT_DIR / "background_indices.npy", bg_idx)
    np.save(OUT_DIR / "feature_indices.npy", feature_indices)
    np.save(OUT_DIR / "union_local.npy", union_local)
    np.save(OUT_DIR / "union_global.npy", union_global)
    np.save(OUT_DIR / "n_cands_per_example.npy", n_cands)
    np.save(OUT_DIR / "cands_per_example.npy", np.asarray(cands_per_example, dtype=object))
    np.save(OUT_DIR / "local_shap.npy", local_shap)
    np.save(OUT_DIR / f"delta_signed_{MODE}.npy", delta_signed)
    np.save(OUT_DIR / f"delta_abs_{MODE}.npy", delta_abs)
    for name, block in faith.items():
        safe = name.replace(" ", "_")
        np.save(OUT_DIR / f"rho_importance_{safe}.npy", block["importance_rho"])
        np.save(OUT_DIR / f"rho_directional_{safe}.npy", block["directional_rho"])
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(
            {
                "mode": MODE,
                "k_local": K_LOCAL,
                "k_global": K_GLOBAL,
                "n_examples": N_EXAMPLES,
                "n_union": int(len(union_local)),
                "mean_n_cands": float(n_cands.mean()),
                "steering_alpha": STEERING_ALPHA,
                "seed": SEED,
                "checkpoint": CHECKPOINT,
                **summary,
            },
            f,
            indent=2,
        )
    (OUT_DIR / "summary.txt").write_text(text + "\n")
    print(f"\nWrote outputs → {OUT_DIR}/")


if __name__ == "__main__":
    main()
