"""
Local steering faithfulness for the linear (logistic) SAE probe.

Parallel to global_linear/9_steer.py, but:
  - Attributions are per-sentence LinearSHAP φ(x) on the filtered feature set
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
from tqdm import tqdm
from transformer_lens import HookedTransformer

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils import load_model, load_splits

# Reuse LinearSHAP + global steer helpers (numeric / sibling modules).
def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_shap7 = _load("shap7", _SRC / "global_linear" / "7_shap_recompute.py")
_steer9 = _load("steer9", _SRC / "global_linear" / "9_steer.py")

LAYER = 7
N_EXAMPLES = 100
N_BACKGROUND = 100
K = 20
SEED = 0
MODE = "steering"  # "ablation" | "steering"
STEERING_ALPHA = 0.6723
# Shared candidate set (parity with global 9_steer): "top" | "random"
SELECTION = "top"
OUT_DIR = Path("outputs/9_local_steer")


def spearman_pair(
    scores: np.ndarray, deltas: np.ndarray, *, importance: bool
) -> tuple[float, float]:
    if len(scores) < 3:
        return float("nan"), float("nan")
    a = np.abs(scores) if importance else scores
    b = np.abs(deltas) if importance else deltas
    rho, p = spearmanr(a, b)
    return float(rho), float(p)


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
    pad_id = _steer9._pad_token_id(model)
    pos_id, neg_id = _steer9.sentiment_token_ids(
        model, _steer9.POS_TOKEN, _steer9.NEG_TOKEN
    )

    def recon_hook(resid_post, hook):
        return sae.decode(sae.encode(resid_post))

    def score_batch(tokens: torch.Tensor, fwd_hooks: list) -> np.ndarray:
        with torch.no_grad():
            with model.hooks(fwd_hooks=fwd_hooks):
                logits = model(tokens)
            return _steer9.last_non_pad_logit_diff(
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
            acts = _steer9._apply_feature_intervention(
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
    cand_local: np.ndarray,
    delta_signed: np.ndarray,
) -> dict[str, dict]:
    """
    method_scores_filtered[name]: (n_filtered,) global scores on filtered axis
    local_shap: (n_examples, n_filtered)
    cand_local: (K,) indices into filtered axis
    delta_signed: (n_examples, K)
    """
    n = local_shap.shape[0]
    out: dict[str, dict] = {}

    # Local SHAP: per-example score rows
    rho_imp, rho_dir = [], []
    for e in range(n):
        s = local_shap[e, cand_local]
        d = delta_signed[e]
        ri, _ = spearman_pair(s, d, importance=True)
        rd, _ = spearman_pair(s, d, importance=False)
        rho_imp.append(ri)
        rho_dir.append(rd)
    out["local_SHAP"] = {
        "importance_rho": np.asarray(rho_imp, dtype=np.float64),
        "directional_rho": np.asarray(rho_dir, dtype=np.float64),
    }

    for name, scores in method_scores_filtered.items():
        s_cand = scores[cand_local]
        rho_imp, rho_dir = [], []
        for e in range(n):
            d = delta_signed[e]
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


def print_summary(summary: dict, mode: str, selection: str, k: int, n: int) -> str:
    lines = [
        "Local steering faithfulness (linear probe)",
        "=" * 50,
        f"mode={mode}  selection={selection}  K={k}  N={n}",
        "",
        "Mean Spearman ρ across examples (± std)",
        "Importance (|attr| vs |Δ|):",
    ]
    order = [
        "local_SHAP",
        "global_SHAP",
        "Probe weights",
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
    lines.append(f"Fraction examples with ρ_local_SHAP > ρ_global_SHAP (importance): {frac:.3f}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    feature_indices = np.load("outputs/3_shap/shap_feature_indices.npy")
    sae_probe = joblib.load("checkpoints/probe_layer_7.joblib")
    sae_probe.verbose = 0

    probe_full = np.asarray(sae_probe.coef_[0], dtype=np.float64)
    ig_full = np.load("outputs/8_rankings_recompute/ig_scores.npy").astype(np.float64)
    ga_full = np.load("outputs/8_rankings_recompute/ga_scores.npy").astype(np.float64)
    shap_global_full = np.load(
        "outputs/7_shap_recompute/phi_sentiment_layer7_signed.npy"
    ).astype(np.float64)

    probe_f = probe_full[feature_indices]
    ig_f = ig_full[feature_indices]
    ga_f = ga_full[feature_indices]
    shap_global_f = shap_global_full[feature_indices]

    # Shared candidate set on filtered axis (same recipe as global 9_steer).
    cand_local, cand_global = _steer9.get_shap_candidates(
        shap_global_f, feature_indices, k=K, selection=SELECTION, seed=SEED
    )
    sel_desc = (
        f"top-{K} by |global Φ|"
        if SELECTION == "top"
        else f"{K} random filtered (seed={SEED})"
    )
    print(f"Shared candidate set: {sel_desc}")
    print(f"  local (filtered) idx: {cand_local.tolist()}")
    print(f"  global SAE idx:       {cand_global.tolist()}")

    # Sample N SHAP-split examples + background; compute local LinearSHAP.
    rng = np.random.default_rng(SEED)
    shap_acts = np.load(f"activations/shap/layer_{LAYER}/activations.npy")
    train_acts = np.load(f"activations/probe_train/layer_{LAYER}/activations.npy")
    pick = rng.choice(len(shap_acts), size=N_EXAMPLES, replace=False)
    bg_idx = rng.choice(len(train_acts), size=N_BACKGROUND, replace=False)
    shap_eval = shap_acts[pick]
    background = train_acts[bg_idx]

    print(f"Computing local LinearSHAP on {N_EXAMPLES} examples × {len(feature_indices)} features…")
    local_shap = _shap7.run_linearshap(
        sae_probe, shap_eval, background, feature_indices
    )
    print(f"  local_shap shape={local_shap.shape}")

    print("Loading model + SAE…")
    model = load_model()
    sae = SAE.from_pretrained("gpt2-small-resid-post-v5-32k", "blocks.7.hook_resid_post")
    sae = sae.to(model.cfg.device)

    _, _, shap_ds = load_splits()
    sentences = [shap_ds[int(i)]["sentence"] for i in pick]
    all_tokens = model.to_tokens(sentences)

    pos_id, neg_id = _steer9.sentiment_token_ids(model)
    print(
        f"Logit-diff readout: {_steer9.POS_TOKEN!r}({pos_id}) - "
        f"{_steer9.NEG_TOKEN!r}({neg_id}); N={N_EXAMPLES}, mode={MODE}"
    )

    delta_signed, delta_abs = get_per_example_intervention_effects(
        model,
        sae,
        all_tokens,
        cand_global,
        layer=LAYER,
        mode=MODE,
        steering_alpha=STEERING_ALPHA,
    )

    method_scores_f = {
        "global_SHAP": shap_global_f,
        "Probe weights": probe_f,
        "IG": ig_f,
        "GA": ga_f,
    }
    faith = per_example_faithfulness(
        method_scores_f, local_shap, cand_local, delta_signed
    )
    summary = summarize_rhos(faith)
    text = print_summary(summary, MODE, SELECTION, K, N_EXAMPLES)
    print("\n" + text)

    # Mean |Δ| ranking preview (averaged only for display, not for ρ).
    mean_abs = delta_abs.mean(axis=0)
    mean_signed = delta_signed.mean(axis=0)
    print("Top candidates by mean_|Δ| (display only):")
    for j in np.argsort(mean_abs)[::-1][:10]:
        print(
            f"  feature {int(cand_global[j])}: |Δ̄|={mean_abs[j]:.4f}  "
            f"Δ̄={mean_signed[j]:+.4f}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "example_indices.npy", pick)
    np.save(OUT_DIR / "background_indices.npy", bg_idx)
    np.save(OUT_DIR / "feature_indices.npy", feature_indices)
    np.save(OUT_DIR / "cand_local.npy", cand_local)
    np.save(OUT_DIR / "cand_global.npy", cand_global)
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
                "selection": SELECTION,
                "k": K,
                "n_examples": N_EXAMPLES,
                "steering_alpha": STEERING_ALPHA,
                "seed": SEED,
                **summary,
            },
            f,
            indent=2,
        )
    (OUT_DIR / "summary.txt").write_text(text + "\n")
    print(f"\nWrote outputs → {OUT_DIR}/")


if __name__ == "__main__":
    main()
