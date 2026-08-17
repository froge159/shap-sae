"""
Local steering faithfulness for the linear (logistic) SAE probe.

Parallel to global_linear/9_steer.py, but:
  - Attributions are per-sentence LinearSHAP φ(x) on the filtered feature set
  - Candidates are per-example: top-k by |φ(x)| ∪ top-k by |Φ| (same recipe as
    local_shap_faithfulness.py) so local and global compete on an equal footing
  - Intervention effects Δ(x, i) are kept per example (never dataset-averaged
    before correlating)
  - Reports mean Spearman ρ of local φ vs Δ, and of global Φ / probe / IG / GA
    vs the same per-example Δ — separating local SHAP soundness from errors
    introduced by averaging into Φ

CLI defaults are held identical to local/14_local_mlp_steer.py (k_local =
k_global = 20, n_examples = 100). They used to be module constants that had
drifted apart — this arm on k=10, the MLP arm on k=20 — which changes both the
union size and the per-example Spearman n, so the two local arms could not be
compared at all. Match the flags across the arms or the comparison is void.
"""

from __future__ import annotations

import argparse
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

from utils import (
    CANONICAL_SEED,
    checkpoint_path,
    eval_sentences,
    load_eval_and_background,
    load_model,
    output_path,
    resolve_steering_alphas,
)


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
# Matched to local/14_local_mlp_steer.py — do not change one arm alone.
K_LOCAL = 20
K_GLOBAL = 20
# Sampling seed is owned by utils.sample_eval_rows; recorded here so the
# results metadata reflects the rows actually used.
SEED = CANONICAL_SEED
MODE = "ablation"  # "ablation" | "steering"
STEERING_ALPHA = 0.6723
ALPHA_MODE = "scaled"
OUT_DIR = output_path("9_local_steer")


def save_local_artifacts(
    out_dir: Path,
    example_indices: np.ndarray,
    n_cands: np.ndarray,
    n_active: np.ndarray,
    faith: dict[str, dict],
    faith_active: dict[str, dict],
) -> None:
    """
    Persist only what later stats need.

    Local SHAP, per-feature Δ, and the ragged candidate lists are cheap to
    recompute; the deliverable is the per-example ρ (Wilcoxon / gap regression)
    plus the eval-row ids so results join the rest of the pipeline.
    """
    arrays: dict[str, np.ndarray] = {
        "n_cands": np.asarray(n_cands, dtype=np.int64),
        "n_active": np.asarray(n_active, dtype=np.int64),
    }
    for prefix, block_set in (("", faith), ("active_", faith_active)):
        for name, block in block_set.items():
            safe = name.replace(" ", "_")
            arrays[f"{prefix}importance_{safe}"] = block["importance_rho"]
            arrays[f"{prefix}directional_{safe}"] = block["directional_rho"]
    np.savez(out_dir / "rhos.npz", **arrays)
    np.save(out_dir / "example_indices.npy", example_indices)


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
    alpha_by_feature: dict[int, float] | None = None,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One-at-a-time SAE interventions; keep per-example Δ (no mean over dataset).

    `alpha_by_feature` gives a per-feature alpha (see
    `utils.resolve_steering_alphas`); when None every feature gets
    `steering_alpha`. Ignored under mode="ablation".

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
                model, logits, tokens, pos_id, neg_id
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
        alpha_i = (
            steering_alpha
            if alpha_by_feature is None
            else float(alpha_by_feature[idx])
        )

        def intervention_hook(
            resid_post,
            hook,
            feature_idx=idx,
            _mode=mode,
            _alpha=alpha_i,
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


def orient_signed_effects(delta_signed: np.ndarray, mode: str) -> np.ndarray:
    """
    Put signed Δ on a scale where "faithful" always means positive ρ.

    Steering adds +α, so a positive attribution predicts a positive Δ. Ablation
    *removes* the feature, so a positive attribution predicts a *negative* Δ and
    a perfectly faithful method scores ρ ≈ −1 on the raw numbers. Negating here
    keeps the two modes comparable and stops a correct result reading as failure.
    """
    if mode == "ablation":
        return -delta_signed
    if mode == "steering":
        return delta_signed
    raise ValueError(f"Unknown mode={mode!r}; expected 'ablation' or 'steering'")


def per_example_faithfulness(
    method_scores_filtered: dict[str, np.ndarray],
    local_shap: np.ndarray,
    cands_per_example: list[np.ndarray],
    union_local: np.ndarray,
    delta_signed: np.ndarray,
    mode: str,
) -> dict[str, dict]:
    """
    method_scores_filtered[name]: (n_filtered,) global scores on filtered axis
    local_shap: (n_examples, n_filtered)
    cands_per_example[e]: filtered-axis indices for example e (may be empty)
    union_local: (n_union,) filtered indices aligning delta_signed columns
    delta_signed: (n_examples, n_union), raw (un-oriented) effects
    mode: "ablation" | "steering" — sets the sign convention for the
        directional correlation, see `orient_signed_effects`
    """
    n = local_shap.shape[0]
    # filtered idx → column in union / delta
    col_of = {int(f): j for j, f in enumerate(union_local)}
    oriented = orient_signed_effects(delta_signed, mode)
    out: dict[str, dict] = {}

    scores_by_name = {"local_SHAP": None, **method_scores_filtered}
    for name, scores in scores_by_name.items():
        rho_imp, rho_dir = [], []
        for e in range(n):
            cand = np.asarray(cands_per_example[e], dtype=np.int64)
            if len(cand) < 3:
                rho_imp.append(float("nan"))
                rho_dir.append(float("nan"))
                continue
            cols = np.asarray([col_of[int(c)] for c in cand], dtype=np.int64)
            # local SHAP is per-example; every other method is one fixed vector
            s = local_shap[e, cand] if scores is None else scores[cand]
            d = oriented[e, cols]
            ri, _ = spearman_pair(s, d, importance=True)
            rd, _ = spearman_pair(s, d, importance=False)
            rho_imp.append(ri)
            rho_dir.append(rd)
        out[name] = {
            "importance_rho": np.asarray(rho_imp, dtype=np.float64),
            "directional_rho": np.asarray(rho_dir, dtype=np.float64),
        }
    return out


def restrict_to_active(
    cands_per_example: list[np.ndarray], active: np.ndarray
) -> list[np.ndarray]:
    """
    Keep only candidates the example actually activates.

    Control for the main confound in this comparison. Candidates are top-k |φ(x)|
    ∪ top-k |Φ|, and the global-Φ half is usually *inactive* in any one sentence —
    ablating an inactive feature moves the logit by exactly 0, and steering it
    barely moves it. Local |φ(x)| = |w·(x − x̄)| tracks activity by construction,
    global |Φ| does not, so a chunk of "local beats global" is really "local knows
    which features are on in this sentence". Re-scoring on the active subset asks
    whether local SHAP still wins once that advantage is removed.

    active: (n_examples, n_filtered) boolean, activation > 0.
    """
    return [
        cand[active[e, cand]] for e, cand in enumerate(cands_per_example)
    ]


def summarize_rhos(faith: dict[str, dict], prefix: str = "") -> dict[str, dict]:
    summary = {}
    for name, block in faith.items():
        for kind in ("importance_rho", "directional_rho"):
            arr = block[kind]
            key = f"{prefix}{name}:{kind}"
            with np.errstate(invalid="ignore"):
                summary[key] = {
                    "mean": float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan"),
                    "std": float(np.nanstd(arr)) if np.isfinite(arr).any() else float("nan"),
                    "n_finite": int(np.isfinite(arr).sum()),
                }
    local_imp = faith["local_SHAP"]["importance_rho"]
    global_imp = faith["global_SHAP"]["importance_rho"]
    finite = np.isfinite(local_imp) & np.isfinite(global_imp)
    summary[f"{prefix}frac_local_gt_global_importance"] = float(
        np.mean(local_imp[finite] > global_imp[finite]) if finite.any() else float("nan")
    )
    return summary


ORDER = ["local_SHAP", "global_SHAP", "Probe weights", "IG", "GA"]


def _block(summary: dict, prefix: str, mode: str) -> list[str]:
    lines = ["Importance (|attr| vs |Δ|):"]
    for name in ORDER:
        key = f"{prefix}{name}:importance_rho"
        if key not in summary:
            continue
        s = summary[key]
        lines.append(
            f"  {name:14s}  ρ={s['mean']:+.3f} ± {s['std']:.3f}  (n={s['n_finite']})"
        )
    effect_desc = "−Δ" if mode == "ablation" else "Δ"
    lines.append(f"Directional (attr vs {effect_desc}; positive = faithful):")
    for name in ORDER:
        key = f"{prefix}{name}:directional_rho"
        if key not in summary:
            continue
        s = summary[key]
        lines.append(
            f"  {name:14s}  ρ={s['mean']:+.3f} ± {s['std']:.3f}  (n={s['n_finite']})"
        )
    frac = summary.get(f"{prefix}frac_local_gt_global_importance", float("nan"))
    lines.append(
        f"Fraction examples with ρ_local_SHAP > ρ_global_SHAP (importance): {frac:.3f}"
    )
    return lines


def print_summary(
    summary: dict, mode: str, k_local: int, k_global: int, n: int, n_union: int
) -> str:
    lines = [
        "Local steering faithfulness (linear probe)",
        "=" * 50,
        f"mode={mode}  candidates=top-{k_local} local ∪ top-{k_global} global  "
        f"N={n}  n_union={n_union}",
        "",
        "[A] All candidates — mean Spearman ρ across examples (± std)",
    ]
    lines += _block(summary, "", mode)
    lines += [
        "",
        "[B] Active candidates only (control)",
        "    Inactive features move the logit by ~0 under ablation, and local |φ(x)|",
        "    tracks activity while global |Φ| does not — so part of any local win in",
        "    [A] is just 'local knows which features are on here'. [B] removes that",
        "    advantage; a local win that survives here is about ranking, not activity.",
    ]
    lines += _block(summary, "active:", mode)
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """CLI surface held identical to local/14_local_mlp_steer.py."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("ablation", "steering"), default=MODE)
    p.add_argument("--k-local", type=int, default=K_LOCAL)
    p.add_argument("--k-global", type=int, default=K_GLOBAL)
    p.add_argument("--n-examples", type=int, default=N_EXAMPLES)
    p.add_argument("--alpha", type=float, default=STEERING_ALPHA)
    p.add_argument(
        "--alpha-mode", choices=("constant", "scaled"), default=ALPHA_MODE
    )
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return p.parse_args()


def main(
    mode: str = MODE,
    k_local: int = K_LOCAL,
    k_global: int = K_GLOBAL,
    n_examples: int = N_EXAMPLES,
    steering_alpha: float = STEERING_ALPHA,
    alpha_mode: str = ALPHA_MODE,
    out_dir: Path = OUT_DIR,
) -> None:
    out_dir = Path(out_dir)
    feature_indices = np.load(output_path("3_shap", "shap_feature_indices.npy"))
    sae_probe = joblib.load(checkpoint_path("probe_layer_7.joblib"))
    sae_probe.verbose = 0

    probe_full = np.asarray(sae_probe.coef_[0], dtype=np.float64)
    ig_full = np.load(output_path("8_rankings_recompute", "ig_scores.npy")).astype(
        np.float64
    )
    ga_full = np.load(output_path("8_rankings_recompute", "ga_scores.npy")).astype(
        np.float64
    )
    shap_global_full = np.load(
        output_path("7_shap_recompute", "phi_sentiment_layer7_signed.npy")
    ).astype(np.float64)

    probe_f = probe_full[feature_indices]
    ig_f = ig_full[feature_indices]
    ga_f = ga_full[feature_indices]
    shap_global_f = shap_global_full[feature_indices]

    # Sample N SHAP-split examples + background; compute local LinearSHAP.
    # Shared sampler => these rows are an ordered prefix of the rows the global
    # scripts explain, so local and global results join by position.
    shap_eval, background, pick, _bg_idx = load_eval_and_background(
        layer=LAYER, n_eval=n_examples, n_background=N_BACKGROUND
    )

    print(
        f"Computing local LinearSHAP on {n_examples} examples × "
        f"{len(feature_indices)} features…"
    )
    local_shap = _shap7.run_linearshap(
        sae_probe, shap_eval, background, feature_indices
    )
    print(f"  local_shap shape={local_shap.shape}")

    cands_per_example, union_local = build_per_example_candidates(
        local_shap, shap_global_f, k_local, k_global
    )
    union_global = feature_indices[union_local]
    n_cands = np.asarray([len(c) for c in cands_per_example], dtype=np.int64)
    print(
        f"Per-example candidates: top-{k_local} |φ(x)| ∪ top-{k_global} |Φ|  "
        f"(mean |cand|={n_cands.mean():.1f}, union={len(union_local)})"
    )

    print("Loading model + SAE…")
    model = load_model()
    sae = SAE.from_pretrained("gpt2-small-resid-post-v5-32k", "blocks.7.hook_resid_post")
    sae = sae.to(model.cfg.device)

    sentences = eval_sentences(pick)
    all_tokens = model.to_tokens(sentences)

    pos_id, neg_id = _steer9.sentiment_token_ids(model)
    print(
        f"Logit-diff readout: {_steer9.POS_TOKEN!r}({pos_id}) - "
        f"{_steer9.NEG_TOKEN!r}({neg_id}); N={n_examples}, mode={mode}"
    )

    if mode == "steering":
        alpha_by_feature, alpha_desc = resolve_steering_alphas(
            union_global, steering_alpha, alpha_mode
        )
        print(f"Steering strength: {alpha_desc}")
    else:
        alpha_by_feature, alpha_desc = None, "n/a (ablation zeroes the feature)"

    # Intervene on the union once; ρ uses each example's own candidate subset.
    delta_signed, delta_abs = get_per_example_intervention_effects(
        model,
        sae,
        all_tokens,
        union_global,
        layer=LAYER,
        mode=mode,
        steering_alpha=steering_alpha,
        alpha_by_feature=alpha_by_feature,
    )

    method_scores_f = {
        "global_SHAP": shap_global_f,
        "Probe weights": probe_f,
        "IG": ig_f,
        "GA": ga_f,
    }
    faith = per_example_faithfulness(
        method_scores_f, local_shap, cands_per_example, union_local, delta_signed, mode
    )

    # Control for the activity confound: rescore on candidates the example
    # actually activates. See `restrict_to_active`.
    active = np.asarray(shap_eval[:, feature_indices], dtype=np.float64) > 0
    active_cands = restrict_to_active(cands_per_example, active)
    n_active = np.asarray([len(c) for c in active_cands], dtype=np.int64)
    print(
        f"Active-candidate control: mean {n_active.mean():.1f} of "
        f"{n_cands.mean():.1f} candidates fire per example"
    )
    faith_active = per_example_faithfulness(
        method_scores_f, local_shap, active_cands, union_local, delta_signed, mode
    )

    summary = {**summarize_rhos(faith), **summarize_rhos(faith_active, "active:")}
    text = print_summary(
        summary, mode, k_local, k_global, n_examples, len(union_local)
    )
    print("\n" + text)

    # Mean |Δ| ranking preview over the union (display only).
    mean_abs = delta_abs.mean(axis=0)
    mean_signed = delta_signed.mean(axis=0)
    print("Top union features by mean_|Δ| (display only):")
    for j in np.argsort(mean_abs)[::-1][:10]:
        print(
            f"  feature {int(union_global[j])}: |Δ̄|={mean_abs[j]:.4f}  "
            f"Δ̄={mean_signed[j]:+.4f}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    save_local_artifacts(out_dir, pick, n_cands, n_active, faith, faith_active)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(
            {
                "mode": mode,
                "k_local": k_local,
                "k_global": k_global,
                "n_examples": n_examples,
                "n_union": int(len(union_local)),
                "mean_n_cands": float(n_cands.mean()),
                "mean_n_active_cands": float(n_active.mean()),
                "steering_alpha": steering_alpha,
                "alpha_mode": alpha_mode if mode == "steering" else None,
                "alpha_desc": alpha_desc,
                "seed": SEED,
                **summary,
            },
            f,
            indent=2,
        )
    (out_dir / "summary.txt").write_text(text + "\n")
    print(f"\nWrote outputs → {out_dir}/")


if __name__ == "__main__":
    _args = parse_args()
    main(
        mode=_args.mode,
        k_local=_args.k_local,
        k_global=_args.k_global,
        n_examples=_args.n_examples,
        steering_alpha=_args.alpha,
        alpha_mode=_args.alpha_mode,
        out_dir=_args.out_dir,
    )
