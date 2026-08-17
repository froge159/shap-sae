"""
Multi-seed replication of the linear arm's global faithfulness result.

Why this exists
---------------
The headline global faithfulness numbers come from a single `--selection random
--seed 0` draw of k=20 features out of the filtered pool. At n=20 the confidence
interval on a Spearman rho spans most of [-1, 1], so "rho is near zero" from one
draw is compatible with both a true null and a real moderate effect. This script
re-runs the same measurement across many seeds and reports the *distribution* of
rho, which is what the null claim actually needs.

How it stays cheap
------------------
Naively this is `9_steer.py --seed s` in a loop, which reloads GPT-2 + the SAE and
recomputes the reconstruction baseline once per seed. Instead: draw every seed's
candidate set with the *unmodified* `9_steer.get_shap_candidates`, take the union,
and run `9_steer.get_model_intervention_effects` once over that union. With k=20,
a 116-feature pool and 10 seeds the union is ~99 features, so this costs about
half of ten independent runs — the same union-then-slice trick the local scripts
already use.

Per-seed faithfulness is then computed by slicing the shared delta dicts, so each
seed's numbers are identical to what a standalone `9_steer.py --seed s` would have
produced (same rows, same alphas, same baseline).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import wilcoxon
from sae_lens import SAE

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils import (
    CANONICAL_SEED,
    benjamini_hochberg,
    checkpoint_path,
    eval_sentences,
    load_model,
    output_path,
    resolve_steering_alphas,
    sample_eval_rows,
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_steer9 = _load("steer9", _SRC / "global_linear" / "9_steer.py")

LAYER = 7
N_BOOTSTRAP = 5000


def parse_seeds(spec: str) -> list[int]:
    """Accept '0-9', '0,3,7' or a mix of both."""
    seeds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    if not seeds:
        raise ValueError(f"No seeds parsed from {spec!r}")
    return seeds


def bootstrap_ci(
    values: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = CANONICAL_SEED
) -> tuple[float, float]:
    """
    Percentile bootstrap CI for the mean, resampling over *seeds*.

    The resampling unit is the seed, not the example: the question is "across
    arbitrary candidate draws of this size, where does rho sit", so each seed's
    rho is one observation.
    """
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def aggregate(per_seed: dict[str, dict], seeds: list[int], ref_seed: int) -> dict:
    """
    Collapse per-seed rho into a distributional summary per (method, kind).

    Three numbers per series:
      - wilcoxon: is the typical rho across draws different from 0? (signed-rank,
        not a t-test: n is ~10 and Spearman rho is bounded, so normality is not
        a safe assumption)
      - bootstrap CI on the mean rho
      - seed0_outlier_p: how extreme the reference seed's |rho| is within this
        set of draws. An exchangeability p-value, so it says "the headline draw
        was/was not typical", not "the effect is significant".
    """
    keys = list(next(iter(per_seed.values())).keys())
    out: dict[str, dict] = {}
    for key in keys:
        rhos = np.asarray(
            [per_seed[str(s)][key]["rho"] for s in seeds], dtype=np.float64
        )
        finite = rhos[np.isfinite(rhos)]
        row: dict = {
            "n_seeds": int(len(finite)),
            "mean": float(np.mean(finite)) if len(finite) else float("nan"),
            "std": float(np.std(finite)) if len(finite) else float("nan"),
            "min": float(np.min(finite)) if len(finite) else float("nan"),
            "max": float(np.max(finite)) if len(finite) else float("nan"),
        }
        if len(finite) >= 2 and np.any(finite != 0):
            stat, pval = wilcoxon(finite)
            row["wilcoxon_stat"] = float(stat)
            row["wilcoxon_p"] = float(pval)
        else:
            row["wilcoxon_stat"] = float("nan")
            row["wilcoxon_p"] = float("nan")
        lo, hi = bootstrap_ci(finite)
        row["bootstrap_ci95"] = [lo, hi]
        ref = per_seed[str(ref_seed)][key]["rho"] if str(ref_seed) in per_seed else np.nan
        row["ref_seed"] = int(ref_seed)
        row["ref_seed_rho"] = float(ref)
        row["ref_seed_outlier_p"] = (
            float(np.mean(np.abs(finite) >= abs(ref)))
            if len(finite) and np.isfinite(ref)
            else float("nan")
        )
        out[key] = row
    return out


def render(agg: dict, per_seed: dict, seeds: list[int], header: str) -> str:
    lines = [header, ""]
    lines.append(
        f"{'test':28s} {'mean rho':>9s} {'std':>7s} {'95% CI':>18s} "
        f"{'wilcoxon p':>11s} {'ref rho':>8s} {'ref p':>7s}"
    )
    lines.append("-" * 95)
    for key, row in agg.items():
        ci = f"[{row['bootstrap_ci95'][0]:+.3f},{row['bootstrap_ci95'][1]:+.3f}]"
        lines.append(
            f"{key:28s} {row['mean']:+9.3f} {row['std']:7.3f} {ci:>18s} "
            f"{row['wilcoxon_p']:11.4f} {row['ref_seed_rho']:+8.3f} "
            f"{row['ref_seed_outlier_p']:7.3f}"
        )
    lines.append("")
    lines.append(
        "Reading: 'wilcoxon p' tests whether the typical rho across draws differs "
        "from 0.\n'ref p' is the fraction of draws with |rho| at least as large as "
        "the reference\nseed's - a small value means the headline run was an "
        "unusually extreme draw."
    )
    lines.append("")
    lines.append("Per-seed rho:")
    keys = list(agg)
    lines.append("  " + f"{'seed':>5s} " + " ".join(f"{k[:14]:>14s}" for k in keys))
    for s in seeds:
        row = per_seed[str(s)]
        lines.append(
            "  "
            + f"{s:5d} "
            + " ".join(f"{row[k]['rho']:+14.3f}" for k in keys)
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("steering", "ablation"), default="steering")
    p.add_argument("--alpha", type=float, default=0.3316)
    p.add_argument("--alpha-mode", choices=("constant", "scaled"), default="scaled")
    p.add_argument("--k", type=int, default=20)
    p.add_argument(
        "--seeds",
        type=str,
        default="0-9",
        help="Seeds for --selection random, e.g. '0-9' or '0,1,5'",
    )
    p.add_argument(
        "--ref-seed",
        type=int,
        default=0,
        help="The seed whose run is the published headline (for the outlier check)",
    )
    p.add_argument("--n-eval", type=int, default=_steer9.N_STEER_EVAL)
    p.add_argument("--out-dir", type=Path, default=output_path("9_seed_sweep"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    mode = args.mode

    feature_indices = np.load(output_path("3_shap", "shap_feature_indices.npy"))
    sae_probe = joblib.load(checkpoint_path("probe_layer_7.joblib"))
    probe_scores = sae_probe.coef_[0]
    ig_scores = np.load(output_path("8_rankings_recompute", "ig_scores.npy"))
    ga_scores = np.load(output_path("8_rankings_recompute", "ga_scores.npy"))
    shap_scores = np.load(
        output_path("7_shap_recompute", "phi_sentiment_layer7_signed.npy")
    )
    shap_filtered = shap_scores[feature_indices]

    # Candidate set per seed, drawn exactly as 9_steer would have.
    per_seed_global: dict[int, np.ndarray] = {}
    for s in seeds:
        _, g = _steer9.get_shap_candidates(
            shap_filtered, feature_indices, k=args.k, selection="random", seed=s
        )
        per_seed_global[s] = g
    union_global = np.unique(np.concatenate([per_seed_global[s] for s in seeds]))
    print(
        f"{len(seeds)} seeds x k={args.k} -> union of {len(union_global)} distinct "
        f"features (of {len(feature_indices)} in the pool)"
    )
    print(
        f"  intervening once over the union instead of {len(seeds)} separate runs "
        f"({len(union_global)} vs {len(seeds) * args.k} feature passes)"
    )

    model = load_model()
    sae = SAE.from_pretrained(
        "gpt2-small-resid-post-v5-32k", f"blocks.{LAYER}.hook_resid_post"
    )
    sae = sae.to(model.cfg.device)

    steer_eval_idx, _ = sample_eval_rows(layer=LAYER, n_eval=args.n_eval)
    steer_tokens = model.to_tokens(eval_sentences(steer_eval_idx))
    print(f"Steering eval: {len(steer_eval_idx)} held-out rows")

    if mode == "steering":
        alpha_by_feature, alpha_desc = resolve_steering_alphas(
            union_global, args.alpha, args.alpha_mode
        )
    else:
        alpha_by_feature, alpha_desc = None, "n/a (ablation zeroes the feature)"
    print(f"Steering strength: {alpha_desc}")

    delta_abs, delta_signed = _steer9.get_model_intervention_effects(
        model,
        sae,
        steer_tokens,
        union_global,
        layer=LAYER,
        mode=mode,
        steering_alpha=args.alpha,
        alpha_by_feature=alpha_by_feature,
    )

    method_scores = {
        "SHAP": shap_scores,
        "Probe weights": probe_scores,
        "IG": ig_scores,
        "GA": ga_scores,
    }

    # Slice the shared deltas per seed: identical to what a standalone run
    # with that seed would have scored.
    per_seed: dict[str, dict] = {}
    for s in seeds:
        feats = [int(f) for f in per_seed_global[s]]
        da = {f: delta_abs[f] for f in feats}
        ds = {f: delta_signed[f] for f in feats}
        per_seed[str(s)] = _steer9.faithfulness_stats(method_scores, da, ds, mode)

    agg = aggregate(per_seed, seeds, args.ref_seed)

    # BH across the whole sweep: every (method, kind, seed) test at once. This is
    # the honest family when the claim is "nothing came out significant anywhere".
    pooled_p = {
        f"seed{s}:{k}": per_seed[str(s)][k]["p"] for s in seeds for k in per_seed[str(s)]
    }
    pooled_bh = benjamini_hochberg(pooled_p)
    n_rej = sum(r["reject"] for r in pooled_bh.values())

    tag = f"{mode}_{args.alpha_mode if mode == 'steering' else 'na'}_k{args.k}"
    header = (
        f"Linear (logistic) arm - multi-seed faithfulness sweep\n"
        f"mode={mode}  selection=random  k={args.k}  seeds={args.seeds}  "
        f"n_eval={len(steer_eval_idx)}\n"
        f"alpha: {alpha_desc}\n"
        f"union features intervened on: {len(union_global)}\n"
        f"Pooled Benjamini-Hochberg over all {len(pooled_bh)} tests "
        f"({len(seeds)} seeds x {len(per_seed[str(seeds[0])])} tests): "
        f"{n_rej} survive at FDR 0.05."
    )
    text = render(agg, per_seed, seeds, header)
    print(text)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"seed_sweep_{tag}.json", "w") as f:
        json.dump(
            {
                "mode": mode,
                "alpha": args.alpha,
                "alpha_mode": args.alpha_mode if mode == "steering" else None,
                "k": args.k,
                "seeds": seeds,
                "ref_seed": args.ref_seed,
                "n_eval": int(len(steer_eval_idx)),
                "n_union_features": int(len(union_global)),
                "per_seed": per_seed,
                "aggregate": agg,
                "pooled_bh": {
                    "family_size": len(pooled_bh),
                    "n_reject": int(n_rej),
                    "tests": pooled_bh,
                },
            },
            f,
            indent=2,
        )
    (out_dir / f"seed_sweep_{tag}.txt").write_text(text)
    np.save(out_dir / f"union_global_{tag}.npy", union_global)
    for s in seeds:
        np.save(out_dir / f"candidates_seed{s}_{tag}.npy", per_seed_global[s])
    print(f"Wrote outputs -> {out_dir}/  (tag: {tag})")


if __name__ == "__main__":
    main()
