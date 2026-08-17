"""
Multi-seed replication of the MLP arm's global faithfulness result.

Mirror of global_linear/9a_seed_sweep.py — same union-then-slice strategy, same
statistics, same CLI — but scores the MLP arm's filtered attributions. See that
module's docstring for why a single k=20 seed draw is not enough to support a
null claim.

The distributional statistics (`parse_seeds`, `bootstrap_ci`, `aggregate`,
`render`) are imported from the linear arm's copy rather than duplicated: they
are arm-independent, and the two arms must summarise their sweeps identically or
the comparison between them is not like-for-like.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sae_lens import SAE

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils import (
    benjamini_hochberg,
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


_steer14 = _load("mlp_steer14", _SRC / "global_mlp" / "14_mlp_steer.py")
_sweep9a = _load("seed_sweep9a", _SRC / "global_linear" / "9a_seed_sweep.py")

LAYER = 7
RANKINGS_DIR = output_path("13_mlp_ranking")
UNSIGNED = frozenset({"Probe saliency"})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # Held identical to global_linear/9a_seed_sweep.py.
    p.add_argument("--mode", choices=("steering", "ablation"), default="steering")
    p.add_argument("--alpha", type=float, default=0.3316)
    p.add_argument("--alpha-mode", choices=("constant", "scaled"), default="scaled")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--seeds", type=str, default="0-9")
    p.add_argument("--ref-seed", type=int, default=0)
    p.add_argument("--n-eval", type=int, default=_steer14.N_STEER_EVAL)
    p.add_argument("--out-dir", type=Path, default=output_path("14_mlp_seed_sweep"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = _sweep9a.parse_seeds(args.seeds)
    mode = args.mode

    feature_indices = np.load(RANKINGS_DIR / "feature_indices.npy")
    ig_scores = np.load(RANKINGS_DIR / "ig_scores.npy")
    ga_scores = np.load(RANKINGS_DIR / "ga_scores.npy")
    probe_scores = np.load(RANKINGS_DIR / "probe_scores.npy")
    shap_scores = np.load(RANKINGS_DIR / "shap_scores.npy")

    per_seed_global: dict[int, np.ndarray] = {}
    for s in seeds:
        _, g = _steer14.get_shap_candidates(
            shap_scores, feature_indices, k=args.k, selection="random", seed=s
        )
        per_seed_global[s] = g
    union_global = np.unique(np.concatenate([per_seed_global[s] for s in seeds]))
    print(
        f"{len(seeds)} seeds x k={args.k} -> union of {len(union_global)} distinct "
        f"features (of {len(feature_indices)} in the pool)"
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

    delta_abs, delta_signed = _steer14.get_model_intervention_effects(
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
        "SHAP": _steer14.expand_to_full(shap_scores, feature_indices),
        "Probe saliency": _steer14.expand_to_full(probe_scores, feature_indices),
        "IG": _steer14.expand_to_full(ig_scores, feature_indices),
        "GA": _steer14.expand_to_full(ga_scores, feature_indices),
    }

    per_seed: dict[str, dict] = {}
    for s in seeds:
        feats = [int(f) for f in per_seed_global[s]]
        da = {f: delta_abs[f] for f in feats}
        ds = {f: delta_signed[f] for f in feats}
        per_seed[str(s)] = _steer14.faithfulness_stats(
            method_scores, da, ds, mode, UNSIGNED
        )

    agg = _sweep9a.aggregate(per_seed, seeds, args.ref_seed)

    pooled_p = {
        f"seed{s}:{k}": per_seed[str(s)][k]["p"] for s in seeds for k in per_seed[str(s)]
    }
    pooled_bh = benjamini_hochberg(pooled_p)
    n_rej = sum(r["reject"] for r in pooled_bh.values())

    tag = f"{mode}_{args.alpha_mode if mode == 'steering' else 'na'}_k{args.k}"
    header = (
        f"MLP arm - multi-seed faithfulness sweep\n"
        f"mode={mode}  selection=random  k={args.k}  seeds={args.seeds}  "
        f"n_eval={len(steer_eval_idx)}\n"
        f"alpha: {alpha_desc}\n"
        f"union features intervened on: {len(union_global)}\n"
        f"Probe saliency is magnitude-only: importance test only, no directional row.\n"
        f"Pooled Benjamini-Hochberg over all {len(pooled_bh)} tests: "
        f"{n_rej} survive at FDR 0.05."
    )
    text = _sweep9a.render(agg, per_seed, seeds, header)
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
