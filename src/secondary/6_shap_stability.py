from __future__ import annotations

import importlib.util
import os
import sys
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import spearmanr

# Allow `from utils import ...` and load core/3_shap_comp.py (numeric filename).
_SRC = Path(__file__).resolve().parents[1]
_ROOT = _SRC.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _load_shap_comp():
    path = _SRC / "core" / "3_shap_comp.py"
    spec = importlib.util.spec_from_file_location("shap_comp", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shap_comp = _load_shap_comp()


def run_shap_with_seed(
    probe,
    shap_eval: np.ndarray,
    background: np.ndarray,
    feature_indices: np.ndarray,
    seed: int,
    n_shap_samples: int = 256,
) -> np.ndarray:
    """One KernelSHAP run → Φ (n_total_features,)."""
    shap_values = shap_comp.run_kernelshap(
        probe,
        shap_eval,
        background,
        feature_indices,
        n_shap_samples=n_shap_samples,
        seed=seed,
        silent=False,
    )
    return shap_comp.build_attribution_matrix(shap_values, feature_indices)


def pairwise_rank_correlations(phis: np.ndarray, feature_indices: np.ndarray):
    """
    Spearman ρ between every pair of runs, scored only on filtered features.

    phis: (n_runs, n_total_features)
    returns: list of (i, j, rho, p), mean_rho, std_rho
    """
    pairs = []
    for i, j in combinations(range(len(phis)), 2):
        a = phis[i, feature_indices]
        b = phis[j, feature_indices]
        rho, p = spearmanr(a, b)
        pairs.append((i, j, float(rho), float(p)))

    rhos = np.array([r for _, _, r, _ in pairs])
    return pairs, float(rhos.mean()), float(rhos.std(ddof=1) if len(rhos) > 1 else 0.0)


def main(
    checkpoint_dir: str = "checkpoints",
    layer: int = 7,
    data_seed: int = 42,
    explainer_seeds: list[int] | None = None,
    n_background: int = 100,
    n_shap: int = 500,
    n_shap_samples: int = 256,
    out_dir: str = "outputs/6_shap_stability",
):
    if explainer_seeds is None:
        explainer_seeds = [42, 43, 44, 45, 46]

    probe = joblib.load(f"{checkpoint_dir}/probe_layer_{layer}.joblib")
    probe.verbose = 0

    # Fixed data so seed variance is from KernelSHAP sampling only.
    shap_eval, background, bg_idx = shap_comp.prepare_shap_inputs(
        layer=layer,
        n_background=n_background,
        n_shap=n_shap,
        seed=data_seed,
    )
    feature_indices = shap_comp.get_shap_feature_mask(probe, shap_eval)

    phis = []
    for seed in explainer_seeds:
        print(f"\n=== KernelSHAP seed={seed} ===")
        phi = run_shap_with_seed(
            probe,
            shap_eval,
            background,
            feature_indices,
            seed=seed,
            n_shap_samples=n_shap_samples,
        )
        phis.append(phi)

    phis = np.stack(phis, axis=0)  # (n_runs, 32768)
    pairs, mean_rho, std_rho = pairwise_rank_correlations(phis, feature_indices)

    print("\nInter-run Spearman rank correlation (Φ on filtered features):")
    for i, j, rho, p in pairs:
        print(
            f"  seed {explainer_seeds[i]} vs {explainer_seeds[j]}: "
            f"ρ={rho:.4f}  (p={p:.2e})"
        )
    print(f"\nMean pairwise ρ: {mean_rho:.4f} ± {std_rho:.4f}")

    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/phis.npy", phis)
    np.save(f"{out_dir}/feature_indices.npy", feature_indices)
    np.save(f"{out_dir}/explainer_seeds.npy", np.array(explainer_seeds))
    np.save(f"{out_dir}/background_indices.npy", bg_idx)
    np.save(
        f"{out_dir}/pairwise_rho.npy",
        np.array([(i, j, rho, p) for i, j, rho, p in pairs], dtype=np.float64),
    )

    summary_path = Path(out_dir) / "stability_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"data_seed={data_seed}\n")
        f.write(f"explainer_seeds={explainer_seeds}\n")
        f.write(f"n_background={n_background} n_shap={n_shap} nsamples={n_shap_samples}\n")
        f.write(f"n_filtered_features={len(feature_indices)}\n")
        f.write(f"mean_pairwise_spearman_rho={mean_rho:.6f}\n")
        f.write(f"std_pairwise_spearman_rho={std_rho:.6f}\n")
        for i, j, rho, p in pairs:
            f.write(
                f"seed_{explainer_seeds[i]}_vs_{explainer_seeds[j]} "
                f"rho={rho:.6f} p={p:.6e}\n"
            )
    print(f"\nWrote summary → {summary_path}")


if __name__ == "__main__":
    main()
