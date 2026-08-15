"""
Recompute SHAP with LinearExplainer (exact for the linear probe).

Historical differences from core/3_shap_comp.py's original KernelSHAP pass
(that computation was later removed from 3_shap_comp.py — nothing downstream
read its outputs — which now only computes and saves the feature mask):
  - LinearExplainer instead of KernelExplainer
  - Explains the **log-odds margin**, not `predict_proba` — magnitudes were on
    a different scale from the old Φ and were not interchangeable with it
    (rank correlations survive the monotone link; effect sizes do not)
  - Phi = mean signed SHAP (not mean absolute)
  - Also reports per-feature sign-consistency (% of examples agreeing on sign)
  - Deliberately explains **all** 32768 features rather than the
    `get_shap_feature_mask` subset: LinearExplainer is exact and cheap, so there
    is no reason to filter. Downstream consumers that need the filtered pool
    slice this Φ by `3_shap/shap_feature_indices.npy`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import shap
from sklearn.linear_model import LogisticRegression

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils import (
    DEFAULT_N_EVAL,
    checkpoint_path,
    load_eval_and_background,
    output_path,
)

# The feature mask lives in utils.get_shap_feature_mask; do not re-derive it here.


def make_filtered_probe(probe: LogisticRegression, feature_indices: np.ndarray):
    """LogisticRegression with coefs restricted to `feature_indices` for LinearExplainer."""
    filtered = LogisticRegression()
    filtered.coef_ = probe.coef_[:, feature_indices].copy()
    filtered.intercept_ = probe.intercept_.copy()
    filtered.classes_ = probe.classes_.copy()
    return filtered


def run_linearshap(
    probe,
    shap_eval: np.ndarray,
    background: np.ndarray,
    feature_indices: np.ndarray,
) -> np.ndarray:
    """
    Exact interventional SHAP for the linear probe on filtered features.

    Returns shap_values of shape (n_eval, n_filtered_features).
    """
    shap_eval_filtered = shap_eval[:, feature_indices]
    background_filtered = background[:, feature_indices]
    filtered_probe = make_filtered_probe(probe, feature_indices)

    # Build the masker explicitly. Handing LinearExplainer a raw array makes it
    # construct `maskers.Independent(bg)`, whose default `max_samples=100` would
    # silently take an *unseeded* random subsample of any background larger than
    # 100 rows. Passing the full count pins it whatever DEFAULT_N_BACKGROUND is.
    masker = shap.maskers.Independent(
        background_filtered, max_samples=len(background_filtered)
    )
    explainer = shap.LinearExplainer(filtered_probe, masker)
    shap_values = explainer.shap_values(shap_eval_filtered)

    return np.asarray(shap_values)


def build_attribution_matrix(
    shap_values: np.ndarray,
    feature_indices: np.ndarray,
    n_total_features: int = 32768,
) -> np.ndarray:
    """
    Phi of shape (n_total_features,) — mean *signed* SHAP per feature.
    Unfiltered features get 0.
    """
    mean_signed_shap = shap_values.mean(axis=0)

    Phi = np.zeros(n_total_features)
    Phi[feature_indices] = mean_signed_shap
    return Phi


def sign_consistency_scores(
    shap_values: np.ndarray,
    feature_indices: np.ndarray,
    n_total_features: int = 32768,
) -> np.ndarray:
    """
    Per-feature fraction of examples that share the majority sign.

    score_j = max(#pos, #neg) / n_examples
    Unfiltered features get 0.
    """
    n = shap_values.shape[0]
    n_pos = (shap_values > 0).sum(axis=0)
    n_neg = (shap_values < 0).sum(axis=0)
    consistency_filtered = np.maximum(n_pos, n_neg) / n

    consistency = np.zeros(n_total_features)
    consistency[feature_indices] = consistency_filtered
    return consistency


def get_top_shap_features(Phi: np.ndarray, k: int = 50):
    """Top-k by |signed mean SHAP| (magnitude of directional attribution)."""
    top_k_idx = np.argsort(np.abs(Phi))[::-1][:k]
    return top_k_idx, Phi[top_k_idx]


def main(
    checkpoint_dir: Path | None = None,
    out_dir: Path | None = None,
    layer: int = 7,
    n_shap: int = DEFAULT_N_EVAL,
):
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else checkpoint_path()
    out_dir = Path(out_dir) if out_dir else output_path("7_shap_recompute")

    probe = joblib.load(checkpoint_dir / f"probe_layer_{layer}.joblib")
    probe.verbose = 0

    # LinearExplainer is exact and cheap, so this can afford far more rows than
    # KernelSHAP. Rows come from the shared sampler so IG/GA in 8_feature_ranking
    # score the same examples.
    shap_eval, background, eval_idx, bg_idx = load_eval_and_background(
        layer=layer, n_eval=n_shap
    )
    print(f"LinearSHAP on {len(shap_eval)} held-out rows / {len(background)} background")

    feature_indices = np.arange(shap_eval.shape[1])
    shap_values = run_linearshap(probe, shap_eval, background, feature_indices)

    Phi = build_attribution_matrix(shap_values, feature_indices)
    consistency = sign_consistency_scores(shap_values, feature_indices)
    top_k_idx, top_k_Phi = get_top_shap_features(Phi)

    print("Top features by |mean signed SHAP|:")
    for idx, phi in zip(top_k_idx[:10], top_k_Phi[:10]):
        print(
            f"  feature {idx:5d}:  Φ={phi:+.6f}  "
            f"sign_consistency={consistency[idx]:.3f}"
        )

    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/shap_values_raw.npy", shap_values)
    # NOT a feature mask: this is the set of columns explained, which here is all
    # 32768. Deliberately not named `shap_feature_indices.npy` — that name is
    # taken by the 75-entry filtered mask in 3_shap/ and 12_mlp_shap/, and three
    # files sharing one name across two index spaces is how a mask/space mix-up
    # gets written.
    np.save(f"{out_dir}/explained_feature_indices.npy", feature_indices)
    np.save(f"{out_dir}/phi_sentiment_layer7_signed.npy", Phi)
    np.save(f"{out_dir}/sign_consistency_layer7.npy", consistency)
    np.save(f"{out_dir}/top_k_idx_layer7.npy", top_k_idx)
    np.save(f"{out_dir}/top_k_Phi_layer7.npy", top_k_Phi)
    np.save(f"{out_dir}/background_indices_layer7.npy", bg_idx)
    np.save(f"{out_dir}/eval_indices_layer7.npy", eval_idx)
    print(f"\nWrote outputs → {out_dir}/")


if __name__ == "__main__":
    main()
