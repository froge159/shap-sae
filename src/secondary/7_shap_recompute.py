"""
Recompute SHAP with LinearExplainer (exact for the linear probe).

Differences from core/3_shap_comp.py:
  - LinearExplainer instead of KernelExplainer
  - Phi = mean signed SHAP (not mean absolute)
  - Also reports per-feature sign-consistency (% of examples agreeing on sign)
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import shap
from sklearn.linear_model import LogisticRegression


def prepare_shap_inputs(
    activations_dir="activations",
    layer=7,
    n_background=100,
    n_shap=500,
    seed=42,
):
    """
    activations: shape (N_sentences, 32768)
    """
    rng = np.random.default_rng(seed)

    train_activations = np.load(
        f"{activations_dir}/probe_train/layer_{layer}/activations.npy"
    )
    shap_activations = np.load(
        f"{activations_dir}/shap/layer_{layer}/activations.npy"
    )

    bg_idx = rng.choice(len(train_activations), size=n_background, replace=False)
    background = train_activations[bg_idx]

    shap_idx = rng.choice(len(shap_activations), size=n_shap, replace=False)
    shap_eval = shap_activations[shap_idx]

    return shap_eval, background, bg_idx


def get_shap_feature_mask(
    probe, activations: np.ndarray, min_activation_freq: float = 0.05
):
    nonzero_mask = probe.coef_[0] != 0
    activation_freq = (activations > 0).mean(axis=0)
    freq_mask = activation_freq >= min_activation_freq

    combined_mask = nonzero_mask & freq_mask
    feature_indices = np.where(combined_mask)[0]

    print(f"Non-zero probe weights: {nonzero_mask.sum()}")
    print(f"After frequency filter: {len(feature_indices)}")

    return feature_indices


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

    explainer = shap.LinearExplainer(filtered_probe, background_filtered)
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


def main(checkpoint_dir: str = "checkpoints", out_dir: str = "outputs/7_shap_recompute"):
    probe = joblib.load(f"{checkpoint_dir}/probe_layer_7.joblib")
    probe.verbose = 0

    shap_eval, background, bg_idx = prepare_shap_inputs()
    feature_indices = get_shap_feature_mask(probe, shap_eval)
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
    np.save(f"{out_dir}/shap_feature_indices.npy", feature_indices)
    np.save(f"{out_dir}/phi_sentiment_layer7_signed.npy", Phi)
    np.save(f"{out_dir}/sign_consistency_layer7.npy", consistency)
    np.save(f"{out_dir}/top_k_idx_layer7.npy", top_k_idx)
    np.save(f"{out_dir}/top_k_Phi_layer7.npy", top_k_Phi)
    np.save(f"{out_dir}/background_indices_layer7.npy", bg_idx)
    print(f"\nWrote outputs → {out_dir}/")


if __name__ == "__main__":
    main()
