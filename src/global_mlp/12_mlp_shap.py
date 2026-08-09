"""
DeepSHAP attributions for the filtered-feature MLP probe.

Same structure as secondary/7_shap_recompute.py (mean signed Phi, sign
consistency, top-k), but:
  - Explains the saved MLP probe (checkpoints/mlp_probe_layer_*.joblib)
  - Uses only the probe's filtered SAE feature indices
  - Uses shap.DeepExplainer on a PyTorch clone of the sklearn MLP
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import shap
import torch
import torch.nn as nn
from sklearn.neural_network import MLPClassifier


def prepare_shap_inputs(
    activations_dir="activations",
    layer=7,
    n_background=100,
    n_shap=500,
    seed=42,
):
    """
    Sample background (train) and eval (SHAP held-out) activations.

    Returns full-width tensors of shape (n, 32768); caller slices to filtered
    features.
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


class TorchMLPProbe(nn.Module):
    """
    PyTorch clone of a fitted sklearn MLPClassifier with one hidden ReLU layer
    and a single logistic output (binary).

    Implemented as nn.Sequential so shap.DeepExplainer's PyTorch backend
    recognizes Linear/ReLU ops (custom forwards break additivity).
    """

    def __init__(self, probe: MLPClassifier):
        super().__init__()
        if len(probe.coefs_) != 2:
            raise ValueError(
                f"Expected a 1-hidden-layer MLP; got {len(probe.coefs_)} weight matrices"
            )
        W1 = torch.tensor(probe.coefs_[0], dtype=torch.float32)  # (n_in, hidden)
        b1 = torch.tensor(probe.intercepts_[0], dtype=torch.float32)
        W2 = torch.tensor(probe.coefs_[1], dtype=torch.float32)  # (hidden, 1)
        b2 = torch.tensor(probe.intercepts_[1], dtype=torch.float32)

        self.net = nn.Sequential(
            nn.Linear(W1.shape[0], W1.shape[1]),
            nn.ReLU(),
            nn.Linear(W2.shape[0], W2.shape[1]),
        )
        with torch.no_grad():
            self.net[0].weight.copy_(W1.T)
            self.net[0].bias.copy_(b1)
            self.net[2].weight.copy_(W2.T)
            self.net[2].bias.copy_(b2.reshape(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Logit (pre-sigmoid); DeepSHAP is unreliable on probability outputs.
        return self.net(x)


def run_deepshap(
    probe: MLPClassifier,
    shap_eval: np.ndarray,
    background: np.ndarray,
    feature_indices: np.ndarray,
) -> np.ndarray:
    """
    DeepSHAP on filtered features for the MLP probe.

    Explains the positive-class *logit* (pre-sigmoid), which DeepExplainer
    handles more reliably than probabilities.

    Returns shap_values of shape (n_eval, n_filtered_features).
    """
    shap_eval_filtered = np.asarray(shap_eval[:, feature_indices], dtype=np.float32)
    background_filtered = np.asarray(background[:, feature_indices], dtype=np.float32)

    if shap_eval_filtered.shape[1] != probe.n_features_in_:
        raise ValueError(
            f"Filtered width {shap_eval_filtered.shape[1]} != "
            f"probe.n_features_in_={probe.n_features_in_}"
        )

    model = TorchMLPProbe(probe)
    model.eval()

    bg_t = torch.tensor(background_filtered, dtype=torch.float32)
    eval_t = torch.tensor(shap_eval_filtered, dtype=torch.float32)

    explainer = shap.DeepExplainer(model, bg_t)
    shap_values = explainer.shap_values(eval_t, check_additivity=True)

    # DeepExplainer may return a list (per-output) or an array.
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)
    # Squeeze trailing singleton output dim if present: (n, F, 1) → (n, F)
    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
        shap_values = shap_values[..., 0]

    return shap_values


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
    checkpoint_path: str = "checkpoints/mlp_probe_layer_7.joblib",
    out_dir: str = "outputs/12_mlp_shap",
    layer: int = 7,
):
    payload = joblib.load(checkpoint_path)
    probe: MLPClassifier = payload["probe"]
    feature_indices = np.asarray(payload["feature_indices"])
    print(
        f"Loaded MLP probe from {checkpoint_path}  "
        f"(n_filtered={len(feature_indices)}, hidden={payload.get('hidden_dim')})"
    )

    shap_eval, background, bg_idx = prepare_shap_inputs(layer=layer)
    print(
        f"DeepSHAP on {len(shap_eval)} eval / {len(background)} background "
        f"examples × {len(feature_indices)} features"
    )

    shap_values = run_deepshap(probe, shap_eval, background, feature_indices)
    print(f"shap_values shape: {shap_values.shape}")

    Phi = build_attribution_matrix(shap_values, feature_indices)
    consistency = sign_consistency_scores(shap_values, feature_indices)
    top_k_idx, top_k_Phi = get_top_shap_features(Phi)

    print("Top features by |mean signed DeepSHAP|:")
    for idx, phi in zip(top_k_idx[:10], top_k_Phi[:10]):
        print(
            f"  feature {idx:5d}:  Φ={phi:+.6f}  "
            f"sign_consistency={consistency[idx]:.3f}"
        )

    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/shap_values_raw.npy", shap_values)
    np.save(f"{out_dir}/shap_feature_indices.npy", feature_indices)
    np.save(f"{out_dir}/phi_sentiment_layer{layer}_signed.npy", Phi)
    np.save(f"{out_dir}/sign_consistency_layer{layer}.npy", consistency)
    np.save(f"{out_dir}/top_k_idx_layer{layer}.npy", top_k_idx)
    np.save(f"{out_dir}/top_k_Phi_layer{layer}.npy", top_k_Phi)
    np.save(f"{out_dir}/background_indices_layer{layer}.npy", bg_idx)
    print(f"\nWrote outputs → {out_dir}/")


if __name__ == "__main__":
    main()
