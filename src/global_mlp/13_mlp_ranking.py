"""
Feature ranking for the filtered-feature MLP probe.

Mirrors secondary/8_feature_ranking.py (signed mean IG/GA, Spearman rank
comparisons), but:
  - Explains the saved MLP probe (checkpoints/mlp_probe_layer_*.joblib)
  - IG/GA use input-dependent gradients through ReLU (not constant ∂f/∂x = w)
  - Scores/ranks are over filtered features only; SHAP scores are Phi[indices]
"""

from __future__ import annotations

import os

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm


def mlp_prob_grad(probe: MLPClassifier, x: np.ndarray) -> np.ndarray:
    """
    ∇_x σ(logit) for a 1-hidden-layer ReLU MLP.

    x: (n_features,) or (batch, n_features)
    returns: same leading shape as x
    """
    W1, W2 = probe.coefs_
    b1, b2 = probe.intercepts_
    single = x.ndim == 1
    if single:
        x = x[None, :]

    pre = x @ W1 + b1                          # (batch, hidden)
    h = np.maximum(pre, 0.0)
    logit = h @ W2[:, 0] + b2[0]               # (batch,)
    sig = 1.0 / (1.0 + np.exp(-logit))
    d_logit_dpre = W2[:, 0] * (pre > 0)        # (batch, hidden)
    d_logit_dx = d_logit_dpre @ W1.T           # (batch, n_features)
    grad = (sig * (1.0 - sig))[:, None] * d_logit_dx

    return grad[0] if single else grad


def integrated_gradients(
    probe: MLPClassifier,
    activations: np.ndarray,
    baseline: np.ndarray,
    n_steps: int = 50,
) -> np.ndarray:
    """
    Returns mean signed IG attributions, shape (n_filtered,).
    """
    all_ig = []
    alphas = np.linspace(0, 1, n_steps)

    for x in tqdm(activations, total=len(activations), desc="Calculating IG", dynamic_ncols=True):
        interpolated = baseline + alphas[:, None] * (x - baseline)  # (n_steps, F)
        grads = mlp_prob_grad(probe, interpolated)                  # (n_steps, F)
        avg_grads = np.trapezoid(grads, alphas, axis=0)             # (F,)
        ig = (x - baseline) * avg_grads
        all_ig.append(ig)

    all_ig = np.array(all_ig)                                       # (n, F)
    return all_ig.mean(axis=0)


def gradient_attribution(
    probe: MLPClassifier,
    activations: np.ndarray,
    baseline: np.ndarray,
) -> np.ndarray:
    """
    Returns mean signed gradient × (input − baseline) attributions.

    Uses the same baseline as IG. Plain `x * grad` (implicit zero baseline)
    disagrees with mean-baseline IG on sparse SAE features (~87% zeros):
    when x=0, GA is 0 but IG is −baseline · avg_grad.
    """
    all_attr = []

    for x in tqdm(activations, total=len(activations), desc="Calculating gradients", dynamic_ncols=True):
        grad = mlp_prob_grad(probe, x)
        all_attr.append((x - baseline) * grad)

    all_attr = np.array(all_attr)
    return all_attr.mean(axis=0)


def probe_saliency(probe: MLPClassifier) -> np.ndarray:
    """L2 norm of first-layer weights into each input (n_filtered,)."""
    return np.linalg.norm(probe.coefs_[0], axis=1)


def compile_rankings(ig_scores, probe_scores, ga_scores, shap_scores):
    def scores_to_ranks(scores):
        order = np.argsort(scores)[::-1]
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(scores) + 1)
        return ranks

    probe_ranks = scores_to_ranks(probe_scores)
    ig_ranks = scores_to_ranks(ig_scores)
    ga_ranks = scores_to_ranks(ga_scores)
    shap_ranks = scores_to_ranks(shap_scores)

    return probe_ranks, ig_ranks, ga_ranks, shap_ranks


def compare_rankings(probe_scores, ig_scores, shap_scores, ga_scores):
    rho_probe_ig, p1 = spearmanr(probe_scores, ig_scores)
    rho_probe_shap, p2 = spearmanr(probe_scores, shap_scores)
    rho_ig_shap, p3 = spearmanr(ig_scores, shap_scores)
    rho_probe_ga, p4 = spearmanr(probe_scores, ga_scores)
    rho_ig_ga, p5 = spearmanr(ig_scores, ga_scores)
    rho_shap_ga, p6 = spearmanr(shap_scores, ga_scores)

    print(f"Probe vs IG:   ρ={rho_probe_ig:.3f},   p={p1:.4f}")
    print(f"Probe vs SHAP: ρ={rho_probe_shap:.3f}, p={p2:.4f}")
    print(f"IG vs SHAP:    ρ={rho_ig_shap:.3f},    p={p3:.4f}")
    print(f"Probe vs GA:   ρ={rho_probe_ga:.3f},   p={p4:.4f}")
    print(f"IG vs GA:      ρ={rho_ig_ga:.3f},      p={p5:.4f}")
    print(f"SHAP vs GA:    ρ={rho_shap_ga:.3f},    p={p6:.4f}")


def feature_ranking(probe, train_activations, Phi, feature_indices):
    """
    IG/GA/probe on filtered activations; SHAP = Phi at filtered global indices.
    """
    baseline = train_activations.mean(axis=0)
    ig_scores = integrated_gradients(probe, train_activations, baseline)
    ga_scores = gradient_attribution(probe, train_activations, baseline)
    probe_scores = probe_saliency(probe)
    shap_scores = Phi[feature_indices]
    return ig_scores, ga_scores, probe_scores, shap_scores


if __name__ == "__main__":
    payload = joblib.load("checkpoints/mlp_probe_layer_7.joblib")
    probe: MLPClassifier = payload["probe"]
    feature_indices = np.asarray(payload["feature_indices"])

    train_activations = np.load("activations/probe_train/layer_7/activations.npy")
    train_activations = np.asarray(
        train_activations[:, feature_indices], dtype=np.float32
    )

    Phi = np.load("outputs/12_mlp_shap/phi_sentiment_layer7_signed.npy")
    shap_feature_indices = np.load("outputs/12_mlp_shap/shap_feature_indices.npy")
    if not np.array_equal(feature_indices, shap_feature_indices):
        raise ValueError(
            "Checkpoint feature_indices disagree with "
            "outputs/12_mlp_shap/shap_feature_indices.npy"
        )

    ig_scores, ga_scores, probe_scores, shap_scores = feature_ranking(
        probe, train_activations, Phi, feature_indices
    )

    os.makedirs("outputs/13_mlp_ranking", exist_ok=True)
    np.save("outputs/13_mlp_ranking/ig_scores.npy", ig_scores)
    np.save("outputs/13_mlp_ranking/ga_scores.npy", ga_scores)
    np.save("outputs/13_mlp_ranking/probe_scores.npy", probe_scores)
    np.save("outputs/13_mlp_ranking/shap_scores.npy", shap_scores)
    np.save("outputs/13_mlp_ranking/feature_indices.npy", feature_indices)

    probe_ranks, ig_ranks, ga_ranks, shap_ranks = compile_rankings(
        ig_scores, probe_scores, ga_scores, shap_scores
    )
    compare_rankings(probe_ranks, ig_ranks, shap_ranks, ga_ranks)
