"""
Feature ranking for the filtered-feature MLP probe.

Mirrors global_linear/8_feature_ranking.py (signed mean IG/GA, Spearman rank
comparisons), but:
  - Explains the saved MLP probe (mlp_probe_layer_*.joblib)
  - IG/GA use input-dependent gradients through ReLU (not constant ∂f/∂x = w)
  - Scores/ranks are over filtered features only; SHAP scores are Phi[indices]

**Attribution target.** All methods here explain the **pre-sigmoid logit**, which
is what 12_mlp_shap's DeepExplainer explains. IG/GA used to carry an extra
`σ'(logit) = σ(1−σ)` factor, i.e. they attributed `σ(logit)` while SHAP attributed
the logit. Per example that factor is a positive scalar and changes no ranking,
but every method here averages over examples first, and two differently-reweighted
means of the same per-example quantity are not related by any monotone map — so
the IG-vs-SHAP Spearman was comparing two different target functions. Unlike the
linear arm (where matching the target collapses IG = GA = SHAP into an identity),
the ReLU keeps IG genuinely distinct from SHAP here, so this is where the real
method comparison lives.

Note on `probe_scores`: unlike the linear arm's signed `coef_`, the MLP's
`probe_saliency` is ‖W1[i,:]‖₂ and is therefore **non-negative**. It carries
magnitude only, so it is reported against the other methods for importance and
excluded from any directional comparison.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils import checkpoint_path, load_eval_and_background, output_path


def mlp_logit_grad(probe: MLPClassifier, x: np.ndarray) -> np.ndarray:
    """
    ∇_x (pre-sigmoid logit) for a 1-hidden-layer ReLU MLP.

    The logit, *not* σ(logit): DeepSHAP in 12_mlp_shap explains the logit, and
    IG/GA have to attribute the same function or the rank comparison in
    `compare_rankings` confounds attribution method with target function. Do not
    reintroduce a `σ(1−σ)` factor here — see the module docstring.

    x: (n_features,) or (batch, n_features)
    returns: same leading shape as x
    """
    W1, W2 = probe.coefs_
    b1, _b2 = probe.intercepts_
    single = x.ndim == 1
    if single:
        x = x[None, :]

    pre = x @ W1 + b1                          # (batch, hidden)
    d_logit_dpre = W2[:, 0] * (pre > 0)        # (batch, hidden)
    grad = d_logit_dpre @ W1.T                 # (batch, n_features)

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
        grads = mlp_logit_grad(probe, interpolated)                 # (n_steps, F)
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
        grad = mlp_logit_grad(probe, x)
        all_attr.append((x - baseline) * grad)

    all_attr = np.array(all_attr)
    return all_attr.mean(axis=0)


def probe_saliency(probe: MLPClassifier) -> np.ndarray:
    """L2 norm of first-layer weights into each input (n_filtered,)."""
    return np.linalg.norm(probe.coefs_[0], axis=1)


def compare_rankings(probe_scores, ig_scores, shap_scores, ga_scores):
    """
    Spearman between methods on **|score|** — an importance comparison only.

    `probe_scores` is ‖W1[i,:]‖₂, which has no sign, so correlating it against
    signed IG/GA/SHAP would be comparing a magnitude with a direction. Taking |·|
    throughout puts all four on the one scale they share. The signed question is
    answered against causal effects in 14_mlp_steer, not here.
    """
    probe_scores = np.abs(probe_scores)
    ig_scores = np.abs(ig_scores)
    shap_scores = np.abs(shap_scores)
    ga_scores = np.abs(ga_scores)

    rho_probe_ig, p1 = spearmanr(probe_scores, ig_scores)
    rho_probe_shap, p2 = spearmanr(probe_scores, shap_scores)
    rho_ig_shap, p3 = spearmanr(ig_scores, shap_scores)
    rho_probe_ga, p4 = spearmanr(probe_scores, ga_scores)
    rho_ig_ga, p5 = spearmanr(ig_scores, ga_scores)
    rho_shap_ga, p6 = spearmanr(shap_scores, ga_scores)

    lines = [
        f"n_features={len(probe_scores)}  (Spearman on |score|)",
        f"Probe vs IG:   ρ={rho_probe_ig:.3f},   p={p1:.4f}",
        f"Probe vs SHAP: ρ={rho_probe_shap:.3f}, p={p2:.4f}",
        f"IG vs SHAP:    ρ={rho_ig_shap:.3f},    p={p3:.4f}",
        f"Probe vs GA:   ρ={rho_probe_ga:.3f},   p={p4:.4f}",
        f"IG vs GA:      ρ={rho_ig_ga:.3f},      p={p5:.4f}",
        f"SHAP vs GA:    ρ={rho_shap_ga:.3f},    p={p6:.4f}",
    ]
    for line in lines:
        print(line)
    return "\n".join(lines) + "\n"


def feature_ranking(probe, activations, baseline, Phi, feature_indices):
    """
    IG/GA/probe on filtered activations; SHAP = Phi at filtered global indices.

    `activations` and `baseline` must be the same held-out eval rows and train
    background that produced Phi, or the Spearman comparisons below confound
    attribution method with choice of data.
    """
    ig_scores = integrated_gradients(probe, activations, baseline)
    ga_scores = gradient_attribution(probe, activations, baseline)
    probe_scores = probe_saliency(probe)
    shap_scores = Phi[feature_indices]
    return ig_scores, ga_scores, probe_scores, shap_scores


if __name__ == "__main__":
    LAYER = 7
    OUT_DIR = output_path("13_mlp_ranking")
    SHAP_DIR = output_path("12_mlp_shap")

    payload = joblib.load(checkpoint_path(f"mlp_probe_layer_{LAYER}.joblib"))
    probe: MLPClassifier = payload["probe"]
    feature_indices = np.asarray(payload["feature_indices"])

    # Same held-out rows DeepSHAP explained, same train background it
    # marginalised over, both sliced to the probe's filtered inputs.
    shap_eval, background, eval_idx, bg_idx = load_eval_and_background(layer=LAYER)
    eval_activations = np.asarray(shap_eval[:, feature_indices], dtype=np.float32)
    baseline = np.asarray(background[:, feature_indices], dtype=np.float32).mean(axis=0)
    print(
        f"IG/GA over {len(eval_activations)} held-out rows x "
        f"{len(feature_indices)} features, baseline = mean of "
        f"{len(background)} train background rows"
    )

    Phi = np.load(SHAP_DIR / "phi_sentiment_layer7_signed.npy")
    shap_feature_indices = np.load(SHAP_DIR / "shap_feature_indices.npy")
    if not np.array_equal(feature_indices, shap_feature_indices):
        raise ValueError(
            f"Checkpoint feature_indices disagree with "
            f"{SHAP_DIR / 'shap_feature_indices.npy'}"
        )

    # Φ and IG/GA must describe the *same* rows, or the Spearman comparisons below
    # confound attribution method with choice of data. 12_mlp_shap has been run at
    # a smaller n_shap than DEFAULT_N_EVAL before, which silently breaks this.
    shap_eval_indices_path = SHAP_DIR / f"eval_indices_layer{LAYER}.npy"
    if shap_eval_indices_path.exists():
        shap_eval_idx = np.load(shap_eval_indices_path)
        if not np.array_equal(eval_idx, shap_eval_idx):
            raise ValueError(
                f"Row mismatch: IG/GA scored on {len(eval_idx)} rows but "
                f"{SHAP_DIR}/ holds SHAP for {len(shap_eval_idx)} rows. "
                f"Re-run 12_mlp_shap with n_shap={len(eval_idx)}."
            )
    else:
        print(
            f"WARNING: {shap_eval_indices_path} missing — cannot verify that Φ and "
            "IG/GA describe the same held-out rows. Re-run 12_mlp_shap."
        )

    ig_scores, ga_scores, probe_scores, shap_scores = feature_ranking(
        probe, eval_activations, baseline, Phi, feature_indices
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(OUT_DIR / "ig_scores.npy", ig_scores)
    np.save(OUT_DIR / "ga_scores.npy", ga_scores)
    np.save(OUT_DIR / "probe_scores.npy", probe_scores)
    np.save(OUT_DIR / "shap_scores.npy", shap_scores)
    np.save(OUT_DIR / "feature_indices.npy", feature_indices)
    np.save(OUT_DIR / "eval_indices.npy", eval_idx)
    np.save(OUT_DIR / "background_indices.npy", bg_idx)

    text = compare_rankings(probe_scores, ig_scores, shap_scores, ga_scores)
    (OUT_DIR / "correlation.txt").write_text(text)
    print(f"\nWrote outputs → {OUT_DIR}/")
