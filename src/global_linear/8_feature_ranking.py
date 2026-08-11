import os
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils import checkpoint_path, load_eval_and_background, output_path


def integrated_gradients(
    probe,
    activations: np.ndarray,      # (n, n_features)
    baseline: np.ndarray,         # (n_features,)
    n_steps: int = 50,
) -> np.ndarray:
    """
    Returns mean signed IG attributions, shape (n_features,).

    Accumulates a running sum rather than stacking per-example attributions:
    at full SAE width the stacked array is n x 32768 float64.
    """
    alphas = np.linspace(0, 1, n_steps)                # (n_steps,)
    ig_sum = np.zeros_like(baseline, dtype=np.float64)

    for x in tqdm(activations, total=len(activations), desc="Calculating IG", dynamic_ncols=True):
        # Interpolate from baseline to input
        interpolated = baseline + alphas[:, None] * (x - baseline)  # (n_steps, F)

        # Gradient at each interpolated point
        # For logistic regression: grad = w * sigmoid(Wx+b) * (1 - sigmoid(Wx+b))
        logits = interpolated @ probe.coef_[0] + probe.intercept_[0]  # (n_steps,)
        sig = 1 / (1 + np.exp(-logits))               # (n_steps,)
        grad_scalar = sig * (1 - sig)                  # (n_steps,)

        # Full gradient: outer product of scalar with weights
        grads = grad_scalar[:, None] * probe.coef_[0]  # (n_steps, F)

        # Integrate (trapezoidal rule) and multiply by (x - baseline)
        avg_grads = np.trapezoid(grads, alphas, axis=0)    # (F,)
        ig_sum += (x - baseline) * avg_grads

    return ig_sum / len(activations)


def gradient_attribution(
    probe,
    activations: np.ndarray,      # shape (n, n_features)
    baseline: np.ndarray,         # shape (n_features,)
) -> np.ndarray:
    """
    Returns mean signed gradient x (input - baseline) attributions.

    Uses the same baseline as IG. Plain `x * grad` (implicit zero baseline)
    disagrees with mean-baseline IG on sparse SAE features (~87% zeros):
    when x=0, GA is 0 but IG is -baseline . avg_grad.
    """
    attr_sum = np.zeros_like(baseline, dtype=np.float64)

    for x in tqdm(activations, total=len(activations), desc="Calculating gradients", dynamic_ncols=True):
        logit = x @ probe.coef_[0] + probe.intercept_[0]
        sig = 1 / (1 + np.exp(-logit))
        grad = sig * (1 - sig) * probe.coef_[0]   # (n_features,)
        attr_sum += (x - baseline) * grad

    return attr_sum / len(activations)

def compile_rankings(ig_scores, probe_scores, ga_scores, shap_scores):
    """Signed scores → ranks (rank 1 = most positive). Monotone, so Spearman is
    unchanged by this step; it exists only for readable per-feature tables."""
    def scores_to_ranks(scores):
        order = np.argsort(scores)[::-1]
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(scores) + 1)
        return ranks

    probe_ranks = scores_to_ranks(probe_scores)
    ig_ranks    = scores_to_ranks(ig_scores)
    ga_ranks    = scores_to_ranks(ga_scores)
    shap_ranks  = scores_to_ranks(shap_scores)

    return probe_ranks, ig_ranks, ga_ranks, shap_ranks


def scored_support(probe_scores: np.ndarray) -> np.ndarray:
    """
    Features the comparison is meaningful over: those with a non-zero probe weight.

    Correlating all 32768 features would be misleading. The L1 probe zeroes ~25k
    of them, and IG, GA and Φ are *identically* zero wherever the weight is —
    IG/GA are proportional to `coef_`, and Φ = w·(x̄_eval − x̄_bg). Spearman over
    one enormous block of exact ties reports agreement about which features are
    excluded, not agreement about which features matter.
    """
    return np.flatnonzero(probe_scores)


def compare_rankings(probe_scores, ig_scores, shap_scores, ga_scores, support=None):
    """Spearman between every pair of methods, restricted to `support`."""
    if support is not None:
        probe_scores = probe_scores[support]
        ig_scores = ig_scores[support]
        shap_scores = shap_scores[support]
        ga_scores = ga_scores[support]

    rho_probe_ig,   p1 = spearmanr(probe_scores, ig_scores)
    rho_probe_shap, p2 = spearmanr(probe_scores, shap_scores)
    rho_ig_shap,    p3 = spearmanr(ig_scores, shap_scores)
    rho_probe_ga,   p4 = spearmanr(probe_scores, ga_scores)
    rho_ig_ga,      p5 = spearmanr(ig_scores, ga_scores)
    rho_shap_ga,    p6 = spearmanr(shap_scores, ga_scores)

    lines = [
        f"n_features={len(probe_scores)}",
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


def feature_ranking(probe, activations, baseline, Phi):
    """
    IG/GA over `activations` with `baseline`; SHAP scores passed through.

    All four score vectors must describe the same rows, or the Spearman
    comparisons below confound attribution method with choice of data.
    """
    ig_scores = integrated_gradients(probe, activations, baseline)
    ga_scores = gradient_attribution(probe, activations, baseline)
    probe_scores = probe.coef_[0]
    shap_scores = Phi
    return ig_scores, ga_scores, probe_scores, shap_scores



if __name__ == "__main__":
    LAYER = 7
    OUT_DIR = output_path("8_rankings_recompute")

    probe = joblib.load(checkpoint_path(f"probe_layer_{LAYER}.joblib"))
    Phi = np.load(output_path("7_shap_recompute", "phi_sentiment_layer7_signed.npy"))

    # Same held-out rows Phi was computed on, and the mean of the same train
    # background SHAP marginalised over -- the single-point analogue of SHAP's
    # interventional background.
    shap_eval, background, eval_idx, bg_idx = load_eval_and_background(layer=LAYER)
    baseline = background.mean(axis=0)
    print(
        f"IG/GA over {len(shap_eval)} held-out rows, "
        f"baseline = mean of {len(background)} train background rows"
    )

    # Feature ranking
    ig_scores, ga_scores, probe_scores, shap_scores = feature_ranking(
        probe, shap_eval, baseline, Phi
    )

    # save everything, create directory if it doesn't exist
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(OUT_DIR / "ig_scores.npy", ig_scores)  # shape (32k,)
    np.save(OUT_DIR / "ga_scores.npy", ga_scores)  # shape (32k,)
    np.save(OUT_DIR / "eval_indices.npy", eval_idx)
    np.save(OUT_DIR / "background_indices.npy", bg_idx)

    support = scored_support(probe_scores)
    print(
        f"\nRank agreement over the {len(support)} features with a non-zero probe "
        f"weight (of {len(probe_scores)}):"
    )
    text = compare_rankings(probe_scores, ig_scores, shap_scores, ga_scores, support)
    (OUT_DIR / "correlation.txt").write_text(text)
    np.save(OUT_DIR / "support.npy", support)
    print(f"\nWrote outputs → {OUT_DIR}/")