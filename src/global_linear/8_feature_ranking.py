"""
IG / GA / probe-weight / SHAP rank agreement for the linear (logistic) SAE probe.

**Attribution target.** Every method here explains the **log-odds margin**
`w·x + b` — the same function `7_shap_recompute.py`'s LinearExplainer explains.
IG and GA used to multiply the gradient by `σ'(logit) = σ(1−σ)`, i.e. they
attributed `σ(w·x + b)` while SHAP attributed the margin. That mismatch is not
harmless. Per example σ' is a positive scalar, so it leaves the within-example
ranking alone — but all three methods *average over examples* before anything is
correlated, and

    Φ_j  = w_j · mean_e[ x_ej − μ_j ]
    IG_j = w_j · mean_e[ c_e · (x_ej − μ_j) ]      c_e = per-example σ' path average

are different weighted means of the same quantity, related by no monotone map.
The near-zero historical IG-vs-SHAP ρ was largely that reweighting, not a
property of the methods.

**Consequence, stated plainly:** with the target matched, IG, GA and
interventional SHAP are *analytically identical* for a linear probe — all three
collapse to `w_j · (x_j − μ_j)`. So this script is a closed-form sanity check
(expect ρ = 1.000 and max|IG − Φ| ≈ 0), not an empirical comparison. The real
method comparison lives in the MLP arm (`global_mlp/13_mlp_ranking.py`), where
the ReLU makes IG ≠ SHAP genuinely. `main` asserts the identity and writes it
into `correlation.txt`; if it ever fails, something upstream drifted.
"""

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
    Returns mean signed IG attributions on the **log-odds margin**, shape (n_features,).

    The margin's gradient is the constant `coef_`, so the path integral
    `∫₀¹ ∇f(baseline + α(x − baseline)) dα` collapses to `coef_` exactly and IG
    reduces to `(x − baseline) · w`. Computing it in closed form rather than
    running an n_steps Riemann sum of a constant is the same number without the
    2000 × 50 × 32768 of arithmetic — `n_steps` is kept in the signature only so
    the call sites stay uniform with the MLP arm, where the path integral is real.

    Do NOT reintroduce a `σ(1−σ)` factor here: that attributes `σ(margin)`, which
    is a different target function from the one 7_shap_recompute explains. See
    the module docstring.
    """
    del n_steps  # constant gradient ⇒ the path integral is exact in closed form

    w = np.asarray(probe.coef_[0], dtype=np.float64)
    attr_sum = np.zeros_like(baseline, dtype=np.float64)

    for x in tqdm(activations, total=len(activations), desc="Calculating IG", dynamic_ncols=True):
        attr_sum += (x - baseline) * w

    return attr_sum / len(activations)


def gradient_attribution(
    probe,
    activations: np.ndarray,      # shape (n, n_features)
    baseline: np.ndarray,         # shape (n_features,)
) -> np.ndarray:
    """
    Returns mean signed gradient × (input − baseline) attributions on the margin.

    Uses the same baseline as IG. Plain `x * grad` (implicit zero baseline)
    disagrees with mean-baseline IG on sparse SAE features (~87% zeros):
    when x=0, GA is 0 but IG is −baseline · grad.

    On the margin the gradient is input-independent, so GA and IG coincide here
    by construction. Kept as a separate function because it does not coincide in
    the MLP arm, and because the two are conceptually distinct methods.
    """
    w = np.asarray(probe.coef_[0], dtype=np.float64)
    attr_sum = np.zeros_like(baseline, dtype=np.float64)

    for x in tqdm(activations, total=len(activations), desc="Calculating gradients", dynamic_ncols=True):
        attr_sum += (x - baseline) * w

    return attr_sum / len(activations)


def linearity_identity_report(
    ig_scores: np.ndarray,
    ga_scores: np.ndarray,
    shap_scores: np.ndarray,
    support: np.ndarray,
    rtol: float = 1e-5,
) -> str:
    """
    Check the closed-form identity IG = GA = Φ over the scored support.

    All three equal `w_j · mean_e[x_ej − μ_j]` for a linear probe explained on the
    margin, so a non-zero gap means a target mismatch has crept back in (or that
    Φ and IG/GA were scored on different rows, which `main` also guards).

    The tolerance is *relative* and sized for float32: activations are stored as
    float32 and `shap` accumulates in float32, so the identity lands around 1e-7
    relative, not at machine zero. `rtol=1e-5` is still four orders tighter than
    any real mismatch — reintroducing the σ' reweighting moves this to O(1).
    """
    d_ig = float(np.max(np.abs(ig_scores[support] - shap_scores[support])))
    d_ga = float(np.max(np.abs(ga_scores[support] - shap_scores[support])))
    scale = float(np.max(np.abs(shap_scores[support]))) or 1.0

    lines = [
        "Closed-form identity check (linear probe, margin target):",
        f"  max|IG - Phi| = {d_ig:.3e}   (relative {d_ig / scale:.3e})",
        f"  max|GA - Phi| = {d_ga:.3e}   (relative {d_ga / scale:.3e})",
        f"  rtol          = {rtol:.1e}   (float32 accumulation floors this near 1e-7)",
    ]
    ok = d_ig <= rtol * scale and d_ga <= rtol * scale
    lines.append(
        "  PASS - IG, GA and SHAP coincide analytically, as they must for a "
        "linear probe. The rank agreement below is an identity, not evidence."
        if ok
        else "  FAIL - the three methods no longer coincide; a target mismatch or "
        "a row mismatch has been reintroduced. Do not quote the correlations."
    )
    for line in lines:
        print(line)
    return "\n".join(lines) + "\n"


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

    SHAP_DIR = output_path("7_shap_recompute")

    probe = joblib.load(checkpoint_path(f"probe_layer_{LAYER}.joblib"))
    Phi = np.load(SHAP_DIR / "phi_sentiment_layer7_signed.npy")

    # Same held-out rows Phi was computed on, and the mean of the same train
    # background SHAP marginalised over -- the single-point analogue of SHAP's
    # interventional background.
    shap_eval, background, eval_idx, bg_idx = load_eval_and_background(layer=LAYER)
    baseline = background.mean(axis=0)
    print(
        f"IG/GA over {len(shap_eval)} held-out rows, "
        f"baseline = mean of {len(background)} train background rows"
    )

    # Phi and IG/GA must describe the *same* rows and the *same* background, or
    # the comparisons below confound attribution method with choice of data. The
    # MLP arm has always checked this (13_mlp_ranking); the linear arm did not.
    shap_eval_indices_path = SHAP_DIR / f"eval_indices_layer{LAYER}.npy"
    shap_bg_indices_path = SHAP_DIR / f"background_indices_layer{LAYER}.npy"
    if shap_eval_indices_path.exists():
        shap_eval_idx = np.load(shap_eval_indices_path)
        if not np.array_equal(eval_idx, shap_eval_idx):
            raise ValueError(
                f"Row mismatch: IG/GA scored on {len(eval_idx)} held-out rows but "
                f"{SHAP_DIR}/ holds Phi for {len(shap_eval_idx)} rows. "
                f"Re-run 7_shap_recompute with n_shap={len(eval_idx)}."
            )
        shap_bg_idx = np.load(shap_bg_indices_path)
        if not np.array_equal(bg_idx, shap_bg_idx):
            raise ValueError(
                f"Background mismatch: IG/GA baseline is the mean of "
                f"{len(bg_idx)} train rows but {SHAP_DIR}/ marginalised over "
                f"{len(shap_bg_idx)} different rows."
            )
        print(f"  eval/background rows match {SHAP_DIR}/")
    else:
        print(
            f"WARNING: {shap_eval_indices_path} missing - cannot verify that Phi "
            "and IG/GA describe the same held-out rows. Re-run 7_shap_recompute."
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
    print()
    identity_text = linearity_identity_report(
        ig_scores, ga_scores, shap_scores, support
    )

    print(
        f"\nRank agreement over the {len(support)} features with a non-zero probe "
        f"weight (of {len(probe_scores)}):"
    )
    corr_text = compare_rankings(probe_scores, ig_scores, shap_scores, ga_scores, support)
    (OUT_DIR / "correlation.txt").write_text(identity_text + "\n" + corr_text)
    np.save(OUT_DIR / "support.npy", support)
    print(f"\nWrote outputs → {OUT_DIR}/")