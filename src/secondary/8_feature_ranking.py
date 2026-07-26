import joblib
from scipy.stats import spearmanr
import numpy as np
from tqdm import tqdm
import os


def integrated_gradients(
    probe,
    activations: np.ndarray,      
    baseline: np.ndarray,         
    n_steps: int = 50,
) -> np.ndarray:
    """
    Returns mean signed IG attributions, shape (65,).
    """
    all_ig = []
    
    for x in tqdm(activations, total=len(activations), desc="Calculating IG", dynamic_ncols=True):
        # Interpolate from baseline to input
        alphas = np.linspace(0, 1, n_steps)          # (n_steps,)
        interpolated = baseline + alphas[:, None] * (x - baseline)  # (n_steps, 65)
        
        # Gradient at each interpolated point
        # For logistic regression: grad = w * sigmoid(Wx+b) * (1 - sigmoid(Wx+b))
        logits = interpolated @ probe.coef_[0] + probe.intercept_[0]  # (n_steps,)
        sig = 1 / (1 + np.exp(-logits))               # (n_steps,)
        grad_scalar = sig * (1 - sig)                  # (n_steps,)
        
        # Full gradient: outer product of scalar with weights
        grads = grad_scalar[:, None] * probe.coef_[0]  # (n_steps, 65)
        
        # Integrate (trapezoidal rule) and multiply by (x - baseline)
        avg_grads = np.trapezoid(grads, alphas, axis=0)    # (65,)
        ig = (x - baseline) * avg_grads                # (65,)
        all_ig.append(ig)
    
    all_ig = np.array(all_ig)                          # (n, 65)
    mean_ig = all_ig.mean(axis=0)                      # (65,)
    return mean_ig

def gradient_attribution(
    probe,
    activations: np.ndarray,      # shape (n, n_features)
) -> np.ndarray:
    """
    Returns mean signed gradient * input attributions, shape (n_features,).
    """
    all_attr = []

    for x in tqdm(activations, total=len(activations), desc="Calculating gradients", dynamic_ncols=True):
        logit = x @ probe.coef_[0] + probe.intercept_[0]
        sig = 1 / (1 + np.exp(-logit))
        grad = sig * (1 - sig) * probe.coef_[0]   # (n_features,)
        attr = x * grad                            # gradient × input
        all_attr.append(attr)

    all_attr = np.array(all_attr)                  # (n, n_features)
    mean_attr = all_attr.mean(axis=0)              # (n_features,)
    return mean_attr

def compile_rankings(ig_scores, probe_scores, ga_scores, shap_scores):
    # Convert scores to ranks (rank 1 = most important)
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

def compare_rankings(probe_scores, ig_scores, shap_scores, ga_scores):
    rho_probe_ig,   p1 = spearmanr(probe_scores, ig_scores)
    rho_probe_shap, p2 = spearmanr(probe_scores, shap_scores)
    rho_ig_shap,    p3 = spearmanr(ig_scores, shap_scores)
    rho_probe_ga,   p4 = spearmanr(probe_scores, ga_scores)
    rho_ig_ga,      p5 = spearmanr(ig_scores, ga_scores)
    rho_shap_ga,    p6 = spearmanr(shap_scores, ga_scores)
    
    print(f"Probe vs IG:   ρ={rho_probe_ig:.3f},   p={p1:.4f}")
    print(f"Probe vs SHAP: ρ={rho_probe_shap:.3f}, p={p2:.4f}")
    print(f"IG vs SHAP:    ρ={rho_ig_shap:.3f},    p={p3:.4f}")
    print(f"Probe vs GA:   ρ={rho_probe_ga:.3f},   p={p4:.4f}")
    print(f"IG vs GA:      ρ={rho_ig_ga:.3f},      p={p5:.4f}")
    print(f"SHAP vs GA:    ρ={rho_shap_ga:.3f},    p={p6:.4f}")


def feature_ranking(probe, train_activations, Phi):
    ig_scores = integrated_gradients(probe, train_activations, train_activations.mean(axis=0))
    ga_scores = gradient_attribution(probe, train_activations)
    probe_scores = probe.coef_[0]
    shap_scores = Phi
    return ig_scores, ga_scores, probe_scores, shap_scores



if __name__ == "__main__":
    # Load data
    probe = joblib.load("checkpoints/probe_layer_7.joblib")
    train_activations = np.load("activations/probe_train/layer_7/activations.npy")
    Phi = np.load("outputs/7_shap_recompute/phi_sentiment_layer7_signed.npy")

    # Feature ranking
    ig_scores, ga_scores, probe_scores, shap_scores = feature_ranking(probe, train_activations, Phi)

    # save everything, create directory if it doesn't exist
    os.makedirs("outputs/8_rankings_recompute", exist_ok=True)
    np.save("outputs/8_rankings_recompute/ig_scores.npy", ig_scores) # shape (32k,)
    np.save("outputs/8_rankings_recompute/ga_scores.npy", ga_scores) # shape (32k,)

    probe_ranks, ig_ranks, ga_ranks, shap_ranks = compile_rankings(probe_scores, ig_scores, ga_scores, shap_scores)
    compare_rankings(probe_ranks, ig_ranks, shap_ranks, ga_ranks)

    
    