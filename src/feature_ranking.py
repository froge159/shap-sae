import joblib
from scipy.stats import spearmanr
import numpy as np
from tqdm import tqdm
import os


"""
Feature ranking
"""

def integrated_gradients(
    probe,
    activations: np.ndarray,      # SHAP split, shape (n, 65)
    baseline: np.ndarray,          # mean training activation, shape (65,)
    n_steps: int = 50,
) -> np.ndarray:
    """
    Returns mean absolute IG attributions, shape (65,).
    """
    all_ig = []
    
    for x in tqdm(activations, total=len(activations), desc="Calculating IG"):
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
    mean_abs_ig = np.abs(all_ig).mean(axis=0)          # (65,)
    return mean_abs_ig

def gradient_attribution(
    probe,
    activations: np.ndarray,      # shape (n, n_features)
) -> np.ndarray:
    """
    Returns mean absolute gradient * input attributions, shape (n_features,).
    """
    all_attr = []

    for x in tqdm(activations, total=len(activations), desc="Calculating gradients"):
        logit = x @ probe.coef_[0] + probe.intercept_[0]
        sig = 1 / (1 + np.exp(-logit))
        grad = sig * (1 - sig) * probe.coef_[0]   # (n_features,)
        attr = x * grad                            # gradient × input
        all_attr.append(attr)

    all_attr = np.array(all_attr)                  # (n, n_features)
    mean_abs_attr = np.abs(all_attr).mean(axis=0)  # (n_features,)
    return mean_abs_attr

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


def feature_ranking(probe, train_activations, feature_indices, Phi):
    ig_scores = integrated_gradients(probe, train_activations, train_activations.mean(axis=0))[feature_indices]
    ga_scores = gradient_attribution(probe, train_activations)[feature_indices]
    probe_scores = np.abs(probe.coef_[0][feature_indices])
    shap_scores = Phi[feature_indices]
    return ig_scores, ga_scores, probe_scores, shap_scores


"""
Ablation
"""

def get_ablation_candidates(probe_ranks, ig_ranks, shap_ranks, ga_ranks, feature_indices, k=20):
    # Top-k by each method
    shap_top  = feature_indices[np.argsort(shap_ranks)[:k]]
    probe_top = feature_indices[np.argsort(probe_ranks)[:k]]
    ig_top    = feature_indices[np.argsort(ig_ranks)[:k]]
    ga_top = feature_indices[np.argsort(ga_ranks)[:k]]
    
    # Union — ablate each unique feature once, reuse results for all methods
    all_candidates = np.unique(np.concatenate([shap_top, probe_top, ig_top, ga_top]))
    return all_candidates, shap_top, probe_top, ig_top, ga_top

def ablate(probe, activations, candidate_indices):
    # return dict {index: mean delta probability}
    delta_probabilities = {}
    for index in candidate_indices:
        delta_probability = probe.predict_proba(activations[:, index])[0][1] - probe.predict_proba(activations[:, index])[0][0]
        delta_probabilities[index] = delta_probability
    return delta_probabilities

def faithfulness_correlation(method_scores, feature_indices, ablation_effects):
    scores = []
    effects = []
    for i, feat_idx in enumerate(feature_indices):
        if feat_idx in ablation_effects:
            scores.append(method_scores[i])
            effects.append(abs(ablation_effects[feat_idx]))
    
    rho, p = spearmanr(scores, effects)
    return rho, p



if __name__ == "__main__":
    # Load data
    probe = joblib.load("checkpoints/probe_layer_7.joblib")
    train_activations = np.load("activations/probe_train/layer_7/activations.npy")
    feature_indices = np.load("outputs/shap/shap_feature_indices.npy")
    Phi = np.load("outputs/shap/phi_sentiment_layer7.npy")

    # Feature ranking
    """
    ig_scores, ga_scores, probe_scores, shap_scores = feature_ranking(probe, train_activations, feature_indices, Phi)

    probe_ranks, ig_ranks, ga_ranks, shap_ranks = compile_rankings(probe_scores, ig_scores, ga_scores, shap_scores)
    compare_rankings(probe_ranks, ig_ranks, shap_ranks, ga_ranks)

    # save everything, create directory if it doesn't exist
    os.makedirs("outputs/rankings", exist_ok=True)
    np.save("outputs/rankings/ig_scores.npy", ig_scores)
    np.save("outputs/rankings/ga_scores.npy", ga_scores)
    """

    # Ablation
    probe_scores = np.abs(probe.coef_[0][feature_indices])
    ig_scores = np.load("outputs/rankings/ig_scores.npy")
    ga_scores = np.load("outputs/rankings/ga_scores.npy")
    shap_scores = np.load("outputs/shap/phi_sentiment_layer7.npy")
    probe_ranks, ig_ranks, ga_ranks, shap_ranks = compile_rankings(probe_scores, ig_scores, ga_scores, shap_scores)
    val_activations = np.load("activations/probe_val/layer_7/activations.npy")

    all_candidates, shap_top, probe_top, ig_top, ga_top = get_ablation_candidates(probe_ranks, ig_ranks, shap_ranks, ga_ranks, feature_indices, k=20) # adjust k
    delta_probabilities = ablate(probe, val_activations, all_candidates)
    print("delta_probabilities:")
    for index, delta_probability in delta_probabilities.items():
        print(f"  {index}: {delta_probability:.3f}")

    rho_shap,  p = faithfulness_correlation(shap_scores,  feature_indices, delta_probabilities)
    rho_probe, p = faithfulness_correlation(probe_scores, feature_indices, delta_probabilities)
    rho_ig,    p = faithfulness_correlation(ig_scores,    feature_indices, delta_probabilities)
    rho_ga,    p = faithfulness_correlation(ga_scores,    feature_indices, delta_probabilities)

    print(f"Faithfulness (Spearman ρ with ablation effects):")
    print(f"  SHAP:         {rho_shap:.3f}")
    print(f"  Probe weights:{rho_probe:.3f}")
    print(f"  IG:           {rho_ig:.3f}")
    print(f"  GA:           {rho_ga:.3f}")

    # save delta_probabilities
    os.makedirs("outputs/ablation", exist_ok=True)
    np.save("outputs/ablation/delta_probabilities.npy", delta_probabilities)
