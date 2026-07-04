import numpy as np
import shap
import joblib
from sklearn.pipeline import Pipeline


def prepare_shap_inputs(activations_dir="activations", layer=7, n_background=100, n_shap=200, seed=42):
    """
    activations: shape (N_sentences, 32768)
    """
    rng = np.random.default_rng(seed)
    
    test_activations = np.load(f"{activations_dir}/probe_val/layer_{layer}/activations.npy")
    train_activations = np.load(f"{activations_dir}/probe_train/layer_{layer}/activations.npy")
    shap_activations = np.load(f"{activations_dir}/shap/layer_{layer}/activations.npy")
    
    # Fixed background: 100 training examples
    bg_idx = rng.choice(len(train_activations), size=n_background, replace=False)
    background = train_activations[bg_idx]        # shape: (100, 32768)
    
    # SHAP evaluation set: start with 200 val examples, then scale up to 1000
    shap_idx = rng.choice(len(shap_activations), size=n_shap, replace=False)
    shap_eval = shap_activations[shap_idx]         # shape: (200, 32768)
    
    return shap_eval, background


def get_shap_feature_mask(probe, activations: np.ndarray,  min_activation_freq: float = 0.05):
    # Filter 1: non-zero probe weights
    nonzero_mask = probe.coef_[0] != 0          # shape: (32768,)
    
    # Filter 2: activation frequency on val set
    activation_freq = (activations > 0).mean(axis=0)   # shape: (32768,)
    freq_mask = activation_freq >= min_activation_freq
    
    combined_mask = nonzero_mask & freq_mask
    feature_indices = np.where(combined_mask)[0]
    
    print(f"Non-zero probe weights: {nonzero_mask.sum()}")
    print(f"After frequency filter: {len(feature_indices)}")
    
    return feature_indices


def make_probe_predict_fn(probe, feature_indices: np.ndarray):
    """
    Returns a function that takes a (n_samples, n_filtered_features) array
    and returns predicted probabilities for the positive class.
    """
    def predict_fn(X_filtered):
        # X_filtered has only the filtered features; reconstruct full vector
        # KernelSHAP passes numpy arrays
        return probe.predict_proba(X_filtered)[:, 1]
    
    return predict_fn


def run_kernelshap(
    probe,
    shap_eval: np.ndarray,       # shape: (n_eval, 32768)
    background: np.ndarray,      # shape: (100, 32768)
    feature_indices: np.ndarray, # indices of filtered features
    n_shap_samples: int = 500,
    seed: int = 42,
) -> np.ndarray:
    """
    Returns shap_values of shape (n_eval, n_filtered_features).
    """
    # Slice to filtered features only
    shap_eval_filtered = shap_eval[:, feature_indices]
    background_filtered = background[:, feature_indices]
    
    predict_fn = make_probe_predict_fn(probe, feature_indices)
    
    explainer = shap.KernelExplainer(
        predict_fn,
        background_filtered,
        seed=seed,
    )
    
    # nsamples controls the coalition sampling budget per explanation
    shap_values = explainer.shap_values(
        shap_eval_filtered,
        nsamples=n_shap_samples,
        silent=False,   # shows progress bar
    )
    # shap_values shape: (n_eval, n_filtered_features)
    return shap_values


def build_attribution_matrix(
    shap_values: np.ndarray,      # (n_eval, n_filtered_features)
    feature_indices: np.ndarray,  # which original features these correspond to
    n_total_features: int = 32768,
) -> np.ndarray:
    """
    Returns Phi of shape (n_total_features,) — mean absolute SHAP per feature.
    Unfiltered features get 0.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)   # (n_filtered_features,)
    
    Phi = np.zeros(n_total_features)
    Phi[feature_indices] = mean_abs_shap
    
    return Phi   # your latent-to-concept attribution vector for sentiment


def get_top_shap_features(Phi: np.ndarray, k: int = 50):
    top_k_idx = np.argsort(Phi)[::-1][:k]
    return top_k_idx, Phi[top_k_idx]


def main():
    probe = joblib.load(f"{checkpoint_dir}/probe_layer7.joblib") 
    shap_eval, background = prepare_shap_inputs()
    feature_indices = get_shap_feature_mask(probe, shap_eval)
    shap_values = run_kernelshap(probe, shap_eval, background, feature_indices)
    Phi = build_attribution_matrix(shap_values, feature_indices)
    top_k_idx, top_k_Phi = get_top_shap_features(Phi)
    print(top_k_idx)
    print(top_k_Phi)

    """
    np.save("outputs/shap_values_raw.npy", shap_values)
    np.save("outputs/shap_feature_indices.npy", feature_indices)
    np.save("outputs/phi_sentiment_layer7.npy", Phi)
    np.save("outputs/top_k_idx_layer7.npy", top_k_idx)
    np.save("outputs/top_k_Phi_layer7.npy", top_k_Phi)
    """


if __name__ == "__main__":
    main()