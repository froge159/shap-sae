import sys
import warnings
from pathlib import Path

import numpy as np
import shap
import joblib
from tqdm import tqdm

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils import (
    DEFAULT_N_BACKGROUND,
    DEFAULT_N_EVAL,
    CANONICAL_SEED,
    checkpoint_path,
    load_activations,
    load_eval_and_background,
    output_path,
)


def prepare_shap_inputs(
    activations_dir="activations",
    layer=7,
    n_background=DEFAULT_N_BACKGROUND,
    n_shap=500,
    seed=CANONICAL_SEED,
):
    """
    Canonical eval/background rows, shape (n, 32768).

    Thin wrapper over utils.load_eval_and_background so KernelSHAP here,
    LinearSHAP/DeepSHAP downstream, and IG/GA in the ranking scripts all
    explain the same rows. n_shap stays modest by default because KernelSHAP
    is O(n_eval x nsamples); the exact explainers can afford far more.
    """
    return load_eval_and_background(
        layer=layer,
        n_eval=n_shap,
        n_background=n_background,
        seed=seed,
        activations_dir=activations_dir,
    )


def get_shap_feature_mask(probe, val_activations: np.ndarray, min_activation_freq: float = 0.05):
    """
    Filtered feature set: non-zero probe weight AND active often enough.

    `val_activations` must come from the *val* split. This mask defines the
    MLP probe's input layer and the steering candidate pool, so deriving it
    from the shap split would let the held-out set shape the models it is
    later used to evaluate.
    """
    # Filter 1: non-zero probe weights
    nonzero_mask = probe.coef_[0] != 0          # shape: (32768,)

    # Filter 2: activation frequency on val set
    activation_freq = (val_activations > 0).mean(axis=0)   # shape: (32768,)
    freq_mask = activation_freq >= min_activation_freq

    combined_mask = nonzero_mask & freq_mask
    feature_indices = np.where(combined_mask)[0]

    print(f"Non-zero probe weights: {nonzero_mask.sum()}")
    print(f"After frequency filter: {len(feature_indices)}")

    return feature_indices


def make_probe_predict_fn(
    probe,
    feature_indices: np.ndarray,
    shap_eval: np.ndarray,
    instance_idx: dict,
):
    def predict_fn(X_filtered):
        # X_filtered: (n_coalitions, len(feature_indices))
        instance_full = shap_eval[instance_idx["i"]]
        X_full = np.repeat(instance_full[np.newaxis, :], X_filtered.shape[0], axis=0)
        X_full[:, feature_indices] = X_filtered
        return probe.predict_proba(X_full)[:, 1]

    return predict_fn


def run_kernelshap(
    probe,
    shap_eval: np.ndarray,
    background: np.ndarray,
    feature_indices: np.ndarray,
    n_shap_samples: int | str=256,
    seed: int = 42,
    silent: bool = False,
) -> np.ndarray:
    shap_eval_filtered = shap_eval[:, feature_indices]
    background_filtered = background[:, feature_indices]

    instance_idx = {"i": 0}
    predict_fn = make_probe_predict_fn(probe, feature_indices, shap_eval, instance_idx)
    explainer = shap.KernelExplainer(
        predict_fn,
        background_filtered,
        seed=seed,
    )

    explanations = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        iterator = range(len(shap_eval_filtered))
        if not silent:
            iterator = tqdm(
                iterator,
                file=sys.stderr,
                dynamic_ncols=False,
                ncols=100,
                mininterval=1.0,
            )
        for i in iterator:
            instance_idx["i"] = i
            explanations.append(
                explainer.explain(
                    shap_eval_filtered[i : i + 1],
                    nsamples=n_shap_samples,
                )
            )

    return np.vstack(explanations)  # (n_eval, n_filtered_features)


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



def main(
    checkpoint_dir: Path | None = None,
    layer: int = 7,
    out_dir: Path | None = None,
    n_shap: int = 500,
):
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else checkpoint_path()
    out_dir = Path(out_dir) if out_dir else output_path("3_shap")

    probe = joblib.load(checkpoint_dir / f"probe_layer_{layer}.joblib")
    probe.verbose = 0

    # Mask comes from val; attributions come from the shap held-out split.
    val_activations = load_activations("val", layer, mmap=True)
    feature_indices = get_shap_feature_mask(probe, val_activations)

    shap_eval, background, eval_idx, bg_idx = prepare_shap_inputs(
        layer=layer, n_shap=n_shap
    )
    print(
        f"KernelSHAP on {len(shap_eval)} of the canonical {DEFAULT_N_EVAL} held-out "
        f"rows ({len(background)} background). Downstream scripts that default to "
        f"DEFAULT_N_EVAL describe a superset of these rows."
    )
    shap_values = run_kernelshap(probe, shap_eval, background, feature_indices)
    Phi = build_attribution_matrix(shap_values, feature_indices)
    top_k_idx, top_k_Phi = get_top_shap_features(Phi)
    print(top_k_idx)
    print(top_k_Phi)

    out = out_dir
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "shap_values_raw.npy", shap_values)
    np.save(out / "shap_feature_indices.npy", feature_indices)
    np.save(out / f"phi_sentiment_layer{layer}.npy", Phi)
    np.save(out / f"top_k_idx_layer{layer}.npy", top_k_idx)
    np.save(out / f"top_k_Phi_layer{layer}.npy", top_k_Phi)
    np.save(out / f"background_indices_layer{layer}.npy", bg_idx)
    np.save(out / f"eval_indices_layer{layer}.npy", eval_idx)
    print(f"\nWrote outputs -> {out}/")


if __name__ == "__main__":
    main()
