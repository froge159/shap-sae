"""
Train a small 2-layer MLP probe on the filtered SAE features used for SHAP.

Unlike the L1 logistic probe in core/2_probes.py, gradients here are
input-dependent, so IG/GA no longer get a built-in advantage from constant
∂f/∂x = w. Nonlinearity also lets feature interactions exist in f.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

LAYERS = tuple(range(7, 8))
FEATURE_INDICES_PATH = Path("outputs/3_shap/shap_feature_indices.npy")
# One hidden layer → two affine maps (input→hidden→logit): a small 2-layer MLP.
HIDDEN_DIM = 32
MAX_ITER = 500
RANDOM_STATE = 42


def load_layer_activations(split_dir: Path, layer: int) -> tuple[np.ndarray, np.ndarray]:
    layer_dir = split_dir / f"layer_{layer}"
    X = np.load(layer_dir / "activations.npy")
    y = np.load(layer_dir / "labels.npy")
    return X, y


def filter_features(X: np.ndarray, feature_indices: np.ndarray) -> np.ndarray:
    """Keep only global SAE indices from the SHAP feature mask."""
    return np.asarray(X[:, feature_indices], dtype=np.float32)


def train_probe(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    probe = MLPClassifier(
        hidden_layer_sizes=(HIDDEN_DIM,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=MAX_ITER,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=RANDOM_STATE,
        verbose=False,
    )
    probe.fit(X_train, y_train)
    return probe


def input_feature_saliency(probe: MLPClassifier) -> np.ndarray:
    """
    Proxy importance per filtered input: L2 norm of first-layer weights
    into that input (shape: n_filtered_features,).
    """
    # coefs_[0]: (n_features, hidden_dim)
    W1 = probe.coefs_[0]
    return np.linalg.norm(W1, axis=1)


def evaluate_probe(
    probe: MLPClassifier,
    X_val: np.ndarray,
    y_val: np.ndarray,
    layer: int,
    feature_indices: np.ndarray,
) -> dict:
    y_pred = probe.predict(X_val)
    y_proba = probe.predict_proba(X_val)[:, 1]
    accuracy = accuracy_score(y_val, y_pred)
    auroc = roc_auc_score(y_val, y_proba)

    saliency = input_feature_saliency(probe)
    top_k = min(20, len(saliency))
    top_local = np.argsort(saliency)[-top_k:][::-1]
    top_features = [
        {
            "feature_idx": int(feature_indices[i]),  # global SAE index
            "local_idx": int(i),
            "first_layer_l2": float(saliency[i]),
        }
        for i in top_local
    ]

    return {
        "layer": layer,
        "accuracy": float(accuracy),
        "auroc": float(auroc),
        "n_features": int(X_val.shape[1]),
        "hidden_dim": HIDDEN_DIM,
        "n_iter": int(getattr(probe, "n_iter_", 0)),
        "top_features": top_features,
    }


def main():
    feature_indices = np.load(FEATURE_INDICES_PATH)
    print(
        f"Using {len(feature_indices)} filtered features from {FEATURE_INDICES_PATH}"
    )

    out_ckpt = Path("checkpoints")
    out_ckpt.mkdir(parents=True, exist_ok=True)
    out_dir = Path("outputs/11_mlp_probe")
    out_dir.mkdir(parents=True, exist_ok=True)

    for LAYER in LAYERS:
        train_dir = Path("activations/probe_train")
        val_dir = Path("activations/probe_val")

        X_train, y_train = load_layer_activations(train_dir, LAYER)
        X_val, y_val = load_layer_activations(val_dir, LAYER)

        X_train = filter_features(X_train, feature_indices)
        X_val = filter_features(X_val, feature_indices)

        probe = train_probe(X_train, y_train)
        results = evaluate_probe(probe, X_val, y_val, LAYER, feature_indices)

        # Bundle indices so downstream code knows inputs are filtered globals.
        payload = {
            "probe": probe,
            "feature_indices": feature_indices,
            "layer": LAYER,
            "hidden_dim": HIDDEN_DIM,
        }
        joblib.dump(payload, out_ckpt / f"mlp_probe_layer_{LAYER}.joblib")
        np.save(out_dir / f"feature_indices_layer_{LAYER}.npy", feature_indices)

        top_features_path = out_dir / f"top_features_layer_{LAYER}.json"
        with open(top_features_path, "w") as f:
            json.dump(results["top_features"], f, indent=2)

        summary_path = out_dir / f"results_layer_{LAYER}.json"
        with open(summary_path, "w") as f:
            json.dump(
                {k: v for k, v in results.items() if k != "top_features"},
                f,
                indent=2,
            )

        print(f"Layer {LAYER} MLP probe results")
        print(f"  Val accuracy:     {results['accuracy']:.4f}")
        print(f"  Val AUROC:        {results['auroc']:.4f}")
        print(f"  Features:         {results['n_features']}")
        print(f"  Hidden dim:       {results['hidden_dim']}")
        print(f"  Adam iterations:  {results['n_iter']}")
        print(f"  Checkpoint:       checkpoints/mlp_probe_layer_{LAYER}.joblib")
        print(f"  Top features →    {top_features_path}")
        print("  Top 20 inputs by ||W1[i,:]||₂ (global SAE index):")
        for rank, feat in enumerate(results["top_features"], start=1):
            print(
                f"    {rank:2d}. feature {feat['feature_idx']:5d}  "
                f"||W1||₂={feat['first_layer_l2']:.6f}"
            )


if __name__ == "__main__":
    main()
