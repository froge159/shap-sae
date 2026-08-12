import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils import checkpoint_path

LAYERS = tuple(range(7, 8))  # manually adjust


def load_layer_activations(split_dir: Path, layer: int) -> tuple[np.ndarray, np.ndarray]:
    layer_dir = split_dir / f"layer_{layer}"
    X = np.load(layer_dir / "activations.npy")
    y = np.load(layer_dir / "labels.npy")
    return X, y


def train_probe(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """
    L1 logistic probe on full-width SAE activations.

    `l1_ratio=1` is the L1 selector, and it is load-bearing rather than cosmetic:
    `utils.get_shap_feature_mask` is `(coef != 0) & frequent_enough`, so
    under L2 almost nothing is exactly zero (~12k non-zero vs ~1.4k on the same
    rows), the sparsity filter degenerates towards a no-op, and the filtered
    feature set — which gates the MLP probe's inputs and the steering candidate
    pool — grows without anything erroring.

    Do not "modernise" this to `penalty="l1"`: sklearn deprecated `penalty` in
    1.8 in favour of `l1_ratio`, and passing both warns about inconsistent values
    (`penalty=l1 with l1_ratio=0.0`). This project pins scikit-learn>=1.9.
    """
    probe = LogisticRegression(
        max_iter=1000,
        C=1,
        solver="liblinear",
        l1_ratio=1,
        verbose=0,
    )
    probe.fit(X_train, y_train)
    return probe


def evaluate_probe(probe: LogisticRegression, X_val: np.ndarray, y_val: np.ndarray, layer: int) -> dict:
    y_pred = probe.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    weights = probe.coef_[0]
    n_nonzero = int(np.count_nonzero(weights))

    top_k = 20
    top_indices = np.argsort(np.abs(weights))[-top_k:][::-1]
    top_features = [
        {"feature_idx": int(idx), "weight": float(weights[idx])}
        for idx in top_indices
    ]

    return {
        "layer": layer,
        "accuracy": float(accuracy),
        "n_nonzero_weights": n_nonzero,
        "n_features": int(len(weights)),
        "top_features": top_features,
    }


def main():
    for LAYER in LAYERS:
        train_dir = Path("activations/probe_train")
        val_dir = Path("activations/probe_val")

        X_train, y_train = load_layer_activations(train_dir, LAYER)
        X_val, y_val = load_layer_activations(val_dir, LAYER)

        probe = train_probe(X_train, y_train)
        results = evaluate_probe(probe, X_val, y_val, LAYER)

        ckpt_dir = checkpoint_path()
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(probe, ckpt_dir / f"probe_layer_{LAYER}.joblib")

        print(f"Layer {LAYER} probe results")
        print(f"  Val accuracy:        {results['accuracy']:.4f}")
        print(f"  Non-zero weights:    {results['n_nonzero_weights']} / {results['n_features']}")
        print("  Top 20 features by |weight|:")

        for rank, feat in enumerate(results["top_features"], start=1):
            print(
                f"    {rank:2d}. feature {feat['feature_idx']:5d}  "
                f"weight {feat['weight']:+.6f}"
            )


if __name__ == "__main__":
    main()
