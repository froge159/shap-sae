import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

LAYER = 10  # manually adjust


def load_layer_activations(split_dir: Path, layer: int) -> tuple[np.ndarray, np.ndarray]:
    layer_dir = split_dir / f"layer_{layer}"
    X = np.load(layer_dir / "activations.npy")
    y = np.load(layer_dir / "labels.npy")
    return X, y


def train_probe(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    probe = LogisticRegression(
        max_iter=1000,
        C=0.01,
        solver="saga",
        penalty="l1",
    )
    probe.fit(X_train, y_train)
    return probe


def evaluate_probe(probe: LogisticRegression, X_val: np.ndarray, y_val: np.ndarray) -> dict:
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
        "layer": LAYER,
        "accuracy": float(accuracy),
        "n_nonzero_weights": n_nonzero,
        "n_features": int(len(weights)),
        "top_features": top_features,
    }


def main():
    train_dir = Path("activations/probe_train")
    val_dir = Path("activations/probe_val")

    X_train, y_train = load_layer_activations(train_dir, LAYER)
    X_val, y_val = load_layer_activations(val_dir, LAYER)

    probe = train_probe(X_train, y_train)
    results = evaluate_probe(probe, X_val, y_val)

    
    out_dir = Path("outputs") / f"layer_{LAYER}"
    out_dir.mkdir(parents=True, exist_ok=True)
    top_features_path = out_dir / "top_features.json"
    with open(top_features_path, "w") as f:
        # json.dump(results["top_features"], f, indent=2)
        pass

    print(f"Layer {LAYER} probe results")
    print(f"  Val accuracy:        {results['accuracy']:.4f}")
    print(f"  Non-zero weights:    {results['n_nonzero_weights']} / {results['n_features']}")
    #print(f"  Top 20 features saved to {top_features_path}")
    #print("  Top 20 features by |weight|:")
    """
    for rank, feat in enumerate(results["top_features"], start=1):
        print(
            f"    {rank:2d}. feature {feat['feature_idx']:5d}  "
            f"weight {feat['weight']:+.6f}"
        )
    """


if __name__ == "__main__":
    main()
