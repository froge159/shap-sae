"""
Plot the MLP-arm attribution-agreement matrix from outputs/13_mlp_ranking/.

A 4x4 Spearman-rho heatmap (Probe, IG, GA, SHAP; |score|, n=116 filtered
features) reads faster than the 6-row correlation table it replaces in the
manuscript's main results: the IG/GA/SHAP cluster and the isolated Probe row
are visible at a glance instead of requiring the reader to scan pairs.

Usage
-----
  uv run python src/global_mlp/plot_ranking_agreement.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils import output_path

RANKING_DIR = output_path("13_mlp_ranking")
OUT_DIR = output_path("13_mlp_ranking")

METHODS = ("probe", "ig", "ga", "shap")
METHOD_LABELS = {"probe": "Probe", "ig": "IG", "ga": "GA", "shap": "SHAP"}
SCORE_FILES = {
    "probe": "probe_scores.npy",
    "ig": "ig_scores.npy",
    "ga": "ga_scores.npy",
    "shap": "shap_scores.npy",
}


def load_scores(ranking_dir: Path) -> dict[str, np.ndarray]:
    return {m: np.load(ranking_dir / SCORE_FILES[m]) for m in METHODS}


def correlation_matrix(scores: dict[str, np.ndarray]) -> np.ndarray:
    """n_methods x n_methods Spearman rho on |score|; diagonal = 1."""
    n = len(METHODS)
    mat = np.eye(n)
    for i, mi in enumerate(METHODS):
        for j, mj in enumerate(METHODS):
            if j <= i:
                continue
            rho, _ = spearmanr(np.abs(scores[mi]), np.abs(scores[mj]))
            mat[i, j] = mat[j, i] = rho
    return mat


def plot_heatmap(mat: np.ndarray, n_features: int, ax: plt.Axes | None = None) -> plt.Axes:
    created_fig = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.8))

    im = ax.imshow(mat, vmin=-1.0, vmax=1.0, cmap="RdBu_r")
    labels = [METHOD_LABELS[m] for m in METHODS]
    ax.set_xticks(range(len(METHODS)))
    ax.set_yticks(range(len(METHODS)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(len(METHODS)):
        for j in range(len(METHODS)):
            val = mat[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            weight = "bold" if i == j else "normal"
            ax.text(
                j, i, f"{val:.3f}", ha="center", va="center",
                color=color, fontsize=10, fontweight=weight,
            )

    ax.set_title(f"MLP-arm attribution agreement\n(Spearman $\\rho$ on $|$score$|$, $n$={n_features})")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman $\\rho$")

    if created_fig:
        plt.tight_layout()
    return ax


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ranking-dir", type=Path, default=RANKING_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scores = load_scores(args.ranking_dir)
    n_features = len(next(iter(scores.values())))
    mat = correlation_matrix(scores)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    plot_heatmap(mat, n_features, ax=ax)
    fig.tight_layout()
    path = args.out_dir / "mlp_ranking_agreement_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
