"""
Plot effect-vs-cost curves from outputs/group_steering_results.json.

Matches the canvas view: one curve per method, points ordered by ascending α,
cost (Δ PPL) on x, effect (Δ logit-diff) on y.

Examples
--------
  # default: logistic · k=20 → outputs/group_steering/effect_vs_cost_logistic_k20.png
  uv run python src/extra/plot_group_steering.py

  # single panel
  uv run python src/extra/plot_group_steering.py --probe mlp --k 5

  # all (probe × k) panels + a k=20 comparison pair
  uv run python src/extra/plot_group_steering.py --all
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = Path("outputs/group_steering_results.json")
OUT_DIR = Path("outputs/group_steering")

METHODS = ("shap", "probe", "ig", "ga", "actdiff")
METHOD_LABELS = {
    "shap": "SHAP",
    "probe": "Probe",
    "ig": "IG",
    "ga": "GA",
    "actdiff": "ActDiff",
}
# Distinct, colorblind-friendly palette (order matches METHODS).
METHOD_COLORS = {
    "shap": "#4C78A8",
    "probe": "#F58518",
    "ig": "#E45756",
    "ga": "#72B7B2",
    "actdiff": "#54A24B",
}


def load_results(path: Path = RESULTS_PATH) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def index_results(
    rows: list[dict],
) -> dict[tuple[str, str, int], list[dict]]:
    """{(probe_type, method, k): rows sorted by alpha}."""
    by: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for r in rows:
        by[(r["probe_type"], r["method"], int(r["k"]))].append(r)
    for key in by:
        by[key].sort(key=lambda r: float(r["alpha"]))
    return by


def plot_effect_vs_cost(
    by: dict[tuple[str, str, int], list[dict]],
    probe: str,
    k: int,
    ax: plt.Axes | None = None,
    title: str | None = None,
    legend: bool = True,
) -> plt.Axes:
    """
    Single panel: cost on x, effect on y, one polyline per method.
    """
    created_fig = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 5.0))

    for method in METHODS:
        pts = by.get((probe, method, k), [])
        if not pts:
            continue
        xs = [float(p["cost"]) for p in pts]
        ys = [float(p["effect"]) for p in pts]
        ax.plot(
            xs,
            ys,
            "-o",
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            markersize=5,
            linewidth=1.8,
            zorder=3,
        )

    ax.axhline(0.0, color="0.55", linewidth=0.9, zorder=1)
    ax.axvline(0.0, color="0.55", linewidth=0.9, zorder=1)
    ax.set_xlabel("Cost (Δ perplexity, steered − clean)")
    ax.set_ylabel("Effect (Δ logit wonderful − awful)")
    ax.set_title(title or f"{probe} · top-{k} features")
    ax.grid(True, alpha=0.3, zorder=0)
    if legend:
        ax.legend(frameon=False, loc="best")

    if created_fig:
        plt.tight_layout()
    return ax


def save_single(
    by: dict[tuple[str, str, int], list[dict]],
    probe: str,
    k: int,
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    plot_effect_vs_cost(by, probe, k, ax=ax)
    fig.suptitle("Group steering: effect vs cost", fontsize=13, y=1.02)
    path = out_dir / f"effect_vs_cost_{probe}_k{k}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_probe_grid(
    by: dict[tuple[str, str, int], list[dict]],
    probe: str,
    ks: list[int],
    out_dir: Path,
) -> Path:
    """2×3 grid over k for one probe type (canvas k selector as a sheet)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(ks)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.0, 4.0 * nrows), squeeze=False)
    flat = axes.ravel()
    for i, k in enumerate(ks):
        plot_effect_vs_cost(
            by, probe, k, ax=flat[i], title=f"k = {k}", legend=(i == 0)
        )
    for j in range(len(ks), len(flat)):
        flat[j].set_visible(False)

    handles, labels = flat[0].get_legend_handles_labels()
    leg = flat[0].get_legend()
    if leg is not None:
        leg.remove()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(METHODS),
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
        )
    fig.suptitle(
        f"Group steering effect vs cost — {probe} attributions",
        fontsize=13,
        y=1.06,
    )
    fig.tight_layout()
    path = out_dir / f"effect_vs_cost_{probe}_grid.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_k_pair(
    by: dict[tuple[str, str, int], list[dict]],
    k: int,
    out_dir: Path,
) -> Path:
    """Side-by-side logistic vs mlp at fixed k (default canvas comparison)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    for ax, probe in zip(axes, ("logistic", "mlp")):
        plot_effect_vs_cost(
            by, probe, k, ax=ax, title=f"{probe} · k = {k}", legend=True
        )
    fig.suptitle("Group steering: effect vs cost by method", fontsize=13)
    fig.tight_layout()
    path = out_dir / f"effect_vs_cost_k{k}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results",
        type=Path,
        default=RESULTS_PATH,
        help="Path to group_steering_results.json",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory for PNG outputs",
    )
    p.add_argument(
        "--probe",
        choices=("logistic", "mlp"),
        default="logistic",
        help="Attribution probe type (default: logistic)",
    )
    p.add_argument(
        "--k",
        type=int,
        default=20,
        help="Feature count for the single-panel plot (default: 20)",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Also write per-probe k-grids and a logistic|mlp pair at --k",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_results(args.results)
    by = index_results(rows)

    available_ks = sorted({int(r["k"]) for r in rows})
    if args.k not in available_ks:
        raise SystemExit(f"--k={args.k} not in results; available: {available_ks}")

    written: list[Path] = []
    written.append(save_single(by, args.probe, args.k, args.out_dir))

    if args.all:
        for probe in ("logistic", "mlp"):
            written.append(save_probe_grid(by, probe, available_ks, args.out_dir))
        written.append(save_k_pair(by, args.k, args.out_dir))

    print(f"Wrote {len(written)} figure(s) → {args.out_dir}/")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
