"""
Formal statistics for the local-vs-global faithfulness comparison. CPU only.

Reanalyses artifacts the local scripts already wrote — no model, no SAE, no new
forward passes. Everything here is free.

What it adds over `summary.txt`
-------------------------------
The local scripts currently report "mean rho +- std" per block and a raw
win-fraction. That is a description, not a test, and it leaves the paper's most
interesting local result (local SHAP wins 100/100 in block [A], then loses in
block [B]) resting on two point estimates. This script adds:

  1. A paired test. Per example, `rho_local - rho_global` is a paired difference,
     so Wilcoxon signed-rank is the right test (bounded, non-normal statistic,
     n=100). Reported for both blocks and both correlation types, with a
     bootstrap CI on the mean gap for effect size.

  2. A decomposition. If the block [A] margin is largely "local knows which
     features are active here", then the per-example gap should track how many
     of that example's candidates are active. Regressing gap on the active
     fraction quantifies how much of the margin activity explains, instead of
     asserting it. Three parameterisations are reported (fraction, raw count,
     and fraction controlling for candidate-set size) so the conclusion is not
     an artifact of one choice.

  3. A continuous control. Block [B] splits candidates on active/inactive, which
     throws away how strongly each feature fires. The partial Spearman of
     attribution vs delta controlling for activation magnitude keeps the whole
     candidate set and removes the activity signal continuously. Computed from
     three pairwise Spearman rhos via the standard residualised-rank identity,
     so it needs no extra dependency.

Usage
-----
    uv run python src/local/analyze_local_vs_global.py
    uv run python src/local/analyze_local_vs_global.py --in-dir $OUT/9_local_steer
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import linregress, spearmanr, wilcoxon

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils import CANONICAL_SEED, load_activations, output_path

N_BOOTSTRAP = 5000
DEFAULT_DIRS = ("9_local_steer", "14_local_mlp_steer", "local_shap_faithfulness")


def bootstrap_ci_mean(
    values: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = CANONICAL_SEED
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean, resampling over examples."""
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_test(local: np.ndarray, glob: np.ndarray) -> dict:
    """
    Wilcoxon signed-rank on the per-example gap, plus effect size.

    Signed-rank rather than a paired t-test: rho is bounded in [-1, 1] and its
    per-example distribution is visibly skewed, so a mean-difference t is not
    safe at n=100.
    """
    finite = np.isfinite(local) & np.isfinite(glob)
    gap = local[finite] - glob[finite]
    row = {
        "n": int(finite.sum()),
        "mean_local": float(np.mean(local[finite])) if finite.any() else float("nan"),
        "mean_global": float(np.mean(glob[finite])) if finite.any() else float("nan"),
        "mean_gap": float(np.mean(gap)) if len(gap) else float("nan"),
        "median_gap": float(np.median(gap)) if len(gap) else float("nan"),
        "frac_local_wins": float(np.mean(gap > 0)) if len(gap) else float("nan"),
    }
    if len(gap) >= 2 and np.any(gap != 0):
        stat, p = wilcoxon(gap)
        row["wilcoxon_stat"] = float(stat)
        row["wilcoxon_p"] = float(p)
    else:
        row["wilcoxon_stat"] = float("nan")
        row["wilcoxon_p"] = float("nan")
    lo, hi = bootstrap_ci_mean(gap)
    row["gap_ci95"] = [lo, hi]
    return row


def regress(y: np.ndarray, x: np.ndarray, label: str) -> dict:
    finite = np.isfinite(y) & np.isfinite(x)
    if finite.sum() < 3 or np.ptp(x[finite]) == 0:
        return {"label": label, "slope": float("nan"), "r2": float("nan"), "p": float("nan")}
    res = linregress(x[finite], y[finite])
    return {
        "label": label,
        "slope": float(res.slope),
        "intercept": float(res.intercept),
        "r2": float(res.rvalue**2),
        "p": float(res.pvalue),
        "n": int(finite.sum()),
    }


def regress_multi(y: np.ndarray, X: np.ndarray, labels: list[str]) -> dict:
    """Least-squares with an intercept; R^2 against the mean model."""
    finite = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if finite.sum() < X.shape[1] + 2:
        return {"labels": labels, "coefs": [float("nan")] * X.shape[1], "r2": float("nan")}
    yy = y[finite]
    design = np.column_stack([np.ones(finite.sum()), X[finite]])
    beta, *_ = np.linalg.lstsq(design, yy, rcond=None)
    resid = yy - design @ beta
    ss_tot = float(np.sum((yy - yy.mean()) ** 2))
    r2 = float(1.0 - np.sum(resid**2) / ss_tot) if ss_tot > 0 else float("nan")
    return {
        "labels": labels,
        "intercept": float(beta[0]),
        "coefs": [float(b) for b in beta[1:]],
        "r2": r2,
        "n": int(finite.sum()),
    }


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """
    Spearman correlation of x and y controlling for z, in closed form.

    rho_{xy.z} = (r_xy - r_xz*r_yz) / sqrt((1 - r_xz^2)(1 - r_yz^2))

    Keeps the whole candidate set (unlike the binary active/inactive split) while
    removing the part of both attribution and effect that activation magnitude
    already explains.
    """
    if len(x) < 4:
        return float("nan")
    r_xy = spearmanr(x, y).statistic
    r_xz = spearmanr(x, z).statistic
    r_yz = spearmanr(y, z).statistic
    if not all(np.isfinite([r_xy, r_xz, r_yz])):
        return float("nan")
    denom = np.sqrt(max((1 - r_xz**2) * (1 - r_yz**2), 0.0))
    if denom < 1e-12:
        return float("nan")
    return float((r_xy - r_xz * r_yz) / denom)


def load_ragged(path: Path) -> list[np.ndarray]:
    return [np.asarray(a, dtype=np.int64) for a in np.load(path, allow_pickle=True)]


def load_packed_rhos(in_dir: Path) -> dict[str, np.ndarray] | None:
    """
    Per-example ρ arrays, plus candidate counts.

    The local steer scripts now write a single `rhos.npz`. Older runs dumped
    one `.npy` per method/block; those keys get remapped to the packed layout.
    """
    packed = in_dir / "rhos.npz"
    if packed.exists():
        with np.load(packed) as z:
            return {k: z[k] for k in z.files}

    arrays: dict[str, np.ndarray] = {}
    for npy in in_dir.glob("*.npy"):
        stem = npy.stem
        if stem.startswith("rho_"):
            arrays[stem[len("rho_") :]] = np.load(npy)
        elif stem.startswith("active_rho_"):
            arrays["active_" + stem[len("active_rho_") :]] = np.load(npy)
    for src, dst in (
        ("n_cands_per_example.npy", "n_cands"),
        ("n_active_cands_per_example.npy", "n_active"),
    ):
        path = in_dir / src
        if path.exists():
            arrays[dst] = np.load(path)
    return arrays or None


def continuous_control(in_dir: Path, mode: str) -> dict | None:
    """
    Partial Spearman controlling for activation magnitude, per example.

    Needs the raw per-example deltas and the candidate sets. The local steer
    scripts no longer save those by default; returns None when they are absent.
    """
    needed = [
        in_dir / "local_shap.npy",
        in_dir / "union_local.npy",
        in_dir / "cands_per_example.npy",
        in_dir / "feature_indices.npy",
        in_dir / "example_indices.npy",
        in_dir / f"delta_signed_{mode}.npy",
    ]
    if not all(p.exists() for p in needed):
        return None

    local_shap = np.load(in_dir / "local_shap.npy")
    union_local = np.load(in_dir / "union_local.npy")
    cands = load_ragged(in_dir / "cands_per_example.npy")
    feature_indices = np.load(in_dir / "feature_indices.npy")
    example_idx = np.load(in_dir / "example_indices.npy")
    delta_signed = np.load(in_dir / f"delta_signed_{mode}.npy")
    # Orientation matches the local scripts: positive == faithful.
    oriented = -delta_signed if mode == "ablation" else delta_signed

    # Activation magnitudes for exactly the rows/features under test.
    acts = load_activations("shap", 7, mmap=True)
    X = np.asarray(acts[example_idx][:, feature_indices], dtype=np.float64)

    global_phi = None
    gp = in_dir / "global_phi_filtered.npy"
    if gp.exists():
        global_phi = np.load(gp).astype(np.float64)

    col_of = {int(f): j for j, f in enumerate(union_local)}
    rows_local, rows_global = [], []
    for e, cand in enumerate(cands):
        if len(cand) < 4:
            rows_local.append(np.nan)
            rows_global.append(np.nan)
            continue
        cols = np.asarray([col_of[int(c)] for c in cand], dtype=np.int64)
        d = np.abs(oriented[e, cols])
        a = np.abs(X[e, cand])
        rows_local.append(partial_spearman(np.abs(local_shap[e, cand]), d, a))
        if global_phi is not None:
            rows_global.append(partial_spearman(np.abs(global_phi[cand]), d, a))
        else:
            rows_global.append(np.nan)

    loc = np.asarray(rows_local, dtype=np.float64)
    glo = np.asarray(rows_global, dtype=np.float64)
    out = {
        "description": (
            "Partial Spearman of |attribution| vs |oriented delta|, controlling "
            "for |activation| — continuous alternative to the binary block [B]."
        ),
        "mean_partial_rho_local": float(np.nanmean(loc)),
        "n_finite_local": int(np.isfinite(loc).sum()),
    }
    if np.isfinite(glo).any():
        out["mean_partial_rho_global"] = float(np.nanmean(glo))
        out["n_finite_global"] = int(np.isfinite(glo).sum())
        out["paired"] = paired_test(loc, glo)
    return out


def analyze_dir(in_dir: Path, mode: str) -> dict | None:
    """Both blocks, both correlation types, for one local-script output dir."""
    if not in_dir.exists():
        print(f"  skip {in_dir} (missing)")
        return None

    result: dict = {"dir": str(in_dir), "blocks": {}}

    # Packed rhos.npz (current local-steer layout), leftover per-file dumps,
    # or local_shap_faithfulness's rho_local / rho_global names.
    rhos = load_packed_rhos(in_dir)
    simple_layout = (in_dir / "rho_local.npy").exists()
    has_local_global = rhos is not None and (
        "importance_local_SHAP" in rhos and "importance_global_SHAP" in rhos
    )
    if not (has_local_global or simple_layout):
        print(f"  skip {in_dir} (no recognised rho arrays)")
        return None

    if has_local_global:
        assert rhos is not None
        for block, prefix in (("A", ""), ("B", "active_")):
            for kind in ("importance", "directional"):
                lk = f"{prefix}{kind}_local_SHAP"
                gk = f"{prefix}{kind}_global_SHAP"
                if lk not in rhos or gk not in rhos:
                    continue
                result["blocks"][f"{block}:{kind}"] = paired_test(rhos[lk], rhos[gk])
    else:
        result["blocks"]["A:importance"] = paired_test(
            np.load(in_dir / "rho_local.npy"), np.load(in_dir / "rho_global.npy")
        )
        for block, prefix in (("B", "active_"),):
            lp = in_dir / f"{prefix}rho_local.npy"
            gp = in_dir / f"{prefix}rho_global.npy"
            if lp.exists() and gp.exists():
                result["blocks"][f"{block}:importance"] = paired_test(
                    np.load(lp), np.load(gp)
                )

    # Decomposition: how much of the block [A] gap does activity explain?
    if (
        has_local_global
        and rhos is not None
        and "n_cands" in rhos
        and "n_active" in rhos
    ):
        n_cands = rhos["n_cands"].astype(np.float64)
        n_active = rhos["n_active"].astype(np.float64)
        frac_active = np.divide(
            n_active, n_cands, out=np.full_like(n_cands, np.nan), where=n_cands > 0
        )
        loc = rhos["importance_local_SHAP"]
        glo = rhos["importance_global_SHAP"]
        gap = loc - glo
        result["decomposition"] = {
            "target": "block A importance gap (rho_local - rho_global)",
            "frac_active": regress(gap, frac_active, "gap ~ frac_active"),
            "n_active_raw": regress(gap, n_active, "gap ~ n_active"),
            "frac_active_ctrl_size": regress_multi(
                gap,
                np.column_stack([frac_active, n_cands]),
                ["frac_active", "n_cands"],
            ),
            "mean_frac_active": float(np.nanmean(frac_active)),
        }

    cont = continuous_control(in_dir, mode)
    if cont:
        result["continuous_control"] = cont
    return result


def render(res: dict) -> str:
    lines = [
        f"Local vs global faithfulness - formal tests",
        "=" * 62,
        f"source: {res['dir']}",
        "",
        "Paired Wilcoxon signed-rank on per-example (rho_local - rho_global).",
        "Positive gap = local SHAP ranks that example's effects better.",
        "",
        f"{'block:kind':22s} {'local':>7s} {'global':>7s} {'gap':>7s} "
        f"{'95% CI':>17s} {'wilcoxon p':>11s} {'win frac':>9s}",
        "-" * 88,
    ]
    for key, row in res["blocks"].items():
        ci = f"[{row['gap_ci95'][0]:+.3f},{row['gap_ci95'][1]:+.3f}]"
        lines.append(
            f"{key:22s} {row['mean_local']:+7.3f} {row['mean_global']:+7.3f} "
            f"{row['mean_gap']:+7.3f} {ci:>17s} {row['wilcoxon_p']:11.2e} "
            f"{row['frac_local_wins']:9.3f}"
        )

    if "decomposition" in res:
        d = res["decomposition"]
        lines += [
            "",
            "Decomposition of the block [A] importance gap:",
            f"  mean active fraction of candidates = {d['mean_frac_active']:.3f}",
        ]
        for key in ("frac_active", "n_active_raw"):
            r = d[key]
            lines.append(
                f"  {r['label']:28s} slope={r['slope']:+.3f}  R2={r['r2']:.3f}  "
                f"p={r['p']:.2e}"
            )
        m = d["frac_active_ctrl_size"]
        lines.append(
            f"  {'gap ~ frac_active + n_cands':28s} "
            f"coefs={[round(c, 3) for c in m['coefs']]}  R2={m['r2']:.3f}"
        )
        lines.append(
            "  A high R2 here means the local-vs-global margin is largely an "
            "activity effect,\n  not a ranking-quality effect."
        )

    if "continuous_control" in res:
        c = res["continuous_control"]
        lines += ["", "Continuous control (partial Spearman | activation magnitude):"]
        lines.append(
            f"  mean partial rho local  = {c['mean_partial_rho_local']:+.3f} "
            f"(n={c['n_finite_local']})"
        )
        if "mean_partial_rho_global" in c:
            lines.append(
                f"  mean partial rho global = {c['mean_partial_rho_global']:+.3f} "
                f"(n={c['n_finite_global']})"
            )
            p = c["paired"]
            lines.append(
                f"  paired gap = {p['mean_gap']:+.3f}  wilcoxon p={p['wilcoxon_p']:.2e}"
            )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--in-dir",
        type=Path,
        nargs="*",
        default=None,
        help="Local-script output dirs (default: all three under $SHAP_SAE_OUTPUTS)",
    )
    p.add_argument(
        "--mode",
        choices=("ablation", "steering"),
        default="ablation",
        help="Mode the local run used (only needed for the optional continuous control)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dirs = args.in_dir or [output_path(d) for d in DEFAULT_DIRS]
    for d in dirs:
        d = Path(d)
        print(f"Analysing {d} ...")
        res = analyze_dir(d, args.mode)
        if res is None:
            continue
        text = render(res)
        print(text)
        with open(d / "paired_tests.json", "w") as f:
            json.dump(res, f, indent=2)
        (d / "paired_tests.txt").write_text(text)
        print(f"Wrote -> {d}/paired_tests.{{json,txt}}\n")


if __name__ == "__main__":
    main()
