"""
Sign-inversion reanalysis of *existing* steer outputs. No GPU, no new forward passes.

Question
--------
`9.5_tuning`'s pilot reports `mean_signed_dp` negative at every alpha on top-|SHAP|
features: steering a feature SHAP calls positive-sentiment in the positive
direction made the layer-11 residual probe *less* confident of "positive". That
readout is not the one the steer scripts use, so the obvious first question is
whether the same inversion is visible in the logit-diff numbers already on disk.

It is answerable for free: `model_deltas_<tag>.json` already stores per-feature
`delta_signed` from the real logit-diff readout, and the signed attribution of each
of those features is in the saved Phi. This script joins the two and runs a
binomial sign test per attribution method.

Scope and honesty
-----------------
On the shipped `steering_top_k20` runs the candidate set is *mixed sign* (11
positive, 9 negative Phi in run2's linear arm), which both dilutes a
sign-specific effect and leaves n=20 — badly underpowered for a binomial test.
Treat what this script prints as motivating, not conclusive, and read it next to
a dedicated sign-pure run:

    9_steer.py --selection top-pos ...
    9_steer.py --selection top-neg ...

which this script also handles (it reads whatever tag it is pointed at, and those
runs additionally embed their own `sign_agreement` block).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils import checkpoint_path, output_path, sign_agreement_test

LAYER = 7


def orient(delta_signed: dict[int, float], mode: str) -> dict[int, float]:
    """Same convention as 9_steer.orient_signed_effects: positive == faithful."""
    if mode == "ablation":
        return {k: -v for k, v in delta_signed.items()}
    if mode == "steering":
        return dict(delta_signed)
    raise ValueError(f"Unknown mode={mode!r}")


def load_linear_scores() -> dict[str, np.ndarray]:
    probe = joblib.load(checkpoint_path(f"probe_layer_{LAYER}.joblib"))
    return {
        "SHAP": np.load(
            output_path("7_shap_recompute", "phi_sentiment_layer7_signed.npy")
        ).astype(np.float64),
        "Probe weights": np.asarray(probe.coef_[0], dtype=np.float64),
        "IG": np.load(output_path("8_rankings_recompute", "ig_scores.npy")).astype(
            np.float64
        ),
        "GA": np.load(output_path("8_rankings_recompute", "ga_scores.npy")).astype(
            np.float64
        ),
    }


def load_mlp_scores() -> dict[str, np.ndarray]:
    """
    Full-width signed scores for the MLP arm.

    `Probe saliency` is omitted on purpose: it is ||W1[i,:]||_2, non-negative, so
    a sign test on it would be asking whether a magnitude predicts a direction.
    """
    d = output_path("13_mlp_ranking")
    fi = np.load(d / "feature_indices.npy")
    out = {}
    for name, fname in (("SHAP", "shap_scores.npy"), ("IG", "ig_scores.npy"), ("GA", "ga_scores.npy")):
        full = np.zeros(32768, dtype=np.float64)
        full[fi] = np.load(d / fname).astype(np.float64)
        out[name] = full
    return out


def analyze(deltas_path: Path, scores: dict[str, np.ndarray]) -> dict:
    with open(deltas_path) as f:
        payload = json.load(f)
    mode = payload["mode"]
    delta_signed = {int(k): float(v) for k, v in payload["delta_signed"].items()}
    oriented = orient(delta_signed, mode)

    result = {
        "source": str(deltas_path),
        "mode": mode,
        "selection": payload.get("selection"),
        "k": payload.get("k"),
        "alpha_mode": payload.get("alpha_mode"),
        "n_eval": payload.get("n_eval"),
        "methods": {},
    }
    for name, vec in scores.items():
        signed = {f: float(vec[f]) for f in oriented}
        block = sign_agreement_test(signed, oriented)
        # Split by the sign of the attribution: an inversion may be specific to
        # positive-Phi features, which a pooled test would wash out.
        for label, keep in (
            ("positive_phi", lambda s: s > 0),
            ("negative_phi", lambda s: s < 0),
        ):
            sub = {f: s for f, s in signed.items() if keep(s)}
            block[label] = sign_agreement_test(
                sub, {f: oriented[f] for f in sub}
            )
        result["methods"][name] = block
    return result


def render(res: dict) -> str:
    lines = [
        "Sign-agreement reanalysis (existing deltas, logit-diff readout)",
        "=" * 66,
        f"source={Path(res['source']).name}  mode={res['mode']}  "
        f"selection={res['selection']}  k={res['k']}  n_eval={res['n_eval']}",
        "",
        "Agreement = sign(attribution) matches sign(oriented delta).",
        "'inversion p' is the one-sided binomial test for agreement BELOW chance.",
        "",
        f"{'method':16s} {'group':14s} {'agree':>9s} {'rate':>7s} "
        f"{'two-sided p':>12s} {'inversion p':>12s}",
        "-" * 76,
    ]
    for name, block in res["methods"].items():
        for label in ("(all)", "positive_phi", "negative_phi"):
            row = block if label == "(all)" else block[label]
            if row["n"] == 0:
                lines.append(f"{name:16s} {label:14s} {'n/a':>9s}")
                continue
            lines.append(
                f"{name:16s} {label:14s} {row['n_agree']:4d}/{row['n']:<4d} "
                f"{row['agreement_rate']:7.3f} {row['binom_two_sided_p']:12.4f} "
                f"{row['binom_less_p']:12.4f}"
            )
    lines.append("")
    n_any = max((b["n"] for b in res["methods"].values()), default=0)
    if n_any < 30:
        lines.append(
            f"CAVEAT: n={n_any} features. A binomial test at this n cannot resolve "
            "anything\nbut a very large inversion. Read as motivating; run "
            "--selection top-pos / top-neg\nfor a sign-pure, adequately-powered test."
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "deltas",
        type=Path,
        nargs="+",
        help="One or more model_deltas_<tag>.json files to reanalyse",
    )
    p.add_argument(
        "--arm",
        choices=("linear", "mlp"),
        default="linear",
        help="Which arm's attribution scores to join against",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write sign_test_<tag>.json (default: beside each input)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scores = load_linear_scores() if args.arm == "linear" else load_mlp_scores()
    for path in args.deltas:
        if not path.exists():
            raise FileNotFoundError(path)
        res = analyze(path, scores)
        text = render(res)
        print(text)
        out_dir = args.out_dir or path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = path.stem.replace("model_deltas_", "")
        with open(out_dir / f"sign_test_{tag}.json", "w") as f:
            json.dump(res, f, indent=2)
        (out_dir / f"sign_test_{tag}.txt").write_text(text)
        print(f"Wrote -> {out_dir}/sign_test_{tag}.{{json,txt}}\n")


if __name__ == "__main__":
    main()
