"""
Calibrate additive steering alpha (a_i ← a_i + α) without a full multi-feature sweep.

Procedure
---------
1. Estimate natural positive-activation scales for ranking candidates.
2. Build an α grid at {0.5, 1, 2, 4} × median scale.
3. Pilot-steer a few high-|score| features on a small *train* subset.
4. Recommend the smallest α with measurable |ΔP| that is not saturated.

Caveat: ΔP here is a layer-11 residual probe's P(positive). The steer scripts
that consume α read out a GPT-2 logit difference instead, so the acceptance
threshold does not transfer between the two scales — α is a rough magnitude,
not a tuned hyperparameter.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from sae_lens import SAE
from tqdm import tqdm

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils import checkpoint_path, load_model, load_splits, output_path

# --- reuse intervention helpers from 9_steer.py (invalid module name) ---
_STEER_PATH = Path(__file__).resolve().parent / "9_steer.py"
_spec = importlib.util.spec_from_file_location("steer9", _STEER_PATH)
steer9 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(steer9)


LAYER = 7
K_CANDIDATES = 20
N_PILOT_FEATURES = 3
N_PILOT_EXAMPLES = 200
SCALE_MULTIPLIERS = (0.5, 1.0, 2.0, 4.0)
MIN_MEAN_ABS_DP = 0.02
MAX_SATURATION_FRAC = 0.25  # fraction of examples with P < 0.02 or P > 0.98
OUT_DIR = output_path("9.5_tuning")
ACTIVATIONS_PATH = f"activations/probe_train/layer_{LAYER}/activations.npy"


def load_ranking_candidates(k: int = K_CANDIDATES):
    """
    Union of each method's top-k features, as global SAE ids.

    Candidates are drawn from the same filtered pool 9_steer intervenes on, so
    the α this script calibrates is calibrated on features that actually get
    steered. All four score vectors are full width; they are sliced to the
    filtered axis before ranking and mapped back through `feature_indices`.
    """
    feature_indices = np.load(output_path("3_shap", "shap_feature_indices.npy"))
    sae_probe = joblib.load(checkpoint_path(f"probe_layer_{LAYER}.joblib"))

    scores_by_method = {
        "probe": np.asarray(sae_probe.coef_[0], dtype=np.float64),
        "ig": np.load(output_path("8_rankings_recompute", "ig_scores.npy")),
        "ga": np.load(output_path("8_rankings_recompute", "ga_scores.npy")),
        "shap": np.load(
            output_path("7_shap_recompute", "phi_sentiment_layer7_signed.npy")
        ),
    }

    globals_per_method = []
    for scores in scores_by_method.values():
        _, global_idx = steer9.get_shap_candidates(
            scores[feature_indices], feature_indices, k=k, selection="top"
        )
        globals_per_method.append(global_idx)
    cand = np.unique(np.concatenate(globals_per_method))

    shap_scores = scores_by_method["shap"]
    # Pilot features: top-|SHAP| within the candidate union
    order = np.argsort(np.abs(shap_scores[cand]))[::-1]
    pilot_features = cand[order[:N_PILOT_FEATURES]]

    return {
        "probe_scores": scores_by_method["probe"],
        "shap_scores": shap_scores,
        "all_candidates": cand,
        "pilot_features": pilot_features,
    }


def positive_activation_scales(
    activations_path: str,
    feature_indices: np.ndarray,
) -> np.ndarray:
    """
    Per-feature scale = mean(a | a > 0), falling back to std if never active.
    Uses memmap so the full activation tensor is not loaded into RAM.
    """
    acts = np.load(activations_path, mmap_mode="r")
    scales = np.zeros(len(feature_indices), dtype=np.float64)
    for i, feat in enumerate(tqdm(feature_indices, desc="Activation scales")):
        col = np.asarray(acts[:, int(feat)], dtype=np.float64)
        pos = col[col > 0]
        scales[i] = float(pos.mean()) if pos.size > 0 else float(col.std() + 1e-8)
    return scales


def decoder_norms(sae: SAE, feature_indices: np.ndarray) -> np.ndarray:
    W_dec = sae.W_dec.detach().float().cpu().numpy()  # (n_features, d_model)
    return np.linalg.norm(W_dec[np.asarray(feature_indices)], axis=1)


def build_alpha_grid(median_scale: float) -> list[float]:
    return [float(m * median_scale) for m in SCALE_MULTIPLIERS]


def pilot_effects_for_alpha(
    model,
    sae: SAE,
    residual_probe,
    tokens_subset,
    pilot_features: np.ndarray,
    alpha: float,
    readout_layer: int,
    baseline_probs: np.ndarray,
) -> dict:
    """
    Additive steering at `alpha` on pilot features / subset.

    Uses a shared SAE-recon baseline (`baseline_probs`) computed once by the caller.
    """
    intervene_point = f"blocks.{LAYER}.hook_resid_post"
    readout_point = f"blocks.{readout_layer}.hook_resid_post"
    resid_store: dict = {}

    def capture_hook(resid_post, hook):
        resid_store["resid"] = resid_post
        return resid_post

    def probe_prob(tokens: torch.Tensor) -> float:
        pooled = steer9.pool_last_non_pad_token(model, tokens, resid_store["resid"])
        return float(residual_probe.predict_proba(pooled.reshape(1, -1))[0, 1])

    delta_abs: dict[int, float] = {}
    delta_signed: dict[int, float] = {}
    sat_flags: list[bool] = []

    for feat in pilot_features:
        idx = int(feat)

        def intervention_hook(resid_post, hook, feature_idx=idx, _alpha=alpha):
            acts = sae.encode(resid_post).clone()
            acts[..., feature_idx] = acts[..., feature_idx] + _alpha
            return sae.decode(acts)

        intervened_probs = []
        for tokens in tokens_subset:
            tokens = steer9._as_batch(tokens, model.cfg.device)
            with torch.no_grad():
                with model.hooks(
                    fwd_hooks=[
                        (intervene_point, intervention_hook),
                        (readout_point, capture_hook),
                    ]
                ):
                    model(tokens)
                p = probe_prob(tokens)
            intervened_probs.append(p)
            sat_flags.append(p < 0.02 or p > 0.98)

        intervened_probs = np.asarray(intervened_probs)
        diffs = intervened_probs - baseline_probs
        delta_abs[idx] = float(np.mean(np.abs(diffs)))
        delta_signed[idx] = float(np.mean(diffs))

    return {
        "alpha": alpha,
        "mean_abs_dp": float(np.mean(list(delta_abs.values()))),
        "mean_signed_dp": float(np.mean(list(delta_signed.values()))),
        "saturation_frac": float(np.mean(sat_flags)) if sat_flags else 0.0,
        "delta_abs": delta_abs,
        "delta_signed": delta_signed,
    }


def compute_recon_baseline_probs(
    model,
    sae: SAE,
    residual_probe,
    tokens_subset,
    readout_layer: int,
) -> np.ndarray:
    intervene_point = f"blocks.{LAYER}.hook_resid_post"
    readout_point = f"blocks.{readout_layer}.hook_resid_post"
    resid_store: dict = {}

    def capture_hook(resid_post, hook):
        resid_store["resid"] = resid_post
        return resid_post

    def recon_hook(resid_post, hook):
        return sae.decode(sae.encode(resid_post))

    probs = []
    for tokens in tqdm(tokens_subset, desc="Pilot baseline"):
        tokens = steer9._as_batch(tokens, model.cfg.device)
        with torch.no_grad():
            with model.hooks(
                fwd_hooks=[(intervene_point, recon_hook), (readout_point, capture_hook)]
            ):
                model(tokens)
            pooled = steer9.pool_last_non_pad_token(model, tokens, resid_store["resid"])
            probs.append(
                float(residual_probe.predict_proba(pooled.reshape(1, -1))[0, 1])
            )
    return np.asarray(probs, dtype=np.float64)


def recommend_alpha(rows: list[dict]) -> dict | None:
    """
    Smallest α with mean|ΔP| >= MIN_MEAN_ABS_DP and saturation below cap.
    Falls back to the α with best mean|ΔP| among unsaturated rows.
    """
    eligible = [
        r
        for r in rows
        if r["mean_abs_dp"] >= MIN_MEAN_ABS_DP
        and r["saturation_frac"] <= MAX_SATURATION_FRAC
    ]
    if eligible:
        return min(eligible, key=lambda r: r["alpha"])

    unsaturated = [r for r in rows if r["saturation_frac"] <= MAX_SATURATION_FRAC]
    if unsaturated:
        return max(unsaturated, key=lambda r: r["mean_abs_dp"])
    return max(rows, key=lambda r: r["mean_abs_dp"]) if rows else None


def print_scale_report(
    feature_indices: np.ndarray,
    scales: np.ndarray,
    norms: np.ndarray,
    shap_scores: np.ndarray,
) -> None:
    print("\nActivation-scale report (candidates):")
    print(
        f"  n={len(scales)}  median_scale={np.median(scales):.4f}  "
        f"mean_scale={scales.mean():.4f}  "
        f"median_||W_dec||={np.median(norms):.4f}"
    )
    order = np.argsort(scales)[::-1][:5]
    print("  Top-5 by positive-activation mean:")
    for i in order:
        f = int(feature_indices[i])
        print(
            f"    feat {f:5d}:  scale={scales[i]:.4f}  "
            f"|Φ|={abs(shap_scores[f]):.5f}  ||W_dec||={norms[i]:.4f}"
        )


def print_pilot_table(rows: list[dict]) -> None:
    header = f"{'α':>8}  {'mean|ΔP|':>10}  {'meanΔP':>10}  {'sat_frac':>8}  {'ok?':>4}"
    print("\nPilot sweep (pilot features × val subset):")
    print(header)
    print("-" * len(header))
    for r in rows:
        ok = (
            r["mean_abs_dp"] >= MIN_MEAN_ABS_DP
            and r["saturation_frac"] <= MAX_SATURATION_FRAC
        )
        print(
            f"{r['alpha']:8.4f}  {r['mean_abs_dp']:10.4f}  {r['mean_signed_dp']:+10.4f}  "
            f"{r['saturation_frac']:8.3f}  {'yes' if ok else 'no':>4}"
        )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Candidate pool + α report → {OUT_DIR}/")

    ranking = load_ranking_candidates()
    candidates = ranking["all_candidates"]
    pilot_features = ranking["pilot_features"]
    shap_scores = ranking["shap_scores"]

    print(f"Candidates (union top-{K_CANDIDATES}): {len(candidates)}")
    print(f"Pilot features (top-|SHAP|): {pilot_features.tolist()}")

    scales = positive_activation_scales(ACTIVATIONS_PATH, candidates)
    median_scale = float(np.median(scales))
    alpha_grid = build_alpha_grid(median_scale)
    print(f"\nMedian positive-activation scale = {median_scale:.4f}")
    print(f"α grid (× {SCALE_MULTIPLIERS}) = {[round(a, 4) for a in alpha_grid]}")

    model = load_model()
    sae = SAE.from_pretrained(
        "gpt2-small-resid-post-v5-32k", f"blocks.{LAYER}.hook_resid_post"
    )
    sae = sae.to(model.cfg.device)
    norms = decoder_norms(sae, candidates)
    print_scale_report(candidates, scales, norms, shap_scores)

    # Residual-space reference: α_resid ≈ α * median||W_dec||
    print(
        "\nFor decoder-normalized steering, equivalent residual L2 nudge ≈ "
        f"α × {np.median(norms):.4f}"
    )

    train_ds, _, _ = load_splits()
    readout_layer = model.cfg.n_layers - 1

    train_tokens = model.to_tokens(list(train_ds["sentence"]))
    train_labels = np.asarray(train_ds["label"])
    residual_probe = steer9.load_or_train_residual_probe(
        model,
        train_tokens,
        train_labels,
        layer=readout_layer,
        checkpoint_path=str(
            checkpoint_path(f"residual_probe_layer_{readout_layer}.joblib")
        ),
    )

    # Pilot on TRAIN, not val or held-out. α only has to land in a range where
    # the effect is measurable and unsaturated — a property of the model, not of
    # labels — so it needs no held-out data, and tuning it on the split that
    # 9_steer later scores would select the hyperparameter on the eval set.
    n_pilot = min(N_PILOT_EXAMPLES, len(train_ds))
    pilot_sentences = list(train_ds["sentence"][:n_pilot])
    pilot_tokens = model.to_tokens(pilot_sentences)
    print(f"\nPilot subset: {n_pilot} train examples × {len(pilot_features)} features")

    baseline_probs = compute_recon_baseline_probs(
        model, sae, residual_probe, pilot_tokens, readout_layer
    )

    rows = []
    for alpha in alpha_grid:
        print(f"\n=== pilot α = {alpha:.4f} ===")
        row = pilot_effects_for_alpha(
            model,
            sae,
            residual_probe,
            pilot_tokens,
            pilot_features,
            alpha=alpha,
            readout_layer=readout_layer,
            baseline_probs=baseline_probs,
        )
        rows.append(row)
        print(
            f"  mean|ΔP|={row['mean_abs_dp']:.4f}  "
            f"meanΔP={row['mean_signed_dp']:+.4f}  "
            f"sat={row['saturation_frac']:.3f}"
        )

    print_pilot_table(rows)
    best = recommend_alpha(rows)

    lines = []
    lines.append("Steering α calibration")
    lines.append(f"median_positive_activation_scale = {median_scale:.6f}")
    lines.append(f"median_decoder_norm = {float(np.median(norms)):.6f}")
    lines.append(f"pilot_features = {pilot_features.tolist()}")
    lines.append(f"n_pilot_examples = {n_pilot} (train split)")
    lines.append(
        f"criteria: mean|ΔP| >= {MIN_MEAN_ABS_DP}, sat_frac <= {MAX_SATURATION_FRAC}"
    )
    lines.append("")
    lines.append(
        f"{'alpha':>10} {'mean_abs_dp':>12} {'mean_signed_dp':>14} {'sat_frac':>10}"
    )
    for r in rows:
        lines.append(
            f"{r['alpha']:10.6f} {r['mean_abs_dp']:12.6f} "
            f"{r['mean_signed_dp']:+14.6f} {r['saturation_frac']:10.6f}"
        )
    lines.append("")
    if best is None:
        lines.append("recommended_alpha = NONE")
        print("\nNo usable α found.")
    else:
        lines.append(f"recommended_alpha = {best['alpha']:.6f}")
        print(
            f"\nRecommended STEERING_ALPHA = {best['alpha']:.4f}  "
            f"(mean|ΔP|={best['mean_abs_dp']:.4f}, sat={best['saturation_frac']:.3f})"
        )
        print(
            "Pass this as --alpha to src/global_linear/9_steer.py (and the MLP / "
            "local steer scripts) and run the full steer once."
        )
        print(
            "CAVEAT: α is calibrated here against this residual probe's P(positive) "
            f"at layer {readout_layer}. The steer scripts read out a GPT-2 logit "
            "difference instead, so the mean|ΔP| ≥ "
            f"{MIN_MEAN_ABS_DP} threshold does not transfer to that scale — treat α "
            "as a rough 'measurable but unsaturated' magnitude, not a tuned value."
        )

    out_txt = Path(OUT_DIR) / "alpha_calibration.txt"
    out_txt.write_text("\n".join(lines) + "\n")
    np.save(Path(OUT_DIR) / "alpha_grid.npy", np.array([r["alpha"] for r in rows]))
    np.save(
        Path(OUT_DIR) / "alpha_pilot_mean_abs_dp.npy",
        np.array([r["mean_abs_dp"] for r in rows]),
    )
    np.save(Path(OUT_DIR) / "candidate_scales.npy", scales)
    np.save(Path(OUT_DIR) / "candidate_indices.npy", candidates)
    print(f"\nWrote {out_txt}")


if __name__ == "__main__":
    main()
