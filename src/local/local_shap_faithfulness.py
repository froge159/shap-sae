"""
Local DeepSHAP vs per-example feature ablation (faithfulness diagnostic).

Question
--------
Is SHAP's *local* ranking faithful for individual sentences, or did averaging
into a global Φ hide / create the error?

Method (plain language)
-----------------------
1. Sample N sentences from the SHAP held-out split (filtered SAE features only).
2. Run DeepSHAP on the MLP probe → one attribution vector φ(x) per sentence.
3. Global ranking Φ: loaded from 12_mlp_shap, which computed it over the full
   canonical eval sample. Deriving Φ from the N sentences under test instead
   would make the "global" arm in-sample and no longer the Φ any other script
   uses; pass --insample-phi only if you specifically want that comparison.
4. For each sentence x, build a candidate feature set = top-k by |φ(x)| ∪ top-k
   by |Φ|.
5. For each candidate feature i on that sentence only:
     - Probe target (default): zero feature i in the probe input, measure
       Δ logit = f(x with i zeroed) − f(x).
     - LM target (optional): SAE encode → zero i → decode at resid_post,
       measure Δ (logit wonderful − logit awful), matching 14_mlp_steer but
       *without* averaging over the dataset.
6. Per sentence, Spearman-correlate |φ(x)| vs |Δ(x,·)| and |Φ| vs |Δ(x,·)|.
7. Summarize mean ρ_local vs mean ρ_global across sentences.

Why not reuse 14_mlp_steer?
---------------------------
Those files store one mean Δ per feature over the whole held-out steer sample.
Local SHAP needs Δ(x, i) for the same x that φ(x) was computed on.

Confound to keep in mind
------------------------
Globally-top features are usually *inactive* in any one sentence, and ablating an
inactive feature moves the logit by exactly 0. Local |φ(x)| tracks activity by
construction; global |Φ| does not. So some of any "local wins" margin is really
"local knows which features fire here". 9_local_steer / 14_local_mlp_steer report
an activity-matched control block for this; treat the headline here as an upper
bound on the local advantage.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from sae_lens import SAE
from scipy.stats import spearmanr
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
from transformer_lens import HookedTransformer

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils import (
    checkpoint_path,
    eval_sentences,
    load_eval_and_background,
    load_model,
    output_path,
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Reuse DeepSHAP and LM-intervention helpers from the global MLP arm.
_mlp_shap = _load("mlp_shap12", _SRC / "global_mlp" / "12_mlp_shap.py")
_steer14 = _load("mlp_steer14", _SRC / "global_mlp" / "14_mlp_steer.py")


LAYER = 7
N_EXAMPLES = 80
N_BACKGROUND = 100
K_LOCAL = 10
K_GLOBAL = 10
SEED = 42
# "probe": ablate in MLP input (matches DeepSHAP's f). "lm": SAE edit + logit-diff.
TARGET = "lm"
CHECKPOINT = checkpoint_path("mlp_probe_layer_7.joblib")
MLP_SHAP_DIR = output_path("12_mlp_shap")
OUT_DIR = output_path("local_shap_faithfulness")


def probe_logit(probe: MLPClassifier, x_filtered: np.ndarray) -> float:
    """
    Positive-class logit for a single filtered activation row.

    MLPClassifier has no decision_function; compute the pre-sigmoid logit from
    the fitted weights so the ablation target matches DeepSHAP (TorchMLPProbe).
    """
    x = np.asarray(x_filtered, dtype=np.float64).reshape(1, -1)
    if len(probe.coefs_) != 2:
        raise ValueError(
            f"Expected a 1-hidden-layer MLP; got {len(probe.coefs_)} weight matrices"
        )
    W1, W2 = probe.coefs_
    b1, b2 = probe.intercepts_
    h = np.maximum(x @ W1 + b1, 0.0)
    return float((h @ W2[:, 0] + b2[0])[0])


def per_example_probe_ablation_deltas(
    probe: MLPClassifier,
    x_filtered: np.ndarray,
    candidate_local_idx: np.ndarray,
) -> np.ndarray:
    """
    One-at-a-time zero ablation on the probe input for a single example.

    Returns Δ logit for each local (filtered) feature index in candidate_local_idx.
    """
    base = probe_logit(probe, x_filtered)
    deltas = np.zeros(len(candidate_local_idx), dtype=np.float64)
    x = np.asarray(x_filtered, dtype=np.float64).copy()
    for j, loc in enumerate(candidate_local_idx):
        x_ab = x.copy()
        x_ab[int(loc)] = 0.0
        deltas[j] = probe_logit(probe, x_ab) - base
    return deltas


def per_example_lm_ablation_deltas(
    model: HookedTransformer,
    sae: SAE,
    tokens: torch.Tensor,
    candidate_global_idx: np.ndarray,
    layer: int = LAYER,
) -> np.ndarray:
    """
    One-at-a-time SAE feature ablation for a *single* sentence; LM logit-diff readout.

    Same intervention as 14_mlp_steer (encode → zero → decode) but Δ is for this
    example only (no dataset average).
    """
    intervene_point = f"blocks.{layer}.hook_resid_post"
    pos_id, neg_id = _steer14.sentiment_token_ids(model)
    batch = tokens.unsqueeze(0).to(model.cfg.device) if tokens.ndim == 1 else tokens.to(
        model.cfg.device
    )

    def recon_hook(resid_post, hook):
        return sae.decode(sae.encode(resid_post))

    with torch.no_grad():
        with model.hooks(fwd_hooks=[(intervene_point, recon_hook)]):
            logits = model(batch)
        base = float(
            _steer14.last_non_pad_logit_diff(model, logits, batch, pos_id, neg_id)[0]
        )

    deltas = np.zeros(len(candidate_global_idx), dtype=np.float64)
    for j, gidx in enumerate(candidate_global_idx):
        gidx = int(gidx)

        def ablation_hook(resid_post, hook, feature_idx=gidx):
            acts = sae.encode(resid_post).clone()
            acts[..., feature_idx] = 0.0
            return sae.decode(acts)

        with torch.no_grad():
            with model.hooks(fwd_hooks=[(intervene_point, ablation_hook)]):
                logits = model(batch)
            steered = float(
                _steer14.last_non_pad_logit_diff(model, logits, batch, pos_id, neg_id)[0]
            )
        deltas[j] = steered - base
    return deltas


def candidate_indices_for_example(
    local_phi: np.ndarray,
    global_phi_filtered: np.ndarray,
    k_local: int,
    k_global: int,
) -> np.ndarray:
    """Union of top-|local| and top-|global| feature indices (into filtered axis)."""
    k_local = min(k_local, len(local_phi))
    k_global = min(k_global, len(global_phi_filtered))
    top_local = np.argsort(np.abs(local_phi))[::-1][:k_local]
    top_global = np.argsort(np.abs(global_phi_filtered))[::-1][:k_global]
    return np.unique(np.concatenate([top_local, top_global]))


def spearman_abs(scores: np.ndarray, deltas: np.ndarray) -> tuple[float, float]:
    if len(scores) < 3:
        return float("nan"), float("nan")
    rho, p = spearmanr(np.abs(scores), np.abs(deltas))
    return float(rho), float(p)


def load_global_phi(feature_indices: np.ndarray) -> np.ndarray:
    """
    Φ on the filtered axis, from 12_mlp_shap's full-sample DeepSHAP run.

    Kept separate from the local φ's on purpose: averaging the N sentences under
    test would make the "global" arm in-sample, and would not be the Φ the rest
    of the pipeline ranks with.
    """
    phi_full = np.load(MLP_SHAP_DIR / "phi_sentiment_layer7_signed.npy")
    shap_feature_indices = np.load(MLP_SHAP_DIR / "shap_feature_indices.npy")
    if not np.array_equal(feature_indices, shap_feature_indices):
        raise ValueError(
            f"Checkpoint feature_indices disagree with "
            f"{MLP_SHAP_DIR / 'shap_feature_indices.npy'}"
        )
    return np.asarray(phi_full[feature_indices], dtype=np.float64)


def main(
    n_examples: int = N_EXAMPLES,
    k_local: int = K_LOCAL,
    k_global: int = K_GLOBAL,
    target: str = TARGET,
    seed: int = SEED,
    checkpoint: Path = CHECKPOINT,
    out_dir: Path = OUT_DIR,
    insample_phi: bool = False,
) -> None:
    if target not in ("probe", "lm"):
        raise ValueError(f"target must be 'probe' or 'lm', got {target!r}")

    payload = joblib.load(checkpoint)
    probe: MLPClassifier = payload["probe"]
    feature_indices = np.asarray(payload["feature_indices"])
    n_filt = len(feature_indices)
    print(
        f"MLP probe: n_filtered={n_filt}, hidden={payload.get('hidden_dim')}, "
        f"target={target}, n_examples={n_examples}"
    )

    # Shared sampler => an ordered prefix of the rows the global scripts explain.
    shap_eval, background, pick, bg_idx = load_eval_and_background(
        layer=LAYER, n_eval=n_examples, n_background=N_BACKGROUND, seed=seed
    )

    print("Computing local DeepSHAP…")
    local_shap = _mlp_shap.run_deepshap(probe, shap_eval, background, feature_indices)
    # (n_examples, n_filt)
    print(f"  local_shap shape={local_shap.shape}")

    if insample_phi:
        print("Global Φ: mean of these N local φ (IN-SAMPLE — diagnostic only)")
        global_phi_filt = local_shap.mean(axis=0)
    else:
        global_phi_filt = load_global_phi(feature_indices)
        print(f"Global Φ: loaded from {MLP_SHAP_DIR}/ (full canonical eval sample)")
    X_filt = np.asarray(shap_eval[:, feature_indices], dtype=np.float64)

    model = None
    sae = None
    tokens = None
    if target == "lm":
        print("Loading GPT-2 + SAE for per-example LM ablation…")
        model = load_model()
        sae = SAE.from_pretrained(
            "gpt2-small-resid-post-v5-32k", "blocks.7.hook_resid_post"
        )
        sae = sae.to(model.cfg.device)
        # Align tokens with the activation rows we picked from the SHAP split.
        sentences = eval_sentences(pick)
        tokens = model.to_tokens(sentences)

    rho_local_list = []
    rho_global_list = []
    rows_meta = []

    # Store ragged results as lists; also a dense summary table.
    for e in tqdm(range(n_examples), desc="Per-example ablation"):
        phi_e = local_shap[e]
        cand_local = candidate_indices_for_example(
            phi_e, global_phi_filt, k_local, k_global
        )
        cand_global = feature_indices[cand_local]

        if target == "probe":
            deltas = per_example_probe_ablation_deltas(probe, X_filt[e], cand_local)
        else:
            assert model is not None and sae is not None and tokens is not None
            deltas = per_example_lm_ablation_deltas(
                model, sae, tokens[e], cand_global, layer=LAYER
            )

        local_scores = phi_e[cand_local]
        global_scores = global_phi_filt[cand_local]
        rho_l, p_l = spearman_abs(local_scores, deltas)
        rho_g, p_g = spearman_abs(global_scores, deltas)
        rho_local_list.append(rho_l)
        rho_global_list.append(rho_g)
        rows_meta.append(
            {
                "example_i": int(e),
                "shap_split_idx": int(pick[e]),
                "n_candidates": int(len(cand_local)),
                "rho_local": rho_l,
                "p_local": p_l,
                "rho_global": rho_g,
                "p_global": p_g,
                "local_wins": bool(rho_l > rho_g)
                if np.isfinite(rho_l) and np.isfinite(rho_g)
                else None,
            }
        )

    rho_local_arr = np.asarray(rho_local_list, dtype=np.float64)
    rho_global_arr = np.asarray(rho_global_list, dtype=np.float64)
    finite = np.isfinite(rho_local_arr) & np.isfinite(rho_global_arr)
    mean_local = float(np.nanmean(rho_local_arr))
    mean_global = float(np.nanmean(rho_global_arr))
    frac_local_wins = float(np.mean(rho_local_arr[finite] > rho_global_arr[finite]))

    summary = {
        "target": target,
        "global_phi_source": "in-sample mean" if insample_phi else str(MLP_SHAP_DIR),
        "n_examples": n_examples,
        "k_local": k_local,
        "k_global": k_global,
        "n_filtered": n_filt,
        "seed": seed,
        "mean_rho_local": mean_local,
        "mean_rho_global": mean_global,
        "std_rho_local": float(np.nanstd(rho_local_arr)),
        "std_rho_global": float(np.nanstd(rho_global_arr)),
        "frac_examples_local_gt_global": frac_local_wins,
        "interpretation": (
            "Local SHAP ranks per-example ablation effects better than global Φ "
            "→ averaging was likely a main error source."
            if mean_local > mean_global + 0.05
            else (
                "Local and global similarly (un)faithful → problem is not just averaging."
                if abs(mean_local - mean_global) <= 0.05
                else "Global Φ unexpectedly beats local rankings on this sample."
            )
        ),
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "example_indices.npy", pick)
    np.save(out_dir / "local_shap_values.npy", local_shap)
    np.save(out_dir / "global_phi_filtered.npy", global_phi_filt)
    np.save(out_dir / "rho_local.npy", rho_local_arr)
    np.save(out_dir / "rho_global.npy", rho_global_arr)
    with open(out_dir / "per_example.json", "w") as f:
        json.dump(rows_meta, f, indent=2)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "Local DeepSHAP vs per-example ablation",
        "=" * 50,
        f"target={target}  n={n_examples}  k_local={k_local}  k_global={k_global}",
        f"mean Spearman ρ(|local φ| vs |Δ|)  = {mean_local:.3f} "
        f"± {summary['std_rho_local']:.3f}",
        f"mean Spearman ρ(|global Φ| vs |Δ|) = {mean_global:.3f} "
        f"± {summary['std_rho_global']:.3f}",
        f"fraction examples with ρ_local > ρ_global = {frac_local_wins:.3f}",
        "",
        summary["interpretation"],
        "",
    ]
    text = "\n".join(lines)
    (out_dir / "summary.txt").write_text(text)
    print("\n" + text)
    print(f"Wrote → {out_dir}/")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-examples", type=int, default=N_EXAMPLES)
    p.add_argument("--k-local", type=int, default=K_LOCAL)
    p.add_argument("--k-global", type=int, default=K_GLOBAL)
    p.add_argument("--target", choices=("probe", "lm"), default=TARGET)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument(
        "--insample-phi",
        action="store_true",
        help="Use mean of these N local φ as the global Φ (in-sample; diagnostic)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        n_examples=args.n_examples,
        k_local=args.k_local,
        k_global=args.k_global,
        target=args.target,
        out_dir=args.out_dir,
        insample_phi=args.insample_phi,
    )
