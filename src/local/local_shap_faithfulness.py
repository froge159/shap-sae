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
3. Global ranking Φ = mean_x φ(x) (same aggregation used elsewhere).
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

Why not reuse outputs/14_mlp_steer?
----------------------------------
Those files store one mean Δ per feature over the whole val set. Local SHAP
needs Δ(x, i) for the same x that φ(x) was computed on.
"""

from __future__ import annotations

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

from utils import load_model, load_splits

# --- reuse DeepSHAP helpers from tertiary/12_mlp_shap.py ---
_SHAP12 = Path(__file__).resolve().parents[1] / "tertiary" / "12_mlp_shap.py"
_spec = importlib.util.spec_from_file_location("mlp_shap12", _SHAP12)
assert _spec is not None and _spec.loader is not None
_mlp_shap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mlp_shap)

# --- reuse LM intervention helpers from tertiary/14_mlp_steer.py ---
_STEER14 = Path(__file__).resolve().parents[1] / "tertiary" / "14_mlp_steer.py"
_spec14 = importlib.util.spec_from_file_location("mlp_steer14", _STEER14)
assert _spec14 is not None and _spec14.loader is not None
_steer14 = importlib.util.module_from_spec(_spec14)
_spec14.loader.exec_module(_steer14)


LAYER = 7
N_EXAMPLES = 80
N_BACKGROUND = 100
K_LOCAL = 10
K_GLOBAL = 10
SEED = 42
# "probe": ablate in MLP input (matches DeepSHAP's f). "lm": SAE edit + logit-diff.
TARGET = "lm"
CHECKPOINT = "checkpoints/mlp_probe_layer_7.joblib"
OUT_DIR = Path("outputs/local_shap_faithfulness")


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
    pad_id = _steer14._pad_token_id(model)
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
            _steer14.last_non_pad_logit_diff(logits, batch, pad_id, pos_id, neg_id)[0]
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
                _steer14.last_non_pad_logit_diff(logits, batch, pad_id, pos_id, neg_id)[0]
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


def main(
    n_examples: int = N_EXAMPLES,
    k_local: int = K_LOCAL,
    k_global: int = K_GLOBAL,
    target: str = TARGET,
    seed: int = SEED,
    checkpoint: str = CHECKPOINT,
    out_dir: Path = OUT_DIR,
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

    # Sample N SHAP-split rows + train background for DeepSHAP (reproducible).
    rng = np.random.default_rng(seed)
    shap_acts = np.load(f"activations/shap/layer_{LAYER}/activations.npy")
    train_acts = np.load(f"activations/probe_train/layer_{LAYER}/activations.npy")
    pick = rng.choice(len(shap_acts), size=n_examples, replace=False)
    bg_idx = rng.choice(len(train_acts), size=N_BACKGROUND, replace=False)
    shap_eval = shap_acts[pick]
    background = train_acts[bg_idx]

    print("Computing local DeepSHAP…")
    local_shap = _mlp_shap.run_deepshap(probe, shap_eval, background, feature_indices)
    # (n_examples, n_filt)
    print(f"  local_shap shape={local_shap.shape}")

    global_phi_filt = local_shap.mean(axis=0)  # mean signed SHAP on this sample
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
        _, _, shap_ds = load_splits()
        # Align tokens with the activation rows we picked from the SHAP split.
        sentences = [shap_ds[int(i)]["sentence"] for i in pick]
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


if __name__ == "__main__":
    main()
