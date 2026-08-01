"""
Steering / ablation faithfulness for MLP-probe attributions.

Mirrors secondary/9_steer.py (logit-diff readout, SAE encode→edit→decode), but:
  - Loads filtered (n_filtered,) scores from outputs/13_mlp_ranking/
  - Maps local → global SAE indices via feature_indices.npy
  - Fixes a single candidate set: top-k by |SHAP| only (no union across methods)
  - Evaluates SHAP / probe / IG / GA faithfulness on that shared set
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from sae_lens import SAE
from scipy.stats import spearmanr
from tqdm import tqdm
from transformer_lens import HookedTransformer

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils import load_model, load_splits

N_TOTAL_FEATURES = 32768
RANKINGS_DIR = Path("outputs/13_mlp_ranking")


def expand_to_full(
    scores: np.ndarray,
    feature_indices: np.ndarray,
    n_total: int = N_TOTAL_FEATURES,
) -> np.ndarray:
    """Scatter filtered scores into a dense (n_total,) vector (zeros elsewhere)."""
    full = np.zeros(n_total, dtype=np.float64)
    full[feature_indices] = scores
    return full


def compile_rankings(ig_scores, probe_scores, ga_scores, shap_scores):
    # Scores may be signed; rank by |score| so rank 1 = largest-magnitude attribution.
    def scores_to_ranks(scores):
        order = np.argsort(np.abs(scores))[::-1]
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(scores) + 1)
        return ranks

    probe_ranks = scores_to_ranks(probe_scores)
    ig_ranks = scores_to_ranks(ig_scores)
    ga_ranks = scores_to_ranks(ga_scores)
    shap_ranks = scores_to_ranks(shap_scores)

    return probe_ranks, ig_ranks, ga_ranks, shap_ranks


def get_shap_candidates(
    shap_scores: np.ndarray,
    feature_indices: np.ndarray,
    k: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fixed candidate set: top-k filtered features by |SHAP|.

    Returns
    -------
    local_top : (k,) indices into the filtered score vectors
    global_top : (k,) global SAE feature indices for intervention
    """
    local_top = np.argsort(np.abs(shap_scores))[::-1][:k]
    global_top = feature_indices[local_top]
    return local_top, global_top


def faithfulness_correlation(method_scores, effects, *, importance: bool):
    """
    Spearman correlation between attribution scores and intervention effects.

    importance=True:  |score| vs |effect|  (effect size / ranking quality)
    importance=False:  score vs effect     (sign / directional agreement)

    `method_scores` is indexed by global SAE feature id (full-width vector).
    """
    scores = []
    vals = []
    for feat_idx, delta in effects.items():
        s = method_scores[feat_idx]
        scores.append(abs(s) if importance else s)
        vals.append(abs(delta) if importance else delta)

    rho, p = spearmanr(scores, vals)
    return rho, p


def report_faithfulness(method_scores_by_name: dict, delta_abs: dict, delta_signed: dict):
    """Print importance and directional faithfulness for each ranking method."""
    print("Importance faithfulness  (Spearman ρ of |attribution| vs |Δ|):")
    for name, scores in method_scores_by_name.items():
        rho, p = faithfulness_correlation(scores, delta_abs, importance=True)
        print(f"  {name:14s}  ρ={rho:.3f}  p={p:.4f}")

    print("Directional faithfulness (Spearman ρ of attribution vs signed Δ):")
    for name, scores in method_scores_by_name.items():
        rho, p = faithfulness_correlation(scores, delta_signed, importance=False)
        print(f"  {name:14s}  ρ={rho:.3f}  p={p:.4f}")


def _pad_token_id(model: HookedTransformer) -> int:
    pad_id = model.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = model.tokenizer.eos_token_id
    return pad_id


def _as_batch(tokens: torch.Tensor, device) -> torch.Tensor:
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)
    return tokens.to(device)


def _apply_feature_intervention(
    acts: torch.Tensor,
    feature_idx: int,
    mode: str,
    steering_alpha: float,
) -> torch.Tensor:
    """Zero or additively steer one feature at every token position on a cloned tensor."""
    acts = acts.clone()
    if mode == "ablation":
        acts[..., feature_idx] = 0.0
    elif mode == "steering":
        acts[..., feature_idx] = acts[..., feature_idx] + steering_alpha
    else:
        raise ValueError(f"Unknown mode={mode!r}; expected 'ablation' or 'steering'")
    return acts


POS_TOKEN = " wonderful"
NEG_TOKEN = " awful"


def sentiment_token_ids(
    model: HookedTransformer,
    pos_token: str = POS_TOKEN,
    neg_token: str = NEG_TOKEN,
) -> tuple[int, int]:
    """Resolve pos/neg strings to single vocabulary ids."""
    pos_ids = model.to_tokens(pos_token, prepend_bos=False)[0]
    neg_ids = model.to_tokens(neg_token, prepend_bos=False)[0]
    if pos_ids.numel() != 1 or neg_ids.numel() != 1:
        raise ValueError(
            f"Expected single-token pair; got {pos_token!r}→{pos_ids.tolist()}, "
            f"{neg_token!r}→{neg_ids.tolist()}"
        )
    return int(pos_ids.item()), int(neg_ids.item())


def last_non_pad_logit_diff(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    pad_id: int,
    pos_id: int,
    neg_id: int,
) -> np.ndarray:
    """
    logit_pos - logit_neg at the last non-padding position.

    logits: (batch, seq, d_vocab)
    tokens: (batch, seq)
    returns: (batch,) numpy
    """
    mask = tokens != pad_id
    last_idx = mask.sum(dim=1) - 1
    batch_idx = torch.arange(tokens.shape[0], device=tokens.device)
    last_logits = logits[batch_idx, last_idx, :]
    diffs = last_logits[:, pos_id] - last_logits[:, neg_id]
    return diffs.detach().float().cpu().numpy()


def get_model_intervention_effects(
    model: HookedTransformer,
    sae: SAE,
    tokens_list,
    candidate_indices: np.ndarray,
    layer: int = 7,
    mode: str = "steering",
    steering_alpha: float = 2.0,
    pos_token: str = POS_TOKEN,
    neg_token: str = NEG_TOKEN,
    batch_size: int = 32,
) -> tuple[dict, dict]:
    """
    Intervene on SAE features at `layer` and measure Δ in sentiment logit difference.

    Readout (no probe): at the last non-pad token,
        s = logit[pos_token] - logit[neg_token]
        Δ = s_intervened - s_baseline

    candidate_indices must be *global* SAE feature ids.
    """
    if mode not in ("ablation", "steering"):
        raise ValueError(f"Unknown mode={mode!r}; expected 'ablation' or 'steering'")

    if isinstance(tokens_list, torch.Tensor) and tokens_list.ndim == 2:
        all_tokens = tokens_list
    else:
        all_tokens = torch.stack(
            [_as_batch(t, "cpu").squeeze(0) for t in tokens_list], dim=0
        )
    n = all_tokens.shape[0]

    intervene_point = f"blocks.{layer}.hook_resid_post"
    pad_id = _pad_token_id(model)
    pos_id, neg_id = sentiment_token_ids(model, pos_token, neg_token)
    delta_abs: dict[int, float] = {}
    delta_signed: dict[int, float] = {}

    def recon_hook(resid_post, hook):
        return sae.decode(sae.encode(resid_post))

    def score_batch(tokens: torch.Tensor, fwd_hooks: list) -> np.ndarray:
        with torch.no_grad():
            with model.hooks(fwd_hooks=fwd_hooks):
                logits = model(tokens)
            return last_non_pad_logit_diff(logits, tokens, pad_id, pos_id, neg_id)

    def score_all(fwd_hooks: list, desc: str) -> np.ndarray:
        chunks = []
        for start in tqdm(range(0, n, batch_size), desc=desc, leave=False):
            batch = all_tokens[start : start + batch_size].to(model.cfg.device)
            chunks.append(score_batch(batch, fwd_hooks))
        return np.concatenate(chunks, axis=0)

    baseline_scores = score_all([(intervene_point, recon_hook)], "Computing baselines")

    loop_desc = "Ablating features" if mode == "ablation" else "Steering features"
    for orig_idx in tqdm(candidate_indices, desc=loop_desc):
        idx = int(orig_idx)

        def intervention_hook(
            resid_post,
            hook,
            feature_idx=idx,
            _mode=mode,
            _steering_alpha=steering_alpha,
        ):
            acts = _apply_feature_intervention(
                sae.encode(resid_post), feature_idx, _mode, _steering_alpha
            )
            return sae.decode(acts)

        intervened_scores = score_all(
            [(intervene_point, intervention_hook)], f"feature {idx}"
        )
        diffs = intervened_scores - baseline_scores
        delta_abs[idx] = float(np.mean(np.abs(diffs)))
        delta_signed[idx] = float(np.mean(diffs))

    return delta_abs, delta_signed


if __name__ == "__main__":
    MODE = "steering"
    STEERING_ALPHA = 0.6723
    K = 20

    feature_indices = np.load(RANKINGS_DIR / "feature_indices.npy")
    ig_scores = np.load(RANKINGS_DIR / "ig_scores.npy")
    ga_scores = np.load(RANKINGS_DIR / "ga_scores.npy")
    probe_scores = np.load(RANKINGS_DIR / "probe_scores.npy")
    shap_scores = np.load(RANKINGS_DIR / "shap_scores.npy")

    n_filt = len(feature_indices)
    for name, arr in [
        ("ig", ig_scores),
        ("ga", ga_scores),
        ("probe", probe_scores),
        ("shap", shap_scores),
    ]:
        if arr.shape != (n_filt,):
            raise ValueError(
                f"{name}_scores shape {arr.shape} != ({n_filt},) filtered width"
            )

    local_top, global_candidates = get_shap_candidates(
        shap_scores, feature_indices, k=K
    )
    print(
        f"Fixed candidate set: top-{K} by |SHAP| "
        f"({len(global_candidates)} global SAE features)"
    )
    print(f"  local indices:  {local_top.tolist()}")
    print(f"  global indices: {global_candidates.tolist()}")

    # Full-width vectors so faithfulness can index by global SAE id (as in 9_steer).
    ig_full = expand_to_full(ig_scores, feature_indices)
    ga_full = expand_to_full(ga_scores, feature_indices)
    probe_full = expand_to_full(probe_scores, feature_indices)
    shap_full = expand_to_full(shap_scores, feature_indices)

    model = load_model()
    sae = SAE.from_pretrained("gpt2-small-resid-post-v5-32k", "blocks.7.hook_resid_post")
    sae = sae.to(model.cfg.device)

    _, val_ds, _ = load_splits()
    val_tokens = model.to_tokens(list(val_ds["sentence"]))

    pos_id, neg_id = sentiment_token_ids(model, POS_TOKEN, NEG_TOKEN)
    print(
        f"Logit-diff readout: {POS_TOKEN!r}({pos_id}) - {NEG_TOKEN!r}({neg_id}) "
        f"at last non-pad token"
    )

    delta_abs, delta_signed = get_model_intervention_effects(
        model,
        sae,
        val_tokens,
        global_candidates,
        layer=7,
        mode=MODE,
        steering_alpha=STEERING_ALPHA,
        pos_token=POS_TOKEN,
        neg_token=NEG_TOKEN,
    )

    label = (
        "Ablated"
        if MODE == "ablation"
        else f"Steered (a_i += {STEERING_ALPHA:+.3g})"
    )
    print(f"{label} {len(delta_abs)} features")
    print("Top features by |Δ| (logit wonderful − awful):")
    for idx, delta in sorted(delta_abs.items(), key=lambda x: -x[1])[:10]:
        print(f"  feature {idx}: |Δ|={delta:.4f}  signedΔ={delta_signed[idx]:+.4f}")

    method_scores = {
        "SHAP": shap_full,
        "Probe saliency": probe_full,
        "IG": ig_full,
        "GA": ga_full,
    }
    report_faithfulness(method_scores, delta_abs, delta_signed)

    out_dir = Path("outputs/14_mlp_steer")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"model_delta_abs_{MODE}.npy", delta_abs)
    np.save(out_dir / f"model_delta_signed_{MODE}.npy", delta_signed)
    np.save(out_dir / "shap_top_local.npy", local_top)
    np.save(out_dir / "shap_top_global.npy", global_candidates)
    print(f"\nWrote outputs → {out_dir}/")
