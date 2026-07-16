import os
from pathlib import Path

import joblib
import numpy as np
import torch
from sae_lens import SAE
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from transformer_lens import HookedTransformer
from utils import load_model, load_splits


"""
Feature ranking
"""


def compile_rankings(ig_scores, probe_scores, ga_scores, shap_scores):
    # Convert scores to ranks (rank 1 = most important)
    def scores_to_ranks(scores):
        order = np.argsort(scores)[::-1]
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(scores) + 1)
        return ranks

    probe_ranks = scores_to_ranks(probe_scores)
    ig_ranks = scores_to_ranks(ig_scores)
    ga_ranks = scores_to_ranks(ga_scores)
    shap_ranks = scores_to_ranks(shap_scores)

    return probe_ranks, ig_ranks, ga_ranks, shap_ranks


def get_ablation_candidates(probe_ranks, ig_ranks, shap_ranks, ga_ranks, feature_indices, k=20):
    # Top-k by each method
    shap_top = feature_indices[np.argsort(shap_ranks)[:k]]
    probe_top = feature_indices[np.argsort(probe_ranks)[:k]]
    ig_top = feature_indices[np.argsort(ig_ranks)[:k]]
    ga_top = feature_indices[np.argsort(ga_ranks)[:k]]

    # Union — ablate each unique feature once, reuse results for all methods
    all_candidates = np.unique(np.concatenate([shap_top, probe_top, ig_top, ga_top]))
    return all_candidates, shap_top, probe_top, ig_top, ga_top


def faithfulness_correlation(method_scores, feature_indices, ablation_effects):
    scores = []
    effects = []
    for i, feat_idx in enumerate(feature_indices):
        if feat_idx in ablation_effects:
            scores.append(method_scores[i])
            effects.append(abs(ablation_effects[feat_idx]))

    rho, p = spearmanr(scores, effects)
    return rho, p


def _pad_token_id(model: HookedTransformer) -> int:
    pad_id = model.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = model.tokenizer.eos_token_id
    return pad_id


def pool_last_non_pad_token(tokens: torch.Tensor, acts: torch.Tensor, pad_id: int) -> np.ndarray:
    """
    Select activations at the last non-padding token (matches extract.py).

    tokens: (batch, seq) or (seq,)
    acts:   (batch, seq, d) or (seq, d)
    returns: (d,) when batch==1, else (batch, d)
    """
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)
    if acts.ndim == 2:
        acts = acts.unsqueeze(0)

    mask = tokens != pad_id
    last_idx = mask.sum(dim=1) - 1
    batch_idx = torch.arange(tokens.shape[0], device=tokens.device)
    pooled = acts[batch_idx, last_idx, :]
    pooled = pooled.detach().cpu().numpy()
    if pooled.shape[0] == 1:
        return pooled[0]
    return pooled


def _as_batch(tokens: torch.Tensor, device) -> torch.Tensor:
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)
    return tokens.to(device)


def collect_last_token_residuals(
    model: HookedTransformer,
    tokens_list,
    layer: int,
    batch_size: int = 32,
) -> np.ndarray:
    """Collect last-non-pad residual vectors at `layer`, shape (n, d_model)."""
    hook_point = f"blocks.{layer}.hook_resid_post"
    pad_id = _pad_token_id(model)
    rows = []

    # tokens_list may be a padded [N, T] tensor or a list of [T] tensors
    if isinstance(tokens_list, torch.Tensor) and tokens_list.ndim == 2:
        n = tokens_list.shape[0]
        indices = range(0, n, batch_size)
        iterator = tqdm(indices, desc=f"Residuals L{layer}")
        for start in iterator:
            batch = tokens_list[start : start + batch_size].to(model.cfg.device)
            with torch.no_grad():
                _, cache = model.run_with_cache(batch, names_filter=hook_point)
            resid = cache[hook_point]
            pooled = pool_last_non_pad_token(batch, resid, pad_id)
            if pooled.ndim == 1:
                pooled = pooled[None, :]
            rows.append(pooled)
    else:
        for tokens in tqdm(tokens_list, desc=f"Residuals L{layer}"):
            tokens = _as_batch(tokens, model.cfg.device)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=hook_point)
            resid = cache[hook_point]
            rows.append(pool_last_non_pad_token(tokens, resid, pad_id))

    return np.asarray(rows, dtype=np.float32)





def get_model_ablation_effects(
    model: HookedTransformer,
    sae: SAE,
    readout_probe,
    tokens_list,
    candidate_indices: np.ndarray,
    layer: int = 7,
    readout_layer: int | None = None,
) -> dict:
    """
    Ablate SAE features at `layer` and measure mean |ΔP| on a downstream residual probe.

    Intervention: encode → zero feature → decode into resid_post at `layer`.
    Readout: last-non-pad residual at `readout_layer` (default: final layer),
    scored by `readout_probe` trained on residual stream (d_model,).

    Baseline runs encode→decode with no zeroing so ΔP is not confounded by
    SAE reconstruction error. Downstream readout lets the ablation propagate
    through later layers before measurement.
    """
    if readout_layer is None:
        readout_layer = model.cfg.n_layers - 1

    ablate_point = f"blocks.{layer}.hook_resid_post"
    readout_point = f"blocks.{readout_layer}.hook_resid_post"
    pad_id = _pad_token_id(model)
    delta_probabilities = {}
    resid_store: dict[str, torch.Tensor] = {}

    def capture_hook(resid_post, hook):
        resid_store["resid"] = resid_post
        return resid_post

    def recon_hook(resid_post, hook):
        return sae.decode(sae.encode(resid_post))

    def probe_prob(tokens: torch.Tensor) -> float:
        pooled = pool_last_non_pad_token(tokens, resid_store["resid"], pad_id)
        return float(readout_probe.predict_proba(pooled.reshape(1, -1))[0, 1])

    # SAE-reconstruction baseline (no feature zeroing), readout downstream
    baseline_probs = []
    for tokens in tqdm(tokens_list, desc="Computing baselines"):
        tokens = _as_batch(tokens, model.cfg.device)
        with torch.no_grad():
            with model.hooks(
                fwd_hooks=[(ablate_point, recon_hook), (readout_point, capture_hook)]
            ):
                model(tokens)
            baseline_probs.append(probe_prob(tokens))
    baseline_probs = np.array(baseline_probs)

    for orig_idx in tqdm(candidate_indices, desc="Ablating features"):
        idx = int(orig_idx)

        def ablation_hook(resid_post, hook, feature_idx=idx):
            acts = sae.encode(resid_post).clone()
            acts[..., feature_idx] = 0.0
            return sae.decode(acts)

        ablated_probs = []
        for tokens in tokens_list:
            tokens = _as_batch(tokens, model.cfg.device)
            with torch.no_grad():
                with model.hooks(
                    fwd_hooks=[
                        (ablate_point, ablation_hook),
                        (readout_point, capture_hook),
                    ]
                ):
                    model(tokens)
                ablated_probs.append(probe_prob(tokens))

        ablated_probs = np.array(ablated_probs)
        delta_probabilities[idx] = float(np.mean(np.abs(ablated_probs - baseline_probs)))

    return delta_probabilities


if __name__ == "__main__":
    # Attribution rankings still come from the layer-7 SAE probe / SHAP pipeline
    sae_probe = joblib.load("checkpoints/probe_layer_7.joblib")
    feature_indices = np.load("outputs/shap/shap_feature_indices.npy")

    probe_scores = np.abs(sae_probe.coef_[0][feature_indices])
    ig_scores = np.load("outputs/rankings/ig_scores.npy")
    ga_scores = np.load("outputs/rankings/ga_scores.npy")
    shap_scores = np.load("outputs/shap/phi_sentiment_layer7.npy")[feature_indices]
    probe_ranks, ig_ranks, ga_ranks, shap_ranks = compile_rankings(
        ig_scores, probe_scores, ga_scores, shap_scores
    )

    all_candidates, shap_top, probe_top, ig_top, ga_top = get_ablation_candidates(
        probe_ranks, ig_ranks, shap_ranks, ga_ranks, feature_indices, k=20
    )

    model = load_model()
    sae = SAE.from_pretrained("gpt2-small-resid-post-v5-32k", "blocks.7.hook_resid_post")
    sae = sae.to(model.cfg.device)

    train_ds, val_ds, _ = load_splits()
    readout_layer = model.cfg.n_layers - 1  # GPT-2 Small: 11

    train_tokens = model.to_tokens(train_ds["sentence"])
    train_labels = np.asarray(train_ds["label"])
    residual_probe = joblib.load("checkpoints/residual_probe_layer_11.joblib")

    val_tokens = model.to_tokens(val_ds["sentence"])
    delta_probabilities = get_model_ablation_effects(
        model,
        sae,
        residual_probe,
        val_tokens,
        all_candidates,
        layer=7,
        readout_layer=readout_layer,
    )

    print(f"Ablated {len(delta_probabilities)} features (readout L{readout_layer})")
    for idx, delta in sorted(delta_probabilities.items(), key=lambda x: -x[1])[:10]:
        print(f"  feature {idx}: ΔP={delta:.4f}")

    rho_shap, _ = faithfulness_correlation(shap_scores, feature_indices, delta_probabilities)
    rho_probe, _ = faithfulness_correlation(probe_scores, feature_indices, delta_probabilities)
    rho_ig, _ = faithfulness_correlation(ig_scores, feature_indices, delta_probabilities)
    rho_ga, _ = faithfulness_correlation(ga_scores, feature_indices, delta_probabilities)
    print("Faithfulness (Spearman ρ with downstream ablation effects):")
    print(f"  SHAP:          {rho_shap:.3f}")
    print(f"  Probe weights: {rho_probe:.3f}")
    print(f"  IG:            {rho_ig:.3f}")
    print(f"  GA:            {rho_ga:.3f}")

    os.makedirs("outputs/ablation", exist_ok=True)
    np.save("outputs/ablation/model_delta_probabilities.npy", delta_probabilities)
