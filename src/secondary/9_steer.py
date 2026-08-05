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


def compile_rankings(ig_scores, probe_scores, ga_scores, shap_scores):
    # Scores are signed; rank by |score| so rank 1 = largest-magnitude attribution.
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
    selection: str = "top",
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fixed candidate set from the filtered feature pool.

    selection="top":    top-k by |SHAP|
    selection="random": k features drawn uniformly (without replacement)

    `shap_scores` and ranking are over the filtered set only (aligned with
    `feature_indices`). Intervention uses the mapped global SAE ids.
    `shap_scores` is unused when selection="random".

    Returns
    -------
    local_idx : (k,) indices into the filtered score vectors
    global_idx : (k,) global SAE feature indices for intervention
    """
    n = len(feature_indices)
    k = min(k, n)
    if selection == "top":
        local_idx = np.argsort(np.abs(shap_scores))[::-1][:k]
    elif selection == "random":
        rng = np.random.default_rng(seed)
        local_idx = rng.choice(n, size=k, replace=False)
    else:
        raise ValueError(f"Unknown selection={selection!r}; expected 'top' or 'random'")
    global_idx = feature_indices[local_idx]
    return local_idx, global_idx


def faithfulness_correlation(method_scores, effects, *, importance: bool):
    """
    Spearman correlation between attribution scores and intervention effects.

    importance=True:  |score| vs |effect|  (effect size / ranking quality)
    importance=False:  score vs effect     (sign / directional agreement)
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
            pooled = pool_last_non_pad_token(tokens, resid, pad_id)
            if pooled.ndim == 1:
                pooled = pooled[None, :]
            rows.append(pooled)

    return np.concatenate(rows, axis=0).astype(np.float32)


def train_residual_probe(
    model: HookedTransformer,
    tokens_list,
    labels: np.ndarray,
    layer: int,
    checkpoint_path: str | None = None,
) -> LogisticRegression:
    """Train a logistic probe on last-token residual stream at `layer`."""
    X = collect_last_token_residuals(model, tokens_list, layer)
    probe = LogisticRegression(max_iter=1000, C=1.0, solver="liblinear", verbose=0)
    probe.fit(X, labels)
    if checkpoint_path is not None:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(probe, checkpoint_path)
    return probe


def load_or_train_residual_probe(
    model: HookedTransformer,
    tokens_list,
    labels: np.ndarray,
    layer: int,
    checkpoint_path: str,
) -> LogisticRegression:
    if os.path.exists(checkpoint_path):
        return joblib.load(checkpoint_path)
    print(f"Training residual probe at layer {layer} → {checkpoint_path}")
    return train_residual_probe(model, tokens_list, labels, layer, checkpoint_path)


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
        # Signed additive steering: a_i ← a_i + α  (α may be negative)
        acts[..., feature_idx] = acts[..., feature_idx] + steering_alpha
    else:
        raise ValueError(f"Unknown mode={mode!r}; expected 'ablation' or 'steering'")
    return acts


# Default sentiment-diagnostic pair (single GPT-2 tokens, leading space).
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
    with default tokens " wonderful" / " awful".

    mode="ablation": encode → zero feature → decode into resid_post.
    mode="steering": encode → a_i += steering_alpha (signed, additive) at every
    token position → decode.

    Returns
    -------
    delta_abs : dict[int, float]
        mean |Δ| per feature (effect magnitude).
    delta_signed : dict[int, float]
        mean Δ per feature (effect direction; + = more "wonderful" vs "awful").

    Baseline runs encode→decode with no feature edit so Δ is not confounded by
    SAE reconstruction error. Examples are scored in batches of `batch_size`.
    """
    if mode not in ("ablation", "steering"):
        raise ValueError(f"Unknown mode={mode!r}; expected 'ablation' or 'steering'")

    # Normalize to a single padded [N, T] tensor for batched forwards.
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


def get_model_ablation_effects(
    model: HookedTransformer,
    sae: SAE,
    tokens_list,
    candidate_indices: np.ndarray,
    layer: int = 7,
    pos_token: str = POS_TOKEN,
    neg_token: str = NEG_TOKEN,
    batch_size: int = 32,
) -> tuple[dict, dict]:
    """Ablate SAE features (zero activations). Thin wrapper around intervention API."""
    return get_model_intervention_effects(
        model,
        sae,
        tokens_list,
        candidate_indices,
        layer=layer,
        mode="ablation",
        pos_token=pos_token,
        neg_token=neg_token,
        batch_size=batch_size,
    )


def get_model_steering_effects(
    model: HookedTransformer,
    sae: SAE,
    tokens_list,
    candidate_indices: np.ndarray,
    layer: int = 7,
    steering_alpha: float = 2.0,
    pos_token: str = POS_TOKEN,
    neg_token: str = NEG_TOKEN,
    batch_size: int = 32,
) -> tuple[dict, dict]:
    """Additively steer each selected feature: a_i ← a_i + α (α may be signed)."""
    return get_model_intervention_effects(
        model,
        sae,
        tokens_list,
        candidate_indices,
        layer=layer,
        mode="steering",
        steering_alpha=steering_alpha,
        pos_token=pos_token,
        neg_token=neg_token,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    # Flip this to switch experiments: "ablation" | "steering"
    MODE = "steering"
    # Signed additive steering strength: a_i ← a_i + α  (use −α to push the other way)
    STEERING_ALPHA = 0.6723
    K = 20
    # Candidate selection: "top" (|SHAP|) | "random" (uniform over filtered set)
    SELECTION = "random"
    SEED = 0

    # Full-width signed attributions; candidates restricted to the SHAP-filtered
    # feature set (outputs/3_shap); no union across methods.
    feature_indices = np.load("outputs/3_shap/shap_feature_indices.npy")
    sae_probe = joblib.load("checkpoints/probe_layer_7.joblib")

    probe_scores = sae_probe.coef_[0]
    ig_scores = np.load("outputs/8_rankings_recompute/ig_scores.npy")
    ga_scores = np.load("outputs/8_rankings_recompute/ga_scores.npy")
    shap_scores = np.load("outputs/7_shap_recompute/phi_sentiment_layer7_signed.npy")

    shap_filtered = shap_scores[feature_indices]
    local_top, global_candidates = get_shap_candidates(
        shap_filtered, feature_indices, k=K, selection=SELECTION, seed=SEED
    )
    if SELECTION == "top":
        sel_desc = f"top-{K} by |SHAP|"
    else:
        sel_desc = f"{K} random (seed={SEED})"
    print(
        f"Fixed candidate set: {sel_desc} among "
        f"{len(feature_indices)} filtered features"
    )
    print(f"  local indices:  {local_top.tolist()}")
    print(f"  global indices: {global_candidates.tolist()}")

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
        "SHAP": shap_scores,
        "Probe weights": probe_scores,
        "IG": ig_scores,
        "GA": ga_scores,
    }
    report_faithfulness(method_scores, delta_abs, delta_signed)

    out_dir = Path("outputs/9_steer_check")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"model_delta_abs_{MODE}.npy", delta_abs)
    np.save(out_dir / f"model_delta_signed_{MODE}.npy", delta_signed)
    np.save(out_dir / "shap_top_local.npy", local_top)
    np.save(out_dir / "shap_top_global.npy", global_candidates)
    print(f"\nWrote outputs → {out_dir}/")
