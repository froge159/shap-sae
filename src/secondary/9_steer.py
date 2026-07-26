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


def get_ablation_candidates(probe_ranks, ig_ranks, shap_ranks, ga_ranks, k=20):
    # Top-k by each method (ranks are over all features; rank 1 = most important)
    shap_top = np.argsort(shap_ranks)[:k]
    probe_top = np.argsort(probe_ranks)[:k]
    ig_top = np.argsort(ig_ranks)[:k]
    ga_top = np.argsort(ga_ranks)[:k]

    # Union — ablate each unique feature once, reuse results for all methods
    all_candidates = np.unique(np.concatenate([shap_top, probe_top, ig_top, ga_top]))
    return all_candidates, shap_top, probe_top, ig_top, ga_top


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
    print("Importance faithfulness  (Spearman ρ of |attribution| vs |ΔP|):")
    for name, scores in method_scores_by_name.items():
        rho, p = faithfulness_correlation(scores, delta_abs, importance=True)
        print(f"  {name:14s}  ρ={rho:.3f}  p={p:.4f}")

    print("Directional faithfulness (Spearman ρ of attribution vs signed ΔP):")
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


def get_model_intervention_effects(
    model: HookedTransformer,
    sae: SAE,
    readout_probe,
    tokens_list,
    candidate_indices: np.ndarray,
    layer: int = 7,
    readout_layer: int | None = None,
    mode: str = "steering",
    steering_alpha: float = 2.0,
) -> tuple[dict, dict]:
    """
    Intervene on SAE features at `layer` and measure downstream residual-probe ΔP.

    mode="ablation": encode → zero feature → decode into resid_post.
    mode="steering": encode → a_i += steering_alpha (signed, additive) at every
    token position → decode.

    Returns
    -------
    delta_abs : dict[int, float]
        mean |P_int - P_base| per feature (effect magnitude).
    delta_signed : dict[int, float]
        mean (P_int - P_base) per feature (effect direction).

    Baseline runs encode→decode with no feature edit so ΔP is not confounded by
    SAE reconstruction error. Downstream readout lets the intervention propagate
    through later layers before measurement.
    """
    if mode not in ("ablation", "steering"):
        raise ValueError(f"Unknown mode={mode!r}; expected 'ablation' or 'steering'")
    if readout_layer is None:
        readout_layer = model.cfg.n_layers - 1

    intervene_point = f"blocks.{layer}.hook_resid_post"
    readout_point = f"blocks.{readout_layer}.hook_resid_post"
    pad_id = _pad_token_id(model)
    delta_abs: dict[int, float] = {}
    delta_signed: dict[int, float] = {}
    resid_store: dict[str, torch.Tensor] = {}

    def capture_hook(resid_post, hook):
        resid_store["resid"] = resid_post
        return resid_post

    def recon_hook(resid_post, hook):
        return sae.decode(sae.encode(resid_post))

    def probe_prob(tokens: torch.Tensor) -> float:
        pooled = pool_last_non_pad_token(tokens, resid_store["resid"], pad_id)
        return float(readout_probe.predict_proba(pooled.reshape(1, -1))[0, 1])

    # SAE-reconstruction baseline (no feature edit), readout downstream
    baseline_probs = []
    for tokens in tqdm(tokens_list, desc="Computing baselines"):
        tokens = _as_batch(tokens, model.cfg.device)
        with torch.no_grad():
            with model.hooks(
                fwd_hooks=[(intervene_point, recon_hook), (readout_point, capture_hook)]
            ):
                model(tokens)
            baseline_probs.append(probe_prob(tokens))
    baseline_probs = np.array(baseline_probs)

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

        intervened_probs = []
        for tokens in tokens_list:
            tokens = _as_batch(tokens, model.cfg.device)
            with torch.no_grad():
                with model.hooks(
                    fwd_hooks=[
                        (intervene_point, intervention_hook),
                        (readout_point, capture_hook),
                    ]
                ):
                    model(tokens)
                intervened_probs.append(probe_prob(tokens))

        intervened_probs = np.array(intervened_probs)
        diffs = intervened_probs - baseline_probs
        delta_abs[idx] = float(np.mean(np.abs(diffs)))
        delta_signed[idx] = float(np.mean(diffs))

    return delta_abs, delta_signed


def get_model_ablation_effects(
    model: HookedTransformer,
    sae: SAE,
    readout_probe,
    tokens_list,
    candidate_indices: np.ndarray,
    layer: int = 7,
    readout_layer: int | None = None,
) -> tuple[dict, dict]:
    """Ablate SAE features (zero activations). Thin wrapper around intervention API."""
    return get_model_intervention_effects(
        model,
        sae,
        readout_probe,
        tokens_list,
        candidate_indices,
        layer=layer,
        readout_layer=readout_layer,
        mode="ablation",
    )


def get_model_steering_effects(
    model: HookedTransformer,
    sae: SAE,
    readout_probe,
    tokens_list,
    candidate_indices: np.ndarray,
    layer: int = 7,
    readout_layer: int | None = None,
    steering_alpha: float = 2.0,
) -> tuple[dict, dict]:
    """Additively steer each selected feature: a_i ← a_i + α (α may be signed)."""
    return get_model_intervention_effects(
        model,
        sae,
        readout_probe,
        tokens_list,
        candidate_indices,
        layer=layer,
        readout_layer=readout_layer,
        mode="steering",
        steering_alpha=steering_alpha,
    )


if __name__ == "__main__":
    # Flip this to switch experiments: "ablation" | "steering"
    MODE = "steering"
    # Signed additive steering strength: a_i ← a_i + α  (use −α to push the other way)
    STEERING_ALPHA = 2.0

    # Attribution rankings still come from the layer-7 SAE probe / SHAP pipeline.
    # Score vectors are signed and shape (n_features,); candidates selected by |score|.
    sae_probe = joblib.load("checkpoints/probe_layer_7.joblib")

    probe_scores = sae_probe.coef_[0]
    ig_scores = np.load("outputs/8_rankings_recompute/ig_scores.npy")
    ga_scores = np.load("outputs/8_rankings_recompute/ga_scores.npy")
    shap_scores = np.load("outputs/7_shap_recompute/phi_sentiment_layer7_signed.npy")
    probe_ranks, ig_ranks, ga_ranks, shap_ranks = compile_rankings(
        ig_scores, probe_scores, ga_scores, shap_scores
    )

    all_candidates, shap_top, probe_top, ig_top, ga_top = get_ablation_candidates(
        probe_ranks, ig_ranks, shap_ranks, ga_ranks, k=20
    )

    model = load_model()
    sae = SAE.from_pretrained("gpt2-small-resid-post-v5-32k", "blocks.7.hook_resid_post")
    sae = sae.to(model.cfg.device)

    train_ds, val_ds, _ = load_splits()
    readout_layer = model.cfg.n_layers - 1  # GPT-2 Small: 11

    train_tokens = model.to_tokens(list(train_ds["sentence"]))
    train_labels = np.asarray(train_ds["label"])
    residual_probe = load_or_train_residual_probe(
        model,
        train_tokens,
        train_labels,
        layer=readout_layer,
        checkpoint_path=f"checkpoints/residual_probe_layer_{readout_layer}.joblib",
    )

    val_tokens = model.to_tokens(list(val_ds["sentence"]))
    delta_abs, delta_signed = get_model_intervention_effects(
        model,
        sae,
        residual_probe,
        val_tokens,
        all_candidates,
        layer=7,
        readout_layer=readout_layer,
        mode=MODE,
        steering_alpha=STEERING_ALPHA,
    )

    label = (
        "Ablated"
        if MODE == "ablation"
        else f"Steered (a_i += {STEERING_ALPHA:+.3g})"
    )
    print(f"{label} {len(delta_abs)} features (readout L{readout_layer})")
    print("Top features by |ΔP|:")
    for idx, delta in sorted(delta_abs.items(), key=lambda x: -x[1])[:10]:
        print(f"  feature {idx}: |ΔP|={delta:.4f}  signedΔP={delta_signed[idx]:+.4f}")

    method_scores = {
        "SHAP": shap_scores,
        "Probe weights": probe_scores,
        "IG": ig_scores,
        "GA": ga_scores,
    }
    report_faithfulness(method_scores, delta_abs, delta_signed)

    out_dir = f"outputs/{MODE}"
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/model_delta_abs.npy", delta_abs)
    np.save(f"{out_dir}/model_delta_signed.npy", delta_signed)
