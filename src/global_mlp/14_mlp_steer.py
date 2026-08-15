"""
Steering / ablation faithfulness for MLP-probe attributions.

Mirrors global_linear/9_steer.py (logit-diff readout, SAE encode→edit→decode), but:
  - Loads filtered (n_filtered,) scores from 13_mlp_ranking/
  - Maps local → global SAE indices via feature_indices.npy
  - Fixes a single candidate set: top-k by |SHAP| or k random filtered features
  - Evaluates SHAP / probe / IG / GA faithfulness on that shared set

`Probe saliency` is ‖W1[i,:]‖₂ and has no sign, so it appears in the importance
table only — a magnitude cannot predict the direction of a signed Δ.

CLI kept byte-for-byte in step with global_linear/9_steer.py — matching `--mode`,
`--selection`, `--alpha`, `--alpha-mode`, `--k` and `--n-eval` is the only way the
two arms are comparable. See that module's docstring for why `--selection random`
and `--alpha-mode scaled` are the defaults.
"""

from __future__ import annotations

import argparse
import json
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

from utils import (
    eval_sentences,
    last_real_token_index,
    load_model,
    output_path,
    resolve_steering_alphas,
    sample_eval_rows,
)

# Same held-out rows (and same prefix) as 9_steer, so the linear and MLP arms
# are scored on identical sentences.
N_STEER_EVAL = 1000

N_TOTAL_FEATURES = 32768
RANKINGS_DIR = output_path("13_mlp_ranking")


def expand_to_full(
    scores: np.ndarray,
    feature_indices: np.ndarray,
    n_total: int = N_TOTAL_FEATURES,
) -> np.ndarray:
    """Scatter filtered scores into a dense (n_total,) vector (zeros elsewhere)."""
    full = np.zeros(n_total, dtype=np.float64)
    full[feature_indices] = scores
    return full


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


def orient_signed_effects(delta_signed: dict, mode: str) -> dict:
    """
    Put signed Δ on a scale where "faithful" always means positive ρ.

    Steering adds +α, so a positive attribution predicts a positive Δ. Ablation
    *removes* the feature, so a positive attribution predicts a *negative* Δ and
    a perfectly faithful method would score ρ ≈ −1 on the raw numbers.
    """
    if mode == "ablation":
        return {k: -v for k, v in delta_signed.items()}
    if mode == "steering":
        return dict(delta_signed)
    raise ValueError(f"Unknown mode={mode!r}; expected 'ablation' or 'steering'")


def report_faithfulness(
    method_scores_by_name: dict,
    delta_abs: dict,
    delta_signed: dict,
    mode: str,
    unsigned_methods: frozenset[str] = frozenset(),
    header: str = "",
) -> str:
    """
    Importance and directional faithfulness for each ranking method.

    `unsigned_methods` are magnitude-only scores (e.g. ‖W1‖₂): they get an
    importance row but are skipped in the directional table, where correlating a
    non-negative score against a signed Δ would be meaningless.

    Returns the rendered table so the caller can write it to disk — these are the
    headline numbers, and stdout is not a result file.
    """
    lines = []
    if header:
        lines.append(header)
    lines.append("Importance faithfulness  (Spearman rho of |attribution| vs |delta|):")
    for name, scores in method_scores_by_name.items():
        rho, p = faithfulness_correlation(scores, delta_abs, importance=True)
        lines.append(f"  {name:14s}  rho={rho:+.3f}  p={p:.4f}")

    oriented = orient_signed_effects(delta_signed, mode)
    effect_desc = "-delta" if mode == "ablation" else "delta"
    lines.append(
        f"Directional faithfulness (Spearman rho of attribution vs {effect_desc}; "
        f"positive = faithful under mode={mode}):"
    )
    for name, scores in method_scores_by_name.items():
        if name in unsigned_methods:
            lines.append(f"  {name:14s}  n/a (magnitude-only score, no direction)")
            continue
        rho, p = faithfulness_correlation(scores, oriented, importance=False)
        lines.append(f"  {name:14s}  rho={rho:+.3f}  p={p:.4f}")

    text = "\n".join(lines) + "\n"
    print(text)
    return text


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
    """
    Zero or additively steer one feature at every token position on a cloned tensor.

    Edits SAE *activation* space (the caller passes `sae.encode(resid_post)` and
    decodes the result), never probe weights.
    """
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
    model: HookedTransformer,
    logits: torch.Tensor,
    tokens: torch.Tensor,
    pos_id: int,
    neg_id: int,
) -> np.ndarray:
    """
    logit_pos - logit_neg at the last non-padding position.

    logits: (batch, seq, d_vocab)
    tokens: (batch, seq)
    returns: (batch,) numpy
    """
    last_idx = last_real_token_index(model, tokens)
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
    alpha_by_feature: dict[int, float] | None = None,
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

    `alpha_by_feature` gives a per-feature alpha (see
    `utils.resolve_steering_alphas`); when None every feature gets
    `steering_alpha`. Ignored under mode="ablation".
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
    pos_id, neg_id = sentiment_token_ids(model, pos_token, neg_token)
    delta_abs: dict[int, float] = {}
    delta_signed: dict[int, float] = {}

    def recon_hook(resid_post, hook):
        return sae.decode(sae.encode(resid_post))

    def score_batch(tokens: torch.Tensor, fwd_hooks: list) -> np.ndarray:
        with torch.no_grad():
            with model.hooks(fwd_hooks=fwd_hooks):
                logits = model(tokens)
            return last_non_pad_logit_diff(model, logits, tokens, pos_id, neg_id)

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
        alpha_i = (
            steering_alpha
            if alpha_by_feature is None
            else float(alpha_by_feature[idx])
        )

        def intervention_hook(
            resid_post,
            hook,
            feature_idx=idx,
            _mode=mode,
            _steering_alpha=alpha_i,
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # Keep this surface identical to global_linear/9_steer.py.
    p.add_argument("--mode", choices=("steering", "ablation"), default="steering")
    p.add_argument("--alpha", type=float, default=0.6723)
    p.add_argument(
        "--alpha-mode",
        choices=("constant", "scaled"),
        default="scaled",
        help=(
            "'scaled' rescales alpha by each feature's own train activation scale "
            "(from 9.5_tuning); 'constant' adds the same alpha everywhere"
        ),
    )
    p.add_argument("--k", type=int, default=20)
    p.add_argument(
        "--selection",
        choices=("top", "random"),
        default="random",
        help=(
            "'random' = uniform over the filtered set (headline); 'top' = top-k by "
            "|SHAP|, which range-restricts SHAP and biases its rho downward"
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-eval", type=int, default=N_STEER_EVAL)
    p.add_argument("--out-dir", type=Path, default=output_path("14_mlp_steer"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    MODE = args.mode
    STEERING_ALPHA = args.alpha
    K = args.k
    SELECTION = args.selection
    SEED = args.seed

    feature_indices = np.load(RANKINGS_DIR / "feature_indices.npy")
    ig_scores = np.load(RANKINGS_DIR / "ig_scores.npy")
    ga_scores = np.load(RANKINGS_DIR / "ga_scores.npy")
    probe_scores = np.load(RANKINGS_DIR / "probe_scores.npy")
    shap_scores = np.load(RANKINGS_DIR / "shap_scores.npy")

    # 13_mlp_ranking already asserts its scores and 12_mlp_shap's Phi describe the
    # same held-out rows; re-check here that the file we are reading is the one
    # that check ran against, so a partial re-run cannot slip through.
    _rank_eval_path = RANKINGS_DIR / "eval_indices.npy"
    if _rank_eval_path.exists():
        print(
            f"Rankings scored on {len(np.load(_rank_eval_path))} held-out rows "
            f"({RANKINGS_DIR.name}/eval_indices.npy)"
        )
    else:
        print(
            f"WARNING: {_rank_eval_path} missing - cannot confirm which held-out "
            "rows these scores describe. Re-run 13_mlp_ranking."
        )

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
        shap_scores, feature_indices, k=K, selection=SELECTION, seed=SEED
    )
    if SELECTION == "top":
        sel_desc = f"top-{K} by |SHAP|"
    else:
        sel_desc = f"{K} random (seed={SEED})"
    print(
        f"Fixed candidate set: {sel_desc} "
        f"({len(global_candidates)} global SAE features)"
    )
    print(f"  local indices:  {local_top.tolist()}")
    print(f"  global indices: {global_candidates.tolist()}")
    if K < 30:
        print(
            f"  NOTE: faithfulness ρ below is computed over n={K} features across "
            "4 methods × 2 correlation types — treat individual p-values as "
            "descriptive, not confirmatory."
        )

    # Full-width vectors so faithfulness can index by global SAE id (as in 9_steer).
    ig_full = expand_to_full(ig_scores, feature_indices)
    ga_full = expand_to_full(ga_scores, feature_indices)
    probe_full = expand_to_full(probe_scores, feature_indices)
    shap_full = expand_to_full(shap_scores, feature_indices)

    model = load_model()
    sae = SAE.from_pretrained("gpt2-small-resid-post-v5-32k", "blocks.7.hook_resid_post")
    sae = sae.to(model.cfg.device)

    steer_eval_idx, _ = sample_eval_rows(layer=7, n_eval=args.n_eval)
    steer_tokens = model.to_tokens(eval_sentences(steer_eval_idx))
    print(f"Steering eval: {len(steer_eval_idx)} held-out rows")

    pos_id, neg_id = sentiment_token_ids(model, POS_TOKEN, NEG_TOKEN)
    print(
        f"Logit-diff readout: {POS_TOKEN!r}({pos_id}) - {NEG_TOKEN!r}({neg_id}) "
        f"at last non-pad token"
    )

    if MODE == "steering":
        alpha_by_feature, alpha_desc = resolve_steering_alphas(
            global_candidates, STEERING_ALPHA, args.alpha_mode
        )
        print(f"Steering strength: {alpha_desc}")
    else:
        alpha_by_feature, alpha_desc = None, "n/a (ablation zeroes the feature)"

    delta_abs, delta_signed = get_model_intervention_effects(
        model,
        sae,
        steer_tokens,
        global_candidates,
        layer=7,
        mode=MODE,
        steering_alpha=STEERING_ALPHA,
        alpha_by_feature=alpha_by_feature,
        pos_token=POS_TOKEN,
        neg_token=NEG_TOKEN,
    )

    label = "Ablated" if MODE == "ablation" else f"Steered ({alpha_desc})"
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
    header = (
        f"MLP arm - faithfulness\n"
        f"mode={MODE}  selection={SELECTION}  k={K}  seed={SEED}  "
        f"n_eval={len(steer_eval_idx)}\n"
        f"alpha: {alpha_desc}\n"
        + (
            "NOTE: selection='top' picks candidates by |SHAP|, which restricts "
            "SHAP's range on the scored set and biases its rho downward relative "
            "to the other methods.\n"
            if SELECTION == "top"
            else ""
        )
        + (
            f"NOTE: rho over n={K} features x 4 methods x 2 correlation types, "
            "no multiple-comparison control - p-values are descriptive.\n"
            if K < 30
            else ""
        )
    )
    faith_text = report_faithfulness(
        method_scores,
        delta_abs,
        delta_signed,
        MODE,
        unsigned_methods=frozenset({"Probe saliency"}),
        header=header,
    )

    # Config-tagged filenames: these used to be config-independent, so a second
    # run into the same directory replaced the first run's candidate set.
    tag = f"{MODE}_{SELECTION}_k{K}"
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    # JSON, not np.save: these are dict[int, float], and np.save would pickle
    # them into a 0-d object array that only reloads with allow_pickle=True.
    with open(out_dir / f"model_deltas_{tag}.json", "w") as f:
        json.dump(
            {
                "mode": MODE,
                "steering_alpha": STEERING_ALPHA,
                "alpha_mode": args.alpha_mode,
                "alpha_by_feature": (
                    {str(k): v for k, v in alpha_by_feature.items()}
                    if alpha_by_feature
                    else None
                ),
                "selection": SELECTION,
                "k": K,
                "seed": SEED,
                "n_eval": int(len(steer_eval_idx)),
                "delta_abs": {str(k): v for k, v in delta_abs.items()},
                "delta_signed": {str(k): v for k, v in delta_signed.items()},
            },
            f,
            indent=2,
        )
    (out_dir / f"faithfulness_{tag}.txt").write_text(faith_text)
    np.save(out_dir / f"shap_top_local_{tag}.npy", local_top)
    np.save(out_dir / f"shap_top_global_{tag}.npy", global_candidates)
    np.save(out_dir / f"eval_indices_{tag}.npy", steer_eval_idx)
    print(f"\nWrote outputs → {out_dir}/  (tag: {tag})")
