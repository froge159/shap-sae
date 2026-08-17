"""
Steering / ablation faithfulness for the linear (logistic) SAE probe.

Readout is the GPT-2 logit difference (" wonderful" − " awful") at the last real
token — deliberately *not* the probe the attributions explain, so faithfulness is
measured against the model rather than against the explainer's own target.

Two mismatches to keep in mind when reading the numbers:
  - Attributions describe the SAE activation vector at one token position;
    `_apply_feature_intervention` edits the feature at *every* token position.
  - `--alpha` comes from 9.5_tuning, which calibrated it against a layer-11
    residual-probe probability, not against this logit-diff scale.

Two flags exist because the defaults were quietly biasing the result:
  - `--selection`: `top` picks candidates by |SHAP|, then scores all four methods
    on them. That restricts SHAP's range on the scored set while leaving the
    others unrestricted, which deflates SHAP's rho relative to methods that did
    not drive selection. Prefer `random` for the headline number.
  - `--alpha-mode`: `constant` adds the same alpha to every feature regardless of
    its natural activation scale, so |delta| partly tracks how hard each feature
    was pushed relative to itself. `scaled` equalises that. See
    `utils.resolve_steering_alphas`.

Outputs are suffixed with `{mode}_{selection}_k{k}` so that two configurations
written to one directory cannot silently overwrite each other's candidate sets.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from sae_lens import SAE
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from transformer_lens import HookedTransformer

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils import (
    CANONICAL_SEED,
    benjamini_hochberg,
    checkpoint_path,
    eval_sentences,
    last_real_token_index,
    load_model,
    output_path,
    resolve_steering_alphas,
    sample_eval_rows,
    sign_agreement_test,
)

# Causal effects are measured on the held-out rows the attributions describe,
# not on val. Nesting makes these the first N_STEER_EVAL rows of the set
# 7_shap_recompute / 8_feature_ranking scored.
N_STEER_EVAL = 1000


SELECTIONS = ("top", "random", "all", "top-pos", "top-neg")


def selection_tag(selection: str) -> str:
    """Filename-safe form of a --selection value ('top-pos' -> 'toppos')."""
    return selection.replace("-", "")


def get_shap_candidates(
    shap_scores: np.ndarray,
    feature_indices: np.ndarray,
    k: int = 20,
    selection: str = "top",
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fixed candidate set from the filtered feature pool.

    selection="top":     top-k by |SHAP|
    selection="random":  k features drawn uniformly (without replacement)
    selection="all":     every filtered feature (k and seed ignored)
    selection="top-pos": top-k by *signed* SHAP among features with Phi > 0
    selection="top-neg": top-k by |SHAP| among features with Phi < 0

    `shap_scores` and ranking are over the filtered set only (aligned with
    `feature_indices`). Intervention uses the mapped global SAE ids.
    `shap_scores` is unused when selection="random" or "all".

    Why "all" exists: the headline faithfulness rho is computed over k=20
    features, which is a small enough n that the confidence interval on Spearman
    rho spans most of [-1, 1]. Scoring the whole pool raises n per test from 20
    to len(feature_indices) without any resampling, and — unlike "random" — is
    deterministic, so it carries no seed-draw luck at all.

    Why the sign-stratified selections exist: "top" mixes positive- and
    negative-Phi features, so a systematic sign inversion (features SHAP calls
    positive moving the logit *down*) partly cancels within the candidate set and
    shows up only as a muddy directional rho. Scoring one sign at a time makes
    the binomial sign test in `utils.sign_agreement_test` interpretable.

    Returns
    -------
    local_idx : (k',) indices into the filtered score vectors
    global_idx : (k',) global SAE feature indices for intervention

    k' == k except for "all" (== n) and the sign-stratified modes, which are
    capped by how many features carry that sign.
    """
    n = len(feature_indices)
    k = min(k, n)
    if selection == "top":
        local_idx = np.argsort(np.abs(shap_scores))[::-1][:k]
    elif selection == "random":
        rng = np.random.default_rng(seed)
        local_idx = rng.choice(n, size=k, replace=False)
    elif selection == "all":
        local_idx = np.arange(n)
    elif selection in ("top-pos", "top-neg"):
        scores = np.asarray(shap_scores, dtype=np.float64)
        pool = np.flatnonzero(scores > 0 if selection == "top-pos" else scores < 0)
        if pool.size == 0:
            raise ValueError(
                f"selection={selection!r} but no filtered feature has that Phi sign"
            )
        order = np.argsort(np.abs(scores[pool]))[::-1][:k]
        local_idx = pool[order]
        if len(local_idx) < k:
            print(
                f"NOTE: selection={selection} yielded only {len(local_idx)} of the "
                f"requested k={k} features (that is how many carry that sign)."
            )
    else:
        raise ValueError(f"Unknown selection={selection!r}; expected one of {SELECTIONS}")
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


def orient_signed_effects(delta_signed: dict, mode: str) -> dict:
    """
    Put signed Δ on a scale where "faithful" always means positive ρ.

    Steering adds +α, so a positive attribution predicts a positive Δ and the
    raw effect is already correctly oriented. Ablation *removes* the feature, so
    a positive attribution predicts a *negative* Δ — a perfectly faithful method
    would score ρ ≈ −1 on the raw numbers. Negating here keeps the two modes
    comparable and stops a correct result from reading as a failure.
    """
    if mode == "ablation":
        return {k: -v for k, v in delta_signed.items()}
    if mode == "steering":
        return dict(delta_signed)
    raise ValueError(f"Unknown mode={mode!r}; expected 'ablation' or 'steering'")


def faithfulness_stats(
    method_scores_by_name: dict,
    delta_abs: dict,
    delta_signed: dict,
    mode: str,
) -> dict[str, dict]:
    """
    Every faithfulness test of one run as {"<method>:<kind>": {"rho", "p"}}.

    Split out from `report_faithfulness` so the numbers can be written to a
    machine-readable sidecar and fed to `utils.benjamini_hochberg` without
    re-parsing the rendered table.
    """
    oriented = orient_signed_effects(delta_signed, mode)
    stats: dict[str, dict] = {}
    for name, scores in method_scores_by_name.items():
        rho_i, p_i = faithfulness_correlation(scores, delta_abs, importance=True)
        stats[f"{name}:importance"] = {"rho": float(rho_i), "p": float(p_i)}
        rho_d, p_d = faithfulness_correlation(scores, oriented, importance=False)
        stats[f"{name}:directional"] = {"rho": float(rho_d), "p": float(p_d)}
    return stats


def report_faithfulness(
    method_scores_by_name: dict,
    delta_abs: dict,
    delta_signed: dict,
    mode: str,
    header: str = "",
    bh_correction: bool = False,
) -> str:
    """
    Importance and directional faithfulness for each ranking method.

    Returns the rendered table so the caller can write it to disk. These are the
    headline numbers of the whole comparison; printing them to stdout only (as
    this used to) meant a completed run left no record of its own result.

    `bh_correction=False` reproduces the historical output byte for byte, so
    existing `faithfulness_*.txt` artifacts stay comparable. Setting it True
    appends a Benjamini-Hochberg block over this run's 8-test family (4 methods x
    2 correlation types) — worth reading before calling any single p<0.05 here a
    result, since 8 uncorrected tests yield one by chance under a complete null.
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
        rho, p = faithfulness_correlation(scores, oriented, importance=False)
        lines.append(f"  {name:14s}  rho={rho:+.3f}  p={p:.4f}")

    if bh_correction:
        stats = faithfulness_stats(
            method_scores_by_name, delta_abs, delta_signed, mode
        )
        corrected = benjamini_hochberg({k: v["p"] for k, v in stats.items()})
        lines.append("")
        lines.append(
            f"Benjamini-Hochberg FDR correction (family = {len(corrected)} tests "
            "in this run, alpha=0.05):"
        )
        for name, row in corrected.items():
            flag = "*" if row["reject"] else " "
            lines.append(
                f"  {name:28s}  p={row['p']:.4f}  q={row['q']:.4f} {flag}"
            )
        n_rej = sum(r["reject"] for r in corrected.values())
        lines.append(f"  {n_rej} of {len(corrected)} survive FDR correction.")

    text = "\n".join(lines) + "\n"
    print(text)
    return text


def pool_last_non_pad_token(
    model: HookedTransformer, tokens: torch.Tensor, acts: torch.Tensor
) -> np.ndarray:
    """
    Select activations at the last non-padding token (matches core/1_extract.py).

    tokens: (batch, seq) or (seq,)
    acts:   (batch, seq, d) or (seq, d)
    returns: (d,) when batch==1, else (batch, d)
    """
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)
    if acts.ndim == 2:
        acts = acts.unsqueeze(0)

    last_idx = last_real_token_index(model, tokens)
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
            pooled = pool_last_non_pad_token(model, batch, resid)
            if pooled.ndim == 1:
                pooled = pooled[None, :]
            rows.append(pooled)
    else:
        for tokens in tqdm(tokens_list, desc=f"Residuals L{layer}"):
            tokens = _as_batch(tokens, model.cfg.device)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=hook_point)
            resid = cache[hook_point]
            pooled = pool_last_non_pad_token(model, tokens, resid)
            if pooled.ndim == 1:
                pooled = pooled[None, :]
            rows.append(pooled)

    return np.concatenate(rows, axis=0).astype(np.float32)


def train_residual_probe(
    model: HookedTransformer,
    tokens_list,
    labels: np.ndarray,
    layer: int,
    ckpt_path: str | None = None,
) -> LogisticRegression:
    """Train a logistic probe on last-token residual stream at `layer`."""
    X = collect_last_token_residuals(model, tokens_list, layer)
    # random_state pins liblinear's internal data shuffling; without it the fit
    # (and so the alpha 9.5_tuning recommends) drifts between runs.
    probe = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="liblinear",
        random_state=CANONICAL_SEED,
        verbose=0,
    )
    probe.fit(X, labels)
    if ckpt_path is not None:
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(probe, ckpt_path)
    return probe


def load_or_train_residual_probe(
    model: HookedTransformer,
    tokens_list,
    labels: np.ndarray,
    layer: int,
    checkpoint_path: str,
) -> LogisticRegression:
    """`checkpoint_path` here is a path string, not utils.checkpoint_path."""
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
    """
    Zero or additively steer one feature at every token position on a cloned tensor.

    Edits SAE *activation* space (the caller passes `sae.encode(resid_post)` and
    decodes the result), never probe weights.
    """
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
    with default tokens " wonderful" / " awful".

    mode="ablation": encode → zero feature → decode into resid_post.
    mode="steering": encode → a_i += alpha_i (signed, additive) at every
    token position → decode.

    `alpha_by_feature` gives a per-feature alpha (see
    `utils.resolve_steering_alphas`); when None every feature gets
    `steering_alpha`. Ignored under mode="ablation", which has no magnitude.

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
    p.add_argument("--mode", choices=("steering", "ablation"), default="steering")
    p.add_argument(
        "--alpha",
        type=float,
        default=0.6723,
        help="Signed additive steering strength: a_i <- a_i + alpha",
    )
    p.add_argument(
        "--alpha-mode",
        choices=("constant", "scaled"),
        default="scaled",
        help=(
            "'scaled' rescales alpha by each feature's own train activation scale "
            "(from 9.5_tuning) so every feature gets a comparable relative nudge; "
            "'constant' adds the same alpha everywhere (what the original runs did)"
        ),
    )
    p.add_argument("--k", type=int, default=20, help="Candidate features")
    p.add_argument(
        "--selection",
        choices=SELECTIONS,
        default="random",
        help=(
            "'random' = uniform over the filtered set (headline; unbiased); "
            "'top' = top-k by |SHAP|, which range-restricts SHAP on the scored "
            "set and deflates its rho relative to the other methods; "
            "'all' = the whole filtered pool (max power, ignores --k/--seed); "
            "'top-pos'/'top-neg' = sign-stratified top-k, for the sign test"
        ),
    )
    p.add_argument("--seed", type=int, default=0, help="Seed for --selection random")
    p.add_argument(
        "--n-eval",
        type=int,
        default=N_STEER_EVAL,
        help="Held-out rows to score (prefix of the canonical sample)",
    )
    p.add_argument(
        "--bh-correction",
        action="store_true",
        help=(
            "Append a Benjamini-Hochberg block to the table and write a "
            "faithfulness_<tag>_bh.json sidecar (does not alter the existing "
            "faithfulness_<tag>.txt body)"
        ),
    )
    p.add_argument("--out-dir", type=Path, default=output_path("9_steer"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    MODE = args.mode
    STEERING_ALPHA = args.alpha
    K = args.k
    SELECTION = args.selection
    SEED = args.seed

    # Full-width signed attributions; candidates restricted to the SHAP-filtered
    # feature set (3_shap); no union across methods.
    SHAP_DIR = output_path("7_shap_recompute")
    RANK_DIR = output_path("8_rankings_recompute")

    feature_indices = np.load(output_path("3_shap", "shap_feature_indices.npy"))
    sae_probe = joblib.load(checkpoint_path("probe_layer_7.joblib"))

    probe_scores = sae_probe.coef_[0]
    ig_scores = np.load(RANK_DIR / "ig_scores.npy")
    ga_scores = np.load(RANK_DIR / "ga_scores.npy")
    shap_scores = np.load(SHAP_DIR / "phi_sentiment_layer7_signed.npy")

    # The four score vectors have to describe the same held-out rows, or the
    # faithfulness table below ranks methods partly by which data they saw.
    _shap_eval_path = SHAP_DIR / "eval_indices_layer7.npy"
    _rank_eval_path = RANK_DIR / "eval_indices.npy"
    if _shap_eval_path.exists() and _rank_eval_path.exists():
        if not np.array_equal(np.load(_shap_eval_path), np.load(_rank_eval_path)):
            raise ValueError(
                f"Row mismatch between {SHAP_DIR}/ and {RANK_DIR}/: SHAP and IG/GA "
                "were scored on different held-out rows. Re-run 7_shap_recompute "
                "and 8_feature_ranking against the same n_eval."
            )
        print(f"Attribution rows match across {SHAP_DIR.name}/ and {RANK_DIR.name}/")
    else:
        print(
            "WARNING: eval_indices missing from 7_shap_recompute/ or "
            "8_rankings_recompute/ - cannot verify SHAP and IG/GA describe the "
            "same rows. Re-run both."
        )

    for _name, _arr in (("ig", ig_scores), ("ga", ga_scores), ("shap", shap_scores)):
        if _arr.shape != probe_scores.shape:
            raise ValueError(
                f"{_name}_scores shape {_arr.shape} != full SAE width "
                f"{probe_scores.shape}; these must be full-width vectors indexed "
                "by global SAE id."
            )

    shap_filtered = shap_scores[feature_indices]
    local_top, global_candidates = get_shap_candidates(
        shap_filtered, feature_indices, k=K, selection=SELECTION, seed=SEED
    )
    # "all" and the sign-stratified modes decide their own size; K drives the
    # output tag and the small-n warning, so re-derive it from what came back.
    K = len(global_candidates)
    sel_desc = {
        "top": f"top-{K} by |SHAP|",
        "random": f"{K} random (seed={SEED})",
        "all": f"all {K} filtered features (deterministic)",
        "top-pos": f"top-{K} by signed SHAP among Phi>0",
        "top-neg": f"top-{K} by |SHAP| among Phi<0",
    }[SELECTION]
    print(
        f"Fixed candidate set: {sel_desc} among "
        f"{len(feature_indices)} filtered features"
    )
    print(f"  local indices:  {local_top.tolist()}")
    print(f"  global indices: {global_candidates.tolist()}")
    if K < 30:
        print(
            f"  NOTE: faithfulness ρ below is computed over n={K} features across "
            "4 methods × 2 correlation types — treat individual p-values as "
            "descriptive, not confirmatory."
        )

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

    label = (
        "Ablated"
        if MODE == "ablation"
        else f"Steered ({alpha_desc})"
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
    header = (
        f"Linear (logistic) arm - faithfulness\n"
        f"mode={MODE}  selection={SELECTION}  k={K}  seed={SEED}  "
        f"n_eval={len(steer_eval_idx)}\n"
        f"alpha: {alpha_desc}\n"
        + (
            "NOTE: selection='top' picks candidates by |SHAP|, which restricts "
            "SHAP's range on the scored set and biases its rho downward relative "
            "to the other three methods.\n"
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
        header=header,
        bh_correction=args.bh_correction,
    )

    # Sign test: does sign(attribution) predict sign(effect) better than chance?
    # Only meaningful on a sign-pure candidate set — on a mixed-sign set a
    # systematic inversion partly cancels. See utils.sign_agreement_test.
    sign_block = None
    if SELECTION in ("top-pos", "top-neg"):
        oriented = orient_signed_effects(delta_signed, MODE)
        sign_block = {
            name: sign_agreement_test(
                {int(f): float(scores[int(f)]) for f in global_candidates}, oriented
            )
            for name, scores in method_scores.items()
        }
        print(f"Sign agreement ({SELECTION}, oriented for mode={MODE}):")
        for name, row in sign_block.items():
            if row["n"] == 0:
                print(f"  {name:14s}  n/a (every delta was exactly 0)")
                continue
            print(
                f"  {name:14s}  {row['n_agree']}/{row['n']} agree "
                f"({row['agreement_rate']:.3f})  two-sided p={row['binom_two_sided_p']:.4f}"
                f"  inversion(less) p={row['binom_less_p']:.4f}"
            )

    # Every artifact is tagged with the configuration that produced it. These
    # files used to have config-independent names, so a second run into the same
    # directory silently replaced the candidate set the first run's JSON refers to.
    tag = f"{MODE}_{selection_tag(SELECTION)}_k{K}"
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
                # Additive: absent for the historical selections, so old readers
                # of this schema keep working.
                **({"sign_agreement": sign_block} if sign_block else {}),
            },
            f,
            indent=2,
        )
    (out_dir / f"faithfulness_{tag}.txt").write_text(faith_text)
    if args.bh_correction:
        stats = faithfulness_stats(method_scores, delta_abs, delta_signed, MODE)
        corrected = benjamini_hochberg({k: v["p"] for k, v in stats.items()})
        with open(out_dir / f"faithfulness_{tag}_bh.json", "w") as f:
            json.dump(
                {
                    "mode": MODE,
                    "selection": SELECTION,
                    "k": K,
                    "family_size": len(corrected),
                    "alpha": 0.05,
                    "tests": {
                        name: {**stats[name], **corrected[name]} for name in stats
                    },
                },
                f,
                indent=2,
            )
    np.save(out_dir / f"shap_top_local_{tag}.npy", local_top)
    np.save(out_dir / f"shap_top_global_{tag}.npy", global_candidates)
    np.save(out_dir / f"eval_indices_{tag}.npy", steer_eval_idx)
    print(f"\nWrote outputs → {out_dir}/  (tag: {tag})")
