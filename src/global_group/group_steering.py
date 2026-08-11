"""
Group steering: effect-vs-cost curves for multi-feature residual steering vectors.

Part 1 — Build unit-norm steering vectors for each (probe_type, method, k):
  attribution methods: top-k by |score|, v = Σ attribution_i · W_dec[i]
  actdiff baseline:    top-k by |mean_pos − mean_neg| activation difference,
                       weights = signed mean difference (no attribution math)

Part 2 — Sweep alpha; add α·v at blocks.7.hook_resid_post (no SAE encode/decode).
  effect = mean Δ logit-diff (wonderful − awful)
  cost   = mean Δ perplexity (steered − clean baseline)

Scoring batches over examples *and* over (vector, alpha) conditions: each token
mini-batch is repeated across C conditions in one forward, with a hook that
adds a different α·v to each condition block.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sae_lens import SAE
from tqdm import tqdm
from transformer_lens import HookedTransformer

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils import (
    checkpoint_path,
    eval_sentences,
    last_real_token_index,
    load_model,
    output_path,
    sample_eval_rows,
)

# Match 9_steer / 14_mlp_steer: causal effects on the held-out rows the
# attributions describe, not on val.
N_STEER_EVAL = 1000

N_TOTAL_FEATURES = 32768
CONFIG_PATH = Path("steering_config.json")
OUT_PATH = output_path("group_steering", "group_steering_results.json")
METHODS = ("shap", "probe", "ig", "ga", "actdiff")
K_SUMMARY = 20
# Every method in every arm ranks over this same pool. Without it the logistic
# scores (full width) would draw top-k from all 32768 features while the MLP
# scores (zero outside the mask) could only draw from the filtered set, so the
# two arms would not be comparable at equal k.
FEATURE_POOL_PATH = output_path("3_shap", "shap_feature_indices.npy")


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def _pad_token_id(model: HookedTransformer) -> int:
    pad_id = model.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = model.tokenizer.eos_token_id
    return int(pad_id)


def sentiment_token_ids(
    model: HookedTransformer, pos_token: str, neg_token: str
) -> tuple[int, int]:
    pos_ids = model.to_tokens(pos_token, prepend_bos=False)[0]
    neg_ids = model.to_tokens(neg_token, prepend_bos=False)[0]
    if pos_ids.numel() != 1 or neg_ids.numel() != 1:
        raise ValueError(
            f"Expected single-token pair; got {pos_token!r}→{pos_ids.tolist()}, "
            f"{neg_token!r}→{neg_ids.tolist()}"
        )
    return int(pos_ids.item()), int(neg_ids.item())


def expand_to_full(
    scores: np.ndarray, feature_indices: np.ndarray, n_total: int = N_TOTAL_FEATURES
) -> np.ndarray:
    full = np.zeros(n_total, dtype=np.float64)
    full[np.asarray(feature_indices)] = np.asarray(scores, dtype=np.float64)
    return full


def load_logistic_scores() -> dict[str, np.ndarray]:
    """Full-width signed attributions for the logistic SAE probe."""
    probe = joblib.load(checkpoint_path("probe_layer_7.joblib"))
    return {
        "shap": np.load(
            output_path("7_shap_recompute", "phi_sentiment_layer7_signed.npy")
        ).astype(np.float64),
        "probe": np.asarray(probe.coef_[0], dtype=np.float64),
        "ig": np.load(output_path("8_rankings_recompute", "ig_scores.npy")).astype(
            np.float64
        ),
        "ga": np.load(output_path("8_rankings_recompute", "ga_scores.npy")).astype(
            np.float64
        ),
    }


def load_mlp_scores() -> dict[str, np.ndarray]:
    """Full-width attributions (zeros outside the filtered MLP feature set)."""
    ranking_dir = output_path("13_mlp_ranking")
    feature_indices = np.load(ranking_dir / "feature_indices.npy")
    return {
        name: expand_to_full(np.load(ranking_dir / f"{name}_scores.npy"), feature_indices)
        for name in ("shap", "probe", "ig", "ga")
    }


def restrict_to_pool(
    scores: np.ndarray, feature_pool: np.ndarray, n_total: int = N_TOTAL_FEATURES
) -> np.ndarray:
    """Zero every feature outside `feature_pool` so top-k ranks over one pool."""
    restricted = np.zeros(n_total, dtype=np.float64)
    restricted[feature_pool] = np.asarray(scores, dtype=np.float64)[feature_pool]
    return restricted


def activation_differences(
    activations_path: str,
    labels_path: str | None = None,
    chunk_size: int = 2048,
) -> np.ndarray:
    """
    Naive baseline scores: mean activation on positive examples minus mean on
    negative examples, per SAE feature. No attribution / probe math.
    """
    acts_path = Path(activations_path)
    if labels_path is None:
        labels_path = str(acts_path.with_name("labels.npy"))
    acts = np.load(acts_path, mmap_mode="r")
    labels = np.asarray(np.load(labels_path))
    if labels.shape[0] != acts.shape[0]:
        raise ValueError(
            f"labels length {labels.shape[0]} != activations n={acts.shape[0]}"
        )

    n, d = acts.shape
    sum_pos = np.zeros(d, dtype=np.float64)
    sum_neg = np.zeros(d, dtype=np.float64)
    n_pos = 0
    n_neg = 0
    for start in tqdm(range(0, n, chunk_size), desc="Activation differences"):
        chunk = np.asarray(acts[start : start + chunk_size], dtype=np.float32)
        lab = labels[start : start + chunk_size]
        pos_mask = lab == 1
        neg_mask = lab == 0
        if pos_mask.any():
            sum_pos += chunk[pos_mask].sum(axis=0, dtype=np.float64)
            n_pos += int(pos_mask.sum())
        if neg_mask.any():
            sum_neg += chunk[neg_mask].sum(axis=0, dtype=np.float64)
            n_neg += int(neg_mask.sum())

    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"Need both classes; got n_pos={n_pos}, n_neg={n_neg}")
    return sum_pos / n_pos - sum_neg / n_neg


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest-|score| entries (stable for ties via argsort)."""
    k = min(k, scores.shape[0])
    return np.argsort(np.abs(scores))[::-1][:k]


def build_steering_vector(
    W_dec: np.ndarray,
    feature_ids: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    v = Σ_i weight_i · W_dec[feature_i], then unit-normalize.

    W_dec: (n_features, d_model)
    """
    feature_ids = np.asarray(feature_ids, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    v = (weights[:, None] * W_dec[feature_ids]).sum(axis=0)
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        return v.astype(np.float32)
    return (v / norm).astype(np.float32)


def build_all_steering_vectors(
    W_dec: np.ndarray,
    scores_by_method: dict[str, np.ndarray],
    ks: list[int],
) -> dict[tuple[str, int], np.ndarray]:
    """
    Returns {(method, k): unit vector}.

    Every method (including actdiff) uses the same recipe: top-k by |score|,
    then v = Σ score_i · W_dec[i], unit-normalized.
    """
    vectors: dict[tuple[str, int], np.ndarray] = {}

    for method, scores in scores_by_method.items():
        for k in ks:
            ids = top_k_indices(scores, k)
            weights = scores[ids]  # signed weights (attribution or mean act-diff)
            vectors[(method, k)] = build_steering_vector(W_dec, ids, weights)

    return vectors


def last_non_pad_logit_diff(
    model: HookedTransformer,
    logits: torch.Tensor,
    tokens: torch.Tensor,
    pos_id: int,
    neg_id: int,
) -> torch.Tensor:
    last_idx = last_real_token_index(model, tokens)
    batch_idx = torch.arange(tokens.shape[0], device=tokens.device)
    last_logits = logits[batch_idx, last_idx, :]
    return last_logits[:, pos_id] - last_logits[:, neg_id]


def per_example_perplexity(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    """
    Per-example perplexity = exp(mean NLL over non-pad next-token targets).

    logits/tokens: (batch, seq, ...) / (batch, seq)
    returns: (batch,)
    """
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    targets = tokens[:, 1:]
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    # Ignore positions whose *target* is padding.
    mask = targets != pad_id
    # Also ignore positions whose source token is padding (no real context).
    # Position 0 is the prepended BOS, which shares the pad id but IS real
    # context, so keep it — dropping it would throw away the first real token's
    # NLL from every example.
    src_ok = tokens[:, :-1] != pad_id
    src_ok[:, 0] = True
    mask = mask & src_ok
    denom = mask.sum(dim=1).clamp(min=1).to(nll.dtype)
    mean_nll = (nll * mask).sum(dim=1) / denom
    return torch.exp(mean_nll)


def score_batches(
    model: HookedTransformer,
    all_tokens: torch.Tensor,
    pos_id: int,
    neg_id: int,
    pad_id: int,
    batch_size: int,
    fwd_hooks: list | None = None,
    desc: str = "Scoring",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (logit_diffs, perplexities), each shape (n,).

    fwd_hooks=None → clean forward (no residual intervention).
    """
    n = all_tokens.shape[0]
    diffs = []
    ppls = []
    for start in tqdm(range(0, n, batch_size), desc=desc, leave=False):
        batch = all_tokens[start : start + batch_size].to(model.cfg.device)
        with torch.no_grad():
            if fwd_hooks:
                with model.hooks(fwd_hooks=fwd_hooks):
                    logits = model(batch)
            else:
                logits = model(batch)
            diffs.append(
                last_non_pad_logit_diff(model, logits, batch, pos_id, neg_id)
                .float()
                .cpu()
                .numpy()
            )
            ppls.append(
                per_example_perplexity(logits, batch, pad_id).float().cpu().numpy()
            )
    return np.concatenate(diffs), np.concatenate(ppls)


def make_multi_condition_hook(
    layer: int,
    scaled_vectors: torch.Tensor,
    examples_per_cond: int,
):
    """
    For an expanded batch of shape (C * B, T, d), add scaled_vectors[c] to
    rows [c*B : (c+1)*B] at every token position.

    scaled_vectors: (C, d_model) — already α · v for each condition.
    """
    point = f"blocks.{layer}.hook_resid_post"

    def hook(resid_post, hook, _scaled=scaled_vectors, _b=examples_per_cond):
        c = resid_post.shape[0] // _b
        add = _scaled[:c].to(dtype=resid_post.dtype).repeat_interleave(_b, dim=0)
        return resid_post + add[:, None, :]

    return point, hook


def score_conditions(
    model: HookedTransformer,
    all_tokens: torch.Tensor,
    conditions: list[tuple[tuple[str, str, int], float, np.ndarray]],
    base_diffs: np.ndarray,
    base_ppls: np.ndarray,
    pos_id: int,
    neg_id: int,
    pad_id: int,
    layer: int,
    batch_size: int,
    condition_batch_size: int,
) -> list[dict]:
    """
    Evaluate many (meta, alpha, vector) conditions with batched forwards.

    Each token mini-batch of size B is repeated across up to
    `condition_batch_size` conditions → one forward of size ≤ C·B, with a
    hook that adds a different α·v to each condition block.
    """
    n = all_tokens.shape[0]
    device = model.cfg.device
    results: list[dict] = []

    for cond_start in tqdm(
        range(0, len(conditions), condition_batch_size), desc="Condition batches"
    ):
        chunk = conditions[cond_start : cond_start + condition_batch_size]
        c = len(chunk)
        scaled = np.stack(
            [float(alpha) * np.asarray(vector, dtype=np.float32) for _, alpha, vector in chunk],
            axis=0,
        )
        scaled_t = torch.as_tensor(scaled, device=device)

        # Accumulators for mean effect / cost over all examples.
        sum_delta_diff = np.zeros(c, dtype=np.float64)
        sum_delta_ppl = np.zeros(c, dtype=np.float64)

        for start in range(0, n, batch_size):
            batch = all_tokens[start : start + batch_size].to(device)
            b = batch.shape[0]
            expanded = batch.repeat(c, 1)  # (C*B, T); first B rows = cond 0
            point, hook = make_multi_condition_hook(layer, scaled_t, b)

            with torch.no_grad():
                with model.hooks(fwd_hooks=[(point, hook)]):
                    logits = model(expanded)
                diffs = last_non_pad_logit_diff(
                    model, logits, expanded, pos_id, neg_id
                ).float().cpu().numpy()
                ppls = (
                    per_example_perplexity(logits, expanded, pad_id)
                    .float()
                    .cpu()
                    .numpy()
                )

            diffs = diffs.reshape(c, b)
            ppls = ppls.reshape(c, b)
            base_d = base_diffs[start : start + b]
            base_p = base_ppls[start : start + b]
            sum_delta_diff += (diffs - base_d[None, :]).sum(axis=1)
            sum_delta_ppl += (ppls - base_p[None, :]).sum(axis=1)

        for i, ((probe_type, method, k), alpha, _) in enumerate(chunk):
            results.append(
                {
                    "probe_type": probe_type,
                    "method": method,
                    "k": int(k),
                    "alpha": float(alpha),
                    "effect": float(sum_delta_diff[i] / n),
                    "cost": float(sum_delta_ppl[i] / n),
                }
            )

    return results


def print_summary_table(results: list[dict], k: int = K_SUMMARY) -> None:
    """Effect/cost at each alpha for k=k, grouped by probe_type and method."""
    probe_types = sorted({r["probe_type"] for r in results})
    methods = [m for m in METHODS if any(r["method"] == m for r in results)]

    print(f"\n=== Effect-vs-cost summary (k={k}) ===")
    for probe_type in probe_types:
        print(f"\n[{probe_type}]")
        for method in methods:
            rows = [
                r
                for r in results
                if r["probe_type"] == probe_type and r["method"] == method and r["k"] == k
            ]
            rows = sorted(rows, key=lambda r: r["alpha"])
            if not rows:
                continue
            print(f"  method={method}")
            print(f"    {'alpha':>8s}  {'effect':>10s}  {'cost':>10s}")
            for r in rows:
                print(f"    {r['alpha']:8g}  {r['effect']:10.4f}  {r['cost']:10.4f}")


def main() -> None:
    cfg = load_config()
    ks = list(cfg["ks"])
    alphas = [float(a) for a in cfg["alphas"]]
    layer = int(cfg["layer"])
    pos_token = cfg["pos_token"]
    neg_token = cfg["neg_token"]
    batch_size = int(cfg.get("batch_size", 32))
    # How many (vector, alpha) conditions to pack into one forward.
    # Peak batch size ≈ batch_size * condition_batch_size.
    condition_batch_size = int(cfg.get("condition_batch_size", 8))

    feature_pool = np.load(FEATURE_POOL_PATH)
    oversized = [k for k in ks if k > len(feature_pool)]
    if oversized:
        print(
            f"Dropping k={oversized}: larger than the shared feature pool "
            f"({len(feature_pool)} features from {FEATURE_POOL_PATH})"
        )
        ks = [k for k in ks if k <= len(feature_pool)]
    if not ks:
        raise ValueError(f"No usable k <= {len(feature_pool)} in config")

    print("Loading attribution scores…")
    score_sets = {
        "logistic": load_logistic_scores(),
        "mlp": load_mlp_scores(),
    }

    print("Computing activation differences (pos − neg)…")
    actdiff = activation_differences(cfg["activations_path"])

    print("Loading model + SAE…")
    model = load_model()
    sae = SAE.from_pretrained(cfg["sae_release"], cfg["sae_id"])
    sae = sae.to(model.cfg.device)
    W_dec = sae.W_dec.detach().float().cpu().numpy()

    steer_eval_idx, _ = sample_eval_rows(layer=layer, n_eval=N_STEER_EVAL)
    steer_tokens = model.to_tokens(eval_sentences(steer_eval_idx))
    pad_id = _pad_token_id(model)
    pos_id, neg_id = sentiment_token_ids(model, pos_token, neg_token)
    print(
        f"Readout: {pos_token!r}({pos_id}) − {neg_token!r}({neg_id}); "
        f"n_eval={steer_tokens.shape[0]} held-out rows, layer={layer}; "
        f"batch_size={batch_size}, condition_batch_size={condition_batch_size}"
    )

    print(
        f"Building steering vectors over a shared pool of {len(feature_pool)} "
        "features (every method and both arms rank the same candidates)…"
    )
    all_vectors: dict[tuple[str, str, int], np.ndarray] = {}
    for probe_type, scores in score_sets.items():
        scores_with_baseline = {**scores, "actdiff": actdiff}
        scores_with_baseline = {
            name: restrict_to_pool(s, feature_pool)
            for name, s in scores_with_baseline.items()
        }
        vecs = build_all_steering_vectors(W_dec, scores_with_baseline, ks)
        for (method, k), v in vecs.items():
            all_vectors[(probe_type, method, k)] = v
            print(
                f"  {probe_type:8s} {method:10s} k={k:2d}  "
                f"||v||={np.linalg.norm(v):.4f}"
            )

    print("Computing clean baseline…")
    base_diffs, base_ppls = score_batches(
        model,
        steer_tokens,
        pos_id,
        neg_id,
        pad_id,
        batch_size,
        fwd_hooks=None,
        desc="Baseline",
    )
    print(
        f"  mean logit-diff={base_diffs.mean():+.4f}  "
        f"mean PPL={base_ppls.mean():.2f}"
    )

    conditions: list[tuple[tuple[str, str, int], float, np.ndarray]] = [
        ((probe_type, method, k), alpha, vector)
        for (probe_type, method, k), vector in all_vectors.items()
        for alpha in alphas
    ]
    print(
        f"Scoring {len(conditions)} conditions "
        f"({len(all_vectors)} vectors × {len(alphas)} alphas)…"
    )
    results = score_conditions(
        model,
        steer_tokens,
        conditions,
        base_diffs,
        base_ppls,
        pos_id,
        neg_id,
        pad_id,
        layer,
        batch_size,
        condition_batch_size,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {len(results)} entries → {OUT_PATH}")

    print_summary_table(results, k=K_SUMMARY)


if __name__ == "__main__":
    main()
