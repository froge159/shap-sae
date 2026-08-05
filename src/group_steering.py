"""
Group steering: effect-vs-cost curves for multi-feature residual steering vectors.

Part 1 — Build unit-norm steering vectors for each (probe_type, method, k):
  attribution methods: top-k by |score|, v = Σ attribution_i · W_dec[i]
  frequency baseline:  top-k by activation frequency, uniform weights

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

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils import load_model, load_splits

N_TOTAL_FEATURES = 32768
CONFIG_PATH = Path("steering_config.json")
OUT_PATH = Path("outputs/group_steering_results.json")
METHODS = ("shap", "probe", "ig", "ga", "frequency")
K_SUMMARY = 20


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
    probe = joblib.load("checkpoints/probe_layer_7.joblib")
    return {
        "shap": np.load("outputs/7_shap_recompute/phi_sentiment_layer7_signed.npy").astype(
            np.float64
        ),
        "probe": np.asarray(probe.coef_[0], dtype=np.float64),
        "ig": np.load("outputs/8_rankings_recompute/ig_scores.npy").astype(np.float64),
        "ga": np.load("outputs/8_rankings_recompute/ga_scores.npy").astype(np.float64),
    }


def load_mlp_scores() -> dict[str, np.ndarray]:
    """Full-width attributions (zeros outside the filtered MLP feature set)."""
    ranking_dir = Path("outputs/13_mlp_ranking")
    feature_indices = np.load(ranking_dir / "feature_indices.npy")
    return {
        name: expand_to_full(np.load(ranking_dir / f"{name}_scores.npy"), feature_indices)
        for name in ("shap", "probe", "ig", "ga")
    }


def activation_frequencies(activations_path: str, chunk_size: int = 2048) -> np.ndarray:
    """Fraction of examples where each SAE feature is active (> 0)."""
    acts = np.load(activations_path, mmap_mode="r")
    n, d = acts.shape
    counts = np.zeros(d, dtype=np.float64)
    for start in tqdm(range(0, n, chunk_size), desc="Activation frequencies"):
        chunk = np.asarray(acts[start : start + chunk_size], dtype=np.float32)
        counts += (chunk > 0).sum(axis=0)
    return counts / max(n, 1)


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
    frequencies: np.ndarray,
    ks: list[int],
) -> dict[tuple[str, int], np.ndarray]:
    """
    Returns {(method, k): unit vector} for attribution methods + frequency baseline.
    """
    vectors: dict[tuple[str, int], np.ndarray] = {}

    for method, scores in scores_by_method.items():
        for k in ks:
            ids = top_k_indices(scores, k)
            weights = scores[ids]  # signed attribution weights
            vectors[(method, k)] = build_steering_vector(W_dec, ids, weights)

    for k in ks:
        ids = top_k_indices(frequencies, k)
        weights = np.ones(len(ids), dtype=np.float64)
        vectors[("frequency", k)] = build_steering_vector(W_dec, ids, weights)

    return vectors


def last_non_pad_logit_diff(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    pad_id: int,
    pos_id: int,
    neg_id: int,
) -> torch.Tensor:
    mask = tokens != pad_id
    last_idx = mask.sum(dim=1) - 1
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
    src_ok = tokens[:, :-1] != pad_id
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
                last_non_pad_logit_diff(logits, batch, pad_id, pos_id, neg_id)
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
                    logits, expanded, pad_id, pos_id, neg_id
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

    print("Loading attribution scores…")
    score_sets = {
        "logistic": load_logistic_scores(),
        "mlp": load_mlp_scores(),
    }

    print("Computing activation frequencies…")
    frequencies = activation_frequencies(cfg["activations_path"])

    print("Loading model + SAE…")
    model = load_model()
    sae = SAE.from_pretrained(cfg["sae_release"], cfg["sae_id"])
    sae = sae.to(model.cfg.device)
    W_dec = sae.W_dec.detach().float().cpu().numpy()

    _, val_ds, _ = load_splits()
    val_tokens = model.to_tokens(list(val_ds["sentence"]))
    pad_id = _pad_token_id(model)
    pos_id, neg_id = sentiment_token_ids(model, pos_token, neg_token)
    print(
        f"Readout: {pos_token!r}({pos_id}) − {neg_token!r}({neg_id}); "
        f"n_val={val_tokens.shape[0]}, layer={layer}; "
        f"batch_size={batch_size}, condition_batch_size={condition_batch_size}"
    )

    print("Building steering vectors…")
    all_vectors: dict[tuple[str, str, int], np.ndarray] = {}
    for probe_type, scores in score_sets.items():
        vecs = build_all_steering_vectors(W_dec, scores, frequencies, ks)
        for (method, k), v in vecs.items():
            all_vectors[(probe_type, method, k)] = v
            print(
                f"  {probe_type:8s} {method:10s} k={k:2d}  "
                f"||v||={np.linalg.norm(v):.4f}"
            )

    print("Computing clean baseline…")
    base_diffs, base_ppls = score_batches(
        model,
        val_tokens,
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
        val_tokens,
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
