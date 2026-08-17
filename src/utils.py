import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from sae_lens import HookedSAETransformer
from scipy.stats import binomtest, false_discovery_control
from transformer_lens import utils as tl_utils

# Canonical sampling parameters. Every script that computes attributions or
# measures causal effects draws its rows from `sample_eval_rows` with these
# defaults so that SHAP / IG / GA / steering are all scored on the same data.
CANONICAL_SEED = 42
DEFAULT_N_EVAL = 2000
DEFAULT_N_BACKGROUND = 100

SPLIT_DIRS = {
    "train": "probe_train",
    "val": "probe_val",
    "shap": "shap",
}

# Artifact roots. Override per run without editing any script:
#   SHAP_SAE_OUTPUTS=runs/exp3 uv run python src/global_mlp/14_mlp_steer.py
# Resolved at import, so module-level constants like `OUT_DIR = output_path(...)`
# pick the override up.
OUTPUTS_DIR = Path(os.environ.get("SHAP_SAE_OUTPUTS", "outputs"))
CHECKPOINTS_DIR = Path(os.environ.get("SHAP_SAE_CHECKPOINTS", "checkpoints"))


def output_path(*parts: str) -> Path:
    """Path under the configured output root. Does not create the directory."""
    return OUTPUTS_DIR.joinpath(*parts)


def checkpoint_path(*parts: str) -> Path:
    """Path under the configured checkpoint root. Does not create the directory."""
    return CHECKPOINTS_DIR.joinpath(*parts)


def get_shap_feature_mask(
    probe, val_activations: np.ndarray, min_activation_freq: float = 0.05
):
    """
    Filtered feature set: non-zero probe weight AND active often enough.

    `val_activations` must come from the *val* split. This mask defines the
    MLP probe's input layer and the steering candidate pool, so deriving it
    from the shap split would let the held-out set shape the models it is
    later used to evaluate.
    """
    # Filter 1: non-zero probe weights
    nonzero_mask = probe.coef_[0] != 0  # shape: (32768,)

    # Filter 2: activation frequency on val set
    activation_freq = (val_activations > 0).mean(axis=0)  # shape: (32768,)
    freq_mask = activation_freq >= min_activation_freq

    combined_mask = nonzero_mask & freq_mask
    feature_indices = np.where(combined_mask)[0]

    print(f"Non-zero probe weights: {nonzero_mask.sum()}")
    print(f"After frequency filter: {len(feature_indices)}")

    return feature_indices


def resolve_steering_alphas(
    candidate_indices: np.ndarray,
    alpha: float,
    alpha_mode: str = "scaled",
    tuning_dir: Path | None = None,
) -> tuple[dict[int, float], str]:
    """
    Per-feature additive steering strength, as {global SAE id: alpha_i}.

    The single definition of how `--alpha` becomes a per-feature nudge, shared by
    the linear and MLP steer scripts so the two arms cannot drift apart.

    alpha_mode="constant"
        alpha_i = alpha for every feature. What the original runs used. Its
        problem: features differ a lot in natural activation scale, so a flat
        alpha shoves a feature whose typical active value is 0.2 about three
        times as hard (in relative terms) as one at 5.0, and the cross-feature
        Spearman of |attribution| vs |delta| then partly ranks how far each
        feature was pushed relative to itself.

    alpha_mode="scaled"
        alpha_i = alpha * scale_i / median(scale), where scale_i is the mean
        positive activation on *train* saved by 9.5_tuning. Equalises the
        relative nudge, and reduces to `constant` for a feature at the median
        scale, so the overall magnitude 9.5_tuning calibrated is preserved.

    No decoder-norm correction is applied, deliberately: this SAE's W_dec rows are
    essentially unit norm (median 0.9996), so ||W_dec[i]|| contributes nothing to
    the confound and dividing by it would only add noise.

    Features missing from the tuning file fall back to the median scale (i.e. to
    `alpha`) with a warning rather than failing the run.
    """
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    if alpha_mode == "constant":
        return {int(f): float(alpha) for f in candidate_indices}, (
            f"constant alpha={alpha:+.4g} for all {len(candidate_indices)} features"
        )
    if alpha_mode != "scaled":
        raise ValueError(
            f"Unknown alpha_mode={alpha_mode!r}; expected 'constant' or 'scaled'"
        )

    tuning_dir = Path(tuning_dir) if tuning_dir else output_path("9.5_tuning")
    # Prefer pool_* (scales for the whole filtered pool, so --selection random can
    # draw any feature and still get its own scale). Fall back to candidate_*,
    # which only covers the calibration union, for runs made before pool_* existed.
    scales_path = tuning_dir / "pool_scales.npy"
    idx_path = tuning_dir / "pool_indices.npy"
    if not (scales_path.exists() and idx_path.exists()):
        scales_path = tuning_dir / "candidate_scales.npy"
        idx_path = tuning_dir / "candidate_indices.npy"
    if not (scales_path.exists() and idx_path.exists()):
        raise FileNotFoundError(
            f"alpha_mode='scaled' needs pool_scales.npy + pool_indices.npy under "
            f"{tuning_dir}. Run src/global_linear/9.5_tuning.py first, or pass "
            "--alpha-mode constant."
        )

    scales = np.load(scales_path)
    tuned_idx = np.load(idx_path)
    scale_of = {int(f): float(s) for f, s in zip(tuned_idx, scales)}
    median_scale = float(np.median(scales))

    missing = [int(f) for f in candidate_indices if int(f) not in scale_of]
    if missing:
        print(
            f"WARNING: {len(missing)} candidate features have no tuned activation "
            f"scale ({missing[:5]}{'...' if len(missing) > 5 else ''}); falling "
            "back to the median scale for those."
        )

    alphas = {
        int(f): float(alpha * scale_of.get(int(f), median_scale) / median_scale)
        for f in candidate_indices
    }
    lo, hi = min(alphas.values()), max(alphas.values())
    desc = (
        f"scaled alpha = {alpha:+.4g} x scale_i / {median_scale:.4f} "
        f"(range {lo:+.4g} .. {hi:+.4g})"
    )
    return alphas, desc


def benjamini_hochberg(
    pvalues: dict[str, float], alpha: float = 0.05
) -> dict[str, dict]:
    """
    Benjamini-Hochberg FDR correction over a caller-defined family of tests.

    The faithfulness tables run 4 methods x 2 correlation types = 8 tests per
    configuration, and a seed sweep multiplies that by the number of seeds, with
    no correction applied historically. At 8 uncorrected tests, one "significant"
    result at p<0.05 is the *expected* yield under a complete null — which is
    exactly what run2's two isolated hits look like.

    The family is whatever the caller passes in: pass the 8 tests of one run to
    correct within a configuration, or the 8*n_seeds of a sweep to correct across
    it. Deliberately not hardcoded, because the right family depends on what is
    being claimed.

    Returns {name: {"p": raw, "q": adjusted, "reject": bool}}. Order of the input
    dict is preserved in the output.
    """
    if not pvalues:
        return {}
    names = list(pvalues)
    raw = np.asarray([float(pvalues[n]) for n in names], dtype=np.float64)
    # NaN p-values (a degenerate Spearman on constant input) would poison the
    # adjustment; hold them out and re-insert as NaN afterwards.
    finite = np.isfinite(raw)
    q = np.full_like(raw, np.nan)
    if finite.any():
        q[finite] = false_discovery_control(raw[finite], method="bh")
    return {
        name: {
            "p": float(raw[i]),
            "q": float(q[i]),
            "reject": bool(np.isfinite(q[i]) and q[i] <= alpha),
        }
        for i, name in enumerate(names)
    }


def sign_agreement_test(
    signed_scores: dict[int, float],
    oriented_deltas: dict[int, float],
) -> dict:
    """
    Does the *sign* of an attribution predict the sign of its causal effect?

    Complements the Spearman directional rho, which can be dragged around by a
    couple of large-|delta| features. This asks the blunter question: across k
    features, how often does sign(attribution) == sign(oriented delta), and is
    that better than a coin flip?

    `oriented_deltas` must already have passed through `orient_signed_effects`,
    so "agreement" means the same thing under steering and ablation.

    Reports the two-sided binomial test (is the agreement rate different from
    chance?) and the one-sided `less` test (is it *worse* than chance — i.e. an
    actual sign inversion, which is the specific hypothesis run1 and run2 both
    hint at). Features with a delta of exactly 0 are counted as non-agreeing but
    also reported separately, since an inactive feature under ablation gives
    delta == 0 by construction and carries no directional information.
    """
    feats = sorted(set(signed_scores) & set(oriented_deltas))
    n_agree = 0
    n_zero_delta = 0
    for f in feats:
        s = float(signed_scores[f])
        d = float(oriented_deltas[f])
        if d == 0.0:
            n_zero_delta += 1
            continue
        if (s > 0 and d > 0) or (s < 0 and d < 0):
            n_agree += 1
    n = len(feats) - n_zero_delta
    if n == 0:
        return {
            "n": 0,
            "n_agree": 0,
            "n_zero_delta": n_zero_delta,
            "agreement_rate": float("nan"),
            "binom_two_sided_p": float("nan"),
            "binom_less_p": float("nan"),
        }
    return {
        "n": int(n),
        "n_agree": int(n_agree),
        "n_zero_delta": int(n_zero_delta),
        "agreement_rate": float(n_agree / n),
        "binom_two_sided_p": float(
            binomtest(n_agree, n, 0.5, alternative="two-sided").pvalue
        ),
        # "less" = agreement is *below* chance = the sign-inversion hypothesis.
        "binom_less_p": float(binomtest(n_agree, n, 0.5, alternative="less").pvalue),
    }


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HookedSAETransformer.from_pretrained_no_processing("gpt2", device=device)
    return model


def last_real_token_index(model, tokens: torch.Tensor) -> torch.Tensor:
    """
    Index of the final non-padding token in each row of `tokens`, shape (batch,).

    The single definition of "the last token" for the whole pipeline: activation
    extraction, the probe readout, and the logit-diff readout all pool here so
    they cannot drift apart.

    Do not hand-roll this as `(tokens != pad_id).sum(dim=1) - 1`. GPT-2 has no
    dedicated pad token, so transformer_lens sets pad = bos = eos =
    "<|endoftext|>"; that expression counts the prepended BOS as padding and
    lands one position early — on the *second*-to-last real token, or on BOS
    itself for a single-token sentence. `get_attention_mask` resolves the
    ambiguity by only treating trailing (or leading) pads as real padding.
    """
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)
    attention_mask = tl_utils.get_attention_mask(
        model.tokenizer, tokens, prepend_bos=bool(model.cfg.default_prepend_bos)
    )
    # clamp(min=0) only bites on an all-padding row (empty input), where there is
    # no real token to point at and position 0 is the least-wrong answer.
    return (attention_mask.sum(dim=1) - 1).clamp(min=0)


def load_splits():
    """
    The fixed three-way split, read from local disk.

    Must stay on `load_from_disk` rather than the Hub: activations were
    extracted in this row order, and scripts pair `shap_ds[i]["sentence"]`
    with `activations/shap/...[i]`. A Hub revision that reorders SST-2 would
    silently decouple the two.
    """
    ds = load_from_disk("data/sst2_train")
    with open("data/three_way_split_indices.json") as f:
        indices = json.load(f)

    train_ds = ds.select(indices["train_indices"])
    val_ds = ds.select(indices["val_indices"])
    shap_ds = ds.select(indices["shap_indices"])

    return train_ds, val_ds, shap_ds


def activations_path(split: str, layer: int, activations_dir: str = "activations") -> Path:
    """Directory holding activations.npy / labels.npy for one split and layer."""
    try:
        split_dir = SPLIT_DIRS[split]
    except KeyError:
        raise ValueError(
            f"Unknown split {split!r}; expected one of {sorted(SPLIT_DIRS)}"
        ) from None
    return Path(activations_dir) / split_dir / f"layer_{layer}"


def load_activations(
    split: str,
    layer: int = 7,
    activations_dir: str = "activations",
    mmap: bool = False,
) -> np.ndarray:
    """Activations for one split, shape (n_sentences, 32768)."""
    path = activations_path(split, layer, activations_dir) / "activations.npy"
    return np.load(path, mmap_mode="r" if mmap else None)


def load_labels(
    split: str, layer: int = 7, activations_dir: str = "activations"
) -> np.ndarray:
    path = activations_path(split, layer, activations_dir) / "labels.npy"
    return np.load(path)


def _n_rows(split: str, layer: int, activations_dir: str) -> int:
    return load_activations(split, layer, activations_dir, mmap=True).shape[0]


def sample_eval_rows(
    layer: int = 7,
    n_eval: int = DEFAULT_N_EVAL,
    n_background: int = DEFAULT_N_BACKGROUND,
    seed: int = CANONICAL_SEED,
    activations_dir: str = "activations",
) -> tuple[np.ndarray, np.ndarray]:
    """
    The single definition of which rows get explained and steered.

    Returns
    -------
    eval_idx : (n_eval,) row indices into the *shap* held-out split
    bg_idx   : (n_background,) row indices into the *train* split

    Both are prefixes of a seeded permutation, not `rng.choice` draws, which
    buys two properties the pipeline depends on:

      - Nesting: for a fixed seed, `sample_eval_rows(n_eval=100)[0]` is exactly
        `sample_eval_rows(n_eval=2000)[0][:100]`, elementwise. So the 100 rows
        the local scripts explain are an ordered prefix of the rows the global
        scripts explain, and results join by position.
      - Independence: `bg_idx` does not shift when `n_eval` changes, and vice
        versa, because each permutation consumes a fixed amount of RNG state.

    Rows are left in permutation order (not sorted) to preserve nesting.
    """
    rng = np.random.default_rng(seed)

    n_shap = _n_rows("shap", layer, activations_dir)
    n_train = _n_rows("train", layer, activations_dir)

    if n_eval > n_shap:
        raise ValueError(f"n_eval={n_eval} exceeds shap split size {n_shap}")
    if n_background > n_train:
        raise ValueError(
            f"n_background={n_background} exceeds train split size {n_train}"
        )

    eval_idx = rng.permutation(n_shap)[:n_eval]
    bg_idx = rng.permutation(n_train)[:n_background]
    return eval_idx, bg_idx


def load_eval_and_background(
    layer: int = 7,
    n_eval: int = DEFAULT_N_EVAL,
    n_background: int = DEFAULT_N_BACKGROUND,
    seed: int = CANONICAL_SEED,
    activations_dir: str = "activations",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Canonical (eval, background) activations plus the indices that produced them.

    eval rows come from the shap held-out split; background rows from train.
    Both are full width (32768); callers slice to their filtered feature set.
    """
    eval_idx, bg_idx = sample_eval_rows(
        layer=layer,
        n_eval=n_eval,
        n_background=n_background,
        seed=seed,
        activations_dir=activations_dir,
    )
    shap_acts = load_activations("shap", layer, activations_dir, mmap=True)
    train_acts = load_activations("train", layer, activations_dir, mmap=True)

    shap_eval = np.asarray(shap_acts[eval_idx])
    background = np.asarray(train_acts[bg_idx])
    return shap_eval, background, eval_idx, bg_idx


def eval_sentences(eval_idx: np.ndarray, split: str = "shap") -> list[str]:
    """
    Sentences aligned row-for-row with `load_activations(split)[eval_idx]`.

    `Dataset.select` preserves the order of the indices it is given, so the
    returned list stays positionally matched to the activation rows — which is
    what lets a steering script attribute example i to explanation i.
    """
    train_ds, val_ds, shap_ds = load_splits()
    ds = {"train": train_ds, "val": val_ds, "shap": shap_ds}[split]
    return list(ds.select([int(i) for i in eval_idx])["sentence"])


def count_indices():
    with open("data/three_way_split_indices.json") as f:
        indices = json.load(f)
    for key in ("train_indices", "val_indices", "shap_indices"):
        print(f"{key}: {len(indices[key])}")


if __name__ == "__main__":
    count_indices()
