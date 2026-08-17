"""
Known-ground-truth validation: minimal-pair sentiment edits.

Why this exists
---------------
Every faithfulness number in this repo is measured on natural SST-2 text, where
the true causal importance of an SAE feature is unknown. So a near-zero
faithfulness rho has two incompatible readings that the main pipeline cannot
separate:

  (a) attribution methods really are unfaithful on this model, or
  (b) the harness (k=20 features, an alpha calibrated on a different readout,
      last-token attribution vs all-token intervention) cannot detect
      faithfulness even when it is there.

This script builds stimuli where a real sentiment change is known by
construction, then asks whether the harness recovers it. If it does, (a) stands
and the null result is about the methods. If it does not, the null is at least
partly about the measurement, and that has to be said in the paper.

The stimulus
------------
A minimal pair differing by exactly one strong sentiment word, drawn from the
lexicon in `10_simplicity_check.py`:

  --edit remove  (default, the cleanest pair): take held-out sentences with
      exactly one lexicon word and delete that token. Everything else is
      byte-identical, so the only thing that changed is the sentiment word.
  --edit insert: take sentences with no lexicon word at all and append
      " it was <word> ." Less minimal (it also adds a clause), but it works in
      the direction the remove-pairs cannot: adding sentiment where there was
      none.

Three things get measured
-------------------------
1. Readout validity. Does the GPT-2 logit difference move in the known direction
   (removing a positive word lowers it; removing a negative word raises it)?
   A readout that fails here cannot support any faithfulness claim downstream,
   so this is the precondition for reading anything else in the pipeline.

2. Feature recovery (the actual ground truth). The edit demonstrably acts through
   whichever SAE features changed activation between the pair. Labelling those
   features positive and the rest negative gives a binary ground truth, and each
   attribution method gets an AUROC for recovering them. Unlike a Spearman
   against an unknown target, this has a defined right answer.

3. Causal recovery. Ablating a feature in the edited sentence undoes some of the
   edit's effect. That per-feature recovery is a continuous ground-truth causal
   score on a controlled stimulus, which the attribution methods are then ranked
   against — the same faithfulness question as `9_steer`, but on text where a
   real effect is known to exist.

Cost: ~n_pairs * (2 + n_candidates) single-sentence forwards. At the defaults
(100 pairs, ~25 candidates) that is a few thousand passes — minutes, not hours.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from sae_lens import SAE
from scipy.stats import spearmanr
from tqdm import tqdm

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils import (
    CANONICAL_SEED,
    checkpoint_path,
    last_real_token_index,
    load_model,
    load_splits,
    output_path,
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_simplicity = _load("simplicity10", _SRC / "global_linear" / "10_simplicity_check.py")
_shap7 = _load("shap7", _SRC / "global_linear" / "7_shap_recompute.py")
_steer9 = _load("steer9", _SRC / "global_linear" / "9_steer.py")

LAYER = 7
SAE_RELEASE = "gpt2-small-resid-post-v5-32k"
SAE_ID = f"blocks.{LAYER}.hook_resid_post"
# A feature counts as "carrying the edit" if its activation moved by at least
# this fraction of the largest single-feature change in that pair. Relative, not
# absolute, because activation scales differ severalfold across features.
MOVED_REL_THRESHOLD = 0.10


def build_remove_pairs(
    sentences: list[str], rng: np.random.Generator, n_pairs: int
) -> list[dict]:
    """
    Sentences containing exactly one lexicon word, paired with that word deleted.

    Exactly one, not at least one: with two sentiment words the remaining one
    keeps carrying sentiment, so the pair is no longer a clean single-cause edit.
    """
    pool = []
    for s in sentences:
        toks = _simplicity.tokenize(s)
        pos_hits = [t for t in toks if t in _simplicity.POS_WORDS]
        neg_hits = [t for t in toks if t in _simplicity.NEG_WORDS]
        if len(pos_hits) + len(neg_hits) != 1:
            continue
        word = (pos_hits + neg_hits)[0]
        polarity = 1 if pos_hits else -1
        # Delete the whole word token, keeping surrounding whitespace sane.
        parts = s.split()
        idx = next((i for i, p in enumerate(parts) if p.strip(".,!?;:'\"") == word), None)
        if idx is None:
            continue
        edited_parts = parts[:idx] + parts[idx + 1 :]
        if len(edited_parts) < 3:
            continue
        pool.append(
            {
                "original": s,
                "edited": " ".join(edited_parts) + " ",
                "word": word,
                "polarity": polarity,
                # Removing a positive word should LOWER the logit diff.
                "expected_sign": -polarity,
            }
        )
    if not pool:
        raise RuntimeError("No single-lexicon-word sentences found")
    pick = rng.permutation(len(pool))[:n_pairs]
    return [pool[int(i)] for i in pick]


def build_insert_pairs(
    sentences: list[str], rng: np.random.Generator, n_pairs: int
) -> list[dict]:
    """Lexicon-free sentences, paired with ' it was <word> .' appended."""
    flags = _simplicity.lexicon_flags(sentences)
    neutral = [s for s, ok in zip(sentences, ~flags["has_any"]) if ok and len(s.split()) >= 3]
    if not neutral:
        raise RuntimeError("No lexicon-free sentences found")
    pos_words = sorted(_simplicity.POS_WORDS)
    neg_words = sorted(_simplicity.NEG_WORDS)
    pick = rng.permutation(len(neutral))[:n_pairs]
    pairs = []
    for j, i in enumerate(pick):
        s = neutral[int(i)]
        polarity = 1 if j % 2 == 0 else -1
        vocab = pos_words if polarity == 1 else neg_words
        word = vocab[int(rng.integers(len(vocab)))]
        pairs.append(
            {
                "original": s,
                "edited": s.rstrip() + f" it was {word} . ",
                "word": word,
                "polarity": polarity,
                # Adding a positive word should RAISE the logit diff.
                "expected_sign": polarity,
            }
        )
    return pairs


def last_token_sae_acts(model, sae, tokens: torch.Tensor) -> np.ndarray:
    """SAE feature activations at the last real token, shape (n_features,)."""
    hook_point = f"blocks.{LAYER}.hook_resid_post"
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook_point)
        acts = sae.encode(cache[hook_point])
    last_idx = last_real_token_index(model, tokens)
    return acts[0, int(last_idx[0]), :].detach().float().cpu().numpy()


def score_with_hooks(model, tokens, pos_id, neg_id, fwd_hooks) -> float:
    with torch.no_grad():
        with model.hooks(fwd_hooks=fwd_hooks):
            logits = model(tokens)
        return float(
            _steer9.last_non_pad_logit_diff(model, logits, tokens, pos_id, neg_id)[0]
        )


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Rank-based AUROC. Ties get average ranks, so it matches the Mann-Whitney form.

    Hand-rolled rather than sklearn's so this stays a pure numpy/scipy path and
    degenerate single-class inputs return nan instead of raising mid-loop.
    """
    labels = np.asarray(labels).astype(bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks within tied score groups.
    uniq, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.zeros(len(uniq))
        np.add.at(sums, inv, ranks)
        ranks = (sums / counts)[inv]
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--edit", choices=("remove", "insert"), default="remove")
    p.add_argument("--n-pairs", type=int, default=100)
    p.add_argument(
        "--k-moved",
        type=int,
        default=15,
        help="Top-|delta activation| features treated as edit-carrying candidates",
    )
    p.add_argument(
        "--k-distractor",
        type=int,
        default=15,
        help="Distractor features added to each pair's candidate set",
    )
    p.add_argument(
        "--distractors",
        choices=("random", "global"),
        default="random",
        help=(
            "'random' = uniform from the filtered pool (headline; unbiased). "
            "'global' = top-|Phi|, which makes the negative class the features "
            "global SHAP ranks highest and drives its AUROC down by construction "
            "- labelled secondary only"
        ),
    )
    p.add_argument("--seed", type=int, default=CANONICAL_SEED)
    p.add_argument("--out-dir", type=Path, default=output_path("24_minimal_pairs"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    _, _, shap_ds = load_splits()
    sentences = list(shap_ds["sentence"])
    builder = build_remove_pairs if args.edit == "remove" else build_insert_pairs
    pairs = builder(sentences, rng, args.n_pairs)
    print(f"Built {len(pairs)} '{args.edit}' minimal pairs from the held-out split")

    feature_indices = np.load(output_path("3_shap", "shap_feature_indices.npy"))
    probe = joblib.load(checkpoint_path("probe_layer_7.joblib"))
    probe.verbose = 0
    phi = np.load(
        output_path("7_shap_recompute", "phi_sentiment_layer7_signed.npy")
    ).astype(np.float64)
    ig = np.load(output_path("8_rankings_recompute", "ig_scores.npy")).astype(np.float64)
    ga = np.load(output_path("8_rankings_recompute", "ga_scores.npy")).astype(np.float64)
    probe_w = np.asarray(probe.coef_[0], dtype=np.float64)

    # Distractor set: features the edit is not expected to move, forming the
    # negative class of the AUROC.
    #
    # `random` is the headline. Picking distractors by top-|Phi| (the `global`
    # option) makes the negative class *definitionally* the features global SHAP
    # ranks highest, which drives its AUROC toward 0 no matter how faithful it
    # is — the same range-restriction trap `--selection top` sets in 9_steer.
    # Kept available as a labelled secondary, never as the headline.
    if args.distractors == "random":
        distractors = rng.choice(
            feature_indices, size=min(args.k_distractor, len(feature_indices)), replace=False
        )
    else:
        distractors = np.argsort(np.abs(phi))[::-1][: args.k_distractor]
    print(
        f"Distractors: {len(distractors)} {args.distractors}"
        + (
            "  (NOTE: chosen by |Phi|, which biases AUROC against global SHAP)"
            if args.distractors == "global"
            else ""
        )
    )

    model = load_model()
    sae = SAE.from_pretrained(SAE_RELEASE, SAE_ID).to(model.cfg.device)
    pos_id, neg_id = _steer9.sentiment_token_ids(model)
    intervene_point = f"blocks.{LAYER}.hook_resid_post"

    # Background for local SHAP: the canonical train background mean, the same
    # one 7_shap_recompute marginalises over.
    #
    # Deliberately NOT the pair's own original activations. With that as the
    # background, local phi_j = w_j * (x_j - orig_j) is proportional to the
    # activation change itself — which is exactly the quantity `moved_label` is
    # derived from, so local SHAP would score near-perfect AUROC by definition
    # rather than by being faithful.
    from utils import load_eval_and_background

    _, background, _, _ = load_eval_and_background(layer=LAYER)
    background_mean = background.mean(axis=0, keepdims=True)

    def recon_hook(resid_post, hook):
        return sae.decode(sae.encode(resid_post))

    rows = []
    per_method_rho = {m: [] for m in ("local_SHAP", "global_SHAP", "Probe weights", "IG", "GA")}
    per_method_auroc = {m: [] for m in per_method_rho}

    for pair in tqdm(pairs, desc="Minimal pairs"):
        tok_o = model.to_tokens([pair["original"]])
        tok_e = model.to_tokens([pair["edited"]])

        base_o = score_with_hooks(model, tok_o, pos_id, neg_id, [(intervene_point, recon_hook)])
        base_e = score_with_hooks(model, tok_e, pos_id, neg_id, [(intervene_point, recon_hook)])
        text_delta = base_e - base_o

        acts_o = last_token_sae_acts(model, sae, tok_o)
        acts_e = last_token_sae_acts(model, sae, tok_e)
        d_act = acts_e - acts_o

        # Ground-truth candidates: features the edit actually moved, plus
        # distractors it did not, so AUROC has both classes.
        moved = np.argsort(np.abs(d_act))[::-1][: args.k_moved]
        cand = np.unique(np.concatenate([moved, distractors]))

        max_move = float(np.abs(d_act[cand]).max())
        moved_label = np.abs(d_act[cand]) >= MOVED_REL_THRESHOLD * max_move

        # Causal recovery: ablate each candidate in the EDITED sentence and see
        # how much of the edited logit-diff it was supplying.
        recovery = np.zeros(len(cand), dtype=np.float64)
        for j, f in enumerate(cand):
            fi = int(f)

            def ablate_hook(resid_post, hook, feature_idx=fi):
                a = sae.encode(resid_post).clone()
                a[..., feature_idx] = 0.0
                return sae.decode(a)

            abl = score_with_hooks(
                model, tok_e, pos_id, neg_id, [(intervene_point, ablate_hook)]
            )
            recovery[j] = base_e - abl

        # Local SHAP on the edited sentence, over the filtered pool, against the
        # canonical train background (see the note where background_mean is set).
        local_phi_full = np.zeros(len(phi), dtype=np.float64)
        local_vals = _shap7.run_linearshap(
            probe, acts_e[None, :], background_mean, feature_indices
        )
        local_phi_full[feature_indices] = np.asarray(local_vals)[0]

        scores_by_method = {
            "local_SHAP": local_phi_full,
            "global_SHAP": phi,
            "Probe weights": probe_w,
            "IG": ig,
            "GA": ga,
        }
        row = {
            "word": pair["word"],
            "polarity": int(pair["polarity"]),
            "expected_sign": int(pair["expected_sign"]),
            "logitdiff_original": base_o,
            "logitdiff_edited": base_e,
            "text_delta": text_delta,
            "sign_correct": bool(np.sign(text_delta) == pair["expected_sign"]),
            "n_candidates": int(len(cand)),
            "n_moved": int(moved_label.sum()),
            "methods": {},
        }
        for name, vec in scores_by_method.items():
            s = np.abs(vec[cand])
            rho = spearmanr(s, np.abs(recovery)).statistic if len(cand) >= 3 else np.nan
            a = auroc(moved_label, s)
            row["methods"][name] = {"rho_recovery": float(rho), "auroc_moved": float(a)}
            if np.isfinite(rho):
                per_method_rho[name].append(float(rho))
            if np.isfinite(a):
                per_method_auroc[name].append(float(a))
        rows.append(row)

    sign_acc = float(np.mean([r["sign_correct"] for r in rows]))
    mean_abs_delta = float(np.mean([abs(r["text_delta"]) for r in rows]))
    summary = {
        "edit": args.edit,
        "n_pairs": len(rows),
        "seed": args.seed,
        "k_moved": args.k_moved,
        "k_distractor": args.k_distractor,
        "distractors": args.distractors,
        "moved_rel_threshold": MOVED_REL_THRESHOLD,
        "readout_sign_accuracy": sign_acc,
        "mean_abs_text_delta": mean_abs_delta,
        "methods": {
            name: {
                "mean_rho_recovery": float(np.mean(per_method_rho[name]))
                if per_method_rho[name]
                else float("nan"),
                "mean_auroc_moved": float(np.mean(per_method_auroc[name]))
                if per_method_auroc[name]
                else float("nan"),
                "n": len(per_method_rho[name]),
            }
            for name in per_method_rho
        },
    }

    lines = [
        "Minimal-pair known-ground-truth validation",
        "=" * 60,
        f"edit={args.edit}  n_pairs={len(rows)}  seed={args.seed}  "
        f"distractors={args.distractors}",
        (
            "NOTE: distractors=global picks the negative class by |Phi|, which "
            "biases AUROC\nagainst global SHAP by construction. Secondary only."
            if args.distractors == "global"
            else ""
        ),
        "",
        "1) Readout validity (does the known edit move the logit diff correctly?)",
        f"   sign accuracy      = {sign_acc:.3f}  (0.5 = chance)",
        f"   mean |text delta|  = {mean_abs_delta:.4f} logits",
        "",
        "2) Ground-truth feature recovery (AUROC for ranking edit-carrying",
        "   features above distractors) and 3) causal recovery (Spearman of",
        "   |attribution| vs |effect of ablating that feature|):",
        "",
        f"   {'method':16s} {'AUROC moved':>12s} {'rho recovery':>13s}",
        "   " + "-" * 43,
    ]
    for name, m in summary["methods"].items():
        lines.append(
            f"   {name:16s} {m['mean_auroc_moved']:12.3f} {m['mean_rho_recovery']:13.3f}"
        )
    lines += [
        "",
        "Reading: AUROC ~0.5 means the method cannot tell which features actually",
        "carried a known sentiment edit — that is a failure on a case with a known",
        "right answer, so it constrains the interpretation of the null result on",
        "natural text. AUROC well above 0.5 with near-zero faithfulness rho on",
        "natural text points the other way: the harness works, and the methods are",
        "genuinely unfaithful there.",
        "",
    ]
    text = "\n".join(lines)
    print("\n" + text)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.edit}_n{len(rows)}"
    with open(out_dir / f"summary_{tag}.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / f"per_pair_{tag}.json", "w") as f:
        json.dump(rows, f, indent=2)
    (out_dir / f"summary_{tag}.txt").write_text(text)
    print(f"Wrote outputs -> {out_dir}/  (tag: {tag})")


if __name__ == "__main__":
    main()
