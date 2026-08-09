"""
SST lexical simplicity check.

Asks whether SST-2 is too lexically simple for SHAP's interaction-aware
attribution to matter:

  1. How many of the filtered SAE features are active per example?
  2. Do top-ranked features mostly track strong sentiment words?
  3. How far does a pure lexicon bag-of-words baseline get vs the SAE probe?

If examples rarely co-activate multiple ranked features and top features
correlate tightly with sentiment lexicon hits, sentiment is largely a lexical
shortcut — little genuine interaction for KernelSHAP to have an edge on.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import pointbiserialr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from datasets import load_from_disk


def load_splits():
    """Same three-way split as extract.py (local disk, no Hub lock)."""
    import json

    ds = load_from_disk("data/sst2_train")
    with open("data/three_way_split_indices.json") as f:
        indices = json.load(f)
    train_ds = ds.select(indices["train_indices"])
    val_ds = ds.select(indices["val_indices"])
    shap_ds = ds.select(indices["shap_indices"])
    return train_ds, val_ds, shap_ds

# ---------------------------------------------------------------------------
# Strong sentiment lexicon (movie-review / SST style). No extra downloads.
# ---------------------------------------------------------------------------
POS_WORDS = frozenset(
    """
    amazing awesome beautiful best brilliant charming delightful excellent
    exceptional fantastic favorite fun funny glad glorious good great happy
    hilarious impressive joy joyful lovely magnificent marvelous masterful
    outstanding perfect pleasant powerful remarkable rich solid splendid
    stunning superb terrific touching wonderful worthwhile witty wise warm
    enjoyable entertaining engaging elegant enchanting extraordinary fine
    fresh heartwarming honest hopeful inspiring interesting magical
    memorable moving nice original poignant pretty refreshing satisfying
    sensational sharp smart spectacular sweet talented thrilling triumphant
    uplifting vibrant vivid
    """.split()
)
NEG_WORDS = frozenset(
    """
    awful bad boring clumsy dull dreadful failed failure flat flawed
    forgettable frustrating grim horrible lame lifeless mediocre messy
    miserable muddy pointless poor poorly predictable pretentious ridiculous
    shallow silly slow sluggish stupid tedious terrible tired ugly
    unconvincing uneven uninspired uninteresting unnecessary unpleasant
    unoriginal unwatchable weak weird worse worst worthless wretched
    annoying awkward bland bloated confusing contrived disappointing
    disastrous embarrassing empty exhausting forced hollow incoherent
    incompetent irritating lackluster loud muddled offensive painful
    pathetic repetitive rotten soggy stale stilted strained tiresome
    trashy vapid wooden wrong
    """.split()
)

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def lexicon_flags(sentences: list[str]) -> dict[str, np.ndarray]:
    """Per-sentence lexicon presence / counts."""
    n = len(sentences)
    has_pos = np.zeros(n, dtype=bool)
    has_neg = np.zeros(n, dtype=bool)
    n_pos = np.zeros(n, dtype=np.int32)
    n_neg = np.zeros(n, dtype=np.int32)
    has_any = np.zeros(n, dtype=bool)
    for i, s in enumerate(sentences):
        toks = set(tokenize(s))
        p = len(toks & POS_WORDS)
        q = len(toks & NEG_WORDS)
        n_pos[i] = p
        n_neg[i] = q
        has_pos[i] = p > 0
        has_neg[i] = q > 0
        has_any[i] = p > 0 or q > 0
    return {
        "has_pos": has_pos,
        "has_neg": has_neg,
        "has_any": has_any,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "lex_score": n_pos.astype(np.float64) - n_neg.astype(np.float64),
    }


def active_feature_stats(activations: np.ndarray, feature_indices: np.ndarray) -> dict:
    """How many filtered features fire (activation > 0) per example."""
    sub = np.asarray(activations[:, feature_indices], dtype=np.float32)
    active = (sub > 0).sum(axis=1)
    # Co-activation among the full filtered set
    frac_ge2 = float((active >= 2).mean())
    frac_ge5 = float((active >= 5).mean())
    return {
        "n_features": int(len(feature_indices)),
        "n_examples": int(len(active)),
        "mean": float(active.mean()),
        "median": float(np.median(active)),
        "std": float(active.std()),
        "p10": float(np.percentile(active, 10)),
        "p90": float(np.percentile(active, 90)),
        "frac_ge2": frac_ge2,
        "frac_ge5": frac_ge5,
        "counts": active.astype(np.int32),
    }


def top_feature_coactivation(
    activations: np.ndarray,
    top_features: np.ndarray,
) -> dict:
    """Co-activation among top-|Φ| features only (stricter interaction proxy)."""
    sub = np.asarray(activations[:, top_features], dtype=np.float32)
    active = (sub > 0).sum(axis=1)
    # Pairwise Jaccard on binary activity
    binary = sub > 0
    k = binary.shape[1]
    jaccards = []
    for i in range(k):
        for j in range(i + 1, k):
            inter = np.logical_and(binary[:, i], binary[:, j]).sum()
            union = np.logical_or(binary[:, i], binary[:, j]).sum()
            if union > 0:
                jaccards.append(inter / union)
    return {
        "n_top": int(len(top_features)),
        "mean_active": float(active.mean()),
        "median_active": float(np.median(active)),
        "frac_exactly_1": float((active == 1).mean()),
        "frac_ge2": float((active >= 2).mean()),
        "mean_pairwise_jaccard": float(np.mean(jaccards)) if jaccards else 0.0,
    }


def feature_lexical_tracking(
    activations: np.ndarray,
    feature_ids: np.ndarray,
    flags: dict[str, np.ndarray],
    phi: np.ndarray,
) -> list[dict]:
    """
    For each feature: does activation track strong sentiment-word presence?

    Uses point-biserial r with has_any / has_pos / has_neg, and Spearman with
    continuous lex_score. Also reports mean activation when lexicon hits vs not.
    """
    rows = []
    has_any = flags["has_any"]
    has_pos = flags["has_pos"]
    has_neg = flags["has_neg"]
    lex_score = flags["lex_score"]

    for f in feature_ids:
        a = np.asarray(activations[:, int(f)], dtype=np.float64)
        row = {
            "feature": int(f),
            "phi": float(phi[int(f)]),
            "abs_phi": float(abs(phi[int(f)])),
            "freq": float((a > 0).mean()),
        }
        # Point-biserial needs both classes present
        for name, binary in (
            ("has_any", has_any),
            ("has_pos", has_pos),
            ("has_neg", has_neg),
        ):
            if binary.any() and (~binary).any():
                r, p = pointbiserialr(binary.astype(np.float64), a)
                row[f"r_{name}"] = float(r)
                row[f"p_{name}"] = float(p)
            else:
                row[f"r_{name}"] = float("nan")
                row[f"p_{name}"] = float("nan")

        rho, p_rho = spearmanr(a, lex_score)
        row["rho_lex_score"] = float(rho)
        row["p_lex_score"] = float(p_rho)
        row["mean_act_lex"] = float(a[has_any].mean()) if has_any.any() else 0.0
        row["mean_act_nolex"] = float(a[~has_any].mean()) if (~has_any).any() else 0.0
        # Among examples where this feature is active, what fraction have a lexicon hit?
        on = a > 0
        row["lex_rate_when_active"] = (
            float(has_any[on].mean()) if on.any() else float("nan")
        )
        row["lex_rate_when_inactive"] = (
            float(has_any[~on].mean()) if (~on).any() else float("nan")
        )
        rows.append(row)
    return rows


def lexicon_baseline(
    train_sentences: list[str],
    train_labels: np.ndarray,
    eval_sentences: list[str],
    eval_labels: np.ndarray,
    sae_probe,
    eval_activations: np.ndarray,
    batch_size: int = 512,
) -> dict:
    """
    Bag-of-lexicon-words logistic regression vs SAE probe on the same split.
    Also majority-lexicon vote (pos_count > neg_count → positive).
    """
    vocab = sorted(POS_WORDS | NEG_WORDS)
    word_to_i = {w: i for i, w in enumerate(vocab)}

    def bow(sentences: list[str]) -> np.ndarray:
        X = np.zeros((len(sentences), len(vocab)), dtype=np.float32)
        for i, s in enumerate(sentences):
            for t in tokenize(s):
                j = word_to_i.get(t)
                if j is not None:
                    X[i, j] += 1.0
        return X

    def probe_predict(acts: np.ndarray):
        preds, probas = [], []
        n = acts.shape[0]
        for start in range(0, n, batch_size):
            batch = np.asarray(acts[start : start + batch_size], dtype=np.float32)
            preds.append(sae_probe.predict(batch))
            probas.append(sae_probe.predict_proba(batch)[:, 1])
        return np.concatenate(preds), np.concatenate(probas)

    X_tr, X_ev = bow(train_sentences), bow(eval_sentences)
    lex_clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear")
    lex_clf.fit(X_tr, train_labels)
    lex_pred = lex_clf.predict(X_ev)
    lex_proba = lex_clf.predict_proba(X_ev)[:, 1]

    sae_pred, sae_proba = probe_predict(eval_activations)

    flags_ev = lexicon_flags(eval_sentences)
    maj = int(np.round(train_labels.mean()))
    vote = np.where(
        flags_ev["n_pos"] > flags_ev["n_neg"],
        1,
        np.where(flags_ev["n_neg"] > flags_ev["n_pos"], 0, maj),
    )
    has_signal = flags_ev["has_any"]

    return {
        "lexicon_vocab_size": len(vocab),
        "frac_eval_with_lexicon_word": float(has_signal.mean()),
        "lex_bow_accuracy": float(accuracy_score(eval_labels, lex_pred)),
        "lex_bow_auroc": float(roc_auc_score(eval_labels, lex_proba)),
        "lex_vote_accuracy": float(accuracy_score(eval_labels, vote)),
        "lex_vote_accuracy_when_signal": float(
            accuracy_score(eval_labels[has_signal], vote[has_signal])
        )
        if has_signal.any()
        else float("nan"),
        "sae_probe_accuracy": float(accuracy_score(eval_labels, sae_pred)),
        "sae_probe_auroc": float(roc_auc_score(eval_labels, sae_proba)),
        "agreement_lex_bow_vs_sae": float((lex_pred == sae_pred).mean()),
    }


def write_summary(
    path: Path,
    active: dict,
    coact: dict,
    tracking: list[dict],
    baseline: dict,
    n_top: int,
) -> None:
    # Heuristic reading of results
    sparse_top = coact["frac_exactly_1"] >= 0.5 or coact["mean_active"] < 1.5
    strong_lex = np.nanmean([abs(r["r_has_any"]) for r in tracking]) >= 0.25
    lex_near_probe = baseline["lex_bow_accuracy"] >= baseline["sae_probe_accuracy"] - 0.03

    lines = []
    lines.append("SST simplicity check")
    lines.append("=" * 60)
    lines.append("")
    lines.append("1) Active filtered features per example")
    lines.append(f"   n_filtered_features = {active['n_features']}")
    lines.append(
        f"   mean={active['mean']:.2f}  median={active['median']:.1f}  "
        f"std={active['std']:.2f}  p10={active['p10']:.1f}  p90={active['p90']:.1f}"
    )
    lines.append(
        f"   frac(≥2 active)={active['frac_ge2']:.3f}  "
        f"frac(≥5 active)={active['frac_ge5']:.3f}"
    )
    lines.append("")
    lines.append(f"2) Co-activation among top-{n_top} |Φ| features")
    lines.append(
        f"   mean active={coact['mean_active']:.2f}  "
        f"median={coact['median_active']:.1f}  "
        f"frac(exactly 1)={coact['frac_exactly_1']:.3f}  "
        f"frac(≥2)={coact['frac_ge2']:.3f}"
    )
    lines.append(
        f"   mean pairwise Jaccard={coact['mean_pairwise_jaccard']:.3f}"
    )
    lines.append("")
    lines.append(f"3) Lexical tracking (top-{n_top} |Φ| features)")
    lines.append(
        f"   {'feat':>6}  {'Φ':>9}  {'r_any':>7}  {'r_pos':>7}  {'r_neg':>7}  "
        f"{'ρ_lex':>7}  {'lex|on':>6}  {'lex|off':>7}"
    )
    for r in tracking:
        lines.append(
            f"   {r['feature']:6d}  {r['phi']:+9.5f}  "
            f"{r['r_has_any']:+7.3f}  {r['r_has_pos']:+7.3f}  {r['r_has_neg']:+7.3f}  "
            f"{r['rho_lex_score']:+7.3f}  "
            f"{r['lex_rate_when_active']:6.3f}  {r['lex_rate_when_inactive']:7.3f}"
        )
    mean_abs_r = float(np.nanmean([abs(r["r_has_any"]) for r in tracking]))
    lines.append(f"   mean |r(activation, has_lexicon_word)| = {mean_abs_r:.3f}")
    lines.append("")
    lines.append("4) Lexicon baseline vs SAE probe (SHAP held-out split)")
    lines.append(
        f"   frac examples with ≥1 lexicon word = "
        f"{baseline['frac_eval_with_lexicon_word']:.3f}"
    )
    lines.append(
        f"   lexicon BoW  acc={baseline['lex_bow_accuracy']:.3f}  "
        f"AUROC={baseline['lex_bow_auroc']:.3f}"
    )
    lines.append(
        f"   lexicon vote acc={baseline['lex_vote_accuracy']:.3f}  "
        f"(when signal: {baseline['lex_vote_accuracy_when_signal']:.3f})"
    )
    lines.append(
        f"   SAE probe    acc={baseline['sae_probe_accuracy']:.3f}  "
        f"AUROC={baseline['sae_probe_auroc']:.3f}"
    )
    lines.append(
        f"   agreement(lex BoW, SAE) = {baseline['agreement_lex_bow_vs_sae']:.3f}"
    )
    lines.append("")
    lines.append("5) Reading")
    if sparse_top:
        lines.append(
            "   Top-ranked features rarely co-activate → limited room for "
            "feature×feature interactions."
        )
    else:
        lines.append(
            "   Top-ranked features often co-activate → interaction structure "
            "is at least possible."
        )
    if strong_lex:
        lines.append(
            "   Top features correlate with strong sentiment words → likely "
            "lexical detectors more than compositional features."
        )
    else:
        lines.append(
            "   Top features are only weakly tied to the sentiment lexicon → "
            "not a pure lexical shortcut (on this lexicon)."
        )
    if lex_near_probe:
        lines.append(
            "   Lexicon BoW nearly matches the SAE probe → sentiment on SST "
            "may be too lexically simple for SHAP's interaction edge to matter."
        )
    else:
        gap = baseline["sae_probe_accuracy"] - baseline["lex_bow_accuracy"]
        lines.append(
            f"   SAE probe beats lexicon BoW by {gap:+.3f} accuracy → some "
            "non-lexical signal remains (necessary but not sufficient for "
            "SHAP to help)."
        )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    print(text)


def main(
    feature_indices_path: str = "outputs/3_shap/shap_feature_indices.npy",
    phi_path: str = "outputs/7_shap_recompute/phi_sentiment_layer7_signed.npy",
    activations_path: str = "activations/shap/layer_7/activations.npy",
    probe_path: str = "checkpoints/probe_layer_7.joblib",
    n_top: int = 20,
    out_dir: str = "outputs/10_simplicity_check",
):
    train_ds, _, shap_ds = load_splits()
    shap_sentences = list(shap_ds["sentence"])
    shap_labels = np.asarray(shap_ds["label"])
    train_sentences = list(train_ds["sentence"])
    train_labels = np.asarray(train_ds["label"])

    feature_indices = np.load(feature_indices_path)
    phi = np.load(phi_path)
    acts = np.load(activations_path, mmap_mode="r")
    probe = joblib.load(probe_path)
    probe.verbose = 0

    assert len(shap_sentences) == acts.shape[0], (
        f"Sentence/activation length mismatch: {len(shap_sentences)} vs {acts.shape[0]}"
    )

    # Rank by |signed Φ|, restrict reporting to filtered set when available
    ranked = np.argsort(np.abs(phi))[::-1]
    # Prefer top features that are in the filtered SHAP set
    filtered_set = set(int(x) for x in feature_indices)
    top_in_filtered = [int(f) for f in ranked if int(f) in filtered_set][:n_top]
    if len(top_in_filtered) < n_top:
        # Fall back to global top-|Φ| if filter set is empty / unused
        top_in_filtered = [int(f) for f in ranked[:n_top]]
    top_features = np.asarray(top_in_filtered, dtype=np.int64)

    flags = lexicon_flags(shap_sentences)
    active = active_feature_stats(acts, feature_indices)
    coact = top_feature_coactivation(acts, top_features)
    tracking = feature_lexical_tracking(acts, top_features, flags, phi)
    baseline = lexicon_baseline(
        train_sentences,
        train_labels,
        shap_sentences,
        shap_labels,
        probe,
        acts,
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "active_counts.npy", active["counts"])
    np.save(out / "top_features.npy", top_features)
    np.save(
        out / "lexical_tracking.npy",
        np.array(
            [
                (
                    r["feature"],
                    r["phi"],
                    r["r_has_any"],
                    r["r_has_pos"],
                    r["r_has_neg"],
                    r["rho_lex_score"],
                    r["lex_rate_when_active"],
                    r["lex_rate_when_inactive"],
                )
                for r in tracking
            ],
            dtype=np.float64,
        ),
    )
    write_summary(out / "simplicity_summary.txt", active, coact, tracking, baseline, n_top)


if __name__ == "__main__":
    main()
