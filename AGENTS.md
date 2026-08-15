# Agent Guide

This project uses SHAP to interpret Sparse Autoencoder (SAE) features in GPT-2 Small for sentiment analysis. Read this before writing or modifying any code.

---

## Project Structure

```
shap-sae/
├── data/                   # SST-2 dataset + three_way_split_indices.json
├── activations/            # saved SAE activation tensors (not in git)
├── checkpoints/            # probe weights
├── outputs/                # SHAP matrices, results, figures
├── notebooks/              # exploration and figure generation only
└── src/
    ├── utils.py                    # shared helpers: splits, sampling, paths, feature mask
    ├── core/
    │   ├── 1_extract.py            # activation extraction pipeline
    │   ├── 2_probes.py             # L1 logistic probe
    │   └── 3_shap_comp.py          # SHAP feature mask (filtered support)
    ├── global_linear/              # the logistic-probe arm
    │   ├── 6_shap_stability.py     # DEPRECATED — KernelSHAP seed stability (see script docstring)
    │   ├── 7_shap_recompute.py     # exact LinearSHAP, signed Phi
    │   ├── 8_feature_ranking.py    # IG / GA / probe / SHAP rank agreement
    │   ├── 9.5_tuning.py           # steering alpha calibration
    │   ├── 9_steer.py              # steering / ablation faithfulness
    │   └── 10_simplicity_check.py  # is SST-2 just a lexicon?
    ├── global_mlp/                 # the MLP-probe arm (11-14, mirrors the above)
    ├── local/                      # per-example faithfulness (9, 14, and the
    │                               #   local_shap_faithfulness diagnostic)
    └── global_group/               # multi-feature steering vectors + plots
```

Scripts are run from the repo root and resolve data paths relative to it:

```bash
uv run python src/core/1_extract.py
```

**Artifact roots are configurable.** `outputs/` and `checkpoints/` are never
hardcoded — every script goes through `utils.output_path()` /
`utils.checkpoint_path()`, which read `SHAP_SAE_OUTPUTS` and
`SHAP_SAE_CHECKPOINTS` (defaults `outputs`, `checkpoints`). To run a variant
without clobbering a previous run:

```bash
SHAP_SAE_OUTPUTS=runs/exp3 uv run python src/global_mlp/14_mlp_steer.py
```

Do not reintroduce a literal `"outputs/..."` or `"checkpoints/..."` string; the
env override is resolved at import, so module-level `OUT_DIR = output_path(...)`
constants pick it up.

---

## Stack

- **Model:** GPT-2 Small via `transformer_lens`
- **SAE:** `gpt2-small-resid-post-v5-32k` from `sae_lens`, `resid_post` hook points, 32k features
- **Dataset:** SST-2 binary sentiment (HuggingFace `stanfordnlp/sst2`)
- **Probing:** scikit-learn `LogisticRegression`, L1 penalty, `saga` solver
- **SHAP:** `shap` library, KernelSHAP primary, DeepSHAP for validation

---

## Environment

This project uses `uv` for package management, not `pip`.

```bash
# install uv if not present
curl -Lsf https://astral.sh/uv/install.sh | sh

# install dependencies
uv sync

# run a script (always from the repo root)
uv run python src/core/1_extract.py
```

Do not use `pip install` directly. If a new dependency is needed, add it with:
```bash
uv add <package>
```

---

## Key Conventions

**Activations are never stored in git.** They are large (5-8 GB) and regenerable. The `activations/` directory is in `.gitignore`. To regenerate:
```bash
uv run python src/core/1_extract.py
```

**Token position:** Use the last token position for sentence-level activation
extraction. Do not change this without explicit instruction.

Get that index from `utils.last_real_token_index` — never hand-roll
`(tokens != pad_id).sum(dim=1) - 1`. GPT-2 has no dedicated pad token, so
`transformer_lens` sets pad = bos = eos = `<|endoftext|>`; that expression counts
the prepended BOS as padding and silently lands one position early (on the
*second*-to-last real token, or on BOS itself for a one-token sentence). The
helper delegates to `transformer_lens`'s own `get_attention_mask`, which resolves
the ambiguity. This affects extraction and every readout, so all of them must use
the one helper or they will disagree about which token they describe.

**SAE hook points:** Always use `resid_post` not `resid_pre` or `resid_mid`. The
pipeline currently runs on **layer 7** only — `TARGET_LAYERS` in `1_extract.py`
and `LAYERS` in the probe scripts.

**Splits:** SST-2 is divided into three fixed splits with a set random seed:
- Probe train (70%, 47,144)
- Probe val (15%, 10,102)
- Held-out (15%, 10,103)

Each split has exactly one job. Keep to this — mixing them silently confounds
every attribution comparison in the repo:

| Split | Used for | Never used for |
|---|---|---|
| train | fitting probes, SHAP background, IG/GA baseline point, steering α tuning | reporting any result |
| val | probe hyperparameters, the activation-frequency feature mask, reported probe accuracy | attribution, steering, ablation |
| held-out | **all** attribution (SHAP/IG/GA) **and all** causal evaluation (ablation/steering) | fitting or tuning anything |

Attribution and causal evaluation deliberately share the *same rows*: nothing is
fit on them, and "did steering example i move the logit the way SHAP said it
would for example i" is only meaningful on the same examples.

**Row sampling:** never draw eval or background rows ad hoc. Call
`utils.sample_eval_rows` / `utils.load_eval_and_background`, which are the single
definition of which rows get explained and steered. They return permutation
prefixes, so a 100-row local sample is an ordered prefix of the 2000-row global
sample and results join by position. Pair sentences to activation rows with
`utils.eval_sentences`, never by indexing a split dataset directly.

**Feature mask:** `utils.get_shap_feature_mask` is the only definition
of the filtered feature set, and its frequency filter runs on **val**. It gates
the MLP probe's input layer and the steering candidate pool, so deriving it from
held-out data would let the eval split shape the models it later scores. A 500-row
estimate of that 5% threshold is badly unstable (mask size ranges 68–81 across
samples); on the full val split with the committed probe it is **75**.

**Probe regularization:** L1 with `solver="liblinear"`, `C=1`, currently untuned.
Select L1 via `l1_ratio=1`, **not** `penalty="l1"` — scikit-learn deprecated
`penalty` in 1.8 and warns about inconsistent values if both are given. The
sparsity is load-bearing: the mask is `(coef != 0) & frequent_enough`, so an L2
fit (~12k non-zero vs ~1.4k) quietly turns the sparsity filter into a no-op.

**Always pass `random_state=CANONICAL_SEED` to `LogisticRegression`.** sklearn
hands it to liblinear to seed the coordinate-descent data shuffling; at `None`
the seed comes from the global NumPy RNG and the fit drifts between runs. For the
L1 probe that drift moves the exactly-zero coefficients, which re-rolls the
75-feature mask that gates the MLP probe's inputs, the steering candidate pool
and the group-steering ranking pool. This applies to all three fits in the repo:
`core/2_probes`, `9_steer.train_residual_probe`, `10_simplicity_check`'s lexicon.

**Every method explains the same target within an arm.** This is the one rule
that makes the IG-vs-SHAP comparison mean anything:

| Arm | Explainer | Target function | IG / GA gradient |
|---|---|---|---|
| `global_linear` (7, 8) | LinearSHAP (exact) | log-odds margin `w·x + b` | ∇(margin) = `coef_` |
| `global_mlp` (12, 13) | DeepSHAP | pre-sigmoid logit | ∇(logit), `mlp_logit_grad` |

**Never reintroduce a `σ(1−σ)` factor into IG/GA.** It makes them attribute
`σ(logit)` while SHAP attributes the logit. Per example that factor is a positive
scalar and changes no ranking — but every method here averages over examples
first, and `mean_e[c_e · v_e]` is not a monotone function of `mean_e[v_e]`, so
the averaged scores are genuinely different quantities. The old near-zero
IG-vs-SHAP ρ was substantially this artifact.

Consequence worth knowing before reading `8_rankings_recompute/correlation.txt`:
with the target matched, **IG, GA and interventional SHAP are analytically
identical for a linear probe** — all three are `w_j · (x_j − μ_j)`. The linear
arm's rank agreement is therefore an identity (ρ = 1.000), not evidence;
`8_feature_ranking.linearity_identity_report` asserts it and writes the residual
into `correlation.txt`. The substantive method comparison lives in the **MLP
arm**, where the ReLU keeps IG genuinely distinct from SHAP.

Magnitudes are still not interchangeable *across* arms (log-odds vs logit).
`core/3_shap_comp` no longer computes an attribution — it only saves the
filtered feature mask (`shap_feature_indices.npy`) that other scripts load.
It used to also run KernelSHAP; that had no downstream readers and was removed.
`6_shap_stability`, which measured that KernelSHAP path's seed stability, is
deprecated as a result (see its module docstring) — it never certified the Φ
that 8/9/10/group_steering actually consume, and now has nothing left to
measure. `7_shap_recompute` deliberately explains all 32768 features rather
than the mask (exact and cheap, so no reason to filter); consumers needing the
filtered pool slice it themselves.

**Steering vs ablation sign convention:** steering adds `+α`, so a positive
attribution predicts a positive Δ. Ablation *removes* the feature, so a positive
attribution predicts a *negative* Δ — a perfectly faithful method scores ρ ≈ −1 on
raw numbers. Every directional report passes Δ through `orient_signed_effects`
first, so **positive always means faithful**. Do not compare a directional ρ
across modes without checking it was oriented.

**Steering α is per-feature.** `utils.resolve_steering_alphas` is the single
definition, shared by all four steer scripts. `--alpha-mode scaled` (the default)
sets `α_i = α · scale_i / median(scale)` from the train activation scales
`9.5_tuning` writes to `pool_scales.npy` / `pool_indices.npy`; `constant` adds a
flat α everywhere. Constant is what the original runs used and it confounds the
`|attribution| vs |Δ|` correlation: features differ severalfold in natural
activation scale, so a flat α pushes some far harder relative to themselves than
others, and |Δ| partly ranks that instead of the attribution. No decoder-norm
correction is applied and none is needed — this SAE's `W_dec` rows are ~unit norm
(median 0.9996). Report `scaled` as primary and `constant` as a robustness check.

**Candidate selection biases the faithfulness table.** `--selection top` picks
the k candidates by |SHAP| and then scores all four methods on them, which
range-restricts SHAP alone and deflates its ρ relative to methods that did not
drive selection. `--selection random` is the default and the headline; run `top`
only as a labelled secondary. (`9.5_tuning` uses the union of every method's
top-k, which is the other unbiased option.)

**Magnitude-only scores never become steering vectors.** `group_steering` builds
`v = Σ score_i · W_dec[i]`, using the score as a *signed* weight, so the MLP arm's
`probe` score (‖W1[i,:]‖₂, non-negative) is **excluded** from that script — it
would add positive- and negative-sentiment decoder directions with the same sign.
The logistic arm keeps its `probe` vector because `coef_[0]` is signed. The
`actdiff` baseline uses no probe, so it is scored once under the internal
`SHARED_ARM` and fanned out to both arms with `arm_independent: true`.

**Magnitude-only scores:** the MLP arm's `probe_saliency` is ‖W1[i,:]‖₂ and has no
sign. It appears in importance tables only; correlating it against a signed Δ, or
against signed IG/GA/SHAP, compares a magnitude with a direction.

---


## What Not to Do

- Do not modify `data/sst2_train` or `data/three_way_split_indices.json`, and do not
  re-download the dataset with a different seed. `1_extract.create_splits()` would
  do exactly that; it is guarded behind `force=True` and is not part of any run
- Do not run SHAP on the probe val split — only on the held-out split
- Do not fit, tune, or select anything on the held-out split (steering α included)
- Do not load splits from the Hub — use `utils.load_splits`, which reads from disk;
  activations were extracted in that row order and a Hub reorder would silently
  decouple sentences from their activations
- Do not compare attribution methods scored on different rows or with different
  baselines — IG/GA must use the same eval rows and the same background mean as SHAP.
  Each output dir carries an `eval_indices.npy` for exactly this check, and every
  script that combines two artifact directories asserts they match:
  `8_feature_ranking`, `9_steer`, `13_mlp_ranking` and `group_steering`. Keep that
  guard when adding a consumer. Mind the two filename spellings — `7_shap_recompute`
  and `12_mlp_shap` write `eval_indices_layer7.npy`, `8` and `13` write
  `eval_indices.npy`
- Do not assume same-named `.npy` files share an index space. `ig_scores.npy` /
  `ga_scores.npy` are full-width (32768) under `8_rankings_recompute/` and
  filtered-width (75) under `13_mlp_ranking/`. Consumers assert the width they
  expect; keep those asserts. `shap_feature_indices.npy` is always the 75-entry
  mask — the full-width column list `7_shap_recompute` explains is deliberately
  named `explained_feature_indices.npy` so the two cannot be confused
- Do not rank-correlate attribution methods over the full 32768 features. The L1
  probe zeroes ~77% of them and IG/GA/Φ are *identically* zero there, so Spearman
  reports agreement about exclusion, not about importance (it inflated Probe-vs-GA
  to 0.997, and made an IG-vs-SHAP ρ of 0.018 look significant). Restrict to the
  non-zero support — `8_feature_ranking.scored_support`
- Do not read "local SHAP beats global Φ" from the local scripts without the
  activity control. Globally-top features are usually inactive in any one sentence
  and ablating an inactive feature gives Δ = 0 exactly, while local |φ(x)| tracks
  activity by construction. Block [B] of those summaries rescores on active
  candidates only; a local win has to survive there to be about ranking
- Do not store activations in float64 — float32 is what `1_extract.py` writes and
  what every loader expects
- Do not add new dependencies without checking with the user first
- Do not write exploration code into `src/` — that belongs in `notebooks/`
- Do not use `pip` — use `uv`

---

## Known caveats when reading results

Real limits of the current design. State them alongside any number you quote;
they are not bugs to fix silently.

- **The committed `outputs/` predate the current scripts.** No directory contains
  the `eval_indices.npy` that 7/8/12/13 now write, and the saved
  `shap_values_raw.npy` files are 500 rows while the IG/GA arrays beside them are
  2000. Treat the checked-in numbers as illustrative until the chain is re-run.
  (`3_shap_comp.py` no longer writes an `eval_indices.npy` at all — it only
  saves the feature mask — and `6_shap_stability` is deprecated.)
- **α is calibrated against a different readout than it is used with.**
  `9.5_tuning` picks α from a layer-11 residual probe's P(positive); the steer
  scripts read out a GPT-2 logit difference. α is a rough "measurable but
  unsaturated" magnitude, not a tuned hyperparameter.
- **Attribution is last-token; intervention is all-token.** Attributions describe
  the SAE vector at one position, but `_apply_feature_intervention` edits the
  feature at every position. Some of any faithfulness gap is this mismatch.
- **The faithfulness ρ are small-n.** Default `--k 20` means 20 features across
  4 methods × 2 correlation types, with no multiple-comparison control. Individual
  p-values are descriptive. The rendered table is saved as
  `faithfulness_{mode}_{selection}_k{k}.txt` beside the deltas JSON — it used to
  be printed to stdout only, so a finished run left no record of its own result.
- **Steer outputs are config-tagged.** `model_deltas_`, `faithfulness_`,
  `shap_top_local_`, `shap_top_global_` and `eval_indices_` all carry a
  `{mode}_{selection}_k{k}` suffix. Before that, a second run into the same
  directory replaced the candidate set the first run's JSON referred to.
- **Group steering is on a different scale from per-feature steering.**
  `group_steering` never touches the SAE at inference, so its baseline is a clean
  forward, while `9_steer` / `14_mlp_steer` baseline through an encode→decode
  reconstruction. Both are internally consistent; compare the *shapes* of the
  effect-vs-cost curves across the two families, never the effect sizes.
- **Match the arms before comparing them.** `--mode`, `--selection`, `--k`,
  `--alpha`, `--alpha-mode` and `--n-eval` are CLI flags on the steer scripts
  precisely so the linear and MLP arms can be run identically; they used to be
  module constants that had drifted apart. The **three local scripts** now carry
  the same surface (`--mode --k-local --k-global --n-examples --alpha
  --alpha-mode --out-dir`) with matched defaults `k_local = k_global = 20`,
  `n_examples = 100`; they had drifted to k=10 (linear) vs k=20 (MLP) vs k=10,
  n=80 (`local_shap_faithfulness`), which changes the union size and the
  per-example Spearman n, so the local arms were not comparable at all. Pass the
  same flags to both arms or the comparison is void.
- **The last-token fix invalidates cached artifacts.** Anything under
  `activations/` produced before it describes the second-to-last token. Re-run
  `1_extract.py` and the full chain before quoting new numbers.