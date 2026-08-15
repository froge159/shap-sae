# CLAUDE.md — pipeline runbook

Operational guide for running this pipeline end to end. **Read `AGENTS.md` too** —
it holds the conventions and the *why* behind the rules below, and it is the file
to update when a convention changes. This file is the *how to run it*.

---

## Current state (2026-08-15)

- **The audit fixes are applied** across 16 files (target mismatch, seeded probe
  fits, per-feature α, config-tagged outputs, index guards, matched CLIs). They
  are committed to the working tree but **not yet committed to git** unless a
  later session did so — check `git status` before assuming.
- **`run1/` is stale.** It predates both the current scripts and the fixes: no
  directory has `eval_indices*.npy`, `12_mlp_shap/shap_values_raw.npy` is 500 rows
  where the arrays beside it are 2000, and its IG/GA were computed against the old
  σ(logit) target. **Do not quote any number from `run1/`.** It is kept only for
  before/after comparison.
- **The clean re-run has not been done.** `activations/` does not exist. Nothing
  in `run2/` exists yet. Every number in the repo is currently un-quotable.

So: the pipeline below has been fixed and statically verified, but never executed
in full. Expect to be the first to run it.

---

## Before you run anything

### Environment

```bash
cd /home/ubuntu/shap-sae
uv sync
```

`uv`, never `pip`. Add deps with `uv add`, and ask the user first.

### Artifact roots — set these in every shell

```bash
export SHAP_SAE_OUTPUTS=run2/outputs
export SHAP_SAE_CHECKPOINTS=run2/checkpoints
```

Every script resolves artifacts through `utils.output_path()` /
`utils.checkpoint_path()`, which read these at import. **If you set them for one
step and forget on the next, the chain silently reads one run and writes
another.** Defaults are bare `outputs/` and `checkpoints/`.

`activations/` is **not** configurable — it is hardcoded in `1_extract.py`,
`2_probes.py`, `11_mlp_probe.py`, `9.5_tuning.py`, `10_simplicity_check.py` and
`steering_config.json`. That is fine: extraction is deterministic given the
committed splits, so one `activations/` tree is shared across runs.

### Resources

| | |
|---|---|
| GPU | needed for step 1 and steps 9-20 (anything that loads GPT-2 + the SAE); an A10 (23 GB) is enough. Steps 2-8 are CPU-only sklearn/numpy/shap |
| RAM | **~18 GB peak at step 2** — the L1 fit is on a dense 47144 × 32768 matrix (5.8 GiB float32) and liblinear makes a float64 copy (11.5 GiB). Steps 1 and 18 also hold multi-GB arrays |
| Disk | ~8.8 GB for `activations/`, ~600 MB for `run2/outputs` (mostly `7_shap_recompute/shap_values_raw.npy` at 524 MB) |
| Wall clock | ~7 h serial for the full chain, dominated by steps 9-17 |
| Network | first run pulls GPT-2 and the SAE from HuggingFace; no revisions are pinned |

Steps are serial — each depends on the previous. Do not try to parallelise across
GPUs without re-checking the dependency graph below.

### Never run these

- **`1_extract.create_splits()`** — guarded behind `force=True`. It would
  overwrite `data/three_way_split_indices.json` and `data/sst2_train`,
  invalidating every extracted activation. The split is committed; step 0 is
  "already done".
- **`src/global_linear/6_shap_stability.py`** — deprecated, raises on entry. It
  measured KernelSHAP's seed variance; KernelSHAP is no longer in the pipeline.

---

## Invariants you can break by accident

Full reasoning in `AGENTS.md`; these are the ones that bite during a run.

1. **Splits.** train = fit/background/α-tuning · val = hyperparameters + the
   feature mask + reported probe accuracy · held-out ("shap") = **all**
   attribution and **all** causal evaluation. Never fit or tune on held-out.
2. **One target per arm.** IG/GA must attribute the same function SHAP does —
   the **margin** in the linear arm, the **pre-sigmoid logit** in the MLP arm.
   Never reintroduce a `σ(1−σ)` factor. Step 5's identity check exists to catch
   exactly this.
3. **Match the arms.** `9_steer` and `14_mlp_steer` take the same flags on
   purpose; so do the three local scripts. Different `--k` or `--mode` between
   arms makes the comparison meaningless.
4. **Row sampling** goes through `utils.sample_eval_rows` /
   `load_eval_and_background` only. Never draw eval or background rows ad hoc.
5. **Last token** comes from `utils.last_real_token_index` only.
6. **Seeds.** `LogisticRegression` always gets `random_state=CANONICAL_SEED`;
   without it the L1 fit drifts and re-rolls the 75-feature mask.

---

## Dependency graph

```
1_extract ──> 2_probes ──> 3_shap_comp ──┬─> 11_mlp_probe ─> 12_mlp_shap ─> 13_mlp_ranking ─┐
    │             │                      │                                                  │
    │             └─> 7_shap_recompute ─> 8_feature_ranking ─┐                               │
    │                        │                    │          │                              │
    └────────────────────────┴────────────────────┴──> 9.5_tuning (α)                        │
                                                          │                                  │
                    ┌─────────────────────────────────────┼──────────────────────────────────┤
                    v                                     v                                  v
              9_steer, 9_local_steer            group_steering                14_mlp_steer, 14_local_mlp_steer,
              (linear arm, needs 3+7+8)         (needs 3+7+8+13)              local_shap_faithfulness (MLP arm)
                                                          │
                                                          v
                                                 plot_group_steering
```

`10_simplicity_check` hangs off `3_shap` + `7_shap_recompute` + the probe and can
run any time after step 4.

---

## The run

`<A>` below is the α from step 9 — **read it out of
`$SHAP_SAE_OUTPUTS/9.5_tuning/alpha_calibration.txt` (`recommended_alpha`)**. The
CLI default is `0.6723`, which is what `run1` recommended, but the fixes changed
the probe fit, so do not assume it carries over. Substitute the real value.

### Steps 1-9 — probes and attributions

| # | Command | Produces | Check | ~time |
|---|---|---|---|---|
| 1 | `uv run python src/core/1_extract.py` | `activations/{probe_train,probe_val,shap}/layer_7/{activations,labels}.npy` | shapes `(47144, 32768)`, `(10102, 32768)`, `(10103, 32768)`, all **float32**, ~8.8 GB | 20–40 m |
| 2 | `uv run python src/core/2_probes.py` | `$CKPT/probe_layer_7.joblib` | prints val accuracy + non-zero weights (~1.4k of 32768) | 10–30 m |
| 3 | `uv run python src/core/3_shap_comp.py` | `3_shap/shap_feature_indices.npy` | **must be `(75,)`.** A different size means the mask moved — re-sanity-check every downstream `k`, and note `ks` in `steering_config.json` tops out at 65 | <1 m |
| 4 | `uv run python src/global_linear/7_shap_recompute.py` | `7_shap_recompute/`: `phi_sentiment_layer7_signed.npy` `(32768,)`, `sign_consistency_layer7.npy`, `top_k_{idx,Phi}_layer7.npy` `(50,)`, `explained_feature_indices.npy` `(32768,)`, `shap_values_raw.npy` `(2000, 32768)` **524 MB**, `eval_indices_layer7.npy` `(2000,)`, `background_indices_layer7.npy` `(100,)` | `explained_feature_indices.npy` is `arange(32768)`, **not** a mask | 5–10 m |
| 5 | `uv run python src/global_linear/8_feature_ranking.py` | `8_rankings_recompute/`: `ig_scores.npy` + `ga_scores.npy` `(32768,)`, `support.npy`, `eval_indices.npy`, `background_indices.npy`, `correlation.txt` | **`correlation.txt` must open with `PASS`** from the identity check, and IG/GA/SHAP pairs must be ρ = 1.000. That is correct, not a bug — see below. Raises if step 4's rows disagree | ~2 m |
| 6 | `uv run python src/global_mlp/11_mlp_probe.py` | `$CKPT/mlp_probe_layer_7.joblib`; `11_mlp_probe/{feature_indices_layer_7.npy, results_layer_7.json, top_features_layer_7.json}` | `n_features` = 75; val accuracy is mildly optimistic (its inputs were selected using a val-derived frequency filter — label-free, so small) | 5–15 m |
| 7 | `uv run python src/global_mlp/12_mlp_shap.py` | `12_mlp_shap/`: `phi_sentiment_layer7_signed.npy` `(32768,)`, `shap_feature_indices.npy` `(75,)`, `shap_values_raw.npy` `(2000, 75)`, `sign_consistency_layer7.npy`, `top_k_*`, `eval_indices_layer7.npy`, `background_indices_layer7.npy` | DeepSHAP runs with `check_additivity=True`; a failure there is real. Cost is 2000 eval × 100 background passes, so this is slower than its tiny probe suggests | 5–20 m |
| 8 | `uv run python src/global_mlp/13_mlp_ranking.py` | `13_mlp_ranking/`: `{ig,ga,probe,shap}_scores.npy` `(75,)`, `feature_indices.npy`, `eval_indices.npy`, `background_indices.npy`, `correlation.txt` | raises if step 7's eval rows disagree. **This is where the real IG-vs-SHAP comparison lives** | 5–10 m |
| 9 | `uv run python src/global_linear/9.5_tuning.py` | `$CKPT/residual_probe_layer_11.joblib`; `9.5_tuning/{alpha_calibration.txt, alpha_grid.npy, alpha_pilot_mean_abs_dp.npy, candidate_scales.npy, candidate_indices.npy, pilot_row_indices.npy, pool_scales.npy (75,), pool_indices.npy (75,)}` | **grab `recommended_alpha` → `<A>`.** `pool_*` are what `--alpha-mode scaled` consumes; without them steps 10+ fail loudly rather than silently | 30–60 m |

> **Step 5's ρ = 1.000 is the expected result.** With the target matched, IG, GA
> and interventional SHAP are analytically identical for a linear probe — all
> three are `w_j·(x_j − μ_j)`. The linear arm is a closed-form sanity check, not
> evidence about methods. `Probe vs SHAP` stays a genuine comparison
> (`Φ_j = w_j·Δ̄_j` against `w_j`). If the identity check **FAILs**, either a
> `σ(1−σ)` factor came back or Φ and IG/GA were scored on different rows — stop
> and fix it before running anything downstream.

### Steps 10-14 — per-feature causal runs

Flags must be **identical between the two arms**. Defaults are already the
headline configuration (`--mode steering --selection random --k 20 --n-eval 1000
--alpha-mode scaled`), but pass them explicitly so the command records the config.

Each run writes files tagged `{mode}_{selection}_k{k}`, so several configs can
share a directory — but keep them separate anyway for legibility.

```bash
A=<recommended_alpha from step 9>

# 10 — headline, linear arm
uv run python src/global_linear/9_steer.py \
  --mode steering --selection random --k 20 --n-eval 1000 \
  --alpha $A --alpha-mode scaled \
  --out-dir $SHAP_SAE_OUTPUTS/9_steer/steering_random_k20_scaled

# 11 — headline, MLP arm (same flags)
uv run python src/global_mlp/14_mlp_steer.py \
  --mode steering --selection random --k 20 --n-eval 1000 \
  --alpha $A --alpha-mode scaled \
  --out-dir $SHAP_SAE_OUTPUTS/14_mlp_steer/steering_random_k20_scaled

# 12 — ablation, both arms
uv run python src/global_linear/9_steer.py --mode ablation --selection random \
  --k 20 --n-eval 1000 --alpha $A --alpha-mode scaled \
  --out-dir $SHAP_SAE_OUTPUTS/9_steer/ablation_random_k20_scaled
uv run python src/global_mlp/14_mlp_steer.py --mode ablation --selection random \
  --k 20 --n-eval 1000 --alpha $A --alpha-mode scaled \
  --out-dir $SHAP_SAE_OUTPUTS/14_mlp_steer/ablation_random_k20_scaled

# 13 — SHAP-selected variant (secondary; label it with the bias caveat)
uv run python src/global_linear/9_steer.py --mode steering --selection top \
  --k 20 --n-eval 1000 --alpha $A --alpha-mode scaled \
  --out-dir $SHAP_SAE_OUTPUTS/9_steer/steering_top_k20_scaled
uv run python src/global_mlp/14_mlp_steer.py --mode steering --selection top \
  --k 20 --n-eval 1000 --alpha $A --alpha-mode scaled \
  --out-dir $SHAP_SAE_OUTPUTS/14_mlp_steer/steering_top_k20_scaled

# 14 — constant-α robustness check (this is the parametrisation run1 used,
#      so it is the only run directly comparable to the committed numbers)
uv run python src/global_linear/9_steer.py --mode steering --selection random \
  --k 20 --n-eval 1000 --alpha $A --alpha-mode constant \
  --out-dir $SHAP_SAE_OUTPUTS/9_steer/steering_random_k20_constant
uv run python src/global_mlp/14_mlp_steer.py --mode steering --selection random \
  --k 20 --n-eval 1000 --alpha $A --alpha-mode constant \
  --out-dir $SHAP_SAE_OUTPUTS/14_mlp_steer/steering_random_k20_constant
```

Each writes, per tag: `model_deltas_<tag>.json` (deltas + the resolved
per-feature α), `faithfulness_<tag>.txt` (**the headline table** — it is saved,
not stdout-only), `shap_top_{local,global}_<tag>.npy`, `eval_indices_<tag>.npy`.
~20–40 min per invocation.

**Why these defaults.** `--selection top` picks candidates by |SHAP| and then
scores all four methods on them, which range-restricts SHAP alone and deflates
its ρ — hence `random` for the headline. `--alpha-mode constant` adds a flat α
regardless of a feature's natural activation scale, so |Δ| partly ranks how hard
each feature got pushed relative to itself — hence `scaled`, which uses the train
activation scales from step 9. No decoder-norm correction is applied or needed
(this SAE's `W_dec` rows are ~unit norm, median 0.9996).

### Steps 15-17 — local (per-example) runs

Defaults already match across all three (`k_local = k_global = 20`,
`n_examples = 100`, `--mode ablation`), so bare invocations are aligned.

```bash
# 15
uv run python src/local/9_local_steer.py \
  --mode ablation --k-local 20 --k-global 20 --n-examples 100 --alpha $A
# 16 — must use the same flags as 15
uv run python src/local/14_local_mlp_steer.py \
  --mode ablation --k-local 20 --k-global 20 --n-examples 100 --alpha $A
# 17 — diagnostic
uv run python src/local/local_shap_faithfulness.py \
  --target lm --n-examples 100 --k-local 20 --k-global 20
```

15 and 16 write `summary.txt` / `summary.json` with **blocks [A] and [B]**, plus
`local_shap.npy` `(100, 75)`, `delta_{signed,abs}_ablation.npy` `(100, n_union)`,
`{,active_}rho_{importance,directional}_*.npy`, and ragged
`cands_per_example.npy`. 17 writes `summary.{txt,json}`, `per_example.json`,
`rho_{local,global}.npy`, `local_shap_values.npy`, `global_phi_filtered.npy`.
~20–40 min each.

**Quote block [B], not block [A].** Globally-top features are usually inactive in
any one sentence, and ablating an inactive feature gives Δ = 0 *exactly*, while
local |φ(x)| tracks activity by construction. Block [A] is dominated by that tie
block, so a local win there is partly just "local knows which features fire
here". Block [B] rescores on active candidates only. Script 17 has **no** activity
control at all — treat it as an upper bound and read it against 16's block [B].

### Steps 18-20 — simplicity check, group steering, figures

```bash
# 18
uv run python src/global_linear/10_simplicity_check.py
# 19
uv run python src/global_group/group_steering.py
# 20
uv run python src/global_group/plot_group_steering.py --all --k 20
```

- **18** → `10_simplicity_check/{simplicity_summary.txt, active_counts.npy (10103,),
  top_features.npy (20,), lexical_tracking.npy (20, 8)}`. ~10–20 m. Runs on the
  **full 10,103-row held-out split**, not the 2000-row canonical sample — the
  summary says so in its header. Do not quote its numbers alongside the
  attribution results as if they described the same rows.
- **19** → `group_steering/group_steering_results.json`. ~50–100 m. Expect
  **378 rows from 336 forward passes**: 7 arm-specific vectors (4 logistic + 3
  mlp) × 6 ks × 7 alphas = 294, plus `actdiff` at 6 × 7 = 42 scored **once** and
  emitted under both arms (84 rows, tagged `arm_independent: true`). Config comes
  from `steering_config.json`.
- **20** → `group_steering/effect_vs_cost_{logistic,mlp}_grid.png`,
  `effect_vs_cost_logistic_k20.png`, `effect_vs_cost_k20.png`. **These are the
  only figures the repo produces.** <1 m.

**The MLP panel legitimately has no `probe` curve.** The MLP's probe score is
`‖W1[i,:]‖₂`, which is unsigned, and `build_steering_vector` uses the score as a
*signed* weight — including it would add positive- and negative-sentiment decoder
directions with the same sign. The panel annotates "excluded: Probe". The logistic
arm keeps its `probe` curve because `coef_[0]` is signed.

**Group-steering effect sizes are not comparable to per-feature steering.** This
script never touches the SAE at inference, so its baseline is a clean forward,
while `9_steer` / `14_mlp_steer` baseline through an encode→decode
reconstruction. Compare curve *shapes* across the two families, never magnitudes.

---

## Verification after the run

```
1. Split discipline  — every eval_indices*.npy under $SHAP_SAE_OUTPUTS is a prefix of
                       utils.sample_eval_rows(n_eval=2000)[0]; every background_indices*.npy
                       equals its bg_idx.
2. Rows agree        — 7_shap_recompute/eval_indices_layer7.npy == 8_rankings_recompute/eval_indices.npy
                       12_mlp_shap/eval_indices_layer7.npy    == 13_mlp_ranking/eval_indices.npy
                       (steps 5/8/10 raise on failure; this is belt-and-braces)
3. Index spaces      — 3_shap/ and 12_mlp_shap/shap_feature_indices.npy both (75,) and equal;
                       7_shap_recompute/explained_feature_indices.npy == arange(32768);
                       no shap_feature_indices.npy of shape (32768,) anywhere.
4. M1 identity       — 8_rankings_recompute/correlation.txt says PASS, rho=1.000 for IG/GA/SHAP.
5. M1 in the MLP arm — 13_mlp_ranking/correlation.txt IG-vs-SHAP should move off the old
                       near-zero value. A rho still ~0 means the targets are still mismatched.
6. Group steering    — no {"probe_type":"mlp","method":"probe"} rows; actdiff rows carry
                       arm_independent:true and the two arms' copies are numerically identical.
7. Steer bookkeeping — each out-dir has one model_deltas_<tag>.json + matching
                       faithfulness_<tag>.txt; shap_top_global_<tag>.npy reproducible from
                       that JSON's delta_abs keys.
8. Determinism       — re-run step 2 into a scratch checkpoint root; probe_layer_7.joblib
                       coef_ must be bit-identical and the step-3 mask still (75,).
9. Alpha sensitivity — compare faithfulness rho from the scaled run vs the constant run.
                       A large gap means run1's constant-alpha numbers were substantially
                       reporting activation scale. Report both regardless.
10. Causal direction — features with positive Phi should on average give positive Delta under
                       steering. run1's alpha calibration showed mean_signed_dp NEGATIVE at every
                       alpha on top-|SHAP| features (against the layer-11 residual probe). If that
                       sign inversion reappears in the logit-diff readout, it is a finding to
                       explain, not a bug to silence.
```

---

## Reporting results

- Re-derive everything from `run2/`. Nothing from `run1/` is quotable.
- The faithfulness ρ are **small-n**: k=20 features × 4 methods × 2 correlation
  types with no multiple-comparison control. p-values are descriptive.
- α is calibrated against a **layer-11 residual probe's P(positive)** but used
  with a **GPT-2 logit-diff** readout. It is a rough "measurable but unsaturated"
  magnitude, not a tuned hyperparameter. State this alongside any steering number.
- Attribution is **last-token**; intervention is **all-token**
  (`_apply_feature_intervention` edits every position). Some of any faithfulness
  gap is that mismatch.
- The MLP's `probe_saliency` is magnitude-only — importance tables only, never a
  directional comparison.
- Magnitudes are not comparable across arms (log-odds margin vs pre-sigmoid
  logit). Rankings within one arm are.
