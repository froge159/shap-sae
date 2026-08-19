# Manuscript Methodology Reference

Internal reference for writing the Methods (and relevant Results/Limitations)
sections of the manuscript. Headings below are structured to be used directly
as the manuscript's section/subsection outline — fill in prose and numbers
under each, pulling numbers only from `run2/` (see CLAUDE.md: `run1/` is
stale and unquotable). Where a number is already available in `run2/outputs/`
as of this writing it is given; anything else is marked `[TODO: pull from run2]`.

Source of truth for every claim below: `AGENTS.md`, `CLAUDE.md`, and the
scripts in `src/`. If code and this document ever disagree, trust the code
and update this document.

---



## 1. Task and Dataset

- **Task**: binary sentiment classification, SST-2 (Stanford Sentiment
Treebank, binary), loaded via HuggingFace `stanfordnlp/sst2`.
- **Split provenance**: the working dataset (`data/sst2_train`) is
three-way split **once**, with a fixed seed (`seed=42`), into:
  - **Probe train** — 70% (n = 47,144)
  - **Probe val** — 15% (n = 10,102)
  - **Held-out ("shap")** — 15% (n = 10,103)
  - Indices are persisted in `data/three_way_split_indices.json` and are
  committed to the repository; they are never regenerated (see §3).
- **Why a three-way split, not train/test**: every method compared in this
study (SHAP, Integrated Gradients, Gradient×Input, causal
steering/ablation) needs its own non-overlapping data for (a) fitting, (b)
hyperparameter/mask selection, and (c) evaluation, or the comparison is
contaminated. See §3 for the exact split→purpose mapping.

**Manuscript note**: report n per split, the SST-2 source/version, and state
explicitly that the split is fixed across every experiment in the paper (one
lineage, not resampled per analysis).

---



## 2. Model and Sparse Autoencoder

- **Language model**: GPT-2 Small, loaded via `transformer_lens`
(`HookedSAETransformer.from_pretrained_no_processing("gpt2")`). No
fine-tuning — GPT-2 is used frozen throughout; all learned components are
probes trained on top of its frozen residual stream.
- **Sparse autoencoder**: `gpt2-small-resid-post-v5-32k` from `sae_lens`,
32,768 latent features, hook point `blocks.{layer}.hook_resid_post`.
- **Layer**: **layer 7 only** (`resid_post`). This is a single-layer study;
state this as scope, not as a limitation to apologize for, unless the
manuscript later extends to multiple layers.
- **Token position**: sentence-level representation = SAE activation at the
**last real (non-padding) token**, resolved via `utils.last_real_token_index`
(delegates to `transformer_lens.utils.get_attention_mask`, which correctly
handles GPT-2's pad=bos=eos=`<|endoftext|>` collision). This is the single
definition used for extraction, probing, and every causal readout.
- **Decoder geometry**: `W_dec` rows are ~unit norm (median 0.9996 across
the SAE's 32,768 features) — material later for why no decoder-norm
correction is applied to steering (§8.3) or to group-steering vectors
(§10).

**Manuscript note**: name the exact SAE release string and hook point
(reviewers will want to look it up), and be explicit that "last token"
pooling is a design choice, not an oversight — it matches how the probes
are trained and how causal readouts are taken.

---



## 3. Data Splits, Sampling, and Row Discipline



### 3.1 Split-to-purpose mapping


| Split                      | Used for                                                                              | Never used for                  |
| -------------------------- | ------------------------------------------------------------------------------------- | ------------------------------- |
| train (47,144)             | fitting probes; SHAP/IG/GA background/baseline; steering α calibration                | reporting any result            |
| val (10,102)               | probe hyperparameters; the activation-frequency feature mask; reported probe accuracy | attribution, steering, ablation |
| held-out / "shap" (10,103) | **all** attribution (SHAP/IG/GA) and **all** causal evaluation (ablation/steering)    | fitting or tuning anything      |


This is the central methodological invariant of the study: attribution and
causal evaluation are computed on the **same held-out rows that nothing was
fit or tuned on**, so "did steering example *i* move the logit the way the
attribution method said it would for example *i*" is a valid faithfulness
question rather than a fit-the-training-data tautology.

### 3.2 Canonical row sampling

- All eval/background rows are drawn through one function,
`utils.sample_eval_rows` (seed = 42 canonical seed), which returns
**prefixes of a seeded permutation**, not independent random draws. This
gives two properties load-bearing for cross-scale comparisons:
  - **Nesting**: the 100-row local sample is exactly the first 100 rows of
  the 2000-row global sample — local and global results join by row
  position.
  - **Independence**: background sampling (from train, n=100) does not
  shift when the eval sample size changes.
- **Canonical sample sizes**: n_eval = 2000 (held-out), n_background = 100
(train). Individual scripts may use a prefix of this (e.g. 1000 for the
headline steering runs, 100 for local per-example runs).
- Sentences are paired to activation rows via `utils.eval_sentences`
(`Dataset.select`, which preserves index order) — never by re-indexing a
freshly loaded split, which would silently decouple sentence from
activation if the Hub ever reorders SST-2.

**Manuscript note**: state the canonical eval n (2000) and background n
(100) once, up front, and reference "the canonical held-out sample" rather
than restating the sampling procedure in every subsection.

---



## 4. Probing Models

Two probe architectures are trained on SAE activations, forming two parallel
"arms" of the whole study (linear and MLP). Every attribution method,
causal-evaluation script, and local/global comparison is duplicated across
both arms.

### 4.1 Linear (logistic) probe — the "linear arm"

- **Architecture**: `sklearn.linear_model.LogisticRegression`, **L1**
penalty (`l1_ratio=1`, not `penalty="l1"` — see reproducibility note
below), `solver="liblinear"`, `C=1`, `max_iter=1000`,
`random_state=42` (canonical seed).
- **Input**: full-width SAE activations (32,768-d) from the last token.
- **Sparsity is load-bearing**: L1 zeroes out the large majority of
coefficients (~1.4k of 32,768 non-zero on val-fit data), and this
sparsity pattern is one of two filters that defines the "filtered feature
set" used everywhere downstream (§5). An L2 fit would leave ~12k non-zero
and silently turn that filter into a no-op.
- **Seeding is load-bearing**: an unseeded fit re-rolls which coefficients
land at exactly zero (liblinear's coordinate descent is seeded from the
global NumPy RNG otherwise), which re-rolls the downstream feature mask,
the MLP probe's inputs, and the steering candidate pool. `random_state`
must be fixed on every `LogisticRegression` fit in the pipeline (probe,
residual probe, lexicon baseline).
- **Reported metric**: validation accuracy (on val split, never held-out),
non-zero weight count.



### 4.2 MLP probe — the "MLP arm"

- **Architecture**: `sklearn.neural_network.MLPClassifier`, one hidden
layer, `hidden_layer_sizes=(32,)`, ReLU activation, Adam solver,
`alpha=1e-4` (L2), `batch_size=256`, `learning_rate_init=1e-3`,
`max_iter=500`, early stopping (`validation_fraction=0.1`,
`n_iter_no_change=20`), `random_state=42`.
- **Input**: the **filtered** feature subset only (§5), not full-width —
75 features as of the current mask.
- **Motivation for a second arm**: the linear probe's gradient is constant
(`∂f/∂x = coef_`), which makes Integrated Gradients and Gradient×Input
collapse to a closed-form identity with SHAP (§6.6) — a sanity check, not
a comparison. The MLP's ReLU nonlinearity makes gradients
input-dependent, so IG/GA and SHAP can genuinely disagree; this is where
the substantive method comparison lives.
- **Reported metric**: validation accuracy and AUROC. Flag as **mildly
optimistic**: its inputs were pre-selected by a val-derived activation
frequency filter (label-free, so the leakage is small but real — state
this explicitly, do not omit it).

**Manuscript note**: Table 1 candidate — probe architecture, input
dimensionality, and val accuracy/AUROC for both arms side by side.

---



## 5. Filtered Feature Set (the "feature mask")

- **Single definition**: `utils.get_shap_feature_mask`. Combines two
filters on the **val split only** (never held-out — using held-out here
would let the eval split shape a model that is later scored on it):
  1. Non-zero linear-probe coefficient.
  2. Activation frequency ≥ 5% on val (`(val_activations > 0).mean(axis=0)
    > = 0.05`).
- **Current mask size: 75** (of 32,768). Explicitly note that this number is
unstable at small val sizes (500-row estimates ranged 68–81); 75 is the
size on the full 10,102-row val split with the committed probe.
- **What the mask gates**: the MLP probe's input layer, the steering
candidate pool (linear and MLP arms alike), and the group-steering
ranking pool. It does **not** gate the linear arm's own SHAP/IG/GA, which
are computed over the full 32,768-width space and only sliced to the mask
by consumers that need it (e.g. steering candidates).

**Manuscript note**: report exact mask size (75) and the two-filter
recipe; this is the "feature selection" step of the pipeline and deserves
its own short paragraph — reviewers will ask why 75 and not "all 32k" or
"top-k by weight."

---



## 6. Attribution Methods

Three attribution methods are computed per arm: exact/approximate SHAP,
Integrated Gradients (IG), and Gradient × (Input − Baseline) (GA). A fourth
"method" — raw probe weight / saliency — is reported alongside them as a
zero-cost baseline.

### 6.1 Target-function matching (the central methodological rule)

**Every method within one arm must explain the same scalar function of the
input**, or the comparison is not about attribution methods — it is
confounded by attribution methods disagreeing about *what they're
attributing*. This was a real bug in earlier iterations of this pipeline
(a `σ(1−σ)` sigmoid-derivative factor made IG/GA attribute `σ(logit)` while
SHAP attributed the raw logit) and is worth a paragraph in Methods
explaining why it matters, not just that it was fixed:

- Per example, `σ'(logit)` is a positive scalar, so it does not change the
*within-example* ranking.
- But every method here first **averages attributions over examples**, and
`mean_e[c_e · v_e]` is not a monotone function of `mean_e[v_e]` for
varying per-example weights `c_e`. Two differently-reweighted means of
the same per-example quantity are different numbers with no rank-preserving
relationship between them.
- Consequence: the historical near-zero IG-vs-SHAP correlation was
substantially an artifact of comparing two different target functions,
not evidence about the methods themselves.


| Arm               | Explainer                    | Target function            | ∇ used by IG/GA                                  |
| ----------------- | ---------------------------- | -------------------------- | ------------------------------------------------ |
| Linear (logistic) | exact `shap.LinearExplainer` | log-odds margin, `w·x + b` | `coef_` (constant)                               |
| MLP               | `shap.DeepExplainer`         | pre-sigmoid logit          | `mlp_logit_grad` (input-dependent, through ReLU) |


Magnitudes are **not** comparable across arms (log-odds margin vs.
pre-sigmoid logit are different scales); only within-arm rankings are.

### 6.2 SHAP — linear arm (§ `7_shap_recompute.py`)

- **Method**: `shap.LinearExplainer`, exact for a linear model (no sampling
approximation).
- **Masker**: `shap.maskers.Independent`, background = 100 train rows,
`max_samples` pinned to the full background count (the library default of
100 would otherwise silently subsample an *unseeded* random 100 rows out
of a larger background — a reproducibility hazard worth naming if the
manuscript discusses implementation details).
- **Scope**: explains **all 32,768 features** (not the filtered mask) —
cheap and exact, so there is no reason to restrict it; downstream
consumers slice to the mask themselves.
- **Reported quantity**: Φ = **mean signed** SHAP value per feature over
the 2000-row canonical eval sample (not mean |SHAP| — sign is preserved
and is the basis for the directional/causal comparisons in §8).
- **Sign consistency**: per-feature fraction of examples whose SHAP value
shares the majority sign — reported as a secondary diagnostic of how
representative the mean-signed Φ is of individual examples.



### 6.3 SHAP — MLP arm (§ `12_mlp_shap.py`)

- **Method**: `shap.DeepExplainer` on a hand-cloned PyTorch
`nn.Sequential(Linear, ReLU, Linear)` reproducing the fitted
`MLPClassifier` exactly (weights copied in; a custom `forward` would break
DeepSHAP's op-level attribution).
- **check_additivity=True** — DeepSHAP's additivity check is left on; a
failure there is treated as a real bug, not a numerical nuisance to
suppress.
- **Target**: pre-sigmoid logit (DeepSHAP is documented as more reliable on
logits than on post-sigmoid probabilities).
- **Scope**: the filtered 75-feature set only (this is the MLP's native
input space).



### 6.4 Integrated Gradients (IG)

- **General definition**: `IG_j = (x_j − baseline_j) · ∫₀¹ ∂f/∂x_j (baseline + α(x−baseline)) dα`, approximated (MLP arm) with a 50-step
Riemann sum (`n_steps=50`, `np.trapezoid`).
- **Linear arm**: the path integral of a constant gradient collapses in
closed form to `IG_j = (x_j − baseline_j) · w_j` — computed directly
rather than by Riemann-summing a constant (same number, far less
compute); `n_steps` is retained in the function signature purely so call
sites stay uniform with the MLP arm.
- **MLP arm**: gradient is genuinely input-dependent
(`mlp_logit_grad`, propagated through the ReLU mask `pre > 0`), so the
50-step Riemann sum is real numerical integration, not a formality.
- **Baseline**: the **mean of the 100-row train background** (not a
zero vector). This choice matters concretely on SAE features, which are
~87% zero: a zero baseline makes GA (§6.5) degenerate to 0 whenever
`x_j = 0`, while a mean baseline does not.



### 6.5 Gradient × (Input − Baseline) (GA)

- **Definition**: `GA_j = (x_j − baseline_j) · ∂f/∂x_j(x)` — a single-point
(non-integrated) gradient attribution, using the **same baseline** as IG
for comparability.
- **Linear arm**: coincides exactly with IG (constant gradient ⇒ the two
formulas are identical), so it is reported as a distinct method mainly
for structural parity with the MLP arm, where it is genuinely different
from IG (single endpoint gradient vs. path-integrated gradient through a
nonlinearity).



### 6.6 The linear-arm identity (why ρ = 1.000 there is expected, not a result)

With the target matched (§6.1), **IG, GA, and interventional SHAP are
analytically identical for a linear probe** — all three reduce to
`w_j · mean_e[x_ej − μ_j]`. `8_feature_ranking.linearity_identity_report`
asserts this identity (`max|IG − Φ| ≤ 1e-5` relative, floored near 1e-7 by
float32 accumulation) and writes a PASS/FAIL into `correlation.txt`.

**Manuscript framing**: present the linear arm's perfect rank agreement
explicitly as a **closed-form sanity check**, not as evidence that "SHAP
agrees with gradient methods." The real empirical comparison is the MLP
arm, where the ReLU breaks the identity and IG/GA/SHAP can and do diverge.
Stating this distinction up front pre-empts an obvious reviewer objection
("why is your headline correlation exactly 1.000?").

---



## 7. Feature-Ranking Agreement (Cross-Method Comparison)

- **Statistic**: Spearman ρ, pairwise, between {probe weight/saliency, IG,
GA, SHAP Φ}.
- **Restricted to the "scored support"**: features with a non-zero probe
score (linear arm: non-zero `coef_`; this restriction is what keeps the
correlation meaningful — see below).
- **Why restriction matters**: the L1 probe zeroes ~77% of the 32,768
features, and IG/GA/Φ are *identically* zero wherever the probe weight is
zero (IG/GA ∝ `coef_`; Φ = `w·(x̄_eval − x̄_bg)`). Correlating over the
full 32,768-width space reports agreement about which features were
*excluded*, not agreement about which excluded-complement features
matter — this historically inflated Probe-vs-GA to 0.997 and made an
IG-vs-SHAP ρ of 0.018 look like a meaningful (near-zero) empirical
result rather than an artifact of a tie block. Always report ρ **on the
non-zero support**, and say so.
- **Linear arm**: ρ = 1.000 for every pair by construction (§6.6) — an
identity check, report as such.
- **MLP arm**: ρ computed on **|score|** across all four methods —
importance-only, because the MLP's `probe_saliency` (‖W1[i,:]‖₂) has no
sign and cannot be compared directionally. This is where the genuine
IG-vs-SHAP (and Probe-vs-SHAP, etc.) empirical numbers live.
`[TODO: pull run2/outputs/13_mlp_ranking/correlation.txt figures]`

**Manuscript note**: this section (linear-arm identity + MLP-arm empirical
ρ table) is likely your core "attribution methods agree/disagree" result.
Present the linear-arm number as validation of implementation correctness;
present the MLP-arm number as the actual finding.

---



## 8. Causal Evaluation: Steering and Ablation

The attribution methods above are correlational (computed from a fixed
model, no intervention). Causal faithfulness is evaluated separately by
directly intervening on SAE feature activations and reading out the
resulting change in model behavior — deliberately **not** the probe the
attributions explain, so faithfulness is scored against the underlying
language model rather than against the explainer's own target.

### 8.1 Intervention definitions

At `blocks.7.hook_resid_post`, for a chosen feature *i*:

- **Ablation**: `encode → set a_i = 0 → decode`. Removes the feature.
- **Steering**: `encode → a_i ← a_i + α_i → decode`. Additive, signed
(`α_i` may be negative), applied at **every token position** — not just
the last token that attributions describe (a mismatch worth stating
explicitly, see §13).
- **Baseline**: encode→decode with **no** feature edit (SAE reconstruction
baseline), so measured Δ is not confounded with SAE reconstruction error.



### 8.2 Readout

- **Metric**: GPT-2 logit difference at the last real token,
`logit(" wonderful") − logit(" awful")`, before/after intervention.
Single-token sentiment probe pair, resolved once and asserted to be
single tokens.
- Deliberately **not** the probe's own prediction — this is what makes
causal evaluation a test of "does this feature causally matter to the
language model's sentiment behavior," not a test of "does this feature
matter to the probe we fit."
- Δ reported both as **|Δ|** (magnitude/importance) and **signed Δ**
(direction).



### 8.3 Steering strength (α) calibration

- `9.5_tuning.py` calibrates a global scalar α via a pilot sweep
({0.5, 1, 2, 4} × median natural positive-activation scale) on 3
top-|SHAP| pilot features, over 200 **train**-split examples (never
val/held-out — α must not be tuned on data anything downstream is scored
on).
- **Readout used for calibration differs from the readout used for
evaluation**: α is picked by its effect on a **layer-11 residual probe's
P(positive)**, but the steer scripts that consume α read out a **GPT-2
logit difference** (§8.2). This is a deliberate, acknowledged mismatch:
α is a rough "measurable, unsaturated" magnitude, not a hyperparameter
tuned against the metric it is later evaluated on. **State this every
time α is reported** (per CLAUDE.md's reporting rule).
- Selection rule: smallest α with mean|ΔP| ≥ 0.02 and saturation fraction
(examples with P<0.02 or P>0.98) ≤ 0.25; falls back to best unsaturated
α if nothing meets the threshold.
- Recommended α is read from `alpha_calibration.txt`'s `recommended_alpha`
at run time — the CLI default (0.6723) is what `run1` recommended and
should not be assumed to carry over to a fresh run.
- `--alpha-mode` (`utils.resolve_steering_alphas`), applied per feature:
  - `scaled` (default/primary): `α_i = α · scale_i / median(scale)`, where
  `scale_i` is feature *i*'s mean positive train activation. Equalizes
  the *relative* nudge across features that differ severalfold in
  natural activation scale.
  - `constant` (secondary/robustness check; what `run1` used): flat α for
  every feature. Confounds the `|attribution| vs |Δ|` correlation,
  because |Δ| then partly reflects how hard a feature was pushed
  *relative to its own scale* rather than the attribution itself.
  - No decoder-norm correction is applied (or needed): `W_dec` rows are
  ~unit norm (median 0.9996), so `‖W_dec[i]‖` contributes negligible
  additional confound.



### 8.4 Candidate selection: top vs. random

- `--selection random` (default/headline): *k* features drawn uniformly
from the filtered pool. Unbiased.
- `--selection top` (secondary, must be labeled with the caveat):
top-*k* by |SHAP|, then all four methods scored on that same set. This
**range-restricts SHAP alone** on the scored candidates (by construction,
SHAP's variance on its own top-k is compressed) while leaving the other
three methods' variance unrestricted, which mechanically deflates SHAP's
ρ relative to methods that did not drive selection. Never present a
`top`-selection faithfulness number without this caveat attached.
- Default `k = 20`, `n_eval = 1000` (headline steering configuration).



### 8.5 Faithfulness metric and sign convention

- **Importance faithfulness**: Spearman ρ(|attribution score|, |Δ|) — does
the method's magnitude ranking predict how much the causal effect moves?
- **Directional faithfulness**: Spearman ρ(signed attribution, oriented Δ)
— does the method's *sign* predict the *direction* of the causal effect?
- **Sign convention (**`orient_signed_effects`**)**: steering adds +α, so a
positive attribution should predict a positive Δ (no transform needed).
Ablation *removes* the feature, so a positive attribution should predict
a **negative** Δ — raw ablation Δ is negated before correlating, so that
**"positive ρ" always means "faithful" regardless of mode**. Never
compare a directional ρ across modes without confirming this orientation
was applied.
- **Statistical caveat to repeat wherever ρ is reported**: default k=20
features × 4 methods × 2 correlation types, with **no multiple-comparison
correction**. Treat individual p-values as descriptive, not
confirmatory.
- Every steer run's config (mode/selection/k/alpha/alpha-mode/n-eval) is
encoded in its output filenames (`{mode}_{selection}_k{k}` tag) — cite
the exact tag alongside any number quoted from these outputs, since
several configurations can share one output directory.

**Manuscript note**: §8 (steering/ablation faithfulness) is likely your
second core result table, parallel across the two probe arms (linear via
`9_steer.py`, MLP via `14_mlp_steer.py`, identical flags by design — see
§13 "match the arms"). Report at minimum: headline (`steering`, `random`,
`scaled`, k=20) for both arms, the `ablation` companion, the `top`-selection
secondary (with bias caveat), and the `constant`-alpha robustness check
against `scaled` (§13, point 9).

---



## 9. Local (Per-Example) Faithfulness

Global Φ is a mean over 2000 examples; averaging can hide per-example
attribution error or manufacture agreement that doesn't hold example-by-
example. Three scripts test this directly (`9_local_steer.py`,
`14_local_mlp_steer.py`, `local_shap_faithfulness.py`), sharing a matched
CLI surface (`k_local = k_global = 20`, `n_examples = 100`) so the two probe
arms are comparable.

### 9.1 Candidate construction

- Per sentence *x*, compute local LinearSHAP/DeepSHAP φ(x) (same explainer
class as the global computation, applied to one row).
- Candidate set for that sentence = **top-k by |φ(x)|** (local) **∪ top-k
by |Φ|** (global) — so local and global attribution compete on equal
footing rather than the global-only candidate set structurally favoring
the global method.
- Per-example intervention effects Δ(x, i) are kept **per example**, never
averaged across the dataset before correlating — this is the entire
point of the local analysis (contrast with §8, where Δ is a per-feature
dataset mean).
- Per sentence: Spearman ρ(|local φ(x)|, |Δ(x,·)|) and ρ(|global Φ|,
|Δ(x,·)|) on the same candidate set; summarized as **mean ρ across
sentences** (± std), plus the fraction of sentences where local beats
global.



### 9.2 The activity confound and its control

- **Confound**: globally-top features (by |Φ|) are usually **inactive** in
any single sentence. Ablating an inactive feature moves the logit by
exactly 0 (there is nothing to remove); local |φ(x)| tracks per-sentence
activity by construction (it is `w·(x−x̄)`, so an inactive feature has
φ≈0 too, but *correctly* so). A chunk of any apparent "local SHAP beats
global Φ" margin is therefore just "local knows which features are on in
this sentence," not evidence about ranking quality.
- **Control (Block [B])**: candidates are restricted to those the sentence
**actually activates** (activation > 0) before re-scoring. A local win
that survives this restriction is evidence about *ranking quality among
active features*, not merely about knowing which features fired.
- **Reporting rule**: always report both blocks. **Quote Block [B], not
Block [A]**, as the headline local-vs-global number — CLAUDE.md is
explicit that Block [A] is "dominated by [the inactive] tie block."
`local_shap_faithfulness.py` (the third, diagnostic script) has **no**
activity control at all — treat its numbers as an upper bound on the
local advantage and read them only alongside the controlled Block [B]
from the other two scripts, never in isolation.

**Manuscript note**: this section directly answers "is averaging the
problem, or is the attribution method itself unfaithful even locally?" —
frame it as that question. Report Block [A] and [B] side by side in one
table so the size of the activity confound is visible, not just the
corrected number.

---



## 10. Multi-Feature (Group) Steering

A separate, larger-scale causal experiment: build a single steering
*direction* (not single-feature intervention) from the top-*k* features of
each attribution method, and sweep effect vs. cost.

- **Vector construction**: for method *m*, take top-*k* by |score| from the
**shared 75-feature pool** (same pool for both arms, so top-k at equal k
draws from the same candidate space — without this, the linear arm's
full-width scores and the MLP arm's mask-zeroed scores would not be
comparable at "the same k"), then `v = Σ_i score_i · W_dec[i]`,
unit-normalized.
- `actdiff` **baseline**: a no-attribution-math baseline — top-k by
`|mean_pos_activation − mean_neg_activation|`, weighted by the signed
difference itself. Scored once (arm-independent, since it uses no
probe) and fanned out to both arms' plots (`arm_independent: true` in
the output).
- **MLP arm has no** `probe` **curve**: the MLP's `probe_saliency`
(‖W1[i,:]‖₂) is unsigned, and the vector-construction formula uses score
as a **signed** weight — summing unsigned magnitudes with the decoder
would add positive- and negative-sentiment directions with the same
sign, producing a direction that means nothing. The logistic arm keeps
its `probe` curve because `coef_[0]` is signed. State this explicitly in
any figure/caption showing the MLP panel — it is a principled omission,
not missing data.
- **Sweep**: `k ∈ {5, 10, 20, 30, 50, 65}` × `α ∈ {0.5, 1, 2, 5, 10, 20, 50}` (from `steering_config.json`), giving 7 vectors (4 logistic + 3
MLP) × 6 k × 7 α = 294 arm-specific rows, plus `actdiff` at 6×7=42 scored
once and duplicated across both arms (84 rows tagged
`arm_independent: true`) — **378 total rows from 336 actual forward
passes**.
- **Effect** = mean Δ logit-diff (wonderful − awful); **cost** = mean Δ
perplexity (steered vs. clean baseline) — a genuine effect-vs-fluency
trade-off curve, not a single scalar.
- **Baseline differs from §8**: this script never touches the SAE at
inference (adds `α·v` directly to the clean residual stream, no
encode/decode), so its baseline is a **clean forward pass**, whereas
`9_steer`/`14_mlp_steer` baseline through an **encode→decode
reconstruction**. Both are internally consistent, but **the two families'
effect sizes are not on the same scale and must never be compared
numerically** — only the *shapes* of the effect-vs-cost curves are
comparable across the two families.

**Manuscript note**: this is your "does the attribution ranking transfer to
a practical steering-vector application" section — present as effect-vs-cost
curves (one figure per arm, per CLAUDE.md's figure list — these are the
*only* figures the pipeline produces), and state the clean-vs-reconstruction
baseline distinction in the figure caption, not just in prose, since readers
will otherwise directly compare magnitudes against §8's numbers.

---



## 11. Lexical Simplicity Control

A control analysis, not a headline result: asks whether SST-2 is lexically
simple enough that SHAP's interaction-awareness has little room to matter
(in which case cross-method attribution differences may reflect little more
than "who best detects a sentiment word," not genuine interaction
sensitivity).

- **Population**: the **full 10,103-row held-out split**, not the
2000-row canonical eval sample used everywhere else — flag this
explicitly whenever quoting these numbers next to attribution/steering
numbers, which describe a different (smaller, sampled) population.
- **Three sub-analyses**:
  1. **Co-activation**: how many of the 75 filtered features are active
    (>0) per example, and pairwise Jaccard among the top-20 |Φ| features
     — a sparse/rarely-co-active top set means little room for
     feature×feature interaction.
  2. **Lexical tracking**: point-biserial r (and Spearman ρ against a
    continuous lexicon score) between each top feature's activation and
     presence of a hand-curated positive/negative sentiment word list
     (movie-review-style, ~70 words per polarity, no external download).
     High |r| ⇒ the feature looks like a lexical detector.
  3. **Lexicon baseline vs. SAE probe**: a bag-of-lexicon-words logistic
    regression (fit on train, scored on held-out) compared against the SAE
     probe's held-out accuracy. A near-tie ⇒ SST-2 sentiment is largely
     lexical and the SAE probe adds little beyond word-spotting.
- **Reading heuristics** (already encoded in the script, reuse the same
thresholds for consistency): "sparse top" if ≥50% of examples have
exactly one top-20 feature active, or mean active <1.5; "strong lexical
tracking" if mean |r(activation, has-any-lexicon-word)| ≥ 0.25;
"lexicon-near-probe" if lexicon BoW accuracy is within 0.03 of SAE probe
accuracy.

**Manuscript note**: use this section to pre-empt the "isn't this just
measuring which method best finds sentiment words?" reviewer question —
report the three numbers (mean active features, mean |r|, accuracy gap)
directly and let them argue for or against the lexical-shortcut reading
rather than asserting it.

---



## 12. Statistical Reporting Standards

Conventions to apply uniformly across every results section, not just
where explicitly reiterated above:

1. **Correlation statistic**: Spearman ρ throughout (rank-based; appropriate
  given attribution scores are compared by relative importance, not
   absolute scale, and scales differ across methods/arms).
2. **Small-n, no multiple-comparison control**: the faithfulness tables are
  k=20 features × 4 methods × 2 correlation types by default. Report
   p-values as descriptive, not confirmatory, every time.
3. **Restrict correlations to the non-zero/scored support** (§7) — never
  correlate over the full 32,768-feature space, which reports agreement
   about exclusion rather than about importance among included features.
4. **State the target function** whenever quoting an attribution magnitude
  (log-odds margin vs. pre-sigmoid logit vs. probability) — magnitudes are
   never comparable across arms; only rankings within one arm are.
5. **State the α-calibration mismatch** (§8.3) alongside any steering
  number: α is a rough, unsaturated magnitude calibrated against a
   different readout than it is evaluated with, not a tuned
   hyperparameter.
6. **State which sample/population** a number is drawn from whenever two
  sections could be confused (the 2000-row canonical sample vs. the full
   10,103-row held-out split in §11; the 1000-row steering sample vs. the
   100-row local sample in §9).
7. **Tag every steering/ablation number with its config**
  (`{mode}_{selection}_k{k}`, alpha, alpha-mode) — several configurations
   can share one output directory.
8. **Report both** `scaled` **and** `constant` **alpha-mode results** where
  feasible, as a robustness check on whether activation-scale confounding
   materially changes the conclusion (§13, point 9 below).
9. Every number quoted in the manuscript must trace to `run2/` (or a later
  fresh run) — never `run1/`, which predates the current scripts and is
   kept only for a documented before/after comparison, itself clearly
   labeled as such if used.

---



## 13. Known Limitations to State Alongside Results (not to silently fix)

These are acknowledged, deliberate properties of the current design — state
them in a Limitations section or as caveats next to the relevant result,
not as bugs:

1. **Attribution is last-token; intervention is all-token.** Attributions
  describe the SAE vector at one sequence position; `_apply_feature_  intervention` edits every token position. Part of any faithfulness gap
   is this mismatch, not (necessarily) attribution error.
2. **α is calibrated against a different readout than it is evaluated
  with** (§8.3) — a layer-11 residual probe's P(positive) at calibration
   time, a GPT-2 logit difference at evaluation time.
3. **Group-steering effect sizes are not comparable to per-feature steering
  effect sizes** (§10) — different baselines (clean forward vs.
   encode→decode reconstruction). Compare curve shapes only.
4. **The MLP arm's** `probe_saliency` **is magnitude-only** (‖W1[i,:]‖₂, no
  sign) — valid for importance/ranking comparisons, invalid for any
   directional/causal-sign comparison. It is excluded from the
   group-steering vector construction for the same reason (§10).
5. `--selection top` **biases SHAP's faithfulness ρ downward** relative to
  the other three methods, by range-restricting only SHAP's scored
   variance (§8.4). Never present a `top` number without this caveat.
6. **Faithfulness ρ are small-n with no multiple-comparison control**
  (§12, point 2).
7. **The MLP probe's reported validation accuracy is mildly optimistic**
  (§4.2) — its inputs were selected using a val-derived (label-free)
   frequency filter.
8. **The linear arm's rank-agreement ρ = 1.000 is a closed-form identity**,
  not an empirical finding (§6.6) — do not present it as "SHAP and
   gradient methods agree" without this qualification.
9. **Alpha-mode sensitivity**: compare `faithfulness_..._scaled` vs.
  `faithfulness_..._constant` runs. A large gap between them indicates
   `run1`'s original constant-alpha numbers substantially reflected
   feature activation scale rather than attribution faithfulness — report
   both regardless of which way this comes out.
10. **Causal-direction sign check**: features with positive Φ should on
  average produce positive Δ under steering. `run1`'s alpha calibration
    showed `mean_signed_dp` **negative** at every α on top-|SHAP| features
    (against the layer-11 residual probe). If this sign inversion
    reappears in the `run2` logit-diff readout, it is a finding to report
    and explain (e.g., readout mismatch, SAE reconstruction artifact), not
    a bug to quietly patch away before writing up.

---



## 14. Follow-Up A: Seed Replication and a Higher-Power Global-Faithfulness Run

Addresses the weakest claim in §8: the global causal-faithfulness null rests on
a single `--selection random --seed 0` draw of k=20 features. At n=20 the
confidence interval on Spearman ρ spans most of [−1, 1], so "ρ ≈ 0 from one
draw" is compatible with both a true null and a real moderate effect. Two
independent fixes, one raising power directly and one characterising the
sampling distribution.

### 14.1 Higher-power run (`--selection all`)

- **Change**: new `all` branch in `get_shap_candidates` (both steer scripts),
scoring every feature in the filtered pool. Deterministic — no RNG, so it
carries no seed-draw luck at all.
- **Effect**: raises n per correlation test from 20 to 116 with no resampling.
- **Tag**: `{mode}_all_k116`. Compute: ~5.6× one k=20 run per arm.
- **Manuscript framing**: this, not the k=20 run, should be the headline global
faithfulness number. Report the k=20 result as the original configuration and
the k=116 result as the powered replication.



### 14.2 Multi-seed replication (`9a_seed_sweep.py` / `14a_seed_sweep.py`)

- **Design**: draws each seed's candidate set via the unmodified
`get_shap_candidates`, takes the union (~99 of 116 for 10 seeds at k=20), and
runs `get_model_intervention_effects` **once** over the union, then slices per
seed. Each seed's numbers are identical to a standalone run's (same rows, same
α, same baseline) at ~half the cost of ten independent reruns.
- **Statistics per (method × correlation type)**:
  - one-sample **Wilcoxon signed-rank** of the 10 per-seed ρ against 0 —
  signed-rank rather than a t-test because n≈10 and Spearman ρ is bounded;
  - **percentile bootstrap CI** (B=5000, seed 42) on the mean ρ, resampling over
  *seeds* (the seed is the unit of analysis, not the example);
  - `ref_seed_outlier_p` — the fraction of draws with |ρ| at least as large as
  the published seed-0 run. An exchangeability p-value: it says whether the
  headline draw was typical, not whether an effect exists.
- **Output**: `seed_sweep_{mode}_{alpha_mode}_k{k}.{json,txt}`.



### 14.3 Multiple-comparison correction

- **New utility**: `utils.benjamini_hochberg(pvalues, alpha=0.05)`, wrapping
`scipy.stats.false_discovery_control(..., method='bh')`. The family is
caller-defined, not hardcoded, because the right family depends on the claim:
the 8 tests of one configuration, or the 8 × n_seeds of a sweep.
- **Backward compatibility**: `report_faithfulness` gains
`bh_correction: bool = False`; the default reproduces the historical
`faithfulness_*.txt` byte-for-byte. When enabled it appends an FDR block and
writes a **new sidecar** `faithfulness_<tag>_bh.json` — the original file is
never rewritten.
- **Why it matters here**: the existing run reports ~30+ uncorrected tests, of
which exactly two cleared p<0.05 — the expected yield under a complete null.
Report the corrected count, not the raw hits.

---



## 15. Follow-Up B: The Sign-Inversion Anomaly

`9.5_tuning` reports `mean_signed_dp` **negative at every α** on top-|SHAP|
pilot features: steering a feature SHAP calls positive-sentiment in the positive
direction made the readout *less* positive. This reproduced independently in
`run1` and `run2` but had never been tested directly.

### 15.1 Free reanalysis of existing artifacts (`9c_sign_reanalysis.py`)

No GPU. `model_deltas_<tag>.json` already stores per-feature `delta_signed` from
the real logit-diff readout; joining it against the saved Φ gives a binomial sign
test for free, stratified by the sign of the attribution.

**Results already obtained on** `run2` (report as preliminary/motivating):


| Arm / set                   | Positive-Φ features agreeing | Rate |
| --------------------------- | ---------------------------- | ---- |
| Linear, `selection=top`     | 4 / 11                       | 0.36 |
| MLP, `selection=top` (SHAP) | 3 / 10                       | 0.30 |
| MLP, `selection=top` (IG)   | 3 / 11                       | 0.27 |
| Linear, `selection=random`  | 6 / 10                       | 0.60 |


Two things worth stating: the inversion appears **only in the top-|SHAP| set**,
not the random set — consistent with `9.5_tuning` having piloted on top-|SHAP|
features — and it appears in **both arms** at similar magnitude. But every
p > 0.1 at n≈10. **This is underpowered and must be labelled as motivating, not
confirmatory.**

### 15.2 Sign-stratified run (`--selection top-pos` / `top-neg`)

- **Design**: select top-k by *signed* Φ within one sign class, so a systematic
inversion cannot cancel within a mixed-sign candidate set. All four methods are
still scored on the resulting fixed set (matching the existing convention).
- **Statistic**: `utils.sign_agreement_test` — two-sided binomial (is agreement
different from chance?) plus one-sided `less` (is agreement *below* chance,
i.e. an actual inversion?). Deltas of exactly 0 are excluded and counted
separately, since an inactive feature under ablation gives Δ=0 by construction
and carries no directional information.
- **Output**: an additive `sign_agreement` block inside `model_deltas_<tag>.json`
(absent for historical selections, so old readers keep working).
- **What it isolates**: these scripts already use the logit-diff readout
exclusively, never the layer-11 probe. So if positive-Φ features still invert
here, the readout mismatch cannot be the sole explanation.



### 15.3 Readout-matched replication of the α pilot (`--readout logitdiff`)

Re-runs the *same* pilot (same 3 features, 200 train examples, same α grid) with
the GPT-2 logit difference substituted for the layer-11 probe probability,
holding every other variable fixed. Writes a `_logitdiff`-tagged file and
recommends no α (the mean|ΔP| ≥ 0.02 criterion is on a probability scale and does
not transfer). **Both readouts negative ⇒ the inversion is a property of the
model, not an artifact of the calibration readout** — a finding to explain, not
a bug to silence.

### 15.4 Deferred: SAE-reconstruction control

Steering the same features via a direct residual-stream add (no encode→decode
round trip), reusing `group_steering`'s machinery, would test whether SAE
reconstruction error explains the inversion. Designed but **not implemented** —
narrowest hypothesis, most new code, and §15.2–15.3 will likely adjudicate first.

---



## 16. Follow-Up C: Formalizing the Activity Confound

§9 reported the local-vs-global comparison as "mean ρ ± std" plus a win-fraction.
That is a description, not a test, and it left the paper's most striking local
result resting on two point estimates. `analyze_local_vs_global.py` (CPU-only,
reanalyses saved arrays — **$0 GPU**) adds three things.

### 16.1 Paired significance tests

Per-example `ρ_local − ρ_global` is a paired difference, so **Wilcoxon
signed-rank** is the appropriate test (bounded, non-normal statistic, n=100),
with a bootstrap CI on the mean gap for effect size. Run for both blocks and both
correlation types.

**Results already obtained on** `run2`**:**


| Block : kind    | local  | global | gap        | Wilcoxon p | win frac |
| --------------- | ------ | ------ | ---------- | ---------- | -------- |
| **Linear arm**  |        |        |            |            |          |
| A : importance  | +0.414 | −0.269 | **+0.682** | 3.9e−18    | 1.000    |
| A : directional | +0.135 | −0.158 | +0.294     | 4.1e−15    | 0.860    |
| B : importance  | +0.202 | +0.285 | −0.083     | 6.6e−02    | 0.420    |
| B : directional | +0.144 | −0.311 | **+0.455** | 9.1e−16    | 0.880    |
| **MLP arm**     |        |        |            |            |          |
| A : importance  | +0.345 | −0.223 | +0.568     | 3.9e−18    | 1.000    |
| A : directional | +0.102 | −0.239 | +0.341     | 1.2e−15    | 0.880    |
| B : importance  | +0.077 | +0.371 | **−0.295** | 9.6e−09    | 0.270    |
| B : directional | +0.094 | −0.438 | **+0.532** | 5.7e−17    | 0.920    |


**This changes the story and is the most important new result in the follow-up
set.** The original reading was "local SHAP wins, then loses once activity is
controlled." The tested version is more specific and more interesting:

- On **importance**, the local advantage does not survive the control — and in
the MLP arm it *significantly reverses* (global beats local, p=9.6e−09).
- On **direction**, local SHAP **survives the activity control decisively** in
both arms (+0.455 / +0.532, p<1e−15), because global Φ goes *negative* there.

So the honest claim is not "local beats global" or "the local win was an
artifact", but: **local attribution carries genuine per-example directional
information that averaging into Φ destroys, while its apparent importance-ranking
advantage is largely the activity confound.** That is a sharper and more
defensible contribution than either simpler claim.

### 16.2 Decomposition of the block [A] margin

Regresses the per-example gap on the active fraction of candidates
(`scipy.stats.linregress`), plus a raw-count and a size-controlled variant
(`np.linalg.lstsq`), so the conclusion is not an artifact of one
parameterisation. On `run2`: R² = 0.019 (linear) / 0.081 (MLP) for the single
regressor, rising to 0.120 / 0.215 controlling for candidate-set size. **Activity
explains a real but modest minority of the block [A] margin linearly** — worth
reporting precisely rather than asserting the confound explains everything.

### 16.3 Continuous activity control

Binary active/inactive discards how strongly each feature fires. A **partial
Spearman** of |attribution| vs |Δ| controlling for |activation| keeps the whole
candidate set and removes the activity signal continuously, computed in closed
form from three pairwise Spearman ρ (no new dependency). On `run2`, local SHAP
retains partial ρ = +0.184 (linear) / +0.150 (MLP).

### 16.4 Retrofit of `local_shap_faithfulness.py`

The third local script was cited as an "upper bound" precisely because it had no
activity control. It now reports the same block [A] / [B] split as the other two.
The control is **free** — the per-candidate Δ are already computed in its main
loop, so [B] is a CPU-side subset of the same numbers. It now also persists
`deltas_per_example.npy` and `cands_per_example.npy`, so any further control can
be added offline without another GPU pass. One re-run is needed to produce those
artifacts.

---



## 17. Follow-Up D: Known-Ground-Truth Validation (Minimal Pairs)

Every number in §7–§10 is measured on natural SST-2 text, where the true causal
importance of an SAE feature is unknown. A near-zero faithfulness ρ therefore has
two incompatible readings the main pipeline cannot separate:

(a) the attribution methods really are unfaithful, or
(b) the harness (k=20, α calibrated on a different readout, last-token
attribution vs all-token intervention) cannot detect faithfulness even when
present.

`24_minimal_pairs.py` builds stimuli where a real sentiment change is known by
construction and asks whether the harness recovers it.

### 17.1 Stimulus construction

- `--edit remove` **(primary)**: held-out sentences containing **exactly one**
lexicon word (from `10_simplicity_check`'s `POS_WORDS`/`NEG_WORDS`), paired
with that single token deleted. Everything else is byte-identical. "Exactly
one", not "at least one" — with two sentiment words the remaining one keeps
carrying sentiment and the pair is no longer a single-cause edit.
- `--edit insert` **(secondary)**: lexicon-free sentences with
`" it was <word> ."` appended. Less minimal (adds a clause) but tests the
direction remove-pairs cannot.
- Expected direction is known: removing a positive word should *lower* the logit
difference; adding one should *raise* it.



### 17.2 Three measurements, in dependency order

1. **Readout validity** — sign accuracy of the text-level Δ against the known
  direction. **This is a precondition for everything else in the paper**: a
   readout that cannot detect a deliberately inserted sentiment word cannot
   support any faithfulness claim. On a 12-pair pilot, ~0.9 (chance = 0.5), so
   the logit-diff readout is sound.
2. **Feature recovery (the actual ground truth)** — features whose activation
  moved between the pair demonstrably carry the edit. Labelling those positive
   and distractors negative gives a **binary ground truth with a defined right
   answer**, scored by AUROC per method. Unlike a Spearman against an unknown
   target, chance level is known to be 0.5.
3. **Causal recovery** — ablating each candidate in the edited sentence and
  measuring how much of the edit's effect it was supplying, then correlating
   that continuous ground-truth causal score against each attribution.



### 17.3 Two confounds that must stay controlled

Both were found and fixed during implementation; both produced dramatically
wrong numbers before the fix, and both must be stated in the manuscript if any
variant is reported.

- **Local-SHAP background.** Using the pair's *own original activations* as the
SHAP background makes local `φ_j = w_j·(x_j − orig_j)` proportional to the
activation change — which is exactly the quantity the ground-truth label is
derived from. Local SHAP then scores a near-perfect AUROC **tautologically**
(0.803 vs 0.513 once corrected to the canonical train background).
- **Distractor selection.** Choosing the negative class by top-|Φ| makes it
definitionally the features global SHAP ranks highest, forcing its AUROC toward
0 regardless of faithfulness (0.112 vs 0.491 once corrected to random
distractors). This is the same range-restriction trap as `--selection top`
in §8.4. `--distractors random` is the headline; `global` is available but
must be labelled.



### 17.4 Preliminary result and how to read it

With both confounds controlled, a 12-pair pilot puts **every method at chance**
(AUROC 0.49–0.51) while readout sign accuracy stays high. Read literally, that
supports interpretation (a): the harness demonstrably detects the known
sentiment edit, and the attribution methods still cannot identify which features
carried it. Run at the full n=100 before quoting; the pilot is not powered.

### 17.5 Deferred alternative: synthetic-label probe

A probe refit on a label defined as a deterministic boolean function (AND/OR/XOR)
of 2–3 chosen SAE features gives *exact* ground-truth importance, and ablation
(`acts[i] = 0`) maps exactly onto "deactivate" under a boolean encoding — so no
α calibration is needed at all. XOR in particular would test SHAP's claimed
interaction-awareness against IG/GA directly, following up §11's interaction
question. Designed but **not implemented**; the minimal-pair design was
prioritised as the more direct validation on real text.

---



## 18. Suggested Priority Order

For a single-GPU budget, in the order the evidence compounds:


| #   | Item                      | Cost                    | What it buys                                                                                                                             |
| --- | ------------------------- | ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | §16.1–16.3, §15.1         | **$0 GPU**              | Formal tests + the sign-inversion reanalysis from artifacts already on disk. Do first — §16.1 already changed the local-vs-global story. |
| 2   | §16.4 retrofit            | 1 re-run, existing cost | Closes an explicitly-flagged inconsistency.                                                                                              |
| 3   | §14.1 (`--selection all`) | ~5.6× one k=20 run      | Highest leverage per GPU-hour: turns the most-quoted null into a powered test.                                                           |
| 4   | §15.2–15.3                | ~2–3× one k=20 run      | Directly resolves a cross-run reproduced anomaly.                                                                                        |
| 5   | §14.2 + §14.3             | ~5× one k=20 run        | Upgrades "one seed was near zero" to a sampling distribution, with honest family-wise correction.                                        |
| 6   | §17 minimal pairs         | ~5 min for n=100        | Highest conceptual payoff: separates "methods unfaithful" from "harness underpowered".                                                   |
| 7   | §15.4, §17.5              | —                       | Deferred by design.                                                                                                                      |


---



## Appendix A — Pipeline-to-Section Map

For navigating back to source when writing:


| Manuscript section                         | Script(s)                                                  | Output directory                                            |
| ------------------------------------------ | ---------------------------------------------------------- | ----------------------------------------------------------- |
| §1 Data                                    | `core/1_extract.py`, `data/three_way_split_indices.json`   | `activations/`                                              |
| §4.1 Linear probe                          | `core/2_probes.py`                                         | `checkpoints/probe_layer_7.joblib`                          |
| §5 Feature mask                            | `core/3_shap_comp.py`                                      | `3_shap/shap_feature_indices.npy`                           |
| §6.2 SHAP (linear)                         | `global_linear/7_shap_recompute.py`                        | `7_shap_recompute/`                                         |
| §6.4–6.6, §7 (linear)                      | `global_linear/8_feature_ranking.py`                       | `8_rankings_recompute/`                                     |
| §4.2 MLP probe                             | `global_mlp/11_mlp_probe.py`                               | `11_mlp_probe/`, `checkpoints/mlp_probe_layer_7.joblib`     |
| §6.3 SHAP (MLP)                            | `global_mlp/12_mlp_shap.py`                                | `12_mlp_shap/`                                              |
| §6.4–6.5, §7 (MLP)                         | `global_mlp/13_mlp_ranking.py`                             | `13_mlp_ranking/`                                           |
| §8.3 Alpha calibration                     | `global_linear/9.5_tuning.py`                              | `9.5_tuning/`, `checkpoints/residual_probe_layer_11.joblib` |
| §8 Steering/ablation (linear)              | `global_linear/9_steer.py`                                 | `9_steer/`                                                  |
| §8 Steering/ablation (MLP)                 | `global_mlp/14_mlp_steer.py`                               | `14_mlp_steer/`                                             |
| §9 Local (linear)                          | `local/9_local_steer.py`                                   | `9_local_steer/`                                            |
| §9 Local (MLP)                             | `local/14_local_mlp_steer.py`                              | `14_local_mlp_steer/`                                       |
| §9 Local (diagnostic, no activity control) | `local/local_shap_faithfulness.py`                         | `local_shap_faithfulness/`                                  |
| §10 Group steering                         | `global_group/group_steering.py`, `plot_group_steering.py` | `group_steering/`                                           |
| §11 Lexical simplicity                     | `global_linear/10_simplicity_check.py`                     | `10_simplicity_check/`                                      |


Follow-up experiments (§14–§18):


| Manuscript section                 | Script(s)                                                        | Output directory                                  |
| ---------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------- |
| §14.1 Higher-power run             | `9_steer.py --selection all`, `14_mlp_steer.py --selection all`  | `9_steer/`, `14_mlp_steer/` (tag `*_all_k116`)    |
| §14.2 Seed replication             | `global_linear/9a_seed_sweep.py`, `global_mlp/14a_seed_sweep.py` | `9_seed_sweep/`, `14_mlp_seed_sweep/`             |
| §14.3 FDR correction               | `utils.benjamini_hochberg`, `--bh-correction`                    | `faithfulness_<tag>_bh.json` sidecars             |
| §15.1 Sign reanalysis (free)       | `global_linear/9c_sign_reanalysis.py`                            | `sign_test_<tag>.{json,txt}` beside each input    |
| §15.2 Sign-stratified run          | `9_steer.py --selection top-pos/top-neg` (+ MLP mirror)          | tags `steering_toppos_k20`, `steering_topneg_k20` |
| §15.3 Readout-matched pilot        | `9.5_tuning.py --readout logitdiff`                              | `9.5_tuning/alpha_calibration_logitdiff.txt`      |
| §16 Activity-confound stats (free) | `local/analyze_local_vs_global.py`                               | `paired_tests.{json,txt}` in each local dir       |
| §16.4 Retrofit                     | `local/local_shap_faithfulness.py` (block [A]/[B])               | `local_shap_faithfulness/`                        |
| §17 Minimal pairs                  | `groundtruth/24_minimal_pairs.py`                                | `24_minimal_pairs/`                               |


Designed but not implemented: §15.4 (SAE-reconstruction control) and §17.5
(synthetic-label probe). Specs are in those sections.

Deprecated, do not cite as methodology: `global_linear/6_shap_stability.py`
(KernelSHAP seed-stability check; the KernelSHAP path it measured was
removed from `3_shap_comp.py` and has no downstream readers).