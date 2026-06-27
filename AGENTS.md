# Agent Guide

This project uses SHAP to interpret Sparse Autoencoder (SAE) features in GPT-2 Small for sentiment analysis. Read this before writing or modifying any code.

---

## Project Structure

```
shap-sae/
├── data/
│   └── sentiment/          # raw + processed SST-2 dataset
├── activations/            # saved SAE activation tensors (not in git)
├── checkpoints/            # probe weights, SAE fine-tune checkpoints
├── outputs/                # SHAP matrices, results, figures
├── notebooks/              # exploration and figure generation only
└── src/
    ├── extract.py          # activation extraction pipeline
    ├── probes.py           # probe training and evaluation
    ├── shap_pipeline.py    # KernelSHAP + DeepSHAP computation
    ├── validate.py         # ablation + steering experiments
    ├── benchmark.py        # comparative attribution methods
    └── utils.py            # shared helpers
```

---

## Stack

- **Model:** GPT-2 Small via `transformer_lens`
- **SAE:** `gpt2-small-res-jb` from `sae_lens`, `resid_post` hook points, layers 8-10, 32k features
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

# run a script
uv run python src/extract.py
```

Do not use `pip install` directly. If a new dependency is needed, add it with:
```bash
uv add <package>
```

---

## Key Conventions

**Activations are never stored in git.** They are large (5-8 GB) and regenerable. The `activations/` directory is in `.gitignore`. To regenerate:
```bash
uv run python src/extract.py
```

**Token position:** Use the last token position for sentence-level activation extraction. Do not change this without explicit instruction.

**SAE hook points:** Always use `resid_post` not `resid_pre` or `resid_mid`. Hook into layers 8, 9, and 10 only.

**Splits:** SST-2 is divided into three fixed splits with a set random seed:
- Probe train (70%)
- Probe val (15%)
- SHAP held-out (15%) — do not use this for anything except SHAP computation

**Probe regularization:** L1 penalty with `saga` solver. Default `C=1.0`, tuned across `[0.01, 0.1, 1.0, 10.0]` on the val split.

---


## What Not to Do

- Do not modify `data/sentiment/` splits or re-download the dataset with a different seed
- Do not run SHAP on the probe val split — only on the SHAP held-out split
- Do not store activations in float64 — use float16 to keep file sizes manageable
- Do not add new dependencies without checking with the user first
- Do not write exploration code into `src/` — that belongs in `notebooks/`
- Do not use `pip` — use `uv`