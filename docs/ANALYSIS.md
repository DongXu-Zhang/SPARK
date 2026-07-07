# Stage-3 analysis tools

`src/analysis/` + CLI scripts under `scripts/` produce paper figures and stats from
JSONL artifacts. Everything listed here is **CPU-only** — no GPU required.

---

## What each script does

| Script | Purpose | Paper role |
|---|---|---|
| `plot_chi_vs_difficulty.py` | χ scatter + binned mean vs difficulty | Raw χ vs **d** (diagnostic; motivates length control) |
| `plot_layer_spectrum.py` | χ across layers, grouped by correctness or d-bin | Φ / layer-wise χ morphology |
| `analyze_chi_correctness.py` | Spearman ρ, Cohen's d, ROC AUC | Correlation / separability tables |
| `analyze_length_control.py` | `log χ ~ log T` fit, **χ_LC**, ρ(χ_LC, d), bin tables, plots | Length-controlled susceptibility (core) |
| `export_length_controlled_records.py` | Per-row χ_LC + residuals for downstream steering | Join table for SPARK-Steering |
| `matched_length_frontier_test.py` | Matched-length hard vs easy χ_LC contrast | Under-activation vs length confound |
| `select_steering_sets.py` | Anchor / target / held-out JSONL for steering | Sample construction |

---

## Quick start (multi-domain, one model)

```bash
cd ~/AAAI2027

# Generate the bell-curve figure for all three domains
python scripts/plot_chi_vs_difficulty.py \
    --multi --model Qwen3-4B \
    --problems_dir data/sanity \
    --calibration_dir data/calibrated \
    --spark_dir data/spark \
    --out_dir data/figures \
    --d_field d_structural \
    --score_field chi_max

# Layer-wise χ spectrum, grouped by correctness
python scripts/plot_layer_spectrum.py \
    --problems data/sanity/symbolic.jsonl \
    --calibration data/calibrated/Qwen3-4B__symbolic.jsonl \
    --spark data/spark/Qwen3-4B__symbolic.jsonl \
    --group_by correctness \
    --out_stem data/figures/layer_spectrum__Qwen3-4B__symbolic

# Stats table (markdown)
python scripts/analyze_chi_correctness.py \
    --multi --model Qwen3-4B \
    --problems_dir data/sanity \
    --calibration_dir data/calibrated \
    --spark_dir data/spark \
    --out_md data/figures/chi_correctness.md
```

---

## What goes where

| Path | Meaning |
|---|---|
| `data/sanity/<domain>.jsonl` | the problem set (id, prompt, ground_truth, d_structural) |
| `data/calibrated/<model>__<domain>.jsonl` | per-problem `correct` flag from `run_eval.py` |
| `data/spark/<model>__<domain>.jsonl` | per-problem χ / Φ / SPK from `compute_spark.py` |
| `data/figures/` | all PNGs / PDFs produced by stage-3 scripts |

---

## Field conventions

* `--d_field` chooses the x-axis difficulty:
  * `d_structural` — generator-defined (always present)
  * `d_empirical` — from multi-model pass rate (only after full calibration)
  * `d_final` — rank-normalised fusion (only in `data/final/`)
* `--score_field` chooses the y-axis χ aggregator:
  * `chi_max` — the per-layer maximum (default)
  * `chi_mean` — per-layer mean
  * `spk` — composite SPARK index

---

## Reading the χ-vs-difficulty figure

* **Scatter (blue + red x)**: every problem is a dot; red = wrong, blue = correct.
* **Black error-bar curve**: binned mean ± standard error.
* Raw χ often co-varies with prompt length on algorithmic tasks — interpret together
  with `analyze_length_control.py` and χ_LC.
* **Annotation block**: Spearman ρ, Cohen's d, ROC AUC.

A healthy SPARK signature looks like a bell shape that **peaks near where
correct/wrong intermix** (the model's capability boundary). With only 30
problems per domain you will not see a clean bell — that needs FRONTIER-1.5K.

---

## Reading the layer spectrum

* y-axis is **log-scaled** because χ varies several orders of magnitude
  across layers.
* SPARK theory predicts the χ peak (the "ignition point") sits in the
  **middle** of the network, not at L=0 or L=last.
* A consistent peak position across difficulty bins is the *universality*
  signature.

---

## Files to upload to the server

| Path | Why |
|---|---|
| `src/analysis/` | the helper module |
| `scripts/plot_chi_vs_difficulty.py` | bell curve |
| `scripts/plot_layer_spectrum.py` | layer spectrum |
| `scripts/analyze_chi_correctness.py` | stats table |
| `docs/ANALYSIS.md` | this file |

`tests/` is for local sanity checks only and need not be uploaded.
