# End-to-end pipeline reference

Artifacts chain through `data/` (gitignored). Logs go to `logs/`.

This repo is aligned with the **SPARK + length control + steering** paper line:

1. Programmatic tasks (three domains).
2. Model inference → accuracy vs difficulty → empirical boundary **d\*** (where applicable).
3. Forward-only **χ**, **Φ**, **SPK** on prompts.
4. **Length control**: fit `log χ ~ log T`, residual **χ_LC** separates input-scale from reasoning-state.
5. Select anchors / targets → critical direction → **activation steering** → eval.

---

## Stage diagram (benchmark + SPARK)

```
                 ┌──────────────────┐
                 │ sanity_check.py  │  CPU — verify generators
                 └────────┬─────────┘
                          ▼
                 data/sanity/*.jsonl
                          │
                 ┌────────┴─────────┐
                 │ generate_all.py  │  CPU — 1500 / domain
                 └────────┬─────────┘
                          ▼
           data/raw/{symbolic,logical,algorithmic,all}.jsonl
                          │
                 ┌────────┴─────────┐
                 │run_calibration.sh│  GPU — vLLM in-process
                 └────────┬─────────┘
                          ▼
              data/calibrated[_full]/<model>__<domain>.jsonl
                          │
          ┌───────────────┴────────────────┐
          │                                │
 ┌────────┴─────────┐              ┌─────┴──────┐
 │ difficulty.py    │              │compute_spark│  GPU — χ / Φ / SPK
 │ (optional fuse)  │              └─────┬──────┘
 └────────┬─────────┘                     ▼
          ▼                    data/spark[_full]/<model>__<domain>.jsonl
 data/final/frontier_1500.jsonl            │
 (stratified 1500)                          │
                                            ▼
                              ┌─────────────────────────┐
                              │ export_length_controlled│  CPU — attach χ_LC
                              │ analyze_length_control   │  CPU — ρ, R², bins, plots
                              └────────────┬────────────┘
                                           ▼
                              data/analysis/…length_controlled…
                                           │
                              ┌────────────┴────────────┐
                              │ select_steering_sets    │
                              │ extract_critical_direction │
                              │ eval_spark_steering     │  GPU
                              └─────────────────────────┘
```

For paper figures on raw χ vs **d** and layer spectra, see `docs/ANALYSIS.md` and
`scripts/make_full_spark_figures.sh`.

---

## Commands (dataset + SPARK)

Bundled dataset path: `scripts/run_full_pipeline.sh` (sanity → generate → calibrate → fuse).

```bash
# 1. sanity (CPU)
python scripts/sanity_check.py

# 2. generate 4500 candidates (CPU)
python scripts/generate_all.py --n_per_domain 1500

# 3. run calibration models (GPU)
bash scripts/run_calibration.sh

# 4. fuse difficulty + sample 1500 (CPU)
python -m src.calibration.difficulty \
    --problems data/raw/all.jsonl \
    --eval_dir data/calibrated \
    --out_dir data/final \
    --n_per_bin 100
```

SPARK + χ_LC + steering prep (after `data/spark_full/…` exists):

```bash
bash scripts/run_spark_steering_prep.sh Qwen3-4B algorithmic
```

Steering eval and summaries: `docs/SPARK_STEERING_NEXT.md`, `scripts/run_frontier_domain_steering_eval.sh`.

---

## Resumability

- `run_calibration.sh` skips any model whose output JSONL is non-empty.
- `difficulty.py` is idempotent — rerunning rebuilds the final dataset from
  whatever model files exist in `data/calibrated/`.

---

## Recommended ordering when bringing up a new server

1. CPU-only: `sanity_check.py` (verifies generators).
2. GPU smoke: short `run_eval.py` / `eval_one_model.sh` with `--limit 10`.
3. Full calibration: `run_calibration.sh`.
4. `compute_spark.py` on sanity or full splits.
5. Length-control export + `analyze_length_control.py` before interpreting χ.
