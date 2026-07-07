# FRONTIER-1.5K: A Procedurally-Generated Multi-Difficulty Reasoning Benchmark

This repository builds **FRONTIER-1.5K**, a 1500-problem reasoning benchmark, and the **SPARK**
analysis stack: latent susceptibility **χ**, hierarchical coherence **Φ**, length-controlled
**χ_LC**, and **SPARK-Steering** (activation intervention on under-activated hard prompts).

The dataset spans 3 reasoning families (symbolic / logical / algorithmic), each with continuous
difficulty calibration via a **within-Qwen-family pool** of 7 models (extensible to 9 on 4-GPU rigs).

---

## Quick start

```bash
# 1. install (Linux server with CUDA 12.1+)
pip install -r requirements.txt

# 2. sanity check (CPU-only, ~30 problems, < 1 min)
python scripts/sanity_check.py

# 3. generate full 4500-problem candidate pool (CPU-only)
python scripts/generate_all.py --n_per_domain 1500 --out_dir data/raw

# 4. run calibration (GPU server)
bash scripts/run_calibration.sh

# 5. fuse difficulty + stratified sample to 1500
python -m src.calibration.difficulty \
    --problems data/raw/all.jsonl \
    --eval_dir data/calibrated \
    --out_dir data/final \
    --n_per_bin 100

# 6. SPARK (χ / Φ / SPK) on a problem JSONL — see docs/SPARK.md
#    python scripts/compute_spark.py --model $MODELS_DIR/Qwen3-4B ...

# 7. length control + steering prep — see docs/SPARK_STEERING_NEXT.md
#    bash scripts/run_spark_steering_prep.sh Qwen3-4B algorithmic
```

---

## Repository layout

```
AAAI2027/
├── README.md                 # this file
├── requirements.txt
├── configs/
│   ├── default.yaml          # main config (paths, sampling, hyperparams)
│   └── models.yaml           # per-model launch parameters
├── src/
│   ├── generators/           # programmatic problem generators
│   ├── calibration/          # vLLM calibration, verify, difficulty fusion
│   ├── spark/                # χ, Φ, SPK (susceptibility + coherence)
│   ├── analysis/             # merge, stats, plotting, length_control (χ_LC)
│   ├── steering/             # activation steering hooks
│   └── utils/
├── scripts/                  # entry points + shell launchers
├── docs/
│   ├── SPARK.md, SPARK_STEERING_NEXT.md, ANALYSIS.md
│   ├── SETUP.md, MODELS.md, PIPELINE.md
├── data/                     # generated artifacts (gitignored)
└── logs/
```

---

## Hardware

- **Minimum**: 2× NVIDIA RTX 3090 (48 GB VRAM total). Calibration finishes in 2–3 days.
- **Recommended**: 4× RTX 3090 (96 GB VRAM total). Adds Qwen3-32B (bf16) + DeepSeek-R1-Distill-32B
  to the pool, making `d_e` more reliable. Calibration time: ~36 h.
- **CPU-only**: All problem generators (`src/generators/*`) and analysis scripts run on CPU.
  Calibration, SPARK (`compute_spark.py`), and steering eval need GPUs.

See `docs/MODELS.md` for the full per-model launch matrix.

---

## License

Code: MIT. Generated dataset: CC-BY 4.0.
