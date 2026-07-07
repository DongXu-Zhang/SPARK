#!/usr/bin/env bash
# Benchmark pipeline: download -> generate -> calibrate -> fuse difficulty.
# SPARK / χ_LC / steering are separate (see docs/PIPELINE.md, docs/SPARK_STEERING_NEXT.md).
#
# Prereq:
#   export MODELS_DIR=/data/models
#
# Usage:
#   bash scripts/run_full_pipeline.sh
#   POOL=4card bash scripts/run_full_pipeline.sh

set -euo pipefail

if [[ -z "${MODELS_DIR:-}" ]]; then
  echo "ERROR: please set MODELS_DIR first, e.g.:"
  echo "    export MODELS_DIR=/data/models"
  exit 1
fi

echo "[pipeline] step 0/4: download models from ModelScope (skips ones already present)"
bash scripts/download_models.sh

echo "[pipeline] step 1/4: sanity check"
python scripts/sanity_check.py

echo "[pipeline] step 2/4: generate 4500 candidates"
python scripts/generate_all.py --n_per_domain 1500 --out_dir data/raw

echo "[pipeline] step 3/4: calibrate with Qwen models"
bash scripts/run_calibration.sh

echo "[pipeline] step 4/4: fuse difficulty + stratified sample"
python -m src.calibration.difficulty \
    --problems data/raw/all.jsonl \
    --eval_dir data/calibrated \
    --out_dir data/final \
    --n_per_bin 100

echo "[pipeline] done. Final stratified set: data/final/frontier_1500.jsonl"
echo "  next: compute_spark.py → run_spark_steering_prep.sh (see docs/PIPELINE.md)"
