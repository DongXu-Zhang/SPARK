#!/usr/bin/env bash
set -euo pipefail

# Prepare all non-generation artifacts for SPARK-Steering.
# This script does not load the LLM and does not require GPU.
#
# Usage:
#   bash scripts/run_spark_steering_prep.sh Qwen3-4B algorithmic

MODEL="${1:-Qwen3-4B}"
DOMAIN="${2:-algorithmic}"

RAW_DIR="${RAW_DIR:-data/raw}"
CALIBRATION_DIR="${CALIBRATION_DIR:-data/calibrated_full}"
SPARK_DIR="${SPARK_DIR:-data/spark_full}"
ANALYSIS_DIR="${ANALYSIS_DIR:-data/analysis}"
STEERING_DIR="${STEERING_DIR:-data/steering}"
FIG_DIR="${FIG_DIR:-figures/matched_length_${MODEL}}"

mkdir -p "${ANALYSIS_DIR}" "${STEERING_DIR}" "${FIG_DIR}"

python scripts/export_length_controlled_records.py \
  --model "${MODEL}" \
  --domains symbolic logical algorithmic \
  --raw_dir "${RAW_DIR}" \
  --calibration_dir "${CALIBRATION_DIR}" \
  --spark_dir "${SPARK_DIR}" \
  --out_dir "${ANALYSIS_DIR}" \
  --score_field chi_max \
  --d_field d_structural

python scripts/matched_length_frontier_test.py \
  --model "${MODEL}" \
  --domain "${DOMAIN}" \
  --analysis_dir "${ANALYSIS_DIR}" \
  --out_dir "${FIG_DIR}" \
  --easy_d_min 0.30 \
  --easy_d_max 0.55 \
  --hard_d_min 0.55 \
  --hard_d_max 0.80 \
  --token_tol_frac 0.20 \
  --max_pairs 200

python scripts/select_steering_sets.py \
  --model "${MODEL}" \
  --domain "${DOMAIN}" \
  --analysis_dir "${ANALYSIS_DIR}" \
  --out_dir "${STEERING_DIR}" \
  --active_d_min 0.30 \
  --active_d_max 0.55 \
  --target_d_min 0.55 \
  --target_d_max 0.80 \
  --n_demos 32 \
  --n_anchors 64 \
  --n_targets 160 \
  --n_test 240

echo "[spark-steering-prep] done"
echo "  analysis: ${ANALYSIS_DIR}"
echo "  steering: ${STEERING_DIR}"
echo "  matched length report: ${FIG_DIR}"

