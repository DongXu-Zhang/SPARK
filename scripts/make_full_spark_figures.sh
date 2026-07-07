#!/usr/bin/env bash
set -euo pipefail

# Make post-SPARK figures and Markdown stats for one model.
#
# Run this only after data/spark_full/<model>__<domain>.jsonl exists.
#
# Usage:
#   bash scripts/make_full_spark_figures.sh Qwen3-4B
#
# Environment overrides:
#   RAW_DIR=/home/.../data/raw
#   CALIBRATION_DIR=/home/.../data/calibrated_full
#   SPARK_DIR=/home/.../data/spark_full
#   FIG_DIR=/home/.../figures/full_Qwen3-4B

MODEL_TAG="${1:-Qwen3-4B}"

RAW_DIR="${RAW_DIR:-/home/zhangdongxu/AAAI2027/data/raw}"
CALIBRATION_DIR="${CALIBRATION_DIR:-/home/zhangdongxu/AAAI2027/data/calibrated_full}"
SPARK_DIR="${SPARK_DIR:-/home/zhangdongxu/AAAI2027/data/spark_full}"
FIG_DIR="${FIG_DIR:-/home/zhangdongxu/AAAI2027/figures/full_${MODEL_TAG}}"
DOMAINS=(symbolic logical algorithmic)

mkdir -p "${FIG_DIR}"

echo "[figures] model:          ${MODEL_TAG}"
echo "[figures] raw dir:        ${RAW_DIR}"
echo "[figures] calibration dir:${CALIBRATION_DIR}"
echo "[figures] spark dir:      ${SPARK_DIR}"
echo "[figures] fig dir:        ${FIG_DIR}"

python scripts/plot_chi_vs_difficulty.py \
  --multi \
  --model "${MODEL_TAG}" \
  --problems_dir "${RAW_DIR}" \
  --calibration_dir "${CALIBRATION_DIR}" \
  --spark_dir "${SPARK_DIR}" \
  --out_dir "${FIG_DIR}" \
  --d_field d_structural \
  --score_field chi_max \
  --n_bins 10 \
  --formats png

python scripts/analyze_chi_correctness.py \
  --multi \
  --model "${MODEL_TAG}" \
  --problems_dir "${RAW_DIR}" \
  --calibration_dir "${CALIBRATION_DIR}" \
  --spark_dir "${SPARK_DIR}" \
  --out_md "${FIG_DIR}/chi_correctness__${MODEL_TAG}.md" \
  --d_field d_structural \
  --score_field chi_max

for DOMAIN in "${DOMAINS[@]}"; do
  python scripts/plot_layer_spectrum.py \
    --problems "${RAW_DIR}/${DOMAIN}.jsonl" \
    --calibration "${CALIBRATION_DIR}/${MODEL_TAG}__${DOMAIN}.jsonl" \
    --spark "${SPARK_DIR}/${MODEL_TAG}__${DOMAIN}.jsonl" \
    --group_by correctness \
    --out_stem "${FIG_DIR}/layer_spectrum__${MODEL_TAG}__${DOMAIN}" \
    --formats png

  python scripts/plot_layer_spectrum.py \
    --problems "${RAW_DIR}/${DOMAIN}.jsonl" \
    --calibration "${CALIBRATION_DIR}/${MODEL_TAG}__${DOMAIN}.jsonl" \
    --spark "${SPARK_DIR}/${MODEL_TAG}__${DOMAIN}.jsonl" \
    --group_by d_bin \
    --d_field d_structural \
    --n_bins 4 \
    --out_stem "${FIG_DIR}/layer_spectrum_by_d__${MODEL_TAG}__${DOMAIN}" \
    --formats png
done

python scripts/analyze_length_control.py \
  --model "${MODEL_TAG}" \
  --raw_dir "${RAW_DIR}" \
  --calibration_dir "${CALIBRATION_DIR}" \
  --spark_dir "${SPARK_DIR}" \
  --out_dir "figures/length_control_${MODEL_TAG}" \
  --d_field d_structural \
  --score_field chi_max

echo "[figures] all done -> ${FIG_DIR}"
echo "[figures] length-control report -> figures/length_control_${MODEL_TAG}/"

