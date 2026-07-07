#!/usr/bin/env bash
# SPARK-Steering train prep for one FRONTIER domain (symbolic | logical | algorithmic).
#
# Prerequisites (GPU, run once per domain if missing):
#   data/calibrated_full/${MODEL}__${DOMAIN}.jsonl
#   data/spark_full/${MODEL}__${DOMAIN}.jsonl
#
# This script (CPU + one GPU step for direction):
#   1) export length-controlled χ_LC records (skip if present)
#   2) select steering demos / anchors / held-out test
#   3) extract critical_direction.pt
#
# Usage:
#   bash scripts/run_frontier_domain_steering_train.sh symbolic
#   SKIP_EXTRACT=1 bash scripts/run_frontier_domain_steering_train.sh logical

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -f "${ROOT}/scripts/env_conda.sh" ]]; then
  # shellcheck source=scripts/env_conda.sh
  source "${ROOT}/scripts/env_conda.sh"
fi

DOMAIN="${1:-}"
if [[ -z "${DOMAIN}" ]]; then
  echo "Usage: bash scripts/run_frontier_domain_steering_train.sh <symbolic|logical|algorithmic>" >&2
  exit 1
fi
case "${DOMAIN}" in
  symbolic|logical|algorithmic) ;;
  *)
    echo "Unknown domain: ${DOMAIN}" >&2
    exit 1
    ;;
esac

MODEL="${MODEL:-Qwen3-4B}"
MODEL_PATH="${MODEL_PATH:-${ROOT}/models/${MODEL}}"
RAW_DIR="${RAW_DIR:-data/raw}"
CALIBRATION_DIR="${CALIBRATION_DIR:-data/calibrated_full}"
SPARK_DIR="${SPARK_DIR:-data/spark_full}"
ANALYSIS_DIR="${ANALYSIS_DIR:-data/analysis}"
STEERING_DIR="${STEERING_DIR:-data/steering}"

CAL_FILE="${CALIBRATION_DIR}/${MODEL}__${DOMAIN}.jsonl"
SPARK_FILE="${SPARK_DIR}/${MODEL}__${DOMAIN}.jsonl"
LC_FILE="${ANALYSIS_DIR}/${MODEL}__${DOMAIN}__length_controlled.jsonl"
PREFIX="${STEERING_DIR}/${MODEL}__${DOMAIN}"
DIRECTION="${PREFIX}__critical_direction.pt"

echo "[frontier-train] model=${MODEL} domain=${DOMAIN}"

if [[ ! -f "${CAL_FILE}" ]]; then
  echo "[frontier-train] missing calibration: ${CAL_FILE}" >&2
  echo "  Run: python -m src.calibration.run_eval --model ${MODEL} --input data/raw/${DOMAIN}.jsonl --out_dir ${CALIBRATION_DIR} --tag ${DOMAIN}" >&2
  exit 1
fi
if [[ ! -f "${SPARK_FILE}" ]]; then
  echo "[frontier-train] missing SPARK: ${SPARK_FILE}" >&2
  echo "  Run: python scripts/compute_spark.py --model ${MODEL_PATH} --input data/raw/${DOMAIN}.jsonl --out ${SPARK_FILE} --resume" >&2
  exit 1
fi

if [[ ! -f "${LC_FILE}" ]]; then
  echo "[frontier-train] exporting length-controlled records..."
  python scripts/export_length_controlled_records.py \
    --model "${MODEL}" \
    --domains "${DOMAIN}" \
    --raw_dir "${RAW_DIR}" \
    --calibration_dir "${CALIBRATION_DIR}" \
    --spark_dir "${SPARK_DIR}" \
    --out_dir "${ANALYSIS_DIR}" \
    --score_field chi_max \
    --d_field d_structural
else
  echo "[frontier-train] length-controlled exists: ${LC_FILE}"
fi

echo "[frontier-train] selecting steering sets..."
python scripts/select_steering_sets.py \
  --model "${MODEL}" \
  --domain "${DOMAIN}" \
  --analysis_dir "${ANALYSIS_DIR}" \
  --out_dir "${STEERING_DIR}" \
  --active_d_min "${ACTIVE_D_MIN:-0.30}" \
  --active_d_max "${ACTIVE_D_MAX:-0.55}" \
  --target_d_min "${TARGET_D_MIN:-0.55}" \
  --target_d_max "${TARGET_D_MAX:-0.80}" \
  --n_demos "${N_DEMOS:-32}" \
  --n_anchors "${N_ANCHORS:-64}" \
  --n_targets "${N_TARGETS:-160}" \
  --n_test "${N_TEST:-240}"

if [[ "${SKIP_EXTRACT:-0}" == "1" ]]; then
  echo "[frontier-train] SKIP_EXTRACT=1, done (no direction)."
  exit 0
fi

if [[ -f "${DIRECTION}" && "${FORCE_EXTRACT:-0}" != "1" ]]; then
  echo "[frontier-train] direction exists: ${DIRECTION} (set FORCE_EXTRACT=1 to redo)"
  exit 0
fi

echo "[frontier-train] extracting critical direction (GPU)..."
python scripts/extract_critical_direction.py \
  --model_path "${MODEL_PATH}" \
  --demos "${PREFIX}__active_demos.jsonl" \
  --anchors "${PREFIX}__active_anchors.jsonl" \
  --out "${DIRECTION}" \
  --layer auto \
  --dtype bfloat16 \
  --device_map auto

echo "[frontier-train] done."
echo "  heldout_test: ${PREFIX}__heldout_test.jsonl"
echo "  direction:    ${DIRECTION}"
echo "Next: bash scripts/run_frontier_domain_steering_eval.sh ${DOMAIN}"
