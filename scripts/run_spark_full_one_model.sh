#!/usr/bin/env bash
set -euo pipefail

# Compute chi / Phi / SPK for the three full raw datasets with one HF model.
#
# Usage:
#   bash scripts/run_spark_full_one_model.sh Qwen3-4B
#   bash scripts/run_spark_full_one_model.sh Qwen3-4B /path/to/local/Qwen3-4B
#
# Environment overrides:
#   RAW_DIR=/home/.../data/raw
#   OUT_DIR=/home/.../data/spark_full
#   LOG_DIR=/home/.../logs
#   CONFIG=configs/spark.yaml
#   DTYPE=bfloat16
#   DEVICE_MAP=auto
#   FORCE=1

MODEL_TAG="${1:-Qwen3-4B}"
DEFAULT_MODELS_DIR="${MODELS_DIR:-/home/zhangdongxu/AAAI2027/models}"
MODEL_PATH="${2:-${DEFAULT_MODELS_DIR}/${MODEL_TAG}}"

RAW_DIR="${RAW_DIR:-/home/zhangdongxu/AAAI2027/data/raw}"
OUT_DIR="${OUT_DIR:-/home/zhangdongxu/AAAI2027/data/spark_full}"
LOG_DIR="${LOG_DIR:-/home/zhangdongxu/AAAI2027/logs}"
CONFIG="${CONFIG:-configs/spark.yaml}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
FORCE="${FORCE:-0}"
LOG_EVERY="${LOG_EVERY:-1}"

DOMAINS=(symbolic logical algorithmic)

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

echo "[spark-full] model tag:  ${MODEL_TAG}"
echo "[spark-full] model path: ${MODEL_PATH}"
echo "[spark-full] raw dir:    ${RAW_DIR}"
echo "[spark-full] out dir:    ${OUT_DIR}"
echo "[spark-full] config:     ${CONFIG}"
echo "[spark-full] dtype:      ${DTYPE}"
echo "[spark-full] device_map: ${DEVICE_MAP}"
echo "[spark-full] force:      ${FORCE}"
echo "[spark-full] log_every:  ${LOG_EVERY}"

for DOMAIN in "${DOMAINS[@]}"; do
  INPUT="${RAW_DIR}/${DOMAIN}.jsonl"
  OUTPUT="${OUT_DIR}/${MODEL_TAG}__${DOMAIN}.jsonl"
  DOMAIN_LOG="${LOG_DIR}/spark_full_${MODEL_TAG}__${DOMAIN}.log"

  if [[ ! -f "${INPUT}" ]]; then
    echo "[spark-full] ERROR: missing input ${INPUT}" >&2
    exit 1
  fi

  if [[ -f "${OUTPUT}" && "${FORCE}" != "1" ]]; then
    echo "[spark-full] skip ${DOMAIN}: ${OUTPUT} exists (set FORCE=1 to rerun)"
    continue
  fi

  echo "[spark-full] start ${DOMAIN}"
  python -u scripts/compute_spark.py \
    --model "${MODEL_PATH}" \
    --input "${INPUT}" \
    --out "${OUTPUT}" \
    --config "${CONFIG}" \
    --dtype "${DTYPE}" \
    --device_map "${DEVICE_MAP}" \
    --resume \
    --log_every "${LOG_EVERY}" \
    2>&1 | tee "${DOMAIN_LOG}"
  echo "[spark-full] done ${DOMAIN}: ${OUTPUT}"
done

echo "[spark-full] all done"
