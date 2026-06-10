#!/usr/bin/env bash
# MATH-500 test eval (500 x α grid) for any model with trained critical direction.
#
# Usage:
#   MODEL=Llama-3.1-8B CUDA_VISIBLE_DEVICES=0 bash scripts/run_math500_model_test_nohup.sh
#   MODEL=deepseek-R1 ENABLE_THINKING=1 CUDA_VISIBLE_DEVICES=1 bash scripts/run_math500_model_test_nohup.sh
#   TEST_LIMIT=20 MODEL=Llama-3.1-8B bash scripts/run_math500_model_test_nohup.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

MODEL="${MODEL:?set MODEL}"
export MODEL
export MODELS_DIR="${MODELS_DIR:-${ROOT}/models}"
export MODEL_PATH="${MODEL_PATH:-${MODELS_DIR}/${MODEL}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DEVICE_MAP="${DEVICE_MAP:-auto}"
export ENABLE_THINKING="${ENABLE_THINKING:-0}"
export TEST_LIMIT="${TEST_LIMIT:-}"
# Qwen3 tech-report non-thinking MATH-500 (Table 18/20): 32768, presence_penalty=1.5, etc.
# Non-Qwen models (Llama/DeepSeek) should NOT inherit pp=1.5 — use EVAL_PROFILE instead.
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
export PRESENCE_PENALTY="${PRESENCE_PENALTY:-}"
export EVAL_PROFILE="${EVAL_PROFILE:-}"
export STEERING_POSITION="${STEERING_POSITION:-all}"
export DIRECTION="${DIRECTION:-data/steering/${MODEL}__math500_train__critical_direction.pt}"

if [[ -z "${EVAL_PROFILE}" ]]; then
  case "${MODEL}" in
    Qwen3-*)
      EVAL_PROFILE="${EVAL_PROFILE:-qwen_official}"
      MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
      PRESENCE_PENALTY="${PRESENCE_PENALTY:-1.5}"
      OUT_SUFFIX="${OUT_SUFFIX:-official_hf_v2}"
      ;;
    Llama-*)
      EVAL_PROFILE="${EVAL_PROFILE:-llama_v2}"
      MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"
      PRESENCE_PENALTY="${PRESENCE_PENALTY:-0}"
      STEERING_POSITION="${STEERING_POSITION:-last}"
      OUT_SUFFIX="${OUT_SUFFIX:-v2_cal_aligned}"
      ;;
    *)
      EVAL_PROFILE="${EVAL_PROFILE:-default}"
      MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"
      PRESENCE_PENALTY="${PRESENCE_PENALTY:-0}"
      OUT_SUFFIX="${OUT_SUFFIX:-steering_eval}"
      ;;
  esac
fi
export EVAL_PROFILE
export RESUME="${RESUME:-1}"
export MAX_RAW_CHARS="${MAX_RAW_CHARS:-8000}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
OUT_SUFFIX="${OUT_SUFFIX:-official_hf_v2}"
export OUT="${OUT:-data/steering/${MODEL}__math500_test__steering_eval__${OUT_SUFFIX}.jsonl}"

if [[ ! -f "${DIRECTION}" ]]; then
  echo "[math500-test] missing ${DIRECTION}; run train first" >&2
  exit 1
fi

LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/math500_${OUT_SUFFIX}__${MODEL}.nohup.log}"
mkdir -p "${LOG_DIR}"

echo "[math500-test] MODEL=${MODEL} GPU=${CUDA_VISIBLE_DEVICES} ENABLE_THINKING=${ENABLE_THINKING}"
echo "[math500-test] profile=${EVAL_PROFILE} max_new=${MAX_NEW_TOKENS:-auto} presence=${PRESENCE_PENALTY:-auto} position=${STEERING_POSITION}"
echo "[math500-test] OUT=${OUT} DIRECTION=${DIRECTION} RESUME=${RESUME}"
echo "[math500-test] log -> ${LOG_FILE}"

nohup env PYTHONUNBUFFERED=1 \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  DEVICE_MAP="${DEVICE_MAP}" \
  MODEL="${MODEL}" \
  MODEL_PATH="${MODEL_PATH}" \
  ENABLE_THINKING="${ENABLE_THINKING}" \
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
  PRESENCE_PENALTY="${PRESENCE_PENALTY}" \
  EVAL_PROFILE="${EVAL_PROFILE}" \
  STEERING_POSITION="${STEERING_POSITION}" \
  DIRECTION="${DIRECTION}" \
  RESUME="${RESUME}" \
  MAX_RAW_CHARS="${MAX_RAW_CHARS}" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
  OUT="${OUT}" \
  bash -c "
    set -euo pipefail
    cd \"${ROOT}\"
    source scripts/env_conda.sh 2>/dev/null || true
    export MODEL=\"${MODEL}\"
    export MODEL_PATH=\"${MODEL_PATH}\"
    export ENABLE_THINKING=\"${ENABLE_THINKING}\"
    export DEVICE_MAP=\"${DEVICE_MAP}\"
    export MAX_NEW_TOKENS=\"${MAX_NEW_TOKENS}\"
    export PRESENCE_PENALTY=\"${PRESENCE_PENALTY}\"
    export EVAL_PROFILE=\"${EVAL_PROFILE}\"
    export STEERING_POSITION=\"${STEERING_POSITION}\"
    export DIRECTION=\"${DIRECTION}\"
    export RESUME=\"${RESUME}\"
    export MAX_RAW_CHARS=\"${MAX_RAW_CHARS}\"
    export PYTORCH_CUDA_ALLOC_CONF=\"${PYTORCH_CUDA_ALLOC_CONF}\"
    export OUT=\"${OUT}\"
    export TEST_LIMIT=${TEST_LIMIT}
    echo \"[\$(date -Iseconds)] [${MODEL}] test eval start\"
    bash scripts/run_math500_test_eval.sh
    echo \"[\$(date -Iseconds)] [${MODEL}] test eval done\"
  " >>"${LOG_FILE}" 2>&1 &

echo "[math500-test] PID=$!  tail -f ${LOG_FILE}"
