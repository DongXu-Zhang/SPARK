#!/usr/bin/env bash
# Llama MATH-500 v2: re-extract direction (deep layer) + cal-aligned eval on GPU 7.
#
# Fixes vs previous official_hf_v2 run:
#   - direction layer: auto with min_layer=3 (avoids layer-1 chi artifact → layer 32)
#   - eval profile llama_v2: greedy + system prompt + rep_penalty 1.05 (match vLLM cal)
#   - presence_penalty=0 (pp=1.5 crushed Llama to ~10%)
#   - steering position=last (safer than all-token injection)
#
# Usage:
#   bash scripts/run_llama_math500_v2.sh
#   CUDA_VISIBLE_DEVICES=7 TEST_LIMIT=20 bash scripts/run_llama_math500_v2.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

MODEL="${MODEL:-Llama-3.1-8B}"
export MODEL
export MODELS_DIR="${MODELS_DIR:-${ROOT}/models}"
export MODEL_PATH="${MODEL_PATH:-${MODELS_DIR}/${MODEL}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export DEVICE_MAP="${DEVICE_MAP:-auto}"

DOMAIN="math500_train"
STEER_PREFIX="data/steering/${MODEL}__${DOMAIN}"
DIRECTION_V2="${STEER_PREFIX}__critical_direction__v2.pt"
OUT_V2="data/steering/${MODEL}__math500_test__steering_eval__v2_cal_aligned.jsonl"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/math500_v2__${MODEL}.nohup.log}"

mkdir -p "${LOG_DIR}" data/steering

if [[ -f "${ROOT}/scripts/env_conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/scripts/env_conda.sh"
fi

echo "[llama-v2] GPU=${CUDA_VISIBLE_DEVICES} MODEL=${MODEL}"
echo "[llama-v2] log -> ${LOG_FILE}"

nohup env PYTHONUNBUFFERED=1 \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  bash -c "
    set -euo pipefail
    cd \"${ROOT}\"
    source scripts/env_conda.sh 2>/dev/null || true

    echo \"[\$(date -Iseconds)] ===== extract direction v2 (layer auto, min_layer=3) =====\"
    python scripts/extract_critical_direction.py \\
      --model_path \"${MODEL_PATH}\" \\
      --demos \"${STEER_PREFIX}__active_demos.jsonl\" \\
      --anchors \"${STEER_PREFIX}__active_anchors.jsonl\" \\
      --out \"${DIRECTION_V2}\" \\
      --layer auto \\
      --min_layer 3 \\
      --device_map \"${DEVICE_MAP}\"

    echo \"[\$(date -Iseconds)] ===== test eval v2 (llama_v2 profile) =====\"
    export MODEL=\"${MODEL}\"
    export MODEL_PATH=\"${MODEL_PATH}\"
    export DIRECTION=\"${DIRECTION_V2}\"
    export OUT=\"${OUT_V2}\"
    export EVAL_PROFILE=llama_v2
    export STEERING_POSITION=last
    export PRESENCE_PENALTY=0
    export MAX_NEW_TOKENS=16384
    export RESUME=1
    export ENABLE_THINKING=0
    export ALPHAS=\"0.0 0.25 0.5 1.0\"
    export TEST_LIMIT=${TEST_LIMIT:-}
    bash scripts/run_math500_test_eval.sh

    echo \"[\$(date -Iseconds)] ===== llama v2 done =====\"
    echo \"  direction: ${DIRECTION_V2}\"
    echo \"  results:   ${OUT_V2}\"
  " >>"${LOG_FILE}" 2>&1 &

echo "[llama-v2] PID=$!  tail -f ${LOG_FILE}"
