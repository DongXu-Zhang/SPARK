#!/usr/bin/env bash
# Evaluate steering on HuggingFaceH4 MATH-500 (500 problems).
#
# Defaults align with Qwen3 tech-report non-thinking MATH-500 (Table 18/20):
#   EvalScope prompt, model-aware system (0.6B user-only; 4B/8B DEFAULT_SYSTEM),
#   temp=0.7, top_p=0.8, top_k=20, presence_penalty=0 on HF,
#   max_new_tokens=32768, per-question seed, sympy grading.
#
# Official targets: 0.6B 55.2% | 4B 84.8% | 8B 87.4%
#
# Usage:
#   bash scripts/run_math500_test_eval.sh
#   TEST_LIMIT=50 MODEL=Qwen3-8B bash scripts/run_math500_test_eval.sh
#   LEGACY_CONFIG=1 bash scripts/run_math500_test_eval.sh   # old proven config (ablation)
#   GREEDY=1 bash scripts/run_math500_test_eval.sh          # ablation only

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -f "${ROOT}/scripts/env_conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/scripts/env_conda.sh"
fi

MODEL="${MODEL:-Qwen3-4B}"
DOMAIN_TRAIN="${DOMAIN_TRAIN:-math500_train}"
DOMAIN_TEST="${DOMAIN_TEST:-math500_test}"
MODEL_PATH="${MODEL_PATH:-${ROOT}/models/${MODEL}}"
DIRECTION="${DIRECTION:-data/steering/${MODEL}__${DOMAIN_TRAIN}__critical_direction.pt}"
INPUT="${INPUT:-data/raw/${DOMAIN_TEST}.jsonl}"
STEERING_DIR="${STEERING_DIR:-data/steering}"
OUT="${OUT:-${STEERING_DIR}/${MODEL}__${DOMAIN_TEST}__steering_eval.jsonl}"
LOG_DIR="${LOG_DIR:-logs}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.8}"
TOP_K="${TOP_K:-20}"
# Default 0: eval_spark_steering.py picks model-aware value (0 for 0.6B, 1.5 for 4B/8B on HF).
PRESENCE_PENALTY="${PRESENCE_PENALTY:-}"
SEED="${SEED:-42}"
TEST_LIMIT="${TEST_LIMIT:-}"
ALPHAS="${ALPHAS:-0.0 0.25 0.5 1.0 1.5 2.0 2.5 3.0 4.0}"
EVAL_PROFILE="${EVAL_PROFILE:-default}"
STEERING_POSITION="${STEERING_POSITION:-all}"
REPETITION_PENALTY="${REPETITION_PENALTY:-}"

mkdir -p "${STEERING_DIR}" "${LOG_DIR}"

if [[ ! -f "${DIRECTION}" ]]; then
  echo "[math500-test] missing direction ${DIRECTION}; run run_math500_train_pipeline.sh first" >&2
  exit 1
fi
if [[ ! -f "${INPUT}" ]]; then
  echo "[math500-test] missing ${INPUT}; run prepare_math500.py" >&2
  exit 1
fi

EVAL_ARGS=(
  --model_path "${MODEL_PATH}"
  --direction "${DIRECTION}"
  --input "${INPUT}"
  --out "${OUT}"
  --alphas ${ALPHAS}
  --repetition_penalty 1.0
  --dtype bfloat16
  --device_map "${DEVICE_MAP:-auto}"
  --position "${STEERING_POSITION}"
  --prompt_style math500_official
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --top_k "${TOP_K}"
  --seed "${SEED}"
  --eval_profile "${EVAL_PROFILE}"
)

if [[ -n "${REPETITION_PENALTY}" ]]; then
  EVAL_ARGS+=(--repetition_penalty "${REPETITION_PENALTY}")
fi

if [[ -n "${PRESENCE_PENALTY}" ]]; then
  EVAL_ARGS+=(--presence_penalty "${PRESENCE_PENALTY}")
fi

if [[ -n "${MAX_NEW_TOKENS}" ]]; then
  EVAL_ARGS+=(--max_new_tokens "${MAX_NEW_TOKENS}")
fi

if [[ -n "${MAX_MEMORY:-}" ]]; then
  EVAL_ARGS+=(--max_memory "${MAX_MEMORY}")
fi

if [[ "${LEGACY_CONFIG:-0}" == "1" ]]; then
  MAX_NEW_TOKENS=16384
  PRESENCE_PENALTY=0
  EVAL_ARGS=(
    --model_path "${MODEL_PATH}"
    --direction "${DIRECTION}"
    --input "${INPUT}"
    --out "${OUT}"
    --alphas ${ALPHAS}
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --repetition_penalty 1.0
    --presence_penalty 0
    --dtype bfloat16
    --device_map auto
    --position all
    --prompt_style math500_official
    --temperature "${TEMPERATURE}"
    --top_p "${TOP_P}"
    --top_k "${TOP_K}"
    --seed "${SEED}"
    --with_system_prompt
  )
fi

if [[ "${GREEDY:-0}" == "1" ]]; then
  EVAL_ARGS+=(--greedy)
fi

if [[ "${NO_SYSTEM_PROMPT:-0}" == "1" ]]; then
  EVAL_ARGS+=(--no_system_prompt)
fi

if [[ "${WITH_SYSTEM_PROMPT:-0}" == "1" ]]; then
  EVAL_ARGS+=(--with_system_prompt)
fi

if [[ -n "${TEST_LIMIT}" ]]; then
  EVAL_ARGS+=(--limit "${TEST_LIMIT}")
else
  EVAL_ARGS+=(--limit -1)
fi

if [[ "${RESUME:-1}" == "1" ]]; then
  EVAL_ARGS+=(--resume)
fi

if [[ "${ENABLE_THINKING:-0}" == "1" ]]; then
  EVAL_ARGS+=(--enable_thinking)
fi

if [[ -n "${MAX_RAW_CHARS:-}" ]]; then
  EVAL_ARGS+=(--max_raw_chars "${MAX_RAW_CHARS}")
fi

echo "[math500-test] model=${MODEL} alphas=${ALPHAS}"
echo "[math500-test] config: profile=${EVAL_PROFILE} temp=${TEMPERATURE} top_p=${TOP_P} top_k=${TOP_K} presence=${PRESENCE_PENALTY:-auto} max_new=${MAX_NEW_TOKENS:-auto} position=${STEERING_POSITION} legacy=${LEGACY_CONFIG:-0}"
echo "[math500-test] cuda: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} device_map=${DEVICE_MAP:-auto} max_memory=${MAX_MEMORY:-auto}"
echo "[math500-test] eval steering on MATH-500 benchmark..."
python -u scripts/eval_spark_steering.py "${EVAL_ARGS[@]}"

SUMMARY_MD="${OUT%.jsonl}.summary.md"
python scripts/summarize_steering_eval.py \
  --input "${OUT}" \
  --out_md "${SUMMARY_MD}" \
  --reverify

echo "[math500-test] results -> ${OUT}"
echo "[math500-test] summary -> ${SUMMARY_MD}"
