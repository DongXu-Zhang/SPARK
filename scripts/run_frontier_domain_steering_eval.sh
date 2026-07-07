#!/usr/bin/env bash
# Evaluate SPARK-Steering on held-out FRONTIER targets (Qwen3-4B default).
#
# Usage:
#   bash scripts/run_frontier_domain_steering_eval.sh algorithmic
#   TEST_LIMIT=20 bash scripts/run_frontier_domain_steering_eval.sh symbolic
#   ALPHAS="0 0.5 1" GREEDY=1 bash scripts/run_frontier_domain_steering_eval.sh logical

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -f "${ROOT}/scripts/env_conda.sh" ]]; then
  # shellcheck source=scripts/env_conda.sh
  source "${ROOT}/scripts/env_conda.sh"
fi

DOMAIN="${1:-}"
if [[ -z "${DOMAIN}" ]]; then
  echo "Usage: bash scripts/run_frontier_domain_steering_eval.sh <symbolic|logical|algorithmic>" >&2
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
STEERING_DIR="${STEERING_DIR:-data/steering}"
LOG_DIR="${LOG_DIR:-logs}"

PREFIX="${STEERING_DIR}/${MODEL}__${DOMAIN}"
DIRECTION="${DIRECTION:-${PREFIX}__critical_direction.pt}"
INPUT="${INPUT:-${PREFIX}__heldout_test.jsonl}"
OUT="${OUT:-${PREFIX}__steering_eval.jsonl}"
SUMMARY_MD="${OUT%.jsonl}.summary.md"

ALPHAS="${ALPHAS:-0.0 0.25 0.5 1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
TEST_LIMIT="${TEST_LIMIT:-}"
GREEDY="${GREEDY:-1}"
SEED="${SEED:-42}"

mkdir -p "${STEERING_DIR}" "${LOG_DIR}"

if [[ ! -f "${DIRECTION}" ]]; then
  echo "[frontier-eval] missing ${DIRECTION}; run run_frontier_domain_steering_train.sh ${DOMAIN} first" >&2
  exit 1
fi
if [[ ! -f "${INPUT}" ]]; then
  echo "[frontier-eval] missing ${INPUT}" >&2
  exit 1
fi

EVAL_ARGS=(
  --model_path "${MODEL_PATH}"
  --direction "${DIRECTION}"
  --input "${INPUT}"
  --out "${OUT}"
  --alphas ${ALPHAS}
  --limit -1
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --repetition_penalty "${REPETITION_PENALTY:-1.05}"
  --dtype bfloat16
  --device_map auto
  --position all
  --prompt_style default
  --seed "${SEED}"
)

if [[ "${GREEDY}" == "1" ]]; then
  EVAL_ARGS+=(--greedy)
else
  EVAL_ARGS+=(
    --temperature "${TEMPERATURE:-0.7}"
    --top_p "${TOP_P:-0.8}"
    --top_k "${TOP_K:-20}"
  )
fi

if [[ -n "${TEST_LIMIT}" ]]; then
  EVAL_ARGS+=(--limit "${TEST_LIMIT}")
fi

# Graph domain: reduce edge-list copying in long CoT
if [[ "${DOMAIN}" == "algorithmic" ]]; then
  EVAL_ARGS+=(--concise_prompt)
fi

echo "[frontier-eval] domain=${DOMAIN} model=${MODEL}"
echo "[frontier-eval] input=${INPUT} alphas=${ALPHAS} max_new=${MAX_NEW_TOKENS} greedy=${GREEDY}"
python -u scripts/eval_spark_steering.py "${EVAL_ARGS[@]}"

python scripts/summarize_steering_eval.py \
  --input "${OUT}" \
  --out_md "${SUMMARY_MD}" \
  --reverify

python scripts/summarize_frontier_steering_report.py \
  --inputs "${OUT}" \
  --labels "${DOMAIN}" \
  --out_md "${OUT%.jsonl}.frontier_report.md"

echo "[frontier-eval] wrote ${OUT}"
echo "[frontier-eval] summary ${SUMMARY_MD}"
echo "[frontier-eval] report  ${OUT%.jsonl}.frontier_report.md"
