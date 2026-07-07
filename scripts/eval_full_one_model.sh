#!/usr/bin/env bash
# Run one model on the three full raw domain files:
#   data/raw/symbolic.jsonl
#   data/raw/logical.jsonl
#   data/raw/algorithmic.jsonl
#
# Outputs are written as:
#   data/calibrated_full/<model>__symbolic.jsonl
#   data/calibrated_full/<model>__logical.jsonl
#   data/calibrated_full/<model>__algorithmic.jsonl
#
# Usage:
#   bash scripts/eval_full_one_model.sh Qwen3-4B
#   LIMIT=50 bash scripts/eval_full_one_model.sh Qwen3-4B
#   DATA_DIR=/path/to/raw OUT_DIR=/path/to/out bash scripts/eval_full_one_model.sh Qwen3-4B

set -euo pipefail

MODEL="${1:?Usage: $0 <model_name>}"
DATA_DIR="${DATA_DIR:-/home/zhangdongxu/AAAI2027/data/raw}"
OUT_DIR="${OUT_DIR:-/home/zhangdongxu/AAAI2027/data/calibrated_full}"
CFG="${CFG:-configs/default.yaml}"
MODELS_CFG="${MODELS_CFG:-configs/models.yaml}"

# Qwen3 + vLLM has been more stable with xformers in this project.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

if [[ -z "${MODELS_DIR:-}" ]]; then
  echo "[eval_full_one_model] WARNING: MODELS_DIR is not set."
  echo "[eval_full_one_model] The loader may fall back to ModelScope download."
  echo "[eval_full_one_model] Recommended:"
  echo "    export MODELS_DIR=/home/zhangdongxu/AAAI2027/models"
fi

mkdir -p "$OUT_DIR" logs

EXTRA_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  EXTRA_ARGS+=(--limit "$LIMIT")
fi

echo "[eval_full_one_model] model=$MODEL"
echo "[eval_full_one_model] data_dir=$DATA_DIR"
echo "[eval_full_one_model] out_dir=$OUT_DIR"
echo "[eval_full_one_model] VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND"
echo

DOMAINS=(symbolic logical algorithmic)

for DOM in "${DOMAINS[@]}"; do
  INPUT="$DATA_DIR/$DOM.jsonl"
  OUT_FILE="$OUT_DIR/${MODEL}__${DOM}.jsonl"
  LOG="logs/eval_full_${MODEL}__${DOM}.log"

  if [[ ! -f "$INPUT" ]]; then
    echo "[eval_full_one_model] ERROR: missing input file: $INPUT" >&2
    exit 1
  fi

  if [[ -s "$OUT_FILE" ]]; then
    echo "[eval_full_one_model] skip ${MODEL} x ${DOM}: exists $OUT_FILE"
    continue
  fi

  echo "[eval_full_one_model] === $MODEL x $DOM ==="
  echo "[eval_full_one_model] input: $INPUT"
  echo "[eval_full_one_model] output: $OUT_FILE"
  echo "[eval_full_one_model] log: $LOG"

  python -m src.calibration.run_eval \
    --model "$MODEL" \
    --input "$INPUT" \
    --out_dir "$OUT_DIR" \
    --config "$CFG" \
    --models_config "$MODELS_CFG" \
    --tag "$DOM" \
    "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG"

  echo
done

echo "[eval_full_one_model] done. Outputs:"
ls -lh "$OUT_DIR"/"${MODEL}"__*.jsonl
