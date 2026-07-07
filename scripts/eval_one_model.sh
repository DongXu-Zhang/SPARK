#!/usr/bin/env bash
# Run a single model on all three reasoning domains and keep results
# side-by-side as data/calibrated/<model>__<domain>.jsonl.
#
# Usage:
#   bash scripts/eval_one_model.sh <model_name> [data_dir]
#
# Examples:
#   bash scripts/eval_one_model.sh Qwen3-4B
#   bash scripts/eval_one_model.sh Qwen3-32B-AWQ data/raw
#   LIMIT=10 bash scripts/eval_one_model.sh Qwen3-0.6B    # quick smoke test
#
# Resumable: skips a (model, domain) pair if its output JSONL already exists
# and is non-empty. Delete that file to force a re-run.

set -euo pipefail

MODEL="${1:?Usage: $0 <model_name> [data_dir]}"
DATA_DIR="${2:-data/sanity}"
OUT_DIR="${OUT_DIR:-data/calibrated}"
CFG="${CFG:-configs/default.yaml}"
MODELS_CFG="${MODELS_CFG:-configs/models.yaml}"

# vLLM v1 engine on torch 2.6 + flashinfer was unstable on Qwen3 in our tests;
# fall back to xformers attention by default. Override if you know what you do.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

if [[ -z "${MODELS_DIR:-}" ]]; then
    echo "[eval_one_model] WARNING: MODELS_DIR is not set; the loader will"
    echo "[eval_one_model] fall back to ModelScope download. Export it once:"
    echo "    export MODELS_DIR=/home/zhangdongxu/AAAI2027/models"
fi

mkdir -p "$OUT_DIR" logs

EXTRA_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
    EXTRA_ARGS+=(--limit "$LIMIT")
fi

echo "[eval_one_model] model=$MODEL  data_dir=$DATA_DIR  out_dir=$OUT_DIR"
echo "[eval_one_model] VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND"
echo

for DOM in symbolic logical algorithmic; do
    INPUT="$DATA_DIR/$DOM.jsonl"
    OUT_FILE="$OUT_DIR/${MODEL}__${DOM}.jsonl"

    if [[ ! -f "$INPUT" ]]; then
        echo "[eval_one_model] WARNING: $INPUT not found, skipping $DOM"
        continue
    fi
    if [[ -s "$OUT_FILE" ]]; then
        echo "[eval_one_model] skip ${MODEL} x ${DOM}: $OUT_FILE already exists"
        continue
    fi

    LOG="logs/eval_${MODEL}__${DOM}.log"
    echo "[eval_one_model] === $MODEL  x  $DOM ==="
    echo "[eval_one_model] log: $LOG"

    python -m src.calibration.run_eval \
        --model "$MODEL" \
        --input "$INPUT" \
        --out_dir "$OUT_DIR" \
        --config "$CFG" \
        --models_config "$MODELS_CFG" \
        --tag "$DOM" \
        "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG"
done

echo
echo "[eval_one_model] $MODEL done."
echo "[eval_one_model] outputs:"
ls -lh "$OUT_DIR"/${MODEL}__*.jsonl 2>/dev/null || true
echo
echo "[eval_one_model] next: run summarize once you have multiple models:"
echo "    python -m src.calibration.summarize"
