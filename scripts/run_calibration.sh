#!/usr/bin/env bash
# Run all 7 (or 9) Qwen-family models in series over the candidate pool.
# Each run writes to data/calibrated/{model}.jsonl.
#
# Usage:
#   bash scripts/run_calibration.sh                # 2x3090 default pool
#   POOL=4card bash scripts/run_calibration.sh     # 4x3090 extended pool
#   LIMIT=200 bash scripts/run_calibration.sh      # quick subset for testing
#
# Resumable: if {model}.jsonl already exists and is non-empty, the script skips
# that model. Delete the file to force a re-run.

set -euo pipefail

INPUT="${INPUT:-data/raw/all.jsonl}"
OUT_DIR="${OUT_DIR:-data/calibrated}"
CFG="${CFG:-configs/default.yaml}"
MODELS_CFG="${MODELS_CFG:-configs/models.yaml}"
POOL="${POOL:-2card}"

mkdir -p "$OUT_DIR" logs

if [[ "$POOL" == "4card" ]]; then
  MODELS=(
    Qwen3-0.6B
    Qwen3-1.7B
    Qwen3-4B
    Qwen3-8B
    Qwen3-14B-AWQ
    DSR1-Distill-Qwen-7B
    Qwen3-32B
    DSR1-Distill-Qwen-32B
  )
else
  MODELS=(
    Qwen3-0.6B
    Qwen3-1.7B
    Qwen3-4B
    Qwen3-8B
    Qwen3-14B-AWQ
    DSR1-Distill-Qwen-7B
    Qwen3-32B-AWQ
  )
fi

EXTRA_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  EXTRA_ARGS+=(--limit "$LIMIT")
fi

for M in "${MODELS[@]}"; do
  OUT="$OUT_DIR/$M.jsonl"
  if [[ -s "$OUT" ]]; then
    echo "[run_calibration] skip $M (already exists: $OUT)"
    continue
  fi
  echo "[run_calibration] === $M ==="
  LOG="logs/calib_${M}.log"
  python -m src.calibration.run_eval \
      --model "$M" \
      --input "$INPUT" \
      --out_dir "$OUT_DIR" \
      --config "$CFG" \
      --models_config "$MODELS_CFG" \
      "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG"
done

echo "[run_calibration] all models done. Now run difficulty fusion:"
echo "    python -m src.calibration.difficulty \\"
echo "        --problems $INPUT --eval_dir $OUT_DIR --out_dir data/final"
