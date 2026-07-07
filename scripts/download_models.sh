#!/usr/bin/env bash
# Download all calibration models from ModelScope (魔塔) into $MODELS_DIR.
#
# Usage:
#   export MODELS_DIR=/data/models           # or wherever you want to store weights
#   bash scripts/download_models.sh                      # 2x3090 default pool (~70 GB)
#   POOL=4card bash scripts/download_models.sh           # 4x3090 extended pool (~200 GB)
#   ONLY=Qwen3-4B bash scripts/download_models.sh        # download just one model
#
# The script is resumable: if a model directory already contains a
# `config.json`, it is considered downloaded and skipped.

set -euo pipefail

if [[ -z "${MODELS_DIR:-}" ]]; then
  echo "ERROR: please set MODELS_DIR first, e.g.:"
  echo "    export MODELS_DIR=/data/models"
  exit 1
fi

mkdir -p "$MODELS_DIR"
echo "[download] target directory: $MODELS_DIR"
df -h "$MODELS_DIR" | tail -n 1

if [[ "${POOL:-2card}" == "4card" ]]; then
  ENTRIES=(
    "Qwen3-0.6B|Qwen/Qwen3-0.6B"
    "Qwen3-1.7B|Qwen/Qwen3-1.7B"
    "Qwen3-4B|Qwen/Qwen3-4B"
    "Qwen3-8B|Qwen/Qwen3-8B"
    "Qwen3-14B-AWQ|Qwen/Qwen3-14B-AWQ"
    "DSR1-Distill-Qwen-7B|deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    "Qwen3-32B|Qwen/Qwen3-32B"
    "DSR1-Distill-Qwen-32B|deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
  )
else
  ENTRIES=(
    "Qwen3-0.6B|Qwen/Qwen3-0.6B"
    "Qwen3-1.7B|Qwen/Qwen3-1.7B"
    "Qwen3-4B|Qwen/Qwen3-4B"
    "Qwen3-8B|Qwen/Qwen3-8B"
    "Qwen3-14B-AWQ|Qwen/Qwen3-14B-AWQ"
    "DSR1-Distill-Qwen-7B|deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    "Qwen3-32B-AWQ|Qwen/Qwen3-32B-AWQ"
  )
fi

for entry in "${ENTRIES[@]}"; do
  LOCAL_NAME="${entry%%|*}"
  MS_ID="${entry##*|}"

  if [[ -n "${ONLY:-}" && "$ONLY" != "$LOCAL_NAME" ]]; then
    continue
  fi

  TARGET="$MODELS_DIR/$LOCAL_NAME"

  if [[ -f "$TARGET/config.json" ]]; then
    echo "[download] skip $LOCAL_NAME (already at $TARGET)"
    continue
  fi

  echo "[download] === $LOCAL_NAME from modelscope://$MS_ID ==="
  mkdir -p "$TARGET"
  modelscope download --model "$MS_ID" --local_dir "$TARGET"
  echo "[download] done $LOCAL_NAME"
done

echo
echo "[download] all done. Weights at $MODELS_DIR"
ls -lh "$MODELS_DIR"
