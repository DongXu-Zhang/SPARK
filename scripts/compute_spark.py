"""GPU server entry point — compute χ / Φ / SPK for a JSONL of problems.

Thin wrapper around src.spark.runner so that scripts/ stays the canonical
location for command-line entries (mirrors the calibration scripts).

Usage:
    cd ~/AAAI2027
    python scripts/compute_spark.py \
        --model /home/zhangdongxu/AAAI2027/models/Qwen3-4B \
        --input data/sanity/symbolic.jsonl \
        --out data/spark/Qwen3-4B__symbolic.jsonl \
        --config configs/spark.yaml \
        --limit 5
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `python scripts/compute_spark.py` to find the `src` package.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.spark.runner import main  # noqa: E402

if __name__ == "__main__":
    main()
