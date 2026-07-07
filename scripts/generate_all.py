"""Generate the 4500-problem candidate pool and merge into one JSONL.

CPU-only. Run after sanity_check.py succeeds.

    python scripts/generate_all.py --n_per_domain 1500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.generators import symbolic, logical, algorithmic
from src.utils.io import save_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_domain", type=int, default=1500)
    ap.add_argument("--out_dir", type=str, default="data/raw")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sym = symbolic.generate_dataset(args.n_per_domain, seed_base=100000)
    log = logical.generate_dataset(args.n_per_domain, seed_base=200000)
    alg = algorithmic.generate_dataset(args.n_per_domain, seed_base=300000)

    save_jsonl(sym, out_dir / "symbolic.jsonl")
    save_jsonl(log, out_dir / "logical.jsonl")
    save_jsonl(alg, out_dir / "algorithmic.jsonl")

    merged = sym + log + alg
    save_jsonl(merged, out_dir / "all.jsonl")

    print(f"\n[generate_all] wrote {len(sym)} symbolic, {len(log)} logical, "
          f"{len(alg)} algorithmic")
    print(f"[generate_all] merged → {out_dir / 'all.jsonl'} ({len(merged)} total)")


if __name__ == "__main__":
    main()
