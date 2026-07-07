#!/usr/bin/env python3
"""Re-score steering eval JSONL with official-style grading (no re-generation)."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.calibration.math_grade import extract_math_answer  # noqa: E402
from src.calibration.verify import extract_answer, verify  # noqa: E402


def _pred_from_row(r: dict) -> str | None:
    raw = r.get("raw_text") or ""
    atype = r.get("answer_type")
    if atype in ("math", "integer") and raw:
        pred = extract_math_answer(raw)
        if not pred and atype == "integer":
            pred = extract_answer(raw)
        return pred
    return r.get("pred")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default=None, help="Optional output JSONL with updated fields")
    args = ap.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_alpha: dict[float, list[dict]] = defaultdict(list)
    for r in rows:
        pred = _pred_from_row(r)
        ok = verify(
            pred,
            r.get("ground_truth"),
            r.get("answer_type", "math"),
            ground_truth_raw=r.get("ground_truth_raw"),
        )
        r["pred_reextracted"] = pred
        r["correct_official"] = int(ok)
        by_alpha[float(r["alpha"])].append(r)

    print(f"input: {args.input}  rows={len(rows)}")
    print("| alpha | n | old_acc | official_acc | delta |")
    print("| --- | --- | --- | --- | --- |")
    for alpha in sorted(by_alpha):
        rs = by_alpha[alpha]
        old = sum(int(x.get("correct", 0)) for x in rs) / len(rs)
        new = sum(int(x.get("correct_official", 0)) for x in rs) / len(rs)
        print(f"| {alpha:g} | {len(rs)} | {100*old:.1f}% | {100*new:.1f}% | {100*(new-old):+.1f}pp |")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
