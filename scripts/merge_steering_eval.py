#!/usr/bin/env python3
"""Merge multiple steering eval JSONL runs and print a unified α curve."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.calibration.verify import verify


def load_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--baseline_alpha", type=float, default=0.0)
    ap.add_argument("--reverify", action="store_true")
    args = ap.parse_args()

    rows = load_rows([Path(p) for p in args.inputs])
    by_alpha: dict[float, list[dict]] = defaultdict(list)
    for r in rows:
        a = float(r["alpha"])
        if args.reverify and "ground_truth" in r and "answer_type" in r:
            r["correct_reverified"] = int(
                verify(r.get("pred"), r.get("ground_truth"), r.get("answer_type"))
            )
        by_alpha[a].append(r)

    key = "correct_reverified" if args.reverify else "correct"

    lines = [
        "# Steering Eval Summary (merged)",
        "",
        "- inputs:",
    ]
    for p in args.inputs:
        lines.append(f"  - `{p}`")
    lines.extend([
        f"- n rows total: {len(rows)}",
        f"- baseline α: {args.baseline_alpha:g}",
        "",
        "| alpha | n | acc | Δ vs baseline | rescued | harmed | net | avg_n_new | pct_length |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])

    baseline_rows = {str(r["id"]): int(r.get(key, 0)) for r in by_alpha.get(float(args.baseline_alpha), [])}

    for alpha in sorted(by_alpha):
        rs = by_alpha[alpha]
        acc = sum(int(r.get(key, 0)) for r in rs) / len(rs) if rs else 0.0
        rescued = harmed = 0
        if baseline_rows and alpha != float(args.baseline_alpha):
            cur = {str(r["id"]): int(r.get(key, 0)) for r in rs}
            for pid, b_ok in baseline_rows.items():
                c_ok = cur.get(pid, 0)
                if b_ok == 0 and c_ok == 1:
                    rescued += 1
                elif b_ok == 1 and c_ok == 0:
                    harmed += 1
        delta = acc - (
            sum(baseline_rows.values()) / len(baseline_rows) if baseline_rows else 0.0
        )
        ntoks = [int(r["n_gen_new_tokens"]) for r in rs if r.get("n_gen_new_tokens") is not None]
        avg_n = sum(ntoks) / len(ntoks) if ntoks else None
        n_len = sum(1 for r in rs if r.get("finish_reason") == "length")
        denom = sum(1 for r in rs if r.get("finish_reason") is not None)
        pct_len = 100.0 * n_len / denom if denom else None
        lines.append(
            f"| {alpha:g} | {len(rs)} | {acc:.3f} | {delta:+.3f} | {rescued} | {harmed} | {rescued - harmed:+d} | "
            f"{'NA' if avg_n is None else f'{avg_n:.0f}'} | "
            f"{'NA' if pct_len is None else f'{pct_len:.1f}%'} |"
        )

    text = "\n".join(lines) + "\n"
    print(text)
    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"[merge] wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
