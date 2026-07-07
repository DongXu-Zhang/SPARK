"""Summarize SPARK-Steering evaluation JSONL files."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.calibration.math_grade import extract_math_answer  # noqa: E402
from src.calibration.verify import verify


def _alpha_metrics(rows: list[dict]) -> dict:
    """Accuracy and generation-length stats for one α slice."""
    n = len(rows)
    acc = sum(int(r.get("correct", 0)) for r in rows) / n if n else None
    ntoks = [int(r["n_gen_new_tokens"]) for r in rows if r.get("n_gen_new_tokens") is not None]
    avg_gen = sum(ntoks) / len(ntoks) if ntoks else None
    median_gen = None
    if ntoks:
        s = sorted(ntoks)
        mid = len(s) // 2
        median_gen = float(s[mid]) if len(s) % 2 else (s[mid - 1] + s[mid]) / 2.0
    frs = [r.get("finish_reason") for r in rows if r.get("finish_reason") is not None]
    pct_length = (100.0 * sum(1 for fr in frs if fr == "length") / len(frs)) if frs else None
    return {
        "n": n,
        "acc": acc,
        "avg_gen_tokens": avg_gen,
        "median_gen_tokens": median_gen,
        "pct_length": pct_length,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_md", default=None)
    ap.add_argument("--reverify", action="store_true",
                    help="Recompute correctness with current verifier if ground_truth is present.")
    args = ap.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_alpha = defaultdict(list)
    for r in rows:
        if args.reverify and "ground_truth" in r and "answer_type" in r:
            raw = r.get("raw_text") or ""
            atype = r.get("answer_type")
            if atype in ("math", "integer") and raw:
                pred = extract_math_answer(raw)
                if not pred and atype == "integer":
                    from src.calibration.verify import extract_answer as _extract_answer
                    pred = _extract_answer(raw)
            else:
                pred = r.get("pred")
            r["correct_reverified"] = int(
                verify(
                    pred,
                    r.get("ground_truth"),
                    r.get("answer_type"),
                    ground_truth_raw=r.get("ground_truth_raw"),
                )
            )
        by_alpha[float(r["alpha"])].append(r)

    lines = [
        f"# Steering Eval Summary",
        "",
        f"- input: `{args.input}`",
        f"- n rows: {len(rows)}",
        f"- reverify: {args.reverify}",
        "",
        "## Metrics per α",
        "",
        "Primary length metric: **avg_gen_tokens** = mean `n_gen_new_tokens` (HF decode output only, excl. prompt).",
        "",
        "| alpha | n | acc | acc_reverified | avg_gen_tokens | median_gen_tokens | pct_length |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    detail_lines = [
        "",
        "## Detail",
        "",
        "| alpha | n | acc | acc_reverified | original_acc | boxed_raw | avg_chars | avg_gen_tokens | pct_length | correct_ids |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for alpha in sorted(by_alpha):
        rs = by_alpha[alpha]
        m = _alpha_metrics(rs)
        acc = m["acc"] if m["acc"] is not None else 0.0
        if args.reverify and any("correct_reverified" in r for r in rs):
            acc_rev = sum(int(r.get("correct_reverified", 0)) for r in rs) / len(rs)
        else:
            acc_rev = None
        orig_vals = [int(r.get("original_correct", 0)) for r in rs if r.get("original_correct") is not None]
        orig_acc = sum(orig_vals) / len(orig_vals) if orig_vals else None
        boxed = sum(1 for r in rs if "\\boxed" in (r.get("raw_text") or ""))
        avg_chars = sum(int(r.get("n_chars") or 0) for r in rs) / len(rs) if rs else 0.0
        avg_gen = m["avg_gen_tokens"]
        median_gen = m["median_gen_tokens"]
        pct_len = m["pct_length"]
        lines.append(
            f"| {alpha:g} | {m['n']} | {acc:.3f} | "
            f"{'NA' if acc_rev is None else f'{acc_rev:.3f}'} | "
            f"{'NA' if avg_gen is None else f'{avg_gen:.1f}'} | "
            f"{'NA' if median_gen is None else f'{median_gen:.1f}'} | "
            f"{'NA' if pct_len is None else f'{pct_len:.1f}%'} |"
        )
        key = "correct_reverified" if args.reverify and acc_rev is not None else "correct"
        correct_ids = ",".join(str(r.get("id")) for r in rs if int(r.get(key, 0)) == 1)
        detail_lines.append(
            f"| {alpha:g} | {m['n']} | {acc:.3f} | "
            f"{'NA' if acc_rev is None else f'{acc_rev:.3f}'} | "
            f"{'NA' if orig_acc is None else f'{orig_acc:.3f}'} | "
            f"{boxed} | {avg_chars:.1f} | "
            f"{'NA' if avg_gen is None else f'{avg_gen:.1f}'} | "
            f"{'NA' if pct_len is None else f'{pct_len:.1f}%'} | "
            f"{correct_ids or 'NA'} |"
        )

    text = "\n".join(lines + detail_lines) + "\n"
    print(text)
    if args.out_md:
        out = Path(args.out_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"[summary] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
