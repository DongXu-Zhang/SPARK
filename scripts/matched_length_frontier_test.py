"""Matched-length frontier test for length-controlled susceptibility.

The length-control regression shows whether the global chi-vs-difficulty trend
is explained by prompt length. This script adds a stricter diagnostic:

For each hard/frontier example, find an easier example with similar token
length, then compare chi_lc and correctness. If hard examples remain lower in
chi_lc after length matching, that supports a genuine under-activation signal.

This script expects the output of export_length_controlled_records.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.length_control import (  # noqa: E402
    as_float,
    as_int,
    fmt,
    fmt_pct,
    mean_or_none,
    std_or_none,
)
from src.utils.io import load_jsonl  # noqa: E402


def md_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(x).replace("|", "\\|") for x in row) + " |")
    return "\n".join(lines)


def pass_filters(records, d_min=None, d_max=None, correct=None, chi_lc_max=None, chi_lc_min=None):
    out = []
    for r in records:
        d = as_float(r.get("d_value"))
        if d is None:
            continue
        if d_min is not None and d < d_min:
            continue
        if d_max is not None and d > d_max:
            continue
        if correct is not None and as_int(r.get("correct")) != int(correct):
            continue
        chi_lc = as_float(r.get("chi_lc"))
        if chi_lc is None:
            continue
        if chi_lc_max is not None and chi_lc > chi_lc_max:
            continue
        if chi_lc_min is not None and chi_lc < chi_lc_min:
            continue
        out.append(r)
    return out


def match_pairs(
    easy_records,
    hard_records,
    token_tol_frac: float,
    max_pairs: int | None,
    seed: int,
):
    rng = random.Random(seed)
    hard = list(hard_records)
    easy = list(easy_records)
    rng.shuffle(hard)
    used_easy = set()
    pairs = []

    for h in hard:
        h_tok = as_float(h.get("n_tokens"))
        if h_tok is None:
            continue
        best = None
        best_dist = None
        for e in easy:
            eid = str(e.get("id"))
            if eid in used_easy:
                continue
            e_tok = as_float(e.get("n_tokens"))
            if e_tok is None:
                continue
            rel = abs(e_tok - h_tok) / max(h_tok, 1.0)
            if rel > token_tol_frac:
                continue
            dist = (rel, abs(as_float(e.get("d_value")) - as_float(h.get("d_value"))))
            if best is None or dist < best_dist:
                best = e
                best_dist = dist
        if best is None:
            continue
        used_easy.add(str(best.get("id")))
        pairs.append((best, h, best_dist[0]))
        if max_pairs is not None and len(pairs) >= max_pairs:
            break
    return pairs


def summarize_pairs(pairs):
    deltas = []
    rows = []
    for e, h, rel_token_diff in pairs:
        e_chi_lc = as_float(e.get("chi_lc"))
        h_chi_lc = as_float(h.get("chi_lc"))
        e_ratio = as_float(e.get("chi_ratio_len"))
        h_ratio = as_float(h.get("chi_ratio_len"))
        e_chi = as_float(e.get("chi_max"))
        h_chi = as_float(h.get("chi_max"))
        delta = None if e_chi_lc is None or h_chi_lc is None else h_chi_lc - e_chi_lc
        if delta is not None:
            deltas.append(delta)
        rows.append({
            "easy_id": e.get("id"),
            "hard_id": h.get("id"),
            "easy_d": as_float(e.get("d_value")),
            "hard_d": as_float(h.get("d_value")),
            "easy_correct": as_int(e.get("correct")),
            "hard_correct": as_int(h.get("correct")),
            "easy_tokens": as_float(e.get("n_tokens")),
            "hard_tokens": as_float(h.get("n_tokens")),
            "rel_token_diff": rel_token_diff,
            "easy_chi": e_chi,
            "hard_chi": h_chi,
            "easy_chi_lc": e_chi_lc,
            "hard_chi_lc": h_chi_lc,
            "delta_hard_minus_easy_chi_lc": delta,
            "easy_chi_ratio_len": e_ratio,
            "hard_chi_ratio_len": h_ratio,
        })

    n_neg = sum(1 for d in deltas if d < 0)
    n_pos = sum(1 for d in deltas if d > 0)
    summary = {
        "n_pairs": len(pairs),
        "mean_delta_chi_lc": mean_or_none(deltas),
        "std_delta_chi_lc": std_or_none(deltas),
        "frac_hard_lower_chi_lc": (n_neg / len(deltas)) if deltas else None,
        "n_hard_lower": n_neg,
        "n_hard_higher": n_pos,
    }
    return rows, summary


def write_pairs_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "easy_id", "hard_id", "easy_d", "hard_d", "easy_correct", "hard_correct",
        "easy_tokens", "hard_tokens", "rel_token_diff", "easy_chi", "hard_chi",
        "easy_chi_lc", "hard_chi_lc", "delta_hard_minus_easy_chi_lc",
        "easy_chi_ratio_len", "hard_chi_ratio_len",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-4B")
    ap.add_argument("--domain", default="algorithmic")
    ap.add_argument("--analysis_dir", default="data/analysis")
    ap.add_argument("--out_dir", default="figures/matched_length_Qwen3-4B")
    ap.add_argument("--easy_d_min", type=float, default=0.30)
    ap.add_argument("--easy_d_max", type=float, default=0.55)
    ap.add_argument("--hard_d_min", type=float, default=0.55)
    ap.add_argument("--hard_d_max", type=float, default=0.80)
    ap.add_argument("--easy_correct", type=int, default=1)
    ap.add_argument("--hard_correct", type=int, default=None)
    ap.add_argument("--hard_chi_lc_max", type=float, default=None,
                    help="Optional: only use hard examples below this chi_lc.")
    ap.add_argument("--token_tol_frac", type=float, default=0.20,
                    help="Maximum relative token-length difference for a pair.")
    ap.add_argument("--max_pairs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    in_path = Path(args.analysis_dir) / f"{args.model}__{args.domain}__length_controlled.jsonl"
    records = load_jsonl(in_path)
    easy = pass_filters(
        records,
        d_min=args.easy_d_min,
        d_max=args.easy_d_max,
        correct=args.easy_correct,
    )
    hard = pass_filters(
        records,
        d_min=args.hard_d_min,
        d_max=args.hard_d_max,
        correct=args.hard_correct,
        chi_lc_max=args.hard_chi_lc_max,
    )
    pairs = match_pairs(
        easy,
        hard,
        token_tol_frac=args.token_tol_frac,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )
    pair_rows, summary = summarize_pairs(pairs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"matched_pairs__{args.model}__{args.domain}.csv"
    md_path = out_dir / f"matched_length_report__{args.model}__{args.domain}.md"
    json_path = out_dir / f"matched_length_summary__{args.model}__{args.domain}.json"
    write_pairs_csv(csv_path, pair_rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump({
            "args": vars(args),
            "n_easy_candidates": len(easy),
            "n_hard_candidates": len(hard),
            "summary": summary,
        }, f, ensure_ascii=False, indent=2)

    report_rows = [
        ["easy candidates", len(easy)],
        ["hard candidates", len(hard)],
        ["matched pairs", summary["n_pairs"]],
        ["mean hard-easy chi_lc", fmt(summary["mean_delta_chi_lc"])],
        ["std hard-easy chi_lc", fmt(summary["std_delta_chi_lc"])],
        ["fraction hard lower chi_lc", fmt_pct(summary["frac_hard_lower_chi_lc"])],
        ["n hard lower/higher", f"{summary['n_hard_lower']}/{summary['n_hard_higher']}"],
    ]
    examples = []
    for row in pair_rows[:20]:
        examples.append([
            row["easy_id"],
            row["hard_id"],
            fmt(row["easy_d"]),
            fmt(row["hard_d"]),
            fmt(row["easy_tokens"], 0),
            fmt(row["hard_tokens"], 0),
            fmt(row["easy_chi_lc"]),
            fmt(row["hard_chi_lc"]),
            fmt(row["delta_hard_minus_easy_chi_lc"]),
        ])

    report = [
        f"# Matched-Length Frontier Test: {args.model} / {args.domain}",
        "",
        "## Configuration",
        "",
        md_table(["key", "value"], [[k, v] for k, v in vars(args).items()]),
        "",
        "## Summary",
        "",
        md_table(["metric", "value"], report_rows),
        "",
        "## First Matched Pairs",
        "",
        md_table([
            "easy_id", "hard_id", "easy_d", "hard_d", "easy_tokens",
            "hard_tokens", "easy_chi_lc", "hard_chi_lc", "delta",
        ], examples),
        "",
        f"- pairs csv: `{csv_path}`",
        f"- summary json: `{json_path}`",
        "",
    ]
    md_path.write_text("\n".join(report), encoding="utf-8")
    print(f"[matched-length] report -> {md_path}")
    print(f"[matched-length] pairs  -> {csv_path}")
    print(f"[matched-length] summary -> {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

