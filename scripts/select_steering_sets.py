"""Select SPARK-Steering samples from length-controlled records.

This script operationalizes the current hypothesis:

  * reasoning-active examples: correct, mid-frontier difficulty, positive
    length-controlled chi, high SPK
  * under-activated targets: hard/frontier examples, preferably wrong, low
    length-controlled chi

The outputs are the canonical files for activation-vector extraction and
steering evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.length_control import as_float, as_int, fmt, fmt_pct  # noqa: E402
from src.utils.io import load_jsonl, save_json, save_jsonl  # noqa: E402


def md_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(x).replace("|", "\\|") for x in row) + " |")
    return "\n".join(lines)


def zscore(records, field):
    vals = [as_float(r.get(field)) for r in records]
    clean = np.asarray([v for v in vals if v is not None], dtype=float)
    if len(clean) < 2:
        mu, sd = 0.0, 1.0
    else:
        mu = float(clean.mean())
        sd = float(clean.std(ddof=1))
        if sd <= 1e-12 or not math.isfinite(sd):
            sd = 1.0
    out = []
    for r in records:
        v = as_float(r.get(field))
        out.append(0.0 if v is None else (v - mu) / sd)
    return out


def in_range(value, lo, hi):
    x = as_float(value)
    if x is None:
        return False
    return lo <= x <= hi


def annotate_scores(records, active_center: float, target_d_min: float):
    z_chi_lc = zscore(records, "chi_lc")
    z_spk = zscore(records, "spk")
    z_phi = zscore(records, "phi")
    z_d = zscore(records, "d_value")
    z_tokens = zscore(records, "n_tokens")

    out = []
    for i, r in enumerate(records):
        rec = dict(r)
        d = as_float(rec.get("d_value")) or 0.0
        correct = as_int(rec.get("correct"))
        wrong_bonus = 1.0 if correct == 0 else 0.0
        correct_bonus = 1.0 if correct == 1 else 0.0
        active_distance_penalty = abs(d - active_center)

        # Reasoning-active: high LC chi, high SPK/Phi, correct, not too far
        # from the frontier-side medium difficulty region.
        rec["active_score"] = (
            1.20 * z_chi_lc[i]
            + 0.80 * z_spk[i]
            + 0.25 * z_phi[i]
            + 0.75 * correct_bonus
            - 0.50 * active_distance_penalty
        )

        # Under-activated target: hard, low LC chi, preferably wrong. A small
        # token penalty prevents selecting only ultra-long outliers.
        rec["underactivation_score"] = (
            -1.20 * z_chi_lc[i]
            + 0.75 * z_d[i]
            + 1.00 * wrong_bonus
            - 0.15 * max(z_tokens[i], 0.0)
            + (0.25 if d >= target_d_min else 0.0)
        )
        out.append(rec)
    return out


def select_unique(records, n, excluded_ids=None):
    excluded = set(excluded_ids or [])
    out = []
    for r in records:
        rid = str(r.get("id"))
        if rid in excluded:
            continue
        out.append(r)
        excluded.add(rid)
        if len(out) >= n:
            break
    return out


def compact_for_report(records, n=12):
    rows = []
    for r in records[:n]:
        rows.append([
            r.get("id"),
            fmt(r.get("d_value")),
            as_int(r.get("correct")),
            fmt(r.get("chi_max")),
            fmt(r.get("chi_lc")),
            fmt(r.get("chi_ratio_len")),
            fmt(r.get("spk")),
            fmt(r.get("n_tokens"), 0),
            fmt(r.get("active_score") if "active_score" in r else r.get("underactivation_score")),
        ])
    return rows


def write_with_role(path: Path, records: list[dict], role: str) -> None:
    out = []
    for rank, r in enumerate(records, start=1):
        rec = dict(r)
        rec["selection_role"] = role
        rec["selection_rank"] = rank
        out.append(rec)
    save_jsonl(out, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-4B")
    ap.add_argument("--domain", default="algorithmic")
    ap.add_argument("--analysis_dir", default="data/analysis")
    ap.add_argument("--out_dir", default="data/steering")

    ap.add_argument("--active_d_min", type=float, default=0.30)
    ap.add_argument("--active_d_max", type=float, default=0.55)
    ap.add_argument("--target_d_min", type=float, default=0.55)
    ap.add_argument("--target_d_max", type=float, default=0.80)
    ap.add_argument("--active_min_chi_lc", type=float, default=0.0)
    ap.add_argument("--target_max_chi_lc", type=float, default=0.05)
    ap.add_argument("--target_wrong_first", action="store_true", default=True)

    ap.add_argument("--n_demos", type=int, default=32)
    ap.add_argument("--n_anchors", type=int, default=64)
    ap.add_argument("--n_targets", type=int, default=160)
    ap.add_argument("--n_test", type=int, default=240)
    args = ap.parse_args()

    in_path = Path(args.analysis_dir) / f"{args.model}__{args.domain}__length_controlled.jsonl"
    records = load_jsonl(in_path)
    records = annotate_scores(
        records,
        active_center=(args.active_d_min + args.active_d_max) / 2.0,
        target_d_min=args.target_d_min,
    )

    active_candidates = [
        r for r in records
        if in_range(r.get("d_value"), args.active_d_min, args.active_d_max)
        and as_int(r.get("correct")) == 1
        and (as_float(r.get("chi_lc")) is not None and as_float(r.get("chi_lc")) >= args.active_min_chi_lc)
    ]
    active_candidates.sort(key=lambda r: r["active_score"], reverse=True)

    target_candidates = [
        r for r in records
        if in_range(r.get("d_value"), args.target_d_min, args.target_d_max)
        and (as_float(r.get("chi_lc")) is not None and as_float(r.get("chi_lc")) <= args.target_max_chi_lc)
    ]
    if args.target_wrong_first:
        target_candidates.sort(
            key=lambda r: (as_int(r.get("correct")) == 0, r["underactivation_score"]),
            reverse=True,
        )
    else:
        target_candidates.sort(key=lambda r: r["underactivation_score"], reverse=True)

    used = set()
    demos = select_unique(active_candidates, args.n_demos, used)
    used.update(str(r.get("id")) for r in demos)
    anchors = select_unique(active_candidates, args.n_anchors, used)
    used.update(str(r.get("id")) for r in anchors)

    targets = select_unique(target_candidates, args.n_targets, used)
    used.update(str(r.get("id")) for r in targets)
    heldout_test = select_unique(target_candidates, args.n_test, used)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.model}__{args.domain}"
    demos_path = out_dir / f"{prefix}__active_demos.jsonl"
    anchors_path = out_dir / f"{prefix}__active_anchors.jsonl"
    targets_path = out_dir / f"{prefix}__underactivated_targets.jsonl"
    test_path = out_dir / f"{prefix}__heldout_test.jsonl"
    config_path = out_dir / f"{prefix}__selection_config.json"
    report_path = out_dir / f"{prefix}__selection_report.md"

    write_with_role(demos_path, demos, "active_demo")
    write_with_role(anchors_path, anchors, "active_anchor")
    write_with_role(targets_path, targets, "underactivated_target")
    write_with_role(test_path, heldout_test, "heldout_test")
    save_json(vars(args), config_path)

    def acc(rows):
        vals = [as_int(r.get("correct")) for r in rows]
        vals = [v for v in vals if v in (0, 1)]
        return None if not vals else sum(vals) / len(vals)

    summary = [
        ["records", len(records)],
        ["active candidates", len(active_candidates)],
        ["target candidates", len(target_candidates)],
        ["demos", len(demos)],
        ["anchors", len(anchors)],
        ["targets", len(targets)],
        ["heldout_test", len(heldout_test)],
        ["target accuracy", fmt_pct(acc(targets))],
        ["heldout accuracy", fmt_pct(acc(heldout_test))],
    ]
    report = [
        f"# SPARK-Steering Set Selection: {args.model} / {args.domain}",
        "",
        "## Summary",
        "",
        md_table(["item", "value"], summary),
        "",
        "## Active Demos",
        "",
        md_table(
            ["id", "d", "correct", "chi", "chi_lc", "ratio", "spk", "tokens", "score"],
            compact_for_report(demos),
        ),
        "",
        "## Active Anchors",
        "",
        md_table(
            ["id", "d", "correct", "chi", "chi_lc", "ratio", "spk", "tokens", "score"],
            compact_for_report(anchors),
        ),
        "",
        "## Under-Activated Targets",
        "",
        md_table(
            ["id", "d", "correct", "chi", "chi_lc", "ratio", "spk", "tokens", "score"],
            compact_for_report(targets),
        ),
        "",
        "## Output Files",
        "",
        f"- demos: `{demos_path}`",
        f"- anchors: `{anchors_path}`",
        f"- targets: `{targets_path}`",
        f"- heldout_test: `{test_path}`",
        f"- config: `{config_path}`",
        "",
    ]
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"[select-steering] demos  -> {demos_path}")
    print(f"[select-steering] anchors -> {anchors_path}")
    print(f"[select-steering] targets -> {targets_path}")
    print(f"[select-steering] test    -> {test_path}")
    print(f"[select-steering] report  -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

