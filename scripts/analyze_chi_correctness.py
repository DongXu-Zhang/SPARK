"""Statistics report: how well does χ predict correctness?

For each (model, domain) supplied, computes:
  * Spearman ρ between χ and difficulty
  * Cohen's d for χ between wrong and correct subgroups
  * ROC AUC of χ as a correctness predictor (treating -χ as positive-class score)
  * mean ± std of χ for correct / wrong

Output is a Markdown table on stdout (also written to a file if --out_md is given).

Usage
-----
Multi-domain mode (one row per domain):

    python scripts/analyze_chi_correctness.py \
        --multi --model Qwen3-4B \
        --problems_dir data/sanity \
        --calibration_dir data/calibrated \
        --spark_dir data/spark \
        --out_md data/figures/chi_correctness.md

Single-domain mode:

    python scripts/analyze_chi_correctness.py \
        --problems data/sanity/symbolic.jsonl \
        --calibration data/calibrated/Qwen3-4B__symbolic.jsonl \
        --spark data/spark/Qwen3-4B__symbolic.jsonl \
        --tag symbolic
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.merge import merge_records
from src.analysis.stats import (
    auc_chi_correctness,
    cohens_d,
    spearman_chi_vs_d,
)


def analyze_one(records, score_field, d_field):
    """Compute all stats for one (model, domain) merged record set."""
    spearman = spearman_chi_vs_d(records, score_field=score_field, d_field=d_field)
    cd = cohens_d(records, score_field=score_field, label_field="correct")
    auc_pos = auc_chi_correctness(records, score_field=score_field,
                                  label_field="correct", invert=False)
    auc_neg = auc_chi_correctness(records, score_field=score_field,
                                  label_field="correct", invert=True)

    chi_values = [r.get(score_field) for r in records if r.get(score_field) is not None]
    chi_arr = np.asarray([v for v in chi_values if isinstance(v, (int, float))],
                         dtype=float)

    return {
        "n": len(records),
        "n_with_chi": int(len(chi_arr)),
        "chi_mean": float(chi_arr.mean()) if len(chi_arr) else float("nan"),
        "chi_std": float(chi_arr.std(ddof=1)) if len(chi_arr) > 1 else 0.0,
        "spearman_rho": spearman["rho"],
        "n_correct": cd["n_correct"],
        "n_wrong": cd["n_wrong"],
        "chi_correct": cd["mean_correct"],
        "chi_wrong": cd["mean_wrong"],
        "cohens_d": cd["d"],
        "auc_pos": auc_pos["auc"],
        "auc_neg": auc_neg["auc"],
    }


def format_md_table(rows: list[dict]) -> str:
    """Render a list of stats dicts as a Markdown table."""
    headers = [
        "tag", "n", "n_corr/n_wrong", "chi (mean±std)",
        "Spearman ρ (vs d)", "chi_corr", "chi_wrong", "Cohen's d",
        "AUC(χ→correct)", "AUC(-χ→correct)",
    ]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        chi_mean = r["chi_mean"]
        chi_std = r["chi_std"]
        chi_pretty = (f"{chi_mean:.2f}±{chi_std:.2f}"
                      if np.isfinite(chi_mean) else "—")
        chi_c = (f"{r['chi_correct']:.2f}"
                 if np.isfinite(r['chi_correct']) else "—")
        chi_w = (f"{r['chi_wrong']:.2f}"
                 if np.isfinite(r['chi_wrong']) else "—")
        cohen = (f"{r['cohens_d']:+.3f}"
                 if np.isfinite(r['cohens_d']) else "—")
        rho = (f"{r['spearman_rho']:+.3f}"
               if np.isfinite(r['spearman_rho']) else "—")
        auc_pos = (f"{r['auc_pos']:.3f}"
                   if np.isfinite(r['auc_pos']) else "—")
        auc_neg = (f"{r['auc_neg']:.3f}"
                   if np.isfinite(r['auc_neg']) else "—")

        lines.append("| " + " | ".join([
            r["tag"],
            str(r["n"]),
            f"{r['n_correct']}/{r['n_wrong']}",
            chi_pretty,
            rho,
            chi_c,
            chi_w,
            cohen,
            auc_pos,
            auc_neg,
        ]) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()

    # Single-domain
    ap.add_argument("--problems")
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--spark", default=None)
    ap.add_argument("--tag", default="dataset")

    # Multi-domain
    ap.add_argument("--multi", action="store_true")
    ap.add_argument("--model", help="Model tag, e.g. Qwen3-4B")
    ap.add_argument("--problems_dir", default="data/sanity")
    ap.add_argument("--calibration_dir", default="data/calibrated")
    ap.add_argument("--spark_dir", default="data/spark")
    ap.add_argument("--domains", nargs="+",
                    default=["symbolic", "logical", "algorithmic"])

    # Common
    ap.add_argument("--score_field", default="chi_max")
    ap.add_argument("--d_field", default="d_structural")
    ap.add_argument("--out_md", default=None,
                    help="If set, also write the table to this file.")

    args = ap.parse_args()

    rows = []
    if args.multi:
        if not args.model:
            ap.error("--model is required when --multi")
        for dom in args.domains:
            problems = Path(args.problems_dir) / f"{dom}.jsonl"
            calib = Path(args.calibration_dir) / f"{args.model}__{dom}.jsonl"
            spark = Path(args.spark_dir) / f"{args.model}__{dom}.jsonl"
            if not problems.exists():
                continue
            recs = merge_records(
                problems_path=problems,
                calibration_path=calib if calib.exists() else None,
                spark_path=spark if spark.exists() else None,
            )
            stats = analyze_one(recs, args.score_field, args.d_field)
            stats["tag"] = f"{args.model}__{dom}"
            rows.append(stats)
    else:
        if not args.problems:
            ap.error("--problems is required in single-domain mode")
        recs = merge_records(args.problems, args.calibration, args.spark)
        stats = analyze_one(recs, args.score_field, args.d_field)
        stats["tag"] = args.tag
        rows.append(stats)

    table = format_md_table(rows)
    print(table)

    if args.out_md:
        out = Path(args.out_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write("# χ correctness analysis\n\n")
            f.write(table)
            f.write("\n")
        print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
