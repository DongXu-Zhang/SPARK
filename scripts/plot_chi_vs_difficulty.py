"""Paper Figure 2 generator: χ vs difficulty (the bell-curve test).

For each (model, domain) combination, plots:
  * scatter of every problem (small alpha=0.4 dots)
  * binned mean ± standard error (large filled circles + error bars)

The expected SPARK signature is a bell shape — χ peaks somewhere in the
middle of the difficulty axis and decays toward both ends.

Usage
-----
Single domain:

    python scripts/plot_chi_vs_difficulty.py \
        --problems data/sanity/symbolic.jsonl \
        --calibration data/calibrated/Qwen3-4B__symbolic.jsonl \
        --spark data/spark/Qwen3-4B__symbolic.jsonl \
        --d_field d_structural \
        --score_field chi_max \
        --out_stem data/figures/chi_vs_d__Qwen3-4B__symbolic

Multi-domain (one figure per domain, plus a combined panel):

    python scripts/plot_chi_vs_difficulty.py \
        --multi --model Qwen3-4B \
        --problems_dir data/sanity \
        --calibration_dir data/calibrated \
        --spark_dir data/spark \
        --out_dir data/figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make `src` importable when run as a script
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.merge import merge_records
from src.analysis.plotting import (
    PALETTE,
    apply_paper_style,
    binned_mean_se,
    save_figure,
)
from src.analysis.stats import auc_chi_correctness, cohens_d, spearman_chi_vs_d


def _filter_records(records, score_field, d_field):
    """Drop records missing the required fields or with non-finite values."""
    out = []
    for r in records:
        s = r.get(score_field)
        d = r.get(d_field)
        if s is None or d is None:
            continue
        try:
            sf = float(s)
            df = float(d)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(sf) and np.isfinite(df)):
            continue
        out.append((sf, df, r))
    return out


def plot_one_domain(records, score_field, d_field, out_stem,
                    title=None, n_bins=10, formats=("png",)):
    """Make a single χ-vs-d figure and save."""
    import matplotlib.pyplot as plt

    apply_paper_style()
    filtered = _filter_records(records, score_field, d_field)
    if not filtered:
        print(f"  WARN: no usable records for {out_stem}")
        return

    s = np.array([t[0] for t in filtered])
    d = np.array([t[1] for t in filtered])
    correct = np.array([
        int(t[2].get("correct", -1)) for t in filtered
    ])

    centres, means, ses, counts = binned_mean_se(d, s, n_bins=n_bins)

    # Compute summary statistics
    spearman = spearman_chi_vs_d(
        [t[2] for t in filtered], score_field=score_field, d_field=d_field
    )
    has_correct = (correct >= 0).any()
    if has_correct:
        cd = cohens_d(
            [t[2] for t in filtered], score_field=score_field, label_field="correct"
        )
        auc = auc_chi_correctness(
            [t[2] for t in filtered], score_field=score_field, label_field="correct",
            invert=False,
        )
    else:
        cd = {"d": float("nan"), "n_correct": 0, "n_wrong": 0}
        auc = {"auc": float("nan")}

    # Plot
    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    # Scatter all points, coloured by correctness if available
    if has_correct:
        wrong_mask = correct == 0
        right_mask = correct == 1
        ax.scatter(d[right_mask], s[right_mask], s=18, alpha=0.45,
                   color=PALETTE[0], edgecolors="none", label=f"correct (n={right_mask.sum()})")
        ax.scatter(d[wrong_mask], s[wrong_mask], s=22, alpha=0.65,
                   color=PALETTE[1], marker="x", label=f"wrong (n={wrong_mask.sum()})")
    else:
        ax.scatter(d, s, s=18, alpha=0.45, color=PALETTE[0],
                   edgecolors="none", label=f"all (n={len(d)})")

    # Binned mean
    valid = ~np.isnan(means)
    if valid.any():
        ax.errorbar(centres[valid], means[valid], yerr=ses[valid],
                    fmt="o-", color="black", ms=6, lw=1.5,
                    capsize=3, label=f"binned mean ({n_bins} bins)")

    ax.set_xlabel(f"difficulty ({d_field})")
    ax.set_ylabel(f"χ ({score_field})")
    if title:
        ax.set_title(title)

    # Annotation block
    annot = (f"Spearman ρ = {spearman['rho']:+.3f}  (n={spearman['n']})\n"
             f"Cohen's d = {cd['d']:+.3f}\n"
             f"ROC AUC (χ→correct) = {auc['auc']:.3f}")
    ax.text(0.02, 0.98, annot, transform=ax.transAxes,
            ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="lightgray", alpha=0.9),
            fontsize=9, family="monospace")

    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.2)

    save_figure(fig, out_stem, formats=formats)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    sub_single = ap.add_argument_group("single-domain mode")
    sub_single.add_argument("--problems")
    sub_single.add_argument("--calibration", default=None)
    sub_single.add_argument("--spark", required=False, default=None)
    sub_single.add_argument("--out_stem")

    sub_multi = ap.add_argument_group("multi-domain mode")
    sub_multi.add_argument("--multi", action="store_true",
                           help="Run on all standard domains for a given model.")
    sub_multi.add_argument("--model", help="Model tag, e.g. Qwen3-4B")
    sub_multi.add_argument("--problems_dir", default="data/sanity")
    sub_multi.add_argument("--calibration_dir", default="data/calibrated")
    sub_multi.add_argument("--spark_dir", default="data/spark")
    sub_multi.add_argument("--out_dir", default="data/figures")
    sub_multi.add_argument("--domains", nargs="+",
                           default=["symbolic", "logical", "algorithmic"])

    ap.add_argument("--d_field", default="d_structural",
                    help="Which difficulty column to use on x-axis "
                         "(d_structural | d_empirical | d_final).")
    ap.add_argument("--score_field", default="chi_max",
                    help="Which χ aggregator to plot (chi_max | chi_mean | spk).")
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument("--formats", nargs="+", default=["png"],
                    help="Output formats: png pdf svg")

    args = ap.parse_args()

    if args.multi:
        if not args.model:
            ap.error("--model is required when --multi is set")
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for dom in args.domains:
            problems = Path(args.problems_dir) / f"{dom}.jsonl"
            calib = Path(args.calibration_dir) / f"{args.model}__{dom}.jsonl"
            spark = Path(args.spark_dir) / f"{args.model}__{dom}.jsonl"
            if not problems.exists():
                print(f"[skip] {dom}: {problems} missing")
                continue
            recs = merge_records(
                problems_path=problems,
                calibration_path=calib if calib.exists() else None,
                spark_path=spark if spark.exists() else None,
            )
            stem = out_dir / f"chi_vs_d__{args.model}__{dom}"
            print(f"[plot] {dom}: {len(recs)} records -> {stem}")
            plot_one_domain(
                recs, args.score_field, args.d_field, str(stem),
                title=f"{args.model}  ·  {dom}",
                n_bins=args.n_bins, formats=tuple(args.formats),
            )
    else:
        if not args.problems or not args.out_stem:
            ap.error("In single-domain mode, --problems and --out_stem are required")
        recs = merge_records(
            problems_path=args.problems,
            calibration_path=args.calibration,
            spark_path=args.spark,
        )
        print(f"[plot] {len(recs)} merged records -> {args.out_stem}")
        plot_one_domain(
            recs, args.score_field, args.d_field, args.out_stem,
            n_bins=args.n_bins, formats=tuple(args.formats),
        )


if __name__ == "__main__":
    main()
