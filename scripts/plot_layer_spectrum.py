"""Paper Figure 4 generator: layer-wise χ spectrum.

Each problem has a ``chi_per_layer`` dict (layer_idx -> χ value). This
script aggregates across problems by some grouping (correctness, difficulty
bin, domain) and shows the typical χ profile across the model's layers.

Findings expected from the SPARK theory:
  * χ peaks somewhere in the middle layers (not at L=0 or L=last)
  * the peak position is similar across difficulties (universality)
  * harder problems have a higher overall χ profile

Usage
-----
    python scripts/plot_layer_spectrum.py \
        --problems data/sanity/symbolic.jsonl \
        --calibration data/calibrated/Qwen3-4B__symbolic.jsonl \
        --spark data/spark/Qwen3-4B__symbolic.jsonl \
        --group_by correctness \
        --out_stem data/figures/layer_spectrum__Qwen3-4B__symbolic
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.merge import merge_records
from src.analysis.plotting import PALETTE, apply_paper_style, save_figure


def _records_to_layer_matrix(records, group_key_fn):
    """Returns dict: group_label -> 2D array [n_problems, n_layers].

    ``group_key_fn(rec)`` returns the group label, or None to skip the record.
    """
    grouped = defaultdict(list)
    for r in records:
        cpl = r.get("chi_per_layer")
        if not cpl:
            continue
        # cpl keys are stringified ints (per runner.compute_one).
        try:
            items = sorted(((int(k), float(v)) for k, v in cpl.items()),
                           key=lambda x: x[0])
        except Exception:
            continue
        if not items:
            continue
        layers = [k for k, _ in items]
        values = [v for _, v in items]

        label = group_key_fn(r)
        if label is None:
            continue
        grouped[label].append((layers, values))

    # Convert to padded array per group
    out = {}
    for label, rows in grouped.items():
        # Take intersection of layer indices across rows; safest for variable
        # numbers of layers (different models). For Qwen3-4B it's always 0..36.
        common = sorted(set.intersection(*(set(L) for L, _ in rows)))
        if not common:
            continue
        mat = np.zeros((len(rows), len(common)), dtype=float)
        for i, (L, V) in enumerate(rows):
            ldict = dict(zip(L, V))
            mat[i] = [ldict[c] for c in common]
        out[label] = (np.asarray(common), mat)
    return out


def _group_by_correctness(rec):
    c = rec.get("correct")
    if c is None:
        return None
    return "correct" if int(c) == 1 else "wrong"


def _group_by_d_bin(d_field, n_bins=4):
    """Build a closure that bins records by d_field into n_bins quantiles.
    Caller must pre-compute bin edges from full data — but we use a simpler
    approach: fixed equal-width bins on [0, 1]."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    def fn(rec):
        d = rec.get(d_field)
        if d is None:
            return None
        try:
            d = float(d)
        except (TypeError, ValueError):
            return None
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            if lo <= d <= hi if i == n_bins - 1 else lo <= d < hi:
                return f"d∈[{lo:.2f}, {hi:.2f})"
        return None
    return fn


def plot_spectrum(records, group_key_fn, out_stem, title=None,
                  formats=("png",)):
    import matplotlib.pyplot as plt
    apply_paper_style()

    grouped = _records_to_layer_matrix(records, group_key_fn)
    if not grouped:
        print(f"  WARN: no usable records for {out_stem}")
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    # Sort labels for stable colour assignment
    labels = sorted(grouped.keys())
    for i, label in enumerate(labels):
        layers, mat = grouped[label]
        mean = mat.mean(axis=0)
        n = mat.shape[0]
        if n > 1:
            se = mat.std(axis=0, ddof=1) / np.sqrt(n)
        else:
            se = np.zeros_like(mean)
        colour = PALETTE[i % len(PALETTE)]
        ax.plot(layers, mean, color=colour, marker="o", ms=3, lw=1.3,
                label=f"{label} (n={n})")
        ax.fill_between(layers, mean - se, mean + se,
                        color=colour, alpha=0.18)

    ax.set_xlabel("transformer layer index")
    ax.set_ylabel("χ")
    ax.set_yscale("log")  # χ varies orders of magnitude across layers
    if title:
        ax.set_title(title)
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(alpha=0.2, which="both")

    save_figure(fig, out_stem, formats=formats)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", required=True)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--spark", required=True)
    ap.add_argument("--out_stem", required=True)
    ap.add_argument("--group_by", default="correctness",
                    choices=["correctness", "d_bin"])
    ap.add_argument("--d_field", default="d_structural",
                    help="Used only if --group_by d_bin.")
    ap.add_argument("--n_bins", type=int, default=4,
                    help="Used only if --group_by d_bin.")
    ap.add_argument("--title", default=None)
    ap.add_argument("--formats", nargs="+", default=["png"])
    args = ap.parse_args()

    recs = merge_records(args.problems, args.calibration, args.spark)
    print(f"[spectrum] {len(recs)} merged records")

    if args.group_by == "correctness":
        key_fn = _group_by_correctness
    else:
        key_fn = _group_by_d_bin(args.d_field, n_bins=args.n_bins)

    plot_spectrum(recs, key_fn, args.out_stem, title=args.title,
                  formats=tuple(args.formats))


if __name__ == "__main__":
    main()
