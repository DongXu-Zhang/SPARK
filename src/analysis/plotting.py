"""Shared matplotlib styling for paper figures.

Importing this module configures sensible defaults: serif fonts, larger
labels, no chartjunk. Individual scripts call ``apply_paper_style()`` once
at the top.
"""
from __future__ import annotations

import math
from typing import Optional

import matplotlib

# Use a non-interactive backend so scripts work over SSH / on headless servers.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def apply_paper_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 100,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


# A small colour cycle for up to 4 conditions.
PALETTE = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#ff7f0e",  # orange
]


def binned_mean_se(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> tuple:
    """Bin x into n_bins equal-width buckets, return (centre, mean, se, n)."""
    if len(x) == 0:
        return (np.array([]),) * 4

    bins = np.linspace(x.min(), x.max(), n_bins + 1)
    centres = 0.5 * (bins[:-1] + bins[1:])
    means = np.full(n_bins, np.nan)
    ses = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (x >= bins[i]) & (x <= bins[i + 1])
        else:
            mask = (x >= bins[i]) & (x < bins[i + 1])
        ys = y[mask]
        if len(ys) > 0:
            means[i] = float(ys.mean())
            counts[i] = len(ys)
            if len(ys) > 1:
                ses[i] = float(ys.std(ddof=1) / math.sqrt(len(ys)))
            else:
                ses[i] = 0.0
    return centres, means, ses, counts


def save_figure(fig, out_path: str, formats: tuple = ("png",)):
    """Save the figure in one or more formats. ``out_path`` is the stem."""
    from pathlib import Path
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        target = p.with_suffix(f".{fmt}")
        fig.savefig(target)
        print(f"  saved -> {target}")
