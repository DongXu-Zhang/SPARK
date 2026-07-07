"""Statistics for χ / SPK analysis.

Lightweight pure-numpy/scipy functions that take merged records and emit
correlations, effect sizes, and ROC AUCs.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def _extract_pairs(records: Iterable[dict], score_field: str,
                   target_field: str) -> tuple[np.ndarray, np.ndarray]:
    """Pull two parallel arrays from records, dropping rows where either is missing."""
    xs, ys = [], []
    for r in records:
        x = r.get(score_field)
        y = r.get(target_field)
        if x is None or y is None:
            continue
        if isinstance(x, float) and not math.isfinite(x):
            continue
        xs.append(float(x))
        ys.append(float(y))
    return np.asarray(xs), np.asarray(ys)


# ---------------------------------------------------------------------------
# χ vs continuous difficulty
# ---------------------------------------------------------------------------

def spearman_chi_vs_d(
    records: list[dict],
    score_field: str = "chi_max",
    d_field: str = "d_structural",
) -> dict:
    """Rank correlation between χ and the difficulty coordinate.

    Returns
    -------
    dict with keys: ``rho``, ``n``, ``pearson``.
    """
    xs, ys = _extract_pairs(records, score_field, d_field)
    if len(xs) < 3:
        return {"rho": float("nan"), "n": int(len(xs)), "pearson": float("nan")}

    # Spearman = Pearson on ranks
    rx = _ranks(xs)
    ry = _ranks(ys)
    rho = _pearson(rx, ry)
    pearson = _pearson(xs, ys)
    return {"rho": float(rho), "n": int(len(xs)), "pearson": float(pearson)}


def _ranks(a: np.ndarray) -> np.ndarray:
    order = a.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(a))
    # Average ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    if counts.max() > 1:
        sums = np.bincount(inv, weights=ranks)
        avg = sums / counts
        ranks = avg[inv]
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    if denom == 0:
        return float("nan")
    return float((x * y).sum() / denom)


# ---------------------------------------------------------------------------
# χ vs binary correctness
# ---------------------------------------------------------------------------

def cohens_d(
    records: list[dict],
    score_field: str = "chi_max",
    label_field: str = "correct",
) -> dict:
    """Standardized mean difference of χ between correct and wrong subgroups.

    Sign convention: positive d means wrong-group χ > correct-group χ
    (i.e. χ rises near the capability boundary, the SPARK prediction).
    """
    correct, wrong = [], []
    for r in records:
        x = r.get(score_field)
        y = r.get(label_field)
        if x is None or y is None:
            continue
        if isinstance(x, float) and not math.isfinite(x):
            continue
        (correct if int(y) == 1 else wrong).append(float(x))

    out = {
        "n_correct": len(correct),
        "n_wrong": len(wrong),
        "mean_correct": float("nan"),
        "mean_wrong": float("nan"),
        "d": float("nan"),
    }
    if not correct or not wrong:
        return out

    a = np.asarray(correct)
    b = np.asarray(wrong)
    s_pooled = np.sqrt(((a.var(ddof=1) * (len(a) - 1)) + (b.var(ddof=1) * (len(b) - 1)))
                       / max(len(a) + len(b) - 2, 1))
    out["mean_correct"] = float(a.mean())
    out["mean_wrong"] = float(b.mean())
    if s_pooled == 0:
        out["d"] = float("nan")
    else:
        out["d"] = float((b.mean() - a.mean()) / s_pooled)
    return out


def auc_chi_correctness(
    records: list[dict],
    score_field: str = "chi_max",
    label_field: str = "correct",
    invert: bool = False,
) -> dict:
    """ROC AUC of χ used as a correctness predictor.

    By the SPARK hypothesis χ should be HIGHER on wrong (boundary) examples.
    The convention in this paper is "score predicts correct" (1 = correct),
    so we invert the χ score by default: predict_correct_score = -χ.

    Parameters
    ----------
    invert : if True, use -χ as the score. AUC > 0.5 means χ ranks
             correct-below-wrong (i.e. χ_wrong > χ_correct), supporting SPARK.
    """
    xs, ys = _extract_pairs(records, score_field, label_field)
    if len(xs) < 3 or len(np.unique(ys)) < 2:
        return {"auc": float("nan"), "n": int(len(xs))}

    score = -xs if invert else xs
    auc = _roc_auc(score, ys.astype(int))
    return {"auc": float(auc), "n": int(len(xs))}


def _roc_auc(score: np.ndarray, label: np.ndarray) -> float:
    """Compute ROC AUC using the rank-sum formula. label in {0,1}."""
    n_pos = int((label == 1).sum())
    n_neg = int((label == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score)
    ranked = label[order]
    # Sum of ranks of positive class (1-indexed)
    ranks_of_pos = np.where(ranked == 1)[0] + 1
    rank_sum = ranks_of_pos.sum()
    auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)
