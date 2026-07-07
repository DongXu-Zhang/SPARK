"""Utilities for length-controlled susceptibility analysis.

Raw chi is sensitive to prompt token length. This module provides the shared
logic used by diagnostics, steering-set construction, and later intervention
experiments:

    log(chi) = a + b * log(n_tokens)
    chi_lc   = log(chi) - (a + b * log(n_tokens))

``chi_lc`` is the length-controlled residual. Positive values mean a problem
elicits stronger latent response than expected for its length; negative values
mean under-activation relative to same-length inputs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np

from src.analysis.merge import merge_records


@dataclass
class LengthControlFit:
    """Linear fit for log(chi) against log(prompt length)."""

    intercept: float
    slope: float
    r2: float
    n: int
    score_field: str
    token_field: str

    def predict_log_chi(self, n_tokens: float) -> float:
        return self.intercept + self.slope * math.log(float(n_tokens))

    def to_dict(self) -> dict:
        return asdict(self)


def as_float(value) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def as_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def fmt(value, digits: int = 3) -> str:
    x = as_float(value)
    if x is None:
        return "NA"
    return f"{x:.{digits}f}"


def fmt_pct(value) -> str:
    x = as_float(value)
    if x is None:
        return "NA"
    return f"{100.0 * x:.1f}%"


def ranks(values: Iterable[float]) -> np.ndarray:
    a = np.asarray(list(values), dtype=float)
    order = a.argsort()
    r = np.empty_like(order, dtype=float)
    r[order] = np.arange(len(a), dtype=float)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    if len(counts) and counts.max() > 1:
        sums = np.bincount(inv, weights=r)
        r = (sums / counts)[inv]
    return r


def pearson(x: Iterable[float], y: Iterable[float]) -> float:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    if len(x_arr) < 3:
        return float("nan")
    x_arr = x_arr - x_arr.mean()
    y_arr = y_arr - y_arr.mean()
    denom = math.sqrt(float((x_arr * x_arr).sum() * (y_arr * y_arr).sum()))
    if denom == 0:
        return float("nan")
    return float((x_arr * y_arr).sum() / denom)


def spearman(x: Iterable[float], y: Iterable[float]) -> float:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    if len(x_arr) < 3:
        return float("nan")
    return pearson(ranks(x_arr), ranks(y_arr))


def _valid_xy(records: list[dict], score_field: str, token_field: str) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for r in records:
        score = as_float(r.get(score_field))
        toks = as_float(r.get(token_field))
        if score is None or toks is None:
            continue
        if score <= 0 or toks <= 0:
            continue
        xs.append(math.log(toks))
        ys.append(math.log(score))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def fit_length_control(
    records: list[dict],
    score_field: str = "chi_max",
    token_field: str = "n_tokens",
) -> LengthControlFit:
    """Fit log(score) ~ log(n_tokens)."""
    x, y = _valid_xy(records, score_field, token_field)
    if len(x) < 3:
        return LengthControlFit(
            intercept=float("nan"),
            slope=float("nan"),
            r2=float("nan"),
            n=int(len(x)),
            score_field=score_field,
            token_field=token_field,
        )
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return LengthControlFit(
        intercept=float(beta[0]),
        slope=float(beta[1]),
        r2=float(r2),
        n=int(len(x)),
        score_field=score_field,
        token_field=token_field,
    )


def add_length_control_fields(
    records: list[dict],
    fit: LengthControlFit,
    score_field: str = "chi_max",
    d_field: str = "d_structural",
    token_field: str = "n_tokens",
) -> list[dict]:
    """Return records with length-control fields added.

    Records missing positive score/token values are dropped. The original keys
    are preserved; additional keys are prefixed with ``lc_`` where appropriate.
    """
    out = []
    for r in records:
        score = as_float(r.get(score_field))
        toks = as_float(r.get(token_field))
        d = as_float(r.get(d_field))
        if score is None or toks is None or score <= 0 or toks <= 0:
            continue

        pred = fit.predict_log_chi(toks)
        log_score = math.log(score)
        resid = log_score - pred
        rec = dict(r)
        rec["d_value"] = d
        rec["lc_score_field"] = score_field
        rec["lc_token_field"] = token_field
        rec["lc_log_chi"] = log_score
        rec["lc_log_tokens"] = math.log(toks)
        rec["lc_pred_log_chi"] = pred
        rec["chi_lc"] = resid
        rec["chi_ratio_len"] = math.exp(resid)
        rec["chi_expected_for_length"] = math.exp(pred)

        correct = as_int(rec.get("correct"))
        if correct in (0, 1):
            rec["correct"] = correct
        out.append(rec)
    return out


def merge_and_length_control(
    model: str,
    domain: str,
    raw_dir: str | Path,
    calibration_dir: str | Path,
    spark_dir: str | Path,
    score_field: str = "chi_max",
    d_field: str = "d_structural",
    token_field: str = "n_tokens",
) -> tuple[list[dict], LengthControlFit]:
    """Merge problem/calibration/SPARK files and add chi_lc fields."""
    raw_dir = Path(raw_dir)
    calibration_dir = Path(calibration_dir)
    spark_dir = Path(spark_dir)
    records = merge_records(
        raw_dir / f"{domain}.jsonl",
        calibration_dir / f"{model}__{domain}.jsonl",
        spark_dir / f"{model}__{domain}.jsonl",
        require_all=True,
    )
    fit = fit_length_control(records, score_field=score_field, token_field=token_field)
    return add_length_control_fields(
        records,
        fit,
        score_field=score_field,
        d_field=d_field,
        token_field=token_field,
    ), fit


def fixed_difficulty_bins(records: list[dict], n_bins: int = 10) -> list[dict]:
    rows = []
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        items = [
            r for r in records
            if r.get("d_value") is not None
            and (lo <= float(r["d_value"]) <= hi if i == n_bins - 1 else lo <= float(r["d_value"]) < hi)
        ]
        correct = [as_int(r.get("correct")) for r in items]
        correct = [c for c in correct if c in (0, 1)]
        rows.append({
            "bin": i,
            "lo": lo,
            "hi": hi,
            "center": (lo + hi) / 2.0,
            "records": items,
            "n": len(items),
            "acc": (sum(correct) / len(correct)) if correct else None,
        })
    return rows


def estimate_dstar(records: list[dict], n_bins: int = 10, min_bin_count: int = 20) -> tuple[float | None, str]:
    bins = fixed_difficulty_bins(records, n_bins=n_bins)
    valid = [b for b in bins if b["n"] >= min_bin_count and b["acc"] is not None]
    if not valid:
        return None, "no valid bins"
    for a, b in zip(valid, valid[1:]):
        y1 = float(a["acc"]) - 0.5
        y2 = float(b["acc"]) - 0.5
        if abs(y1) < 1e-12:
            return float(a["center"]), "exact"
        if y1 * y2 < 0:
            t = (0.5 - float(a["acc"])) / (float(b["acc"]) - float(a["acc"]))
            return float(a["center"]) + t * (float(b["center"]) - float(a["center"])), "interpolated"
    if min(float(b["acc"]) for b in valid) > 0.5:
        return None, "too easy"
    if max(float(b["acc"]) for b in valid) < 0.5:
        return None, "too hard"
    return None, "no adjacent crossing"


def mean_or_none(values: Iterable[float | None]) -> float | None:
    xs = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def std_or_none(values: Iterable[float | None]) -> float | None:
    xs = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(xs) < 2:
        return 0.0 if xs else None
    return float(np.asarray(xs, dtype=float).std(ddof=1))

