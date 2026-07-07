"""Length-control diagnostics for SPARK/chi experiments.

The current full-scale results show that chi decreases on harder problems,
especially in Algorithmic Reasoning. This script tests whether that pattern is
mostly a prompt-length artifact or remains after controlling for token length.

Inputs:
  - data/raw/<domain>.jsonl
  - data/calibrated_full/<model>__<domain>.jsonl
  - data/spark_full/<model>__<domain>.jsonl

Outputs:
  - Markdown report with correlations, length regression, and collapse summary
  - CSV tables for fixed difficulty bins and token-quantile bins
  - PNG plots:
      * chi vs n_tokens
      * residualized chi vs difficulty
      * chi vs difficulty within token quantile groups
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.merge import merge_records  # noqa: E402


DOMAINS = ["symbolic", "logical", "algorithmic"]


@dataclass
class Regression:
    intercept: float
    slope: float
    r2: float


def as_float(value):
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def fmt(value, digits=3):
    if value is None:
        return "NA"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"


def fmt_pct(value):
    if value is None:
        return "NA"
    return f"{100.0 * value:.1f}%"


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(x).replace("|", "\\|") for x in row) + " |")
    return "\n".join(lines)


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


def ranks(values: Iterable[float]) -> np.ndarray:
    a = np.asarray(list(values), dtype=float)
    order = a.argsort()
    r = np.empty_like(order, dtype=float)
    r[order] = np.arange(len(a), dtype=float)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    if counts.max() > 1:
        sums = np.bincount(inv, weights=r)
        r = (sums / counts)[inv]
    return r


def spearman(x: Iterable[float], y: Iterable[float]) -> float:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    if len(x_arr) < 3:
        return float("nan")
    return pearson(ranks(x_arr), ranks(y_arr))


def linear_regression(x: Iterable[float], y: Iterable[float]) -> Regression:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    if len(x_arr) < 3:
        return Regression(float("nan"), float("nan"), float("nan"))
    X = np.column_stack([np.ones_like(x_arr), x_arr])
    beta, *_ = np.linalg.lstsq(X, y_arr, rcond=None)
    pred = X @ beta
    ss_res = float(((y_arr - pred) ** 2).sum())
    ss_tot = float(((y_arr - y_arr.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return Regression(float(beta[0]), float(beta[1]), float(r2))


def enrich_records(records: list[dict], score_field: str, d_field: str) -> list[dict]:
    enriched = []
    for r in records:
        d = as_float(r.get(d_field))
        chi = as_float(r.get(score_field))
        toks = as_float(r.get("n_tokens"))
        correct = as_int(r.get("correct"))
        if d is None or chi is None or toks is None:
            continue
        if chi <= 0 or toks <= 0:
            continue
        rec = dict(r)
        rec["_d"] = d
        rec["_chi"] = chi
        rec["_log_chi"] = math.log(chi)
        rec["_tokens"] = toks
        rec["_log_tokens"] = math.log(toks)
        rec["_correct"] = correct if correct in (0, 1) else None
        enriched.append(rec)

    if len(enriched) < 3:
        return enriched

    reg = linear_regression(
        (r["_log_tokens"] for r in enriched),
        (r["_log_chi"] for r in enriched),
    )
    for r in enriched:
        pred = reg.intercept + reg.slope * r["_log_tokens"]
        r["_log_chi_pred_len"] = pred
        r["_log_chi_resid_len"] = r["_log_chi"] - pred
        # A multiplicative residual. 1.0 means exactly expected chi for length.
        r["_chi_ratio_len"] = math.exp(r["_log_chi_resid_len"])
    return enriched


def estimate_dstar(records: list[dict], n_bins: int, min_bin_count: int) -> tuple[float | None, str]:
    rows = difficulty_bins(records, n_bins)
    valid = [b for b in rows if b["n"] >= min_bin_count and b["acc"] is not None]
    if not valid:
        return None, "no valid bins"
    for a, b in zip(valid, valid[1:]):
        y1 = a["acc"] - 0.5
        y2 = b["acc"] - 0.5
        if abs(y1) < 1e-12:
            return a["center"], "exact"
        if y1 * y2 < 0:
            t = (0.5 - a["acc"]) / (b["acc"] - a["acc"])
            return a["center"] + t * (b["center"] - a["center"]), "interpolated"
    if min(b["acc"] for b in valid) > 0.5:
        return None, "too easy"
    if max(b["acc"] for b in valid) < 0.5:
        return None, "too hard"
    return None, "no adjacent crossing"


def stat(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return mean(values), pstdev(values) if len(values) > 1 else 0.0


def difficulty_bins(records: list[dict], n_bins: int) -> list[dict]:
    out = []
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        rows = [
            r for r in records
            if (lo <= r["_d"] <= hi if i == n_bins - 1 else lo <= r["_d"] < hi)
        ]
        corr = [r["_correct"] for r in rows if r["_correct"] in (0, 1)]
        acc = sum(corr) / len(corr) if corr else None
        chi_m, chi_s = stat([r["_chi"] for r in rows])
        resid_m, resid_s = stat([r["_log_chi_resid_len"] for r in rows])
        ratio_m, ratio_s = stat([r["_chi_ratio_len"] for r in rows])
        tok_m, tok_s = stat([r["_tokens"] for r in rows])
        phi_m, _ = stat([as_float(r.get("phi")) for r in rows if as_float(r.get("phi")) is not None])
        spk_m, _ = stat([as_float(r.get("spk")) for r in rows if as_float(r.get("spk")) is not None])
        out.append({
            "bin": i,
            "lo": lo,
            "hi": hi,
            "center": (lo + hi) / 2.0,
            "n": len(rows),
            "acc": acc,
            "chi_mean": chi_m,
            "chi_std": chi_s,
            "log_chi_resid_mean": resid_m,
            "log_chi_resid_std": resid_s,
            "chi_ratio_len_mean": ratio_m,
            "chi_ratio_len_std": ratio_s,
            "n_tokens_mean": tok_m,
            "n_tokens_std": tok_s,
            "phi_mean": phi_m,
            "spk_mean": spk_m,
        })
    return out


def token_quantile_bins(records: list[dict], n_bins: int) -> list[dict]:
    if not records:
        return []
    toks = np.asarray([r["_tokens"] for r in records], dtype=float)
    edges = np.quantile(toks, np.linspace(0.0, 1.0, n_bins + 1))
    # Guard against repeated edges; still keep stable labels.
    out = []
    for i in range(n_bins):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if i == n_bins - 1:
            rows = [r for r in records if lo <= r["_tokens"] <= hi]
        else:
            rows = [r for r in records if lo <= r["_tokens"] < hi]
        if len(rows) < 3:
            rho_chi_d = float("nan")
            rho_resid_d = float("nan")
        else:
            rho_chi_d = spearman((r["_chi"] for r in rows), (r["_d"] for r in rows))
            rho_resid_d = spearman(
                (r["_log_chi_resid_len"] for r in rows),
                (r["_d"] for r in rows),
            )
        corr = [r["_correct"] for r in rows if r["_correct"] in (0, 1)]
        acc = sum(corr) / len(corr) if corr else None
        out.append({
            "qbin": i,
            "token_lo": lo,
            "token_hi": hi,
            "n": len(rows),
            "acc": acc,
            "d_min": min((r["_d"] for r in rows), default=None),
            "d_max": max((r["_d"] for r in rows), default=None),
            "chi_mean": mean([r["_chi"] for r in rows]) if rows else None,
            "resid_mean": mean([r["_log_chi_resid_len"] for r in rows]) if rows else None,
            "spearman_chi_d": rho_chi_d,
            "spearman_resid_d": rho_resid_d,
        })
    return out


def peak_to_hard_drop(bin_rows: list[dict], min_bin_count: int) -> dict:
    valid = [b for b in bin_rows if b["n"] >= min_bin_count and b["chi_mean"] is not None]
    if not valid:
        return {}
    peak = max(valid, key=lambda b: b["chi_mean"])
    hard = max(valid, key=lambda b: b["center"])
    return {
        "chi_peak_d": peak["center"],
        "chi_peak": peak["chi_mean"],
        "hard_d": hard["center"],
        "hard_chi": hard["chi_mean"],
        "chi_drop_hard_minus_peak": hard["chi_mean"] - peak["chi_mean"],
        "resid_peak": peak["log_chi_resid_mean"],
        "resid_hard": hard["log_chi_resid_mean"],
        "resid_drop_hard_minus_peak": (
            None
            if peak["log_chi_resid_mean"] is None or hard["log_chi_resid_mean"] is None
            else hard["log_chi_resid_mean"] - peak["log_chi_resid_mean"]
        ),
        "ratio_peak": peak["chi_ratio_len_mean"],
        "ratio_hard": hard["chi_ratio_len_mean"],
        "ratio_drop_hard_minus_peak": (
            None
            if peak["chi_ratio_len_mean"] is None or hard["chi_ratio_len_mean"] is None
            else hard["chi_ratio_len_mean"] - peak["chi_ratio_len_mean"]
        ),
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_domain(domain: str, model: str, records: list[dict], dstar, out_dir: Path) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[length-control] plot skipped for {domain}: {e}", file=sys.stderr)
        return []

    saved = []
    out_dir.mkdir(parents=True, exist_ok=True)

    d = np.asarray([r["_d"] for r in records], dtype=float)
    chi = np.asarray([r["_chi"] for r in records], dtype=float)
    toks = np.asarray([r["_tokens"] for r in records], dtype=float)
    resid = np.asarray([r["_log_chi_resid_len"] for r in records], dtype=float)
    correct = np.asarray([
        -1 if r["_correct"] is None else int(r["_correct"]) for r in records
    ])

    # 1. chi vs token length
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    mask_c = correct == 1
    mask_w = correct == 0
    ax.scatter(toks[mask_c], chi[mask_c], s=12, alpha=0.35, label=f"correct (n={mask_c.sum()})")
    ax.scatter(toks[mask_w], chi[mask_w], s=16, marker="x", alpha=0.5, label=f"wrong (n={mask_w.sum()})")
    ax.set_xscale("log")
    ax.set_xlabel("n_tokens (log scale)")
    ax.set_ylabel("chi_max")
    ax.set_title(f"{model} - {domain}: chi vs prompt length")
    ax.grid(alpha=0.25, which="both")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    path = out_dir / f"chi_vs_tokens__{model}__{domain}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    saved.append(path)

    # 2. length-residualized log chi vs difficulty
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.scatter(d[mask_c], resid[mask_c], s=12, alpha=0.35, label=f"correct (n={mask_c.sum()})")
    ax.scatter(d[mask_w], resid[mask_w], s=16, marker="x", alpha=0.5, label=f"wrong (n={mask_w.sum()})")
    bins = difficulty_bins(records, 10)
    xs = [b["center"] for b in bins if b["log_chi_resid_mean"] is not None and b["n"] > 0]
    ys = [b["log_chi_resid_mean"] for b in bins if b["log_chi_resid_mean"] is not None and b["n"] > 0]
    ax.plot(xs, ys, color="black", marker="o", linewidth=1.8, label="binned mean")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    if dstar is not None:
        ax.axvline(dstar, color="#d62728", linestyle=":", linewidth=1.4, label=f"d*={dstar:.3f}")
    ax.set_xlabel("difficulty (d_structural)")
    ax.set_ylabel("log chi residual after length control")
    ax.set_title(f"{model} - {domain}: length-controlled chi vs difficulty")
    ax.grid(alpha=0.25)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    path = out_dir / f"chi_residual_vs_d__{model}__{domain}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    saved.append(path)

    # 3. chi vs difficulty within token quantile groups
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    q_edges = np.quantile(toks, np.linspace(0.0, 1.0, 5))
    for i in range(4):
        lo = q_edges[i]
        hi = q_edges[i + 1]
        if i == 3:
            group = [r for r in records if lo <= r["_tokens"] <= hi]
        else:
            group = [r for r in records if lo <= r["_tokens"] < hi]
        if len(group) < 5:
            continue
        bins_g = difficulty_bins(group, 10)
        xs = [b["center"] for b in bins_g if b["chi_mean"] is not None and b["n"] > 0]
        ys = [b["chi_mean"] for b in bins_g if b["chi_mean"] is not None and b["n"] > 0]
        ax.plot(xs, ys, marker="o", linewidth=1.4, label=f"tokens Q{i+1} [{lo:.0f},{hi:.0f}]")
    if dstar is not None:
        ax.axvline(dstar, color="#d62728", linestyle=":", linewidth=1.4)
    ax.set_xlabel("difficulty (d_structural)")
    ax.set_ylabel("chi_max")
    ax.set_title(f"{model} - {domain}: chi vs d within length bins")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    path = out_dir / f"chi_vs_d_by_length__{model}__{domain}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    saved.append(path)

    return saved


def domain_report(
    domain: str,
    model: str,
    records: list[dict],
    n_bins: int,
    token_bins: int,
    min_bin_count: int,
    out_dir: Path,
) -> tuple[list[str], list[dict], list[dict], dict]:
    dstar, dstar_note = estimate_dstar(records, n_bins, min_bin_count)
    reg = linear_regression(
        (r["_log_tokens"] for r in records),
        (r["_log_chi"] for r in records),
    )

    rho_chi_d = spearman((r["_chi"] for r in records), (r["_d"] for r in records))
    rho_chi_tokens = spearman((r["_chi"] for r in records), (r["_tokens"] for r in records))
    rho_resid_d = spearman(
        (r["_log_chi_resid_len"] for r in records),
        (r["_d"] for r in records),
    )
    rho_resid_tokens = spearman(
        (r["_log_chi_resid_len"] for r in records),
        (r["_tokens"] for r in records),
    )

    corr = [r["_correct"] for r in records if r["_correct"] in (0, 1)]
    acc = sum(corr) / len(corr) if corr else None

    d_bins = difficulty_bins(records, n_bins)
    q_bins = token_quantile_bins(records, token_bins)
    collapse = peak_to_hard_drop(d_bins, min_bin_count)

    plot_paths = plot_domain(domain, model, records, dstar, out_dir)

    interpretation = "inconclusive"
    resid_drop = collapse.get("resid_drop_hard_minus_peak")
    if resid_drop is not None:
        if resid_drop < -0.10 and rho_resid_d < -0.15:
            interpretation = "supports length-independent activation collapse"
        elif abs(rho_resid_d) < 0.10 and abs(resid_drop) < 0.10:
            interpretation = "mostly explained by prompt length"
        else:
            interpretation = "mixed: length matters, residual trend remains"

    lines = [
        f"## {domain}",
        "",
        f"- n: {len(records)}",
        f"- accuracy: {fmt_pct(acc)}",
        f"- dstar_50: {fmt(dstar)} ({dstar_note})",
        f"- Spearman chi vs difficulty: {fmt(rho_chi_d)}",
        f"- Spearman chi vs n_tokens: {fmt(rho_chi_tokens)}",
        f"- Length regression: log(chi) = {fmt(reg.intercept)} + {fmt(reg.slope)} * log(n_tokens), R2={fmt(reg.r2)}",
        f"- Spearman length-residual chi vs difficulty: {fmt(rho_resid_d)}",
        f"- Spearman length-residual chi vs n_tokens: {fmt(rho_resid_tokens)}",
        f"- chi_peak_d: {fmt(collapse.get('chi_peak_d'))}",
        f"- hard_d: {fmt(collapse.get('hard_d'))}",
        f"- raw chi hard-minus-peak: {fmt(collapse.get('chi_drop_hard_minus_peak'))}",
        f"- residual hard-minus-peak: {fmt(collapse.get('resid_drop_hard_minus_peak'))}",
        f"- chi ratio hard-minus-peak: {fmt(collapse.get('ratio_drop_hard_minus_peak'))}",
        f"- interpretation: **{interpretation}**",
        "",
        "### Difficulty bins",
        "",
    ]

    d_rows = []
    for b in d_bins:
        d_rows.append([
            b["bin"],
            f"[{b['lo']:.2f},{b['hi']:.2f})" if b["bin"] < n_bins - 1 else f"[{b['lo']:.2f},{b['hi']:.2f}]",
            b["n"],
            fmt_pct(b["acc"]),
            fmt(b["chi_mean"]),
            fmt(b["log_chi_resid_mean"]),
            fmt(b["chi_ratio_len_mean"]),
            fmt(b["n_tokens_mean"], 1),
            fmt(b["spk_mean"]),
        ])
    lines.append(md_table(
        ["bin", "d_range", "n", "acc", "chi", "log_chi_resid", "chi_ratio_len", "tokens", "spk"],
        d_rows,
    ))
    lines.extend(["", "### Token quantile bins", ""])

    q_rows = []
    for b in q_bins:
        q_rows.append([
            b["qbin"],
            f"[{b['token_lo']:.0f},{b['token_hi']:.0f}]",
            b["n"],
            fmt_pct(b["acc"]),
            f"{fmt(b['d_min'])}-{fmt(b['d_max'])}",
            fmt(b["chi_mean"]),
            fmt(b["resid_mean"]),
            fmt(b["spearman_chi_d"]),
            fmt(b["spearman_resid_d"]),
        ])
    lines.append(md_table(
        ["qbin", "token_range", "n", "acc", "d_range", "chi", "resid", "rho(chi,d)", "rho(resid,d)"],
        q_rows,
    ))
    lines.extend(["", "### Plots", ""])
    for p in plot_paths:
        lines.append(f"- `{p}`")
    lines.append("")

    summary = {
        "domain": domain,
        "n": len(records),
        "accuracy": acc,
        "dstar": dstar,
        "dstar_note": dstar_note,
        "rho_chi_d": rho_chi_d,
        "rho_chi_tokens": rho_chi_tokens,
        "len_reg_slope": reg.slope,
        "len_reg_r2": reg.r2,
        "rho_resid_d": rho_resid_d,
        "rho_resid_tokens": rho_resid_tokens,
        "chi_peak_d": collapse.get("chi_peak_d"),
        "hard_d": collapse.get("hard_d"),
        "chi_drop_hard_minus_peak": collapse.get("chi_drop_hard_minus_peak"),
        "resid_drop_hard_minus_peak": collapse.get("resid_drop_hard_minus_peak"),
        "interpretation": interpretation,
    }
    return lines, d_bins, q_bins, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-4B")
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--calibration_dir", default="data/calibrated_full")
    ap.add_argument("--spark_dir", default="data/spark_full")
    ap.add_argument("--out_dir", default="figures/length_control_Qwen3-4B")
    ap.add_argument("--domains", nargs="+", default=DOMAINS)
    ap.add_argument("--d_field", default="d_structural")
    ap.add_argument("--score_field", default="chi_max")
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument("--token_bins", type=int, default=5)
    ap.add_argument("--min_bin_count", type=int, default=20)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    calibration_dir = Path(args.calibration_dir)
    spark_dir = Path(args.spark_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_parts = [
        f"# Length-Control Diagnostics: {args.model}",
        "",
        f"- score_field: `{args.score_field}`",
        f"- d_field: `{args.d_field}`",
        f"- spark_dir: `{spark_dir}`",
        "",
    ]
    all_d_bins = []
    all_q_bins = []
    summaries = []

    for domain in args.domains:
        problems = raw_dir / f"{domain}.jsonl"
        calibration = calibration_dir / f"{args.model}__{domain}.jsonl"
        spark = spark_dir / f"{args.model}__{domain}.jsonl"
        if not problems.exists() or not calibration.exists() or not spark.exists():
            report_parts.extend([
                f"## {domain}",
                "",
                f"Skipped because one input is missing: `{problems}`, `{calibration}`, `{spark}`",
                "",
            ])
            continue
        merged = merge_records(problems, calibration, spark, require_all=True)
        records = enrich_records(merged, args.score_field, args.d_field)
        lines, d_bins, q_bins, summary = domain_report(
            domain,
            args.model,
            records,
            args.n_bins,
            args.token_bins,
            args.min_bin_count,
            out_dir,
        )
        report_parts.extend(lines)
        summaries.append(summary)
        for b in d_bins:
            b = dict(b)
            b["domain"] = domain
            all_d_bins.append(b)
        for b in q_bins:
            b = dict(b)
            b["domain"] = domain
            all_q_bins.append(b)

    summary_rows = []
    for s in summaries:
        summary_rows.append([
            s["domain"],
            s["n"],
            fmt_pct(s["accuracy"]),
            fmt(s["dstar"]),
            fmt(s["rho_chi_d"]),
            fmt(s["rho_chi_tokens"]),
            fmt(s["len_reg_slope"]),
            fmt(s["len_reg_r2"]),
            fmt(s["rho_resid_d"]),
            fmt(s["chi_peak_d"]),
            fmt(s["hard_d"]),
            fmt(s["chi_drop_hard_minus_peak"]),
            fmt(s["resid_drop_hard_minus_peak"]),
            s["interpretation"],
        ])
    summary_block = [
        "## Summary",
        "",
        md_table(
            [
                "domain", "n", "acc", "dstar", "rho(chi,d)", "rho(chi,tokens)",
                "len_slope", "len_R2", "rho(resid,d)", "chi_peak_d", "hard_d",
                "raw_drop", "resid_drop", "interpretation",
            ],
            summary_rows,
        ),
        "",
    ]
    report_parts[6:6] = summary_block

    report_path = out_dir / f"length_control__{args.model}.md"
    d_csv = out_dir / f"length_control_difficulty_bins__{args.model}.csv"
    q_csv = out_dir / f"length_control_token_bins__{args.model}.csv"
    report_path.write_text("\n".join(report_parts), encoding="utf-8")

    write_csv(
        d_csv,
        all_d_bins,
        [
            "domain", "bin", "lo", "hi", "center", "n", "acc",
            "chi_mean", "chi_std", "log_chi_resid_mean", "log_chi_resid_std",
            "chi_ratio_len_mean", "chi_ratio_len_std", "n_tokens_mean",
            "n_tokens_std", "phi_mean", "spk_mean",
        ],
    )
    write_csv(
        q_csv,
        all_q_bins,
        [
            "domain", "qbin", "token_lo", "token_hi", "n", "acc",
            "d_min", "d_max", "chi_mean", "resid_mean",
            "spearman_chi_d", "spearman_resid_d",
        ],
    )

    print(f"[length-control] wrote {report_path}")
    print(f"[length-control] wrote {d_csv}")
    print(f"[length-control] wrote {q_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

