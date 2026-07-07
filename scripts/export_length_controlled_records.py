"""Export merged records with length-controlled susceptibility fields.

This creates the canonical analysis file used by the next-stage experiments:

    data/analysis/<model>__<domain>__length_controlled.jsonl

Each row merges raw problem metadata, calibration correctness, SPARK metrics,
and additional fields:

    chi_lc
    chi_ratio_len
    chi_expected_for_length
    lc_pred_log_chi
    lc_log_tokens
    lc_log_chi

The companion metadata JSON stores the fitted length-control model.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.length_control import (  # noqa: E402
    estimate_dstar,
    merge_and_length_control,
    spearman,
)
from src.utils.io import save_json, save_jsonl  # noqa: E402


DOMAINS = ["symbolic", "logical", "algorithmic"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-4B")
    ap.add_argument("--domains", nargs="+", default=DOMAINS)
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--calibration_dir", default="data/calibrated_full")
    ap.add_argument("--spark_dir", default="data/spark_full")
    ap.add_argument("--out_dir", default="data/analysis")
    ap.add_argument("--score_field", default="chi_max")
    ap.add_argument("--d_field", default="d_structural")
    ap.add_argument("--token_field", default="n_tokens")
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument("--min_bin_count", type=int, default=20)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for domain in args.domains:
        records, fit = merge_and_length_control(
            model=args.model,
            domain=domain,
            raw_dir=args.raw_dir,
            calibration_dir=args.calibration_dir,
            spark_dir=args.spark_dir,
            score_field=args.score_field,
            d_field=args.d_field,
            token_field=args.token_field,
        )
        out_path = out_dir / f"{args.model}__{domain}__length_controlled.jsonl"
        meta_path = out_dir / f"{args.model}__{domain}__length_controlled.meta.json"
        save_jsonl(records, out_path)

        dstar, dstar_note = estimate_dstar(
            records, n_bins=args.n_bins, min_bin_count=args.min_bin_count
        )
        d_vals = [float(r["d_value"]) for r in records if r.get("d_value") is not None]
        chi_vals = [float(r[args.score_field]) for r in records]
        token_vals = [float(r[args.token_field]) for r in records]
        resid_vals = [float(r["chi_lc"]) for r in records]
        correct = [int(r["correct"]) for r in records if r.get("correct") in (0, 1)]

        meta = {
            "model": args.model,
            "domain": domain,
            "n": len(records),
            "accuracy": (sum(correct) / len(correct)) if correct else None,
            "dstar": dstar,
            "dstar_note": dstar_note,
            "length_control_fit": fit.to_dict(),
            "spearman_chi_d": spearman(chi_vals, d_vals) if d_vals else None,
            "spearman_chi_tokens": spearman(chi_vals, token_vals) if token_vals else None,
            "spearman_chi_lc_d": spearman(resid_vals, d_vals) if d_vals else None,
            "spearman_chi_lc_tokens": spearman(resid_vals, token_vals) if token_vals else None,
            "inputs": {
                "raw_dir": args.raw_dir,
                "calibration_dir": args.calibration_dir,
                "spark_dir": args.spark_dir,
                "score_field": args.score_field,
                "d_field": args.d_field,
                "token_field": args.token_field,
            },
        }
        save_json(meta, meta_path)
        summary[domain] = meta
        print(f"[export-lc] {domain}: {len(records)} records -> {out_path}")
        print(f"[export-lc] {domain}: metadata -> {meta_path}")

    save_json(summary, out_dir / f"{args.model}__length_controlled.summary.json")
    print(f"[export-lc] summary -> {out_dir / f'{args.model}__length_controlled.summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

