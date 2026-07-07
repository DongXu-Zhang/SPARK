"""Fuse d_structural with d_empirical (from multi-model pass rates),
then stratified-sample the final FRONTIER-1.5K set.

Outputs:
    data/final/frontier_1500.jsonl       — final 1500 problems with full metadata
    data/final/calibration_summary.json  — per-domain difficulty diagnostics
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

from src.utils.io import load_jsonl, save_jsonl, save_json


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _index_eval_records(eval_records: list[dict]) -> dict:
    """Map id -> {model_name -> 0|1}."""
    idx: dict[str, dict[str, int]] = defaultdict(dict)
    for r in eval_records:
        idx[r["id"]][r["model"]] = r["correct"]
    return idx


def fuse_difficulty(
    problems: list[dict],
    eval_index: dict,
    alpha: float = 0.5,
    beta: float = 0.5,
    normalize: str = "rank",
) -> list[dict]:
    """Attach d_empirical + d_final + bin to each problem (in-place).

    Args:
        alpha: weight on d_structural.
        beta: weight on d_empirical.
        normalize: 'rank' (uniform over [0,1]) or 'minmax'.
    """
    out = []
    by_domain = defaultdict(list)
    for p in problems:
        rates = eval_index.get(p["id"], {})
        if rates:
            d_e = 1.0 - float(np.mean(list(rates.values())))
        else:
            d_e = float("nan")
        p = dict(p)
        p["d_empirical"] = round(d_e, 4) if not np.isnan(d_e) else None
        p["model_pass_rate"] = rates
        by_domain[p["domain"]].append(p)

    for dom, probs in by_domain.items():
        # Need d_empirical for fusion; drop problems missing any model's eval.
        have_e = [p for p in probs if p["d_empirical"] is not None]
        skipped = len(probs) - len(have_e)
        if skipped:
            print(f"[difficulty] {dom}: skipped {skipped} problems "
                  f"with incomplete model coverage")
        if not have_e:
            continue
        d_raw = np.array([alpha * p["d_structural"] + beta * p["d_empirical"]
                          for p in have_e])

        if normalize == "rank":
            ranks = stats.rankdata(d_raw, method="average") / len(d_raw)
            d_final = ranks
        elif normalize == "minmax":
            lo, hi = float(np.min(d_raw)), float(np.max(d_raw))
            d_final = (d_raw - lo) / (hi - lo + 1e-9)
        else:
            raise ValueError(f"Unknown normalize: {normalize}")

        for p, d in zip(have_e, d_final):
            p["d_final"] = float(round(d, 4))
            p["bin"] = min(4, int(p["d_final"] * 5))
        out.extend(have_e)

    return out


def stratified_sample(
    problems: list[dict],
    n_per_bin: int = 100,
    seed: int = 42,
) -> list[dict]:
    """Return n_per_bin problems for each (domain, bin) combination."""
    rng = np.random.default_rng(seed)
    by_db: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for p in problems:
        by_db[(p["domain"], p["bin"])].append(p)

    sampled = []
    print(f"[stratified] (domain, bin) -> available / requested:")
    for key in sorted(by_db.keys()):
        pool = by_db[key]
        if len(pool) >= n_per_bin:
            chosen_idx = rng.choice(len(pool), size=n_per_bin, replace=False)
            chosen = [pool[i] for i in chosen_idx]
        else:
            print(f"  WARNING: {key} has only {len(pool)} (< {n_per_bin}); using all")
            chosen = pool
        print(f"  {key}: {len(pool)} → {len(chosen)}")
        sampled.extend(chosen)

    rng.shuffle(sampled)
    return sampled


def domain_summary(problems: list[dict]) -> dict:
    out = {}
    by_dom = defaultdict(list)
    for p in problems:
        by_dom[p["domain"]].append(p)
    for dom, probs in by_dom.items():
        ds = [p["d_structural"] for p in probs]
        des = [p["d_empirical"] for p in probs if p.get("d_empirical") is not None]
        df = [p["d_final"] for p in probs if p.get("d_final") is not None]
        out[dom] = {
            "n": len(probs),
            "d_structural_mean": float(np.mean(ds)) if ds else None,
            "d_empirical_mean": float(np.mean(des)) if des else None,
            "d_final_mean": float(np.mean(df)) if df else None,
            "bin_counts": [
                sum(1 for p in probs if p.get("bin") == b) for b in range(5)
            ],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", type=str, required=True,
                    help="JSONL of all candidate problems (raw, post-generation).")
    ap.add_argument("--eval_dir", type=str, required=True,
                    help="Directory holding {model}.jsonl per model.")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--n_per_bin", type=int, default=100)
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    diff_cfg = cfg["difficulty"]

    problems = load_jsonl(args.problems)
    print(f"[difficulty] {len(problems)} candidate problems loaded")

    # Load all per-model eval files.
    eval_records = []
    eval_dir = Path(args.eval_dir)
    for f in sorted(eval_dir.glob("*.jsonl")):
        recs = load_jsonl(f)
        eval_records.extend(recs)
        print(f"  loaded {len(recs)} records from {f.name}")

    eval_index = _index_eval_records(eval_records)

    fused = fuse_difficulty(
        problems, eval_index,
        alpha=diff_cfg["alpha"],
        beta=diff_cfg["beta"],
        normalize=diff_cfg["normalize"],
    )
    print(f"[difficulty] {len(fused)} problems with full eval coverage")

    sampled = stratified_sample(fused, n_per_bin=args.n_per_bin)
    print(f"[difficulty] sampled {len(sampled)} into FRONTIER-{len(sampled)}")

    out_dir = Path(args.out_dir)
    save_jsonl(sampled, out_dir / f"frontier_{len(sampled)}.jsonl")
    save_jsonl(fused, out_dir / "all_calibrated.jsonl")
    save_json(domain_summary(sampled), out_dir / "calibration_summary.json")
    print(f"[difficulty] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
