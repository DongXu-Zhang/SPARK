"""Merge data sources into a tidy per-problem dictionary.

Three input streams:
  * problems        — data/sanity/<dom>.jsonl OR data/raw/<dom>.jsonl OR data/final/...
                      contains ``id``, ``prompt``, ``ground_truth``, ``answer_type``,
                      ``d_structural``, optionally ``d_empirical``, ``d_final``,
                      ``model_pass_rate``.
  * calibration     — data/calibrated/<model>__<dom>.jsonl
                      contains ``id``, ``model``, ``pred``, ``correct``,
                      ``finish_reason``, ``n_gen_tokens``, ``raw_text``.
  * spark           — data/spark/<model>__<dom>.jsonl
                      contains ``id``, ``chi_max``, ``chi_mean``, ``chi_per_layer``,
                      ``phi``, ``spk``.

This module provides the join. Plotting / stats consume the merged records.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional


def _load_jsonl(path: str | Path) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _index_by_id(records: Iterable[dict]) -> dict[str, dict]:
    return {r["id"]: r for r in records if "id" in r}


def merge_records(
    problems_path: str | Path,
    calibration_path: Optional[str | Path] = None,
    spark_path: Optional[str | Path] = None,
    require_all: bool = False,
) -> list[dict]:
    """Join up to three data sources by ``id``.

    Parameters
    ----------
    problems_path : the canonical "what is this problem" file (always required).
    calibration_path : optional ``<model>__<dom>.jsonl`` from `run_eval`.
    spark_path : optional ``<model>__<dom>.jsonl`` from `compute_spark`.
    require_all : if True, drop any id not present in every supplied file.
                  Default False keeps partial records (calibration but no spark, etc.).

    Returns
    -------
    A list of merged dicts. For each record the keys from problems are kept;
    calibration adds ``correct``, ``pred``, ``model``, ``finish_reason``,
    ``n_gen_tokens``; spark adds ``chi_max``, ``chi_mean``, ``chi_per_layer``,
    ``phi``, ``spk``, ``chi_argmax_layer``, ``sigma``.
    """
    problems = _index_by_id(_load_jsonl(problems_path))

    calib = _index_by_id(_load_jsonl(calibration_path)) if calibration_path else {}
    spark = _index_by_id(_load_jsonl(spark_path)) if spark_path else {}

    merged = []
    for pid, prob in problems.items():
        rec = dict(prob)

        if calib:
            cr = calib.get(pid)
            if cr is None and require_all:
                continue
            if cr is not None:
                # Pull fields from calibration. Prefix-collision-safe: only
                # copy keys that don't already exist in problem record.
                for k in ("model", "pred", "correct", "finish_reason",
                          "n_gen_tokens", "n_chars"):
                    if k in cr:
                        rec[k] = cr[k]

        if spark:
            sr = spark.get(pid)
            if sr is None and require_all:
                continue
            if sr is not None:
                for k in ("chi_max", "chi_mean", "chi_argmax_layer",
                          "chi_per_layer", "phi", "spk", "sigma", "n_tokens"):
                    if k in sr:
                        rec[k] = sr[k]

        merged.append(rec)

    return merged
