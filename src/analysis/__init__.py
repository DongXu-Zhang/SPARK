"""Stage-3 analysis utilities — data merging, statistics, plotting.

Public API:
    merge_records        — join problems + calibration + spark into one table
    spearman_chi_vs_d    — rank correlation between χ and difficulty
    auc_chi_correctness  — ROC AUC of χ as a correctness predictor
"""
from src.analysis.merge import merge_records
from src.analysis.stats import (
    auc_chi_correctness,
    cohens_d,
    spearman_chi_vs_d,
)

__all__ = [
    "merge_records",
    "spearman_chi_vs_d",
    "auc_chi_correctness",
    "cohens_d",
]
