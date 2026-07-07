"""SPARK: Latent Susceptibility for emergence detection.

Public API:
    compute_chi       — χ (susceptibility) for a single problem
    compute_phi       — Φ (cross-layer coherence)
    compute_spark_index — composite of χ and Φ
    ChiConfig / PhiConfig / SparkConfig — dataclass configs
    ChiResult / PhiResult / SparkResult — typed results
"""
from src.spark.coherence import compute_phi
from src.spark.spark_index import compute_spark_index
from src.spark.susceptibility import compute_chi
from src.spark.types import (
    ChiConfig,
    ChiResult,
    PhiConfig,
    PhiResult,
    SparkConfig,
    SparkResult,
)

__all__ = [
    "compute_chi",
    "compute_phi",
    "compute_spark_index",
    "ChiConfig",
    "ChiResult",
    "PhiConfig",
    "PhiResult",
    "SparkConfig",
    "SparkResult",
]
