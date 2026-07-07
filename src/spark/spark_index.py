"""SPARK Index — composite of χ and Φ.

    SPK(x) = log χ̂(x) + λ · Φ(x)

The log compresses the heavy-tailed χ distribution and the additive Φ
brings in the "ignition" signal as a corroborator. λ ≈ 1 is a reasonable
default; tune on a held-out probe set against ground-truth difficulty.
"""
from __future__ import annotations

import math


def compute_spark_index(
    chi: float,
    phi: float,
    lambda_: float = 1.0,
    chi_floor: float = 1e-12,
) -> float:
    """Combine χ and Φ into the SPARK index.

    Parameters
    ----------
    chi : a single χ value (typically ``chi_max`` from ChiResult).
    phi : the Φ value (mean cross-layer cosine, in [-1, 1]).
    lambda_ : weight on Φ.
    chi_floor : lower bound to keep ``log`` well-defined.
    """
    chi_safe = max(chi, chi_floor)
    return math.log(chi_safe) + lambda_ * phi
