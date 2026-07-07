"""Shared dataclasses for the SPARK module.

Kept in a separate file so configs and results can be imported without
pulling in torch (some downstream callers, e.g. plotting, only need the
types).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# χ (latent susceptibility)
# ---------------------------------------------------------------------------

@dataclass
class ChiConfig:
    """Configuration for χ computation.

    Attributes
    ----------
    K : number of perturbations to sample (K+1 forwards per problem total).
    epsilon : noise scale relative to the per-token embedding norm.
    layers : transformer layer indices to probe. ``None`` = all layers
        (excluding the embedding layer).
    pool : how to reduce the [T, d] hidden state to [d] before measuring
        distance: ``"mean"``, ``"last"``, or ``"first"``.
    seed : RNG seed for the Gaussian perturbations.
    use_attention_mask : when pooling, ignore padding positions if True.
    """
    K: int = 8
    epsilon: float = 0.01
    layers: Optional[list] = None
    pool: str = "mean"
    seed: int = 42
    use_attention_mask: bool = True
    # When True (default), χ is the *relative* susceptibility:
    #     χ = ⟨‖Δh‖² / ‖h‖²⟩ / ε²
    # which is dimensionless and comparable across layers. When False,
    # χ is the *absolute* form ⟨‖Δh‖²⟩ / σ² which grows with hidden norm
    # and hence trivially peaks at the last layer.
    normalize_by_h: bool = True


@dataclass
class ChiResult:
    """Result of χ computation for a single problem."""
    chi_per_layer: dict          # {layer_idx: float}
    chi_max: float               # max across probed layers
    chi_mean: float              # mean across probed layers
    chi_argmax_layer: int        # which layer had the peak
    n_tokens: int                # number of (non-pad) tokens in the input
    sigma: float                 # actual noise scale used (ε * embedding_norm)
    raw_distances_sq: list = field(default_factory=list)
    # raw_distances_sq[layer_idx] = list[K] of ‖Δh‖² values (for debug)


# ---------------------------------------------------------------------------
# Φ (hierarchical coherence)
# ---------------------------------------------------------------------------

@dataclass
class PhiConfig:
    """Configuration for Φ computation."""
    layers: Optional[list] = None
    pool: str = "mean"
    use_attention_mask: bool = True


@dataclass
class PhiResult:
    """Result of Φ computation for a single problem."""
    phi: float                              # mean cosine across consecutive layers
    cos_per_pair: dict                      # {(layer_i, layer_i+1): float}
    n_tokens: int


# ---------------------------------------------------------------------------
# SPK (composite SPARK index)
# ---------------------------------------------------------------------------

@dataclass
class SparkConfig:
    """Top-level configuration combining χ, Φ, and the composite weight λ."""
    chi: ChiConfig = field(default_factory=ChiConfig)
    phi: PhiConfig = field(default_factory=PhiConfig)
    lambda_: float = 1.0
    chi_aggregator: str = "max"     # "max" | "mean" — which χ value enters SPK


@dataclass
class SparkResult:
    """Combined per-problem output."""
    chi: ChiResult
    phi: PhiResult
    spk: float
