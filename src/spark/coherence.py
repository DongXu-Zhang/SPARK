"""Hierarchical Coherence (Φ) — cross-layer hidden-state alignment.

Inspired by Dehaene's *neuronal global workspace ignition* hypothesis:
during a successful "thought event", information is broadcast across
cortical regions — i.e. cross-region representations align. We measure the
analogous quantity in transformers as the mean cosine between consecutive
layer outputs:

    Φ(x) = (1/(L-1)) Σ_ℓ cos(h̄_ℓ, h̄_{ℓ+1})

Φ goes up when many layers settle into a consistent representation, which
empirically tracks the "ignition" component of emergence. Combined with χ
(which captures the instability/criticality component) it forms the
SPARK index.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from src.spark.types import PhiConfig, PhiResult


def _pool_hidden(h: torch.Tensor,
                 attention_mask: torch.Tensor,
                 pool: str,
                 use_attention_mask: bool) -> torch.Tensor:
    """Reduce [B, T, d] -> [B, d]. Same logic as in susceptibility.py but
    duplicated here so the two modules can be used independently."""
    if pool == "mean":
        if use_attention_mask:
            mask = attention_mask.unsqueeze(-1).to(h.dtype)
            denom = mask.sum(dim=1).clamp(min=1.0)
            return (h * mask).sum(dim=1) / denom
        return h.mean(dim=1)
    if pool == "last":
        if use_attention_mask:
            seq_lens = attention_mask.sum(dim=1).long() - 1
            seq_lens = seq_lens.clamp(min=0)
            return h[torch.arange(h.size(0), device=h.device), seq_lens]
        return h[:, -1, :]
    if pool == "first":
        return h[:, 0, :]
    raise ValueError(f"Unknown pool: {pool!r}")


@torch.no_grad()
def compute_phi(
    model,
    tokenizer,
    text: str,
    config: Optional[PhiConfig] = None,
) -> PhiResult:
    """Compute Φ for a single problem."""
    cfg = config or PhiConfig()
    device = next(model.parameters()).device

    enc = tokenizer(text, return_tensors="pt", truncation=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = attention_mask.to(device)
    n_tokens = int(attention_mask.sum().item())

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    all_layers = out.hidden_states
    n_total = len(all_layers)

    if cfg.layers is None:
        layers_to_probe = list(range(1, n_total))
    else:
        layers_to_probe = sorted({
            int(ell) for ell in cfg.layers if 0 <= ell < n_total
        })
        if not layers_to_probe:
            raise ValueError(f"No valid layers in {cfg.layers}")

    # Pool each requested layer once.
    pooled: dict = {}
    for ell in layers_to_probe:
        h = all_layers[ell]                       # [1, T, d]
        h_pool = _pool_hidden(h, attention_mask, cfg.pool, cfg.use_attention_mask)
        pooled[ell] = h_pool.squeeze(0).float()   # [d]

    # Cosine between *consecutive probed layers* (as ordered by layer_idx).
    cos_per_pair: dict = {}
    cos_values = []
    sorted_layers = sorted(pooled.keys())
    for i in range(len(sorted_layers) - 1):
        a_idx = sorted_layers[i]
        b_idx = sorted_layers[i + 1]
        a = pooled[a_idx]
        b = pooled[b_idx]
        cos = float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
        cos_per_pair[(a_idx, b_idx)] = cos
        cos_values.append(cos)

    if not cos_values:
        # Degenerate: only one layer probed
        phi = 0.0
    else:
        phi = sum(cos_values) / len(cos_values)

    return PhiResult(
        phi=phi,
        cos_per_pair=cos_per_pair,
        n_tokens=n_tokens,
    )
