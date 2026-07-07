"""Latent Susceptibility (χ) — core implementation.

For a problem text x, we measure how much the model's hidden state changes
under K small Gaussian perturbations to the input embeddings:

    χ_ℓ(x) = (1/K) Σ_k ‖h_ℓ(x + ε·η_k) - h_ℓ(x)‖² / ε²

where h_ℓ is the (pooled) hidden state at transformer layer ℓ and ε is the
noise scale (relative to the embedding norm). χ peaks near the model's
capability boundary — the "critical-susceptibility" signature of a
second-order phase transition (Mora & Bialek 2011; Beggs & Plenz 2003).

Implementation notes
--------------------
* We perturb **input embeddings**, not token IDs, so each perturbation is a
  truly continuous, infinitesimal nudge — needed for the ε→0 limit
  interpretation as Fisher information.
* All K+1 forwards (1 clean + K perturbed) are batched into one
  ``model.forward(inputs_embeds=...)`` call. This is faster and exactly
  matches the math.
* We do not need the model to *generate* — χ is a probe on the prompt-side
  hidden states alone, which is what we want for a sample-free signal.
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch

from src.spark.types import ChiConfig, ChiResult


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------

def _pool_hidden(
    h: torch.Tensor,            # [B, T, d]
    attention_mask: torch.Tensor,  # [B, T]
    pool: str,
    use_attention_mask: bool,
) -> torch.Tensor:
    """Reduce [B, T, d] -> [B, d] using the requested pooling strategy."""
    if pool == "mean":
        if use_attention_mask:
            mask = attention_mask.unsqueeze(-1).to(h.dtype)   # [B, T, 1]
            denom = mask.sum(dim=1).clamp(min=1.0)            # [B, 1]
            return (h * mask).sum(dim=1) / denom              # [B, d]
        return h.mean(dim=1)
    if pool == "last":
        if use_attention_mask:
            # last non-pad token per row
            seq_lens = attention_mask.sum(dim=1).long() - 1   # [B]
            seq_lens = seq_lens.clamp(min=0)
            return h[torch.arange(h.size(0), device=h.device), seq_lens]
        return h[:, -1, :]
    if pool == "first":
        return h[:, 0, :]
    raise ValueError(f"Unknown pool: {pool!r} (expected 'mean'|'last'|'first')")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_chi(
    model,
    tokenizer,
    text: str,
    config: Optional[ChiConfig] = None,
) -> ChiResult:
    """Compute χ (and a per-layer spectrum) for one problem.

    Parameters
    ----------
    model : a HF ``transformers`` causal-LM (already on device, in eval mode).
    tokenizer : the matching tokenizer.
    text : the raw problem string.
    config : ChiConfig. If None, defaults are used.
    """
    cfg = config or ChiConfig()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # ----- 1. Tokenise -----
    enc = tokenizer(text, return_tensors="pt", truncation=True)
    input_ids = enc["input_ids"].to(device)                # [1, T]
    attention_mask = enc.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = attention_mask.to(device)             # [1, T]
    n_tokens = int(attention_mask.sum().item())

    # ----- 2. Get input embeddings -----
    embed_layer = model.get_input_embeddings()
    base_embeds = embed_layer(input_ids)                   # [1, T, d]
    base_embeds = base_embeds.to(dtype)

    # ----- 3. Compute σ from per-token embedding norm -----
    # Use the mean L2 norm of the token embeddings as the reference scale.
    with torch.no_grad():
        per_token_norm = base_embeds.norm(dim=-1)          # [1, T]
        if cfg.use_attention_mask:
            ref_norm = (per_token_norm * attention_mask).sum() / attention_mask.sum().clamp(min=1)
        else:
            ref_norm = per_token_norm.mean()
    sigma = float(cfg.epsilon) * float(ref_norm.item())

    # Guard against degenerate case (zero-norm embeddings, very short prompts)
    if sigma <= 0 or not math.isfinite(sigma):
        raise RuntimeError(f"Non-positive sigma computed: {sigma}. "
                           f"Check epsilon/embedding norm.")

    # ----- 4. Build batch of K+1 inputs -----
    K = int(cfg.K)
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")

    # Use a CPU generator to pick noise on CPU then transfer — most stable
    # across cuda / mps / cpu.
    rng = torch.Generator(device="cpu").manual_seed(int(cfg.seed))

    # noise: [K, T, d] in float32 for numeric stability, then cast.
    noise_cpu = torch.randn(K, *base_embeds.shape[1:], generator=rng,
                            dtype=torch.float32) * sigma
    noise = noise_cpu.to(device=device, dtype=dtype)

    # batch_embeds: [K+1, T, d] (clean + K perturbed)
    batch_embeds = base_embeds.expand(K + 1, -1, -1).contiguous().clone()
    batch_embeds[1:] = base_embeds.expand(K, -1, -1) + noise

    batch_mask = attention_mask.expand(K + 1, -1).contiguous()

    # ----- 5. Forward pass -----
    out = model(
        inputs_embeds=batch_embeds,
        attention_mask=batch_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    # out.hidden_states is a tuple of length L+1:
    #   hidden_states[0] = embedding output
    #   hidden_states[1..L] = transformer layer outputs
    all_layers = out.hidden_states
    n_total_layers = len(all_layers)

    # ----- 6. Decide which layers to probe -----
    if cfg.layers is None:
        # All real transformer layers (skip the embedding output)
        layers_to_probe = list(range(1, n_total_layers))
    else:
        # Validate user-supplied indices
        layers_to_probe = []
        for ell in cfg.layers:
            if 0 <= ell < n_total_layers:
                layers_to_probe.append(int(ell))
            else:
                # Silently skip out-of-range; report later in result if all skipped
                continue
        if not layers_to_probe:
            raise ValueError(f"No valid layer indices in {cfg.layers} "
                             f"(model has {n_total_layers} hidden states 0..{n_total_layers-1}).")

    # ----- 7. Per-layer χ -----
    chi_per_layer: dict = {}
    raw_distances: list = []

    for layer_idx in layers_to_probe:
        h = all_layers[layer_idx]                          # [K+1, T, d]
        # Pool to [K+1, d]
        h_pool = _pool_hidden(
            h, batch_mask, cfg.pool, cfg.use_attention_mask
        )

        # Cast to float32 for distance math (avoid bf16 precision loss)
        h_pool = h_pool.float()

        h_clean = h_pool[0:1]                              # [1, d]
        h_perturbed = h_pool[1:]                           # [K, d]
        delta = h_perturbed - h_clean                      # [K, d]
        sq_dist = (delta * delta).sum(dim=-1)              # [K]

        if cfg.normalize_by_h:
            # Scale-invariant susceptibility:
            #     χ = ⟨‖Δh‖² / ‖h‖²⟩ / ε²
            # `‖h‖²` cancels the per-layer hidden-magnitude growth so different
            # layers can be compared on the same scale, and ε replaces σ since
            # σ = ε · ⟨emb_norm⟩ already absorbs the input-side scale.
            h_clean_sq = (h_clean * h_clean).sum(dim=-1).clamp(min=1e-12)  # [1]
            relative_sq = sq_dist / h_clean_sq                              # [K]
            chi_value = float(relative_sq.mean().item()) / (cfg.epsilon ** 2)
        else:
            # Absolute form (kept for diagnostics / unit tests):
            #     χ = ⟨‖Δh‖²⟩ / σ²
            chi_value = float(sq_dist.mean().item()) / (sigma * sigma)

        chi_per_layer[layer_idx] = chi_value
        raw_distances.append([float(v) for v in sq_dist.tolist()])

    # ----- 8. Aggregate -----
    layer_keys = list(chi_per_layer.keys())
    chi_values = [chi_per_layer[k] for k in layer_keys]
    chi_max = max(chi_values)
    chi_mean = sum(chi_values) / len(chi_values)
    chi_argmax_layer = layer_keys[chi_values.index(chi_max)]

    return ChiResult(
        chi_per_layer=chi_per_layer,
        chi_max=chi_max,
        chi_mean=chi_mean,
        chi_argmax_layer=chi_argmax_layer,
        n_tokens=n_tokens,
        sigma=sigma,
        raw_distances_sq=raw_distances,
    )
