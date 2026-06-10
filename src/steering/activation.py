"""Minimal HuggingFace activation steering helpers.

The SPARK pipeline uses vLLM for normal calibration, but activation injection
requires access to internal residual streams. These helpers use Transformers
forward hooks and are intentionally small and explicit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


DEFAULT_SYSTEM = (
    "You are a careful reasoner. Think step by step, "
    "then put your final answer inside \\boxed{}."
)


def format_prompt(
    tokenizer,
    problem_text: str,
    use_chat_template: bool = True,
    system_prompt: str | None = DEFAULT_SYSTEM,
    enable_thinking: bool = False,
) -> str:
    if not use_chat_template:
        return problem_text
    if system_prompt is None:
        messages = [{"role": "user", "content": problem_text}]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem_text},
        ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return problem_text


def build_demo_prompt(
    target_prompt: str,
    demos: list[dict],
    k: int,
    answer_key: str = "ground_truth",
) -> str:
    """Build a compact demonstration prompt for activation extraction."""
    parts = []
    for i, demo in enumerate(demos[:k], start=1):
        ans = demo.get(answer_key, demo.get("answer", ""))
        parts.append(
            f"Example {i}\n"
            f"Question:\n{demo['prompt']}\n"
            f"Final answer: \\boxed{{{ans}}}\n"
        )
    parts.append(f"Now solve the target problem.\nQuestion:\n{target_prompt}")
    return "\n\n".join(parts)


def load_hf_model_and_tokenizer(
    model_path: str,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    max_memory: dict | None = None,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype, torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    load_kw: dict = {
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if max_memory is not None:
        load_kw["max_memory"] = max_memory
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kw)
    model.eval()
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_decoder_layers(model) -> list:
    """Return decoder block modules for common CausalLM architectures."""
    candidates = [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ]
    for root_name, layer_name in candidates:
        root = getattr(model, root_name, None)
        if root is not None and hasattr(root, layer_name):
            layers = getattr(root, layer_name)
            return list(layers)
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return list(model.model.decoder.layers)
    raise AttributeError(
        "Could not find decoder layers. Expected model.model.layers, "
        "model.transformer.h, gpt_neox.layers, or model.model.decoder.layers."
    )


def hidden_index_to_module_index(hidden_state_index: int) -> int:
    """Map hidden_states index to decoder module index.

    SPARK chi uses hidden_states[1..L] for real transformer layer outputs.
    The corresponding decoder block is therefore index hidden_state_index - 1.
    """
    idx = int(hidden_state_index) - 1
    if idx < 0:
        raise ValueError("hidden_state_index must be >= 1 for decoder layers")
    return idx


def _last_token_index(attention_mask: torch.Tensor) -> torch.Tensor:
    return attention_mask.sum(dim=1).long().clamp(min=1) - 1


@torch.no_grad()
def collect_hidden_vector(
    model,
    tokenizer,
    prompt_text: str,
    hidden_state_index: int,
    pool: str = "last",
    use_chat_template: bool = True,
    enable_thinking: bool = False,
    max_length: int | None = None,
) -> torch.Tensor:
    """Collect a pooled hidden vector from one prompt.

    ``hidden_state_index`` follows Transformers hidden_states indexing:
    0 is embedding output, 1..L are decoder layer outputs.
    """
    formatted = format_prompt(
        tokenizer,
        prompt_text,
        use_chat_template=use_chat_template,
        enable_thinking=enable_thinking,
    )
    enc = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=max_length is not None,
        max_length=max_length,
    )
    device = next(model.parameters()).device
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = attention_mask.to(device)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    hidden_states = out.hidden_states
    if hidden_state_index < 0 or hidden_state_index >= len(hidden_states):
        raise IndexError(
            f"hidden_state_index={hidden_state_index} out of range; "
            f"model returned {len(hidden_states)} hidden states"
        )
    h = hidden_states[hidden_state_index]
    if pool == "last":
        idx = _last_token_index(attention_mask)
        vec = h[torch.arange(h.size(0), device=h.device), idx]
    elif pool == "mean":
        mask = attention_mask.unsqueeze(-1).to(h.dtype)
        vec = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    else:
        raise ValueError("pool must be 'last' or 'mean'")
    return vec.squeeze(0).detach().float().cpu()


@dataclass
class SteeringHandle:
    handle: object

    def close(self):
        self.handle.remove()


def add_steering_hook(
    model,
    hidden_state_index: int,
    vector: torch.Tensor,
    alpha: float,
    position: str = "all",
) -> SteeringHandle:
    """Install a residual steering hook on one decoder block.

    Parameters
    ----------
    hidden_state_index:
        SPARK/Transformers hidden-state index. For Qwen3-4B, layer 36 maps to
        decoder block 35.
    position:
        ``all`` adds to every token position. ``last`` adds only to the last
        position in the current forward call. During autoregressive decoding,
        the current forward usually has sequence length 1, so both work for
        decode steps; ``all`` also steers the prompt prefill.
    """
    layers = get_decoder_layers(model)
    module_idx = hidden_index_to_module_index(hidden_state_index)
    if module_idx >= len(layers):
        raise IndexError(
            f"hidden_state_index={hidden_state_index} maps to module {module_idx}, "
            f"but model has only {len(layers)} decoder layers"
        )
    module = layers[module_idx]

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        v = vector.to(device=hidden.device, dtype=hidden.dtype) * float(alpha)
        if hidden.ndim != 3:
            return output
        if position == "all":
            hidden2 = hidden + v.view(1, 1, -1)
        elif position == "last":
            hidden2 = hidden.clone()
            hidden2[:, -1:, :] = hidden2[:, -1:, :] + v.view(1, 1, -1)
        else:
            raise ValueError("position must be 'all' or 'last'")

        if rest is None:
            return hidden2
        return (hidden2, *rest)

    return SteeringHandle(module.register_forward_hook(hook))


def top_layer_from_records(records: Iterable[dict], default: int = 36) -> int:
    """Legacy helper: mode of ``chi_argmax_layer`` (may pick shallow layer 1)."""
    layer, _meta = select_steering_layer(records, default=default)
    return layer


def select_steering_layer(
    records: Iterable[dict],
    *,
    default: int = 36,
    min_layer: int = 3,
    shallow_dominance: float = 0.5,
) -> tuple[int, dict]:
    """Pick a steering hidden-state index from SPARK ``chi_argmax_layer`` votes.

    SPARK indexes ``hidden_states[1..L]`` (layer 1 = first transformer block).
    On Llama/DeepSeek, χ often peaks at layer 1 due to embedding-adjacent artifacts;
    when a shallow layer dominates (≥ ``shallow_dominance``), fall back to the
    deepest layer with the highest vote count among layers ≥ ``min_layer``.
    """
    counts: dict[int, int] = {}
    for r in records:
        try:
            ell = int(r.get("chi_argmax_layer"))
        except (TypeError, ValueError):
            continue
        counts[ell] = counts.get(ell, 0) + 1

    meta: dict = {"min_layer": min_layer, "shallow_dominance": shallow_dominance}
    if not counts:
        meta.update({"reason": "empty_counts", "layer": default, "counts": {}})
        return default, meta

    total = sum(counts.values())
    meta["counts"] = dict(sorted(counts.items()))
    top_layer, top_count = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
    deep_counts = {layer: c for layer, c in counts.items() if layer >= min_layer}

    if not deep_counts:
        meta.update({"reason": "no_deep_layers", "layer": top_layer})
        return top_layer, meta

    if top_layer < min_layer and top_count / total >= shallow_dominance:
        chosen = max(deep_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        meta.update({
            "reason": "shallow_dominance_fallback",
            "shallow_top": top_layer,
            "shallow_frac": round(top_count / total, 4),
            "layer": chosen,
        })
        return chosen, meta

    chosen = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
    if chosen < min_layer:
        chosen = max(deep_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        meta.update({"reason": "bumped_from_shallow", "layer": chosen})
    else:
        meta.update({"reason": "mode", "layer": chosen})
    return chosen, meta
