"""Run a single Qwen-family model over all candidate problems.

Designed for vLLM on a Linux server. CPU-only fallback (transformers) is
provided for sanity checks, but never use it for full calibration.

Outputs one JSON file per (model, run): list of
    {"id": ..., "model": ..., "pred": ..., "correct": 0|1, "raw_text": ...}

Usage:
    python -m src.calibration.run_eval \
        --model Qwen3-4B \
        --input data/raw/all.jsonl \
        --out_dir data/calibrated/ \
        --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
from pathlib import Path

import yaml

from src.calibration.math_grade import extract_math_answer  # noqa: E402
from src.calibration.verify import extract_answer, verify
from src.utils.io import load_jsonl, save_jsonl


SYSTEM_PROMPT = (
    "You are a careful reasoner. Think step by step, "
    "then put your final answer inside \\boxed{}."
)


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _expand_env(value):
    """Replace ${VAR} tokens in strings with the corresponding env var."""
    if isinstance(value, str):
        return re.sub(r"\$\{(\w+)\}",
                      lambda m: os.environ.get(m.group(1), m.group(0)),
                      value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(x) for x in value]
    return value


def _resolve_model_source(mcfg: dict) -> str:
    """Pick the best model source:
       1. local_path if it exists on disk and contains config.json
       2. modelscope_id (and set VLLM_USE_MODELSCOPE=True so vLLM downloads)
       3. hf_id (legacy fallback)
    """
    local = mcfg.get("local_path")
    if local and Path(local).expanduser().joinpath("config.json").exists():
        return str(Path(local).expanduser())

    ms_id = mcfg.get("modelscope_id")
    if ms_id:
        os.environ["VLLM_USE_MODELSCOPE"] = "True"
        # vLLM/transformers will resolve `Qwen/Qwen3-4B` against ModelScope
        # if VLLM_USE_MODELSCOPE is set.
        return ms_id

    hf_id = mcfg.get("hf_id")
    if hf_id:
        return hf_id

    raise KeyError(
        "model entry must specify one of: local_path, modelscope_id, hf_id"
    )


def _build_full_prompt(tokenizer, user_text: str, enable_thinking: bool = False) -> str:
    """Build a chat-formatted prompt using the model's chat template.

    For Qwen3 / R1-Distill we go through `apply_chat_template` so the prompt
    matches what the model was trained on.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        # Older tokenizers don't accept `enable_thinking`.
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def run_vllm(
    problems: list[dict],
    model_cfg: dict,
    sampling_cfg: dict,
    enable_thinking: bool = False,
) -> list[dict]:
    """Run a single model over all problems via vLLM and return per-problem records.

    Uses **per-prompt** SamplingParams: each problem gets a max_tokens budget
    estimated from its structural parameters. This avoids the bad tradeoff
    of a single global max_tokens (either too small for hard problems or so
    large that vLLM's KV cache thrashes on short ones).

    Also caps vLLM's concurrency via ``max_num_seqs`` so the KV cache never
    overcommits — the root cause of the 11000-preemption / 11-hour run we
    saw with the previous version.
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from src.calibration.token_budget import (
        estimate_max_tokens,
        summarize_budget_distribution,
    )

    source = _resolve_model_source(model_cfg)
    print(f"[run_eval] loading model from {source} "
          f"(TP={model_cfg.get('tensor_parallel_size', 1)})")

    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    prompts = [_build_full_prompt(tokenizer, p["prompt"], enable_thinking) for p in problems]

    # ----- vLLM engine config -----
    # max_num_seqs: cap concurrent sequences. With max_tokens up to 12288 and
    # 4B model (~360 KB KV / token), 16 seqs * 12288 * 360KB ≈ 70 GB worst case
    # -> fits in 2x24 GB cards with TP=2 + bf16 weights.
    max_num_seqs = int(sampling_cfg.get("max_num_seqs", 16))
    max_model_len = int(model_cfg.get(
        "max_model_len",
        sampling_cfg.get("max_model_len", 16384),
    ))

    llm_kwargs = dict(
        model=source,
        tensor_parallel_size=model_cfg.get("tensor_parallel_size", 1),
        max_model_len=max_model_len,
        dtype=model_cfg.get("dtype", "bfloat16"),
        gpu_memory_utilization=model_cfg.get("gpu_memory_utilization", 0.85),
        enable_prefix_caching=True,
        trust_remote_code=True,
        max_num_seqs=max_num_seqs,
    )
    if model_cfg.get("enforce_eager", False) or sampling_cfg.get("enforce_eager", False):
        llm_kwargs["enforce_eager"] = True
    if "quantization" in model_cfg:
        llm_kwargs["quantization"] = model_cfg["quantization"]

    print(f"[run_eval] vLLM config: max_num_seqs={max_num_seqs}, "
          f"max_model_len={llm_kwargs['max_model_len']}, "
          f"gpu_mem_util={llm_kwargs['gpu_memory_utilization']}, "
          f"enforce_eager={llm_kwargs.get('enforce_eager', False)}")

    llm = LLM(**llm_kwargs)

    # ----- Per-prompt SamplingParams via adaptive token budget -----
    use_dynamic = bool(sampling_cfg.get("dynamic_max_tokens", True))
    floor = int(sampling_cfg.get("max_tokens_floor", 1024))
    ceiling = int(sampling_cfg.get("max_tokens_ceiling", 12288))
    fallback = int(sampling_cfg.get("max_tokens", 4096))
    context_margin = int(sampling_cfg.get("context_margin_tokens", 32))
    temperature = float(sampling_cfg.get("temperature", 0.0))
    seed = int(sampling_cfg.get("seed", 42))
    repetition_penalty = float(sampling_cfg.get("repetition_penalty", 1.05))

    if use_dynamic:
        dist = summarize_budget_distribution(problems)
        if dist.get("n", 0) > 0:
            print(f"[run_eval] adaptive max_tokens distribution: "
                  f"min={dist['min']}, median={dist['median']}, "
                  f"p90={dist['p90']}, max={dist['max']}, "
                  f"mean={dist['mean']:.0f}")
        per_prompt_max = [
            estimate_max_tokens(p, floor=floor, ceiling=ceiling,
                                fallback=fallback)
            for p in problems
        ]
    else:
        # Override for ablation / debugging — use the single fallback.
        budget = int(model_cfg.get("max_tokens_override", fallback))
        per_prompt_max = [budget] * len(problems)
        print(f"[run_eval] dynamic_max_tokens=False; using uniform "
              f"max_tokens={budget}")

    # vLLM enforces prompt_tokens + max_tokens <= max_model_len. The adaptive
    # budget estimates reasoning length, then this pass clips it against the
    # actual chat-formatted prompt length so long-CoT runs use as much context
    # as available without failing at generation time.
    prompt_enc = tokenizer(
        prompts,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    prompt_lens = [len(ids) for ids in prompt_enc["input_ids"]]
    clipped = []
    n_capped, n_too_long = 0, 0
    for requested, prompt_len in zip(per_prompt_max, prompt_lens):
        room = max_model_len - int(prompt_len) - context_margin
        if room < floor:
            n_too_long += 1
        allowed = max(1, room)
        mt = min(int(requested), allowed)
        if mt < int(requested):
            n_capped += 1
        clipped.append(mt)
    per_prompt_max = clipped
    if prompt_lens:
        print(f"[run_eval] prompt tokens: min={min(prompt_lens)}, "
              f"median={sorted(prompt_lens)[len(prompt_lens)//2]}, "
              f"max={max(prompt_lens)}")
    if n_capped:
        print(f"[run_eval] capped max_tokens for {n_capped}/{len(problems)} "
              f"prompts to fit max_model_len={max_model_len}")
    if n_too_long:
        print(f"[run_eval] WARNING: {n_too_long} prompts have less than "
              f"floor={floor} generation tokens available; raise max_model_len "
              f"or shorten prompts for best performance.")

    sampling_list = [
        SamplingParams(
            temperature=temperature,
            max_tokens=mt,
            seed=seed,
            repetition_penalty=repetition_penalty,
        )
        for mt in per_prompt_max
    ]

    t0 = time.time()
    outputs = llm.generate(prompts, sampling_list)
    dt = time.time() - t0
    print(f"[run_eval] generated {len(prompts)} completions in {dt:.1f}s "
          f"({len(prompts)/max(dt,1e-3):.2f} req/s)")

    records = []
    n_stop, n_length, n_other = 0, 0, 0
    for prob, out in zip(problems, outputs):
        if out.outputs:
            comp = out.outputs[0]
            text = comp.text or ""
            # vLLM exposes:
            #   - finish_reason: "stop" | "length" | "abort"
            #   - token_ids:    list of generated tokens (post-prompt)
            finish_reason = getattr(comp, "finish_reason", None) or "unknown"
            n_gen_tokens = len(getattr(comp, "token_ids", []) or [])
        else:
            text = ""
            finish_reason = "no_output"
            n_gen_tokens = 0

        if finish_reason == "stop":
            n_stop += 1
        elif finish_reason == "length":
            n_length += 1
        else:
            n_other += 1

        if prob.get("answer_type") in ("math", "integer"):
            pred = extract_math_answer(text)
            if not pred and prob.get("answer_type") == "integer":
                pred = extract_answer(text)
        else:
            pred = extract_answer(text)
        ok = verify(
            pred,
            prob["ground_truth"],
            prob["answer_type"],
            ground_truth_raw=prob.get("ground_truth_raw"),
        )
        records.append({
            "id": prob["id"],
            "model": model_cfg["name"],
            "pred": pred,
            "correct": int(ok),
            "finish_reason": finish_reason,
            "n_gen_tokens": n_gen_tokens,
            "n_chars": len(text),
            # Save the FULL completion. This makes per-problem JSON ~10-30 KB
            # but lets you actually see the model's real ending (not just the
            # first 1500 chars). Disk cost for 1500 problems x 7 models is
            # under 500 MB total — acceptable.
            "raw_text": text,
        })

    print(f"[run_eval] finish_reason breakdown: "
          f"stop={n_stop}, length={n_length}, other={n_other}")

    # Free GPU memory before the next model.
    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True,
                    help="Model name as in configs/models.yaml.")
    ap.add_argument("--input", type=str, required=True,
                    help="JSONL of candidate problems.")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--models_config", type=str, default="configs/models.yaml")
    ap.add_argument("--enable_thinking", action="store_true",
                    help="For Qwen3 thinking-mode tokenizer support.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional: only evaluate first N problems (sanity).")
    ap.add_argument("--tag", type=str, default=None,
                    help="Optional suffix appended to output filename "
                         "(e.g. 'symbolic'). Use this to keep results from "
                         "the same model on different inputs side-by-side: "
                         "the output file becomes {model}__{tag}.jsonl.")
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    mcfg_all = _load_yaml(args.models_config)
    if args.model not in mcfg_all:
        raise KeyError(f"{args.model} not in {args.models_config}. "
                       f"Available: {sorted(mcfg_all.keys())}")
    mcfg = _expand_env(dict(mcfg_all[args.model]))
    mcfg["name"] = args.model

    problems = load_jsonl(args.input)
    if args.limit is not None:
        problems = problems[: args.limit]
    print(f"[run_eval] {len(problems)} problems loaded from {args.input}")

    sampling_cfg = cfg["calibration"]["sampling"]
    records = run_vllm(problems, mcfg, sampling_cfg, args.enable_thinking)

    suffix = f"__{args.tag}" if args.tag else ""
    out_path = Path(args.out_dir) / f"{args.model}{suffix}.jsonl"
    save_jsonl(records, out_path)
    pass_rate = sum(r["correct"] for r in records) / len(records) if records else 0.0
    print(f"[run_eval] {args.model}{suffix}: pass_rate={pass_rate:.3f} → {out_path}")


if __name__ == "__main__":
    main()
