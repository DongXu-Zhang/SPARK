"""Evaluate a SPARK critical direction with activation steering.

This is a minimal HF/Transformers evaluation loop. It is intentionally meant
for small batches first (e.g. --limit 30) to validate whether the direction
improves hard under-activated examples before scaling up.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.calibration.math_grade import extract_math_answer  # noqa: E402
from src.calibration.verify import extract_answer, verify  # noqa: E402
from src.datasets.gsm8k import format_gsm8k_prompt, strip_gsm8k_prompt_suffix  # noqa: E402
from src.datasets.math500 import format_math500_prompt, strip_math_prompt_suffix  # noqa: E402
from src.steering.activation import (  # noqa: E402
    add_steering_hook,
    format_prompt,
    load_hf_model_and_tokenizer,
)
from src.utils.io import append_jsonl, load_jsonl, save_jsonl  # noqa: E402

# Official Qwen3 non-thinking MATH-500 targets (Table 18 / Table 20, tech report).
OFFICIAL_MATH500_TARGETS = {
    "Qwen3-0.6B": 0.552,
    "Qwen3-4B": 0.848,
    "Qwen3-8B": 0.874,
}


def _alpha_metrics(rows: list[dict]) -> dict:
    """Accuracy and generation-length stats for one α slice."""
    n = len(rows)
    acc = sum(int(r.get("correct", 0)) for r in rows) / n if n else None
    ntoks = [int(r["n_gen_new_tokens"]) for r in rows if r.get("n_gen_new_tokens") is not None]
    avg_gen = sum(ntoks) / len(ntoks) if ntoks else None
    median_gen = None
    if ntoks:
        s = sorted(ntoks)
        mid = len(s) // 2
        median_gen = float(s[mid]) if len(s) % 2 else (s[mid - 1] + s[mid]) / 2.0
    frs = [r.get("finish_reason") for r in rows if r.get("finish_reason") is not None]
    pct_length = (100.0 * sum(1 for fr in frs if fr == "length") / len(frs)) if frs else None
    return {
        "n": n,
        "acc": acc,
        "avg_gen_tokens": avg_gen,
        "median_gen_tokens": median_gen,
        "pct_length": pct_length,
    }


class _PresencePenaltyLogitsProcessor:
    """OpenAI-style presence penalty when HF GenerationConfig lacks the field."""

    def __init__(self, penalty: float, prompt_len: int):
        self.penalty = float(penalty)
        self.prompt_len = int(prompt_len)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.penalty <= 0:
            return scores
        for tid in input_ids[0, self.prompt_len :].tolist():
            tid = int(tid)
            if 0 <= tid < scores.shape[-1]:
                scores[0, tid] -= self.penalty
        return scores


class _RepeatWindowStoppingCriteria:
    """Stop when the same token window repeats (prevents MATH-500 loop-to-max_tokens)."""

    def __init__(self, prompt_len: int, window: int = 48, max_repeats: int = 3):
        from transformers import StoppingCriteria

        class _Impl(StoppingCriteria):
            def __init__(self, outer):
                self._outer = outer

            def __call__(self, input_ids, scores, **kwargs) -> bool:
                return self._outer._should_stop(input_ids)

        self.prompt_len = int(prompt_len)
        self.window = int(window)
        self.max_repeats = int(max_repeats)
        self._counts: dict[tuple[int, ...], int] = {}
        self.criteria = _Impl(self)

    def _should_stop(self, input_ids: torch.LongTensor) -> bool:
        gen_len = int(input_ids.shape[1]) - self.prompt_len
        if gen_len < self.window * self.max_repeats:
            return False
        # Only copy the trailing window (O(window)), not the full generation (O(gen_len)).
        start = self.prompt_len + gen_len - self.window
        key = tuple(int(t) for t in input_ids[0, start:].tolist())
        self._counts[key] = self._counts.get(key, 0) + 1
        return self._counts[key] >= self.max_repeats


def _cap_max_new_tokens(model, tokenizer, n_prompt_tokens: int, requested: int) -> int:
    """Clip generation budget so prompt + new tokens fit model context."""
    ctx = getattr(model.config, "max_position_embeddings", None)
    if ctx is None:
        ctx = getattr(tokenizer, "model_max_length", None)
    if ctx is None or int(ctx) > 200_000:
        return int(requested)
    budget = int(ctx) - int(n_prompt_tokens) - 32
    return max(256, min(int(requested), budget))


def _truncate_text(text: str | None, max_chars: int) -> str | None:
    if text is None or max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...[truncated {len(text) - max_chars} chars]"


def _pred_for_log(pred) -> str:
    s = "" if pred is None else str(pred)
    if len(s) > 120:
        return s[:120] + "..."
    return s


def _question_seed(base_seed: int | None, record_id: str) -> int | None:
    """Stable per-question seed for sampling (reproducible, not identical across items)."""
    if base_seed is None:
        return None
    h = abs(hash(str(record_id))) & 0x7FFFFFFF
    return int(base_seed) ^ h


def _default_presence_penalty(model_path: str, prompt_style: str) -> float:
    """HF-friendly presence penalty defaults.

    Official Qwen3 non-thinking uses 1.5 on vLLM/API. On HF ``generate()``, 1.5
    inflates outputs and can collapse small models; MATH-500 HF eval uses 0 for all
    sizes (repeat-window stopping handles loops). Override via ``--presence_penalty``.
    """
    if prompt_style not in ("math500_official", "math500"):
        return 0.0
    return 0.0


def _default_max_new_tokens(model_path: str, prompt_style: str) -> int:
    """Context budget: edge models rarely need 32k; 4B/8B keep official 32768."""
    if prompt_style not in ("math500_official", "math500"):
        return 32768
    name = Path(model_path).name.lower()
    if "0.6b" in name:
        return 8192
    if "llama" in name or "deepseek" in name:
        return 16384
    return 32768


def _default_no_system_prompt(model_path: str, prompt_style: str) -> bool:
    """Whether to omit the extra system message (user-only chat template).

    EvalScope official MATH-500 is user-only, but HF ablations showed:
    - Qwen3-0.6B: user-only aligns better with the tech-report baseline
    - Qwen3-4B/8B: DEFAULT_SYSTEM (+2–3 pp vs user-only at alpha=0)
    """
    if prompt_style not in ("math500_official", "math500"):
        return False
    name = Path(model_path).name.lower()
    if "0.6b" in name:
        return True
    return False


def _apply_eval_profile(args) -> None:
    """Apply a named decoding/prompt profile."""
    profile = getattr(args, "eval_profile", "default")
    if profile in (None, "", "default"):
        return
    name = Path(args.model_path).name.lower()
    if profile == "qwen_official":
        args.temperature = 0.7
        args.top_p = 0.8
        args.top_k = 20
        args.repetition_penalty = 1.0
        args.with_system_prompt = False
        args.no_system_prompt = _default_no_system_prompt(
            args.model_path, getattr(args, "prompt_style", "math500_official")
        )
        if args.presence_penalty is None:
            args.presence_penalty = 0.0
    elif profile in ("cal_aligned", "llama_v2"):
        args.greedy = True
        args.temperature = 0.0
        args.top_p = 1.0
        args.top_k = 0
        args.repetition_penalty = 1.05
        args.with_system_prompt = True
        args.no_system_prompt = False
        if args.presence_penalty is None:
            args.presence_penalty = 0.0
        if args.max_new_tokens is None:
            args.max_new_tokens = 16384
        if profile == "llama_v2" and args.position == "all":
            args.position = "last"
        if profile == "llama_v2" and "llama" not in name:
            print(f"[eval-steering] warn: llama_v2 profile on non-Llama model {name}", flush=True)
    else:
        raise ValueError(f"Unknown eval_profile={profile!r}")
    print(f"[eval-steering] eval_profile={profile} applied", flush=True)


def _load_done_keys(out_path: Path, *, retry_errors: bool = True) -> set[tuple[str, float]]:
    if not out_path.is_file():
        return set()
    done: set[tuple[str, float]] = set()
    for row in load_jsonl(out_path):
        rid = row.get("id")
        if rid is None:
            continue
        if retry_errors and row.get("error"):
            continue
        done.add((str(rid), float(row.get("alpha", 0.0))))
    return done


def _maybe_empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_eval_problem_text(
    problem_text: str,
    concise: bool = False,
    prompt_style: str = "default",
) -> str:
    if prompt_style in ("math500", "math500_official"):
        return format_math500_prompt(problem_text)
    if prompt_style in ("gsm8k", "gsm8k_official"):
        return format_gsm8k_prompt(problem_text)
    if prompt_style == "gsm8k_legacy":
        return (
            f"{strip_gsm8k_prompt_suffix(problem_text)}\n\n"
            "Important:\n"
            "- Show concise step-by-step reasoning.\n"
            "- End with exactly one line: \\boxed{integer}.\n"
            "- The value inside \\boxed{} must be the final integer only."
        )
    if prompt_style == "math500_legacy":
        return (
            f"{strip_math_prompt_suffix(problem_text)}\n\n"
            "Important:\n"
            "- Show step-by-step reasoning.\n"
            "- End with exactly one line: \\boxed{answer}.\n"
            "- The value inside \\boxed{} must be only the final answer "
            "(number, fraction, or expression)."
        )
    if not concise:
        return problem_text
    return (
        f"{problem_text}\n\n"
        "Important instructions:\n"
        "- Do not copy or restate the edge list.\n"
        "- Compute the requested graph quantity.\n"
        "- End with exactly one final line in the form: \\boxed{answer}.\n"
        "- The answer inside \\boxed{} must be only the final integer or label."
    )


@torch.no_grad()
def generate_one(
    model,
    tokenizer,
    problem_text: str,
    max_new_tokens: int,
    use_chat_template: bool = True,
    concise_prompt: bool = False,
    prompt_style: str = "default",
    enable_thinking: bool = False,
    no_system_prompt: bool = False,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    seed: int | None = None,
) -> tuple[str, dict]:
    problem_text = build_eval_problem_text(
        problem_text, concise=concise_prompt, prompt_style=prompt_style
    )
    system_prompt = None
    if not no_system_prompt:
        from src.steering.activation import DEFAULT_SYSTEM

        system_prompt = DEFAULT_SYSTEM
    prompt = format_prompt(
        tokenizer,
        problem_text,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
        enable_thinking=enable_thinking,
    )
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    n_prompt_tokens = int(input_ids.shape[1])
    effective_max_new = _cap_max_new_tokens(model, tokenizer, n_prompt_tokens, max_new_tokens)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    gen_kw: dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": effective_max_new,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        gen_kw["do_sample"] = True
        gen_kw["temperature"] = float(temperature)
        gen_kw["top_p"] = float(top_p)
        if top_k > 0:
            gen_kw["top_k"] = int(top_k)
    else:
        gen_kw["do_sample"] = False
    # HF >=4.4 rejects `generator` as model_kwargs; seed via torch instead.
    if seed is not None:
        torch.manual_seed(int(seed) & 0x7FFFFFFF)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed) & 0x7FFFFFFF)
    if repetition_penalty is not None and repetition_penalty > 1.0:
        gen_kw["repetition_penalty"] = float(repetition_penalty)
    if presence_penalty is not None and presence_penalty > 0.0:
        from transformers import LogitsProcessorList

        gen_kw["logits_processor"] = LogitsProcessorList(
            [_PresencePenaltyLogitsProcessor(presence_penalty, n_prompt_tokens)]
        )
    if prompt_style in ("math500_official", "math500"):
        from transformers import StoppingCriteriaList

        repeat_stop = _RepeatWindowStoppingCriteria(n_prompt_tokens)
        gen_kw["stopping_criteria"] = StoppingCriteriaList([repeat_stop.criteria])
    out = model.generate(**gen_kw)
    gen_ids = out[0, input_ids.shape[1]:]
    n_new = int(gen_ids.shape[0])
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    last_id = int(gen_ids[-1].item()) if n_new > 0 else None
    ended_on_stop = last_id is not None and (
        last_id == eos_id or (pad_id is not None and last_id == pad_id)
    )
    # Greedy decode: if we emitted exactly max_new_tokens and did not end on EOS/PAD,
    # treat as context-length truncation (same convention as many OpenAI-style APIs).
    if n_new >= effective_max_new and not ended_on_stop:
        finish_reason = "length"
    else:
        finish_reason = "stop"
    meta = {
        "n_prompt_tokens": n_prompt_tokens,
        "max_new_tokens_requested": int(max_new_tokens),
        "max_new_tokens_effective": int(effective_max_new),
        "n_gen_new_tokens": n_new,
        "n_total_tokens": n_prompt_tokens + n_new,
        "finish_reason": finish_reason,
    }
    return text, meta


def evaluate_record(
    model,
    tokenizer,
    record,
    max_new_tokens,
    direction=None,
    alpha=0.0,
    layer=None,
    position="all",
    concise_prompt: bool = False,
    prompt_style: str = "default",
    enable_thinking: bool = False,
    no_system_prompt: bool = False,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    seed: int | None = None,
):
    handle = None
    try:
        if direction is not None and alpha != 0.0:
            handle = add_steering_hook(
                model,
                hidden_state_index=int(layer),
                vector=direction,
                alpha=float(alpha),
                position=position,
            )
        text, gen_meta = generate_one(
            model,
            tokenizer,
            record["prompt"],
            max_new_tokens=max_new_tokens,
            use_chat_template=True,
            concise_prompt=concise_prompt,
            prompt_style=prompt_style,
            enable_thinking=enable_thinking,
            no_system_prompt=no_system_prompt,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
        )
        if record.get("answer_type") in ("math", "integer"):
            pred = extract_math_answer(text)
            if not pred and record.get("answer_type") == "integer":
                pred = extract_answer(text)
        else:
            pred = extract_answer(text)
        ok = verify(
            pred,
            record["ground_truth"],
            record["answer_type"],
            ground_truth_raw=record.get("ground_truth_raw"),
        )
        return {
            "pred": pred,
            "correct": int(ok),
            "raw_text": text,
            "n_chars": len(text),
            "n_prompt_tokens": gen_meta.get("n_prompt_tokens"),
            "n_gen_new_tokens": gen_meta.get("n_gen_new_tokens"),
            "n_total_tokens": gen_meta.get("n_total_tokens"),
            "finish_reason": gen_meta.get("finish_reason"),
        }
    finally:
        if handle is not None:
            handle.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--direction", required=True)
    ap.add_argument("--input", default="data/steering/Qwen3-4B__algorithmic__heldout_test.jsonl")
    ap.add_argument("--out", default="data/steering/Qwen3-4B__algorithmic__steering_eval.jsonl")
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Max new tokens per problem. Default: 8192 (0.6B) / 32768 (4B/8B) for MATH-500.",
    )
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument(
        "--max_memory",
        default=None,
        help="Per-device memory budget for device_map=auto, e.g. '0:22GiB,1:22GiB,cpu:30GiB'.",
    )
    ap.add_argument("--position", default="all", choices=["all", "last"])
    ap.add_argument("--concise_prompt", action="store_true",
                    help="Graph-domain strict final-answer instructions (edge-list copying).")
    ap.add_argument(
        "--prompt_style",
        default="default",
        choices=["default", "gsm8k", "gsm8k_official", "gsm8k_legacy", "math500", "math500_official", "math500_legacy"],
        help="Prompt style. gsm8k/gsm8k_official and math500/math500_official = EvalScope prompts.",
    )
    ap.add_argument("--enable_thinking", action="store_true",
                    help="Enable Qwen3 thinking mode. Default off (official non-thinking).")
    ap.add_argument(
        "--no_system_prompt",
        action="store_true",
        help="Force user-only chat template (no extra system). "
        "Default for math500_official: True on Qwen3-0.6B, False on 4B/8B.",
    )
    ap.add_argument(
        "--with_system_prompt",
        action="store_true",
        help="Force default system message even for math500_official.",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature. Official Qwen3 non-thinking MATH-500 uses 0.7; 0 = greedy.",
    )
    ap.add_argument(
        "--top_p",
        type=float,
        default=0.8,
        help="Nucleus sampling top_p (official non-thinking: 0.8).",
    )
    ap.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Top-k sampling (official non-thinking: 20). 0 disables.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed when temperature > 0.",
    )
    ap.add_argument(
        "--greedy",
        action="store_true",
        help="Force greedy decoding (temperature=0). Overrides --temperature.",
    )
    ap.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="HF generate repetition_penalty (1.0 disables). Official MATH-500 uses 1.0.",
    )
    ap.add_argument(
        "--presence_penalty",
        type=float,
        default=None,
        help="Presence penalty (official Qwen3 non-thinking MATH-500: 1.5). "
        "Default: 1.5 for math500_official, else 0.",
    )
    ap.add_argument(
        "--eval_profile",
        default="default",
        choices=["default", "qwen_official", "cal_aligned", "llama_v2"],
        help="Named decoding profile. llama_v2 = cal_aligned + position=last.",
    )
    ap.add_argument("--shuffle", action="store_true",
                    help="Shuffle input records before applying --limit.")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Append to --out and skip (id, alpha) pairs already present.",
    )
    ap.add_argument(
        "--no-retry-errors",
        action="store_true",
        help="When resuming, keep rows that recorded an error (default: re-run failed rows).",
    )
    ap.add_argument(
        "--max_raw_chars",
        type=int,
        default=8000,
        help="Truncate raw_text in output (0 = keep full text). Default 8000 saves disk.",
    )
    args = ap.parse_args()

    _apply_eval_profile(args)

    if args.greedy:
        args.temperature = 0.0

    if args.presence_penalty is None:
        args.presence_penalty = _default_presence_penalty(args.model_path, args.prompt_style)

    if args.max_new_tokens is None:
        args.max_new_tokens = _default_max_new_tokens(args.model_path, args.prompt_style)

    # MATH-500: model-aware system prompt (see _default_no_system_prompt).
    if args.prompt_style in ("math500_official", "math500"):
        if args.with_system_prompt:
            args.no_system_prompt = False
        elif args.no_system_prompt:
            pass  # explicit --no_system_prompt
        else:
            args.no_system_prompt = _default_no_system_prompt(args.model_path, args.prompt_style)

    print(
        f"[eval-steering] sampling: temp={args.temperature} top_p={args.top_p} top_k={args.top_k} "
        f"presence_penalty={args.presence_penalty} no_system={args.no_system_prompt} "
        f"system={'off' if args.no_system_prompt else 'DEFAULT_SYSTEM'} "
        f"max_new_tokens={args.max_new_tokens}",
        flush=True,
    )

    records = load_jsonl(args.input)
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(records)
    if args.limit is not None and args.limit > 0:
        records = records[: args.limit]
    if not records:
        raise RuntimeError(f"No records loaded from {args.input}")

    payload = torch.load(args.direction, map_location="cpu")
    direction = payload["vector"].float().cpu()
    layer = int(payload["hidden_state_index"])

    max_memory = None
    if args.max_memory:
        max_memory = {}
        for part in str(args.max_memory).split(","):
            part = part.strip()
            if not part:
                continue
            key, val = part.split(":", 1)
            key = key.strip()
            max_memory[int(key) if key.isdigit() else key] = val.strip()

    n_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(
        f"[eval-steering] loading model: {args.model_path} "
        f"(device_map={args.device_map}, visible_cuda={n_cuda})",
        flush=True,
    )
    model, tokenizer = load_hf_model_and_tokenizer(
        args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        max_memory=max_memory,
    )

    out_path = Path(args.out)
    done_keys = (
        _load_done_keys(out_path, retry_errors=not args.no_retry_errors)
        if args.resume
        else set()
    )
    if args.resume and done_keys:
        print(f"[eval-steering] resume: {len(done_keys)} (id, alpha) pairs already in {out_path}", flush=True)
    elif out_path.is_file() and not args.resume:
        print(f"[eval-steering] overwriting existing {out_path}", flush=True)

    rows: list[dict] = []
    if args.resume and out_path.is_file():
        raw_rows = load_jsonl(out_path)
        # Keep the latest row per (id, alpha); prefer successful over error rows.
        by_key: dict[tuple[str, float], dict] = {}
        for r in raw_rows:
            rid = r.get("id")
            if rid is None:
                continue
            key = (str(rid), float(r.get("alpha", 0.0)))
            prev = by_key.get(key)
            if prev is None:
                by_key[key] = r
            elif prev.get("error") and not r.get("error"):
                by_key[key] = r
            elif not prev.get("error") and r.get("error"):
                pass
            else:
                by_key[key] = r
        rows = list(by_key.values())
        if not args.no_retry_errors:
            n_err = sum(1 for r in rows if r.get("error"))
            if n_err:
                rows = [r for r in rows if not r.get("error")]
                print(
                    f"[eval-steering] resume: will retry {n_err} failed rows "
                    f"({len(rows)} successful rows kept)",
                    flush=True,
                )

    t0 = time.time()
    for alpha in args.alphas:
        n_ok = 0
        n_skip = 0
        for i, rec in enumerate(records, start=1):
            key = (str(rec.get("id")), float(alpha))
            if key in done_keys:
                n_skip += 1
                subset = [r for r in rows if str(r.get("id")) == key[0] and float(r.get("alpha")) == key[1]]
                if subset:
                    n_ok += int(subset[-1].get("correct", 0))
                continue
            rec_seed = _question_seed(args.seed, str(rec.get("id", i)))
            try:
                result = evaluate_record(
                    model,
                    tokenizer,
                    rec,
                    max_new_tokens=args.max_new_tokens,
                    direction=direction,
                    alpha=alpha,
                    layer=layer,
                    position=args.position,
                    concise_prompt=args.concise_prompt,
                    prompt_style=args.prompt_style,
                    enable_thinking=args.enable_thinking,
                    no_system_prompt=args.no_system_prompt,
                    repetition_penalty=args.repetition_penalty,
                    presence_penalty=args.presence_penalty,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    seed=rec_seed,
                )
            except Exception as e:
                result = {
                    "pred": None,
                    "correct": 0,
                    "raw_text": "",
                    "n_chars": 0,
                    "n_gen_new_tokens": None,
                    "finish_reason": None,
                    "error": f"{type(e).__name__}: {e}",
                }
            if result.get("error"):
                print(f"[eval-steering] ERROR id={rec.get('id')}: {result['error']}", flush=True)
            n_ok += int(result.get("correct", 0))
            _maybe_empty_cuda_cache()
            row = {
                "id": rec.get("id"),
                "alpha": alpha,
                "layer": layer,
                "position": args.position,
                "repetition_penalty": args.repetition_penalty,
                "presence_penalty": args.presence_penalty,
                "no_system_prompt": int(args.no_system_prompt),
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "seed": args.seed,
                "d_value": rec.get("d_value"),
                "chi_lc": rec.get("chi_lc"),
                "chi_ratio_len": rec.get("chi_ratio_len"),
                "original_correct": rec.get("correct"),
                "ground_truth": rec.get("ground_truth"),
                "answer_type": rec.get("answer_type"),
                "pred": result.get("pred"),
                "correct": result.get("correct"),
                "n_chars": result.get("n_chars"),
                "n_prompt_tokens": result.get("n_prompt_tokens"),
                "n_gen_new_tokens": result.get("n_gen_new_tokens"),
                "n_total_tokens": result.get("n_total_tokens"),
                "finish_reason": result.get("finish_reason"),
                "raw_text": _truncate_text(result.get("raw_text"), args.max_raw_chars),
            }
            if "error" in result:
                row["error"] = result["error"]
            rows.append(row)
            done_keys.add(key)
            append_jsonl(row, out_path)
            nt = row.get("n_gen_new_tokens")
            fr = row.get("finish_reason")
            print(
                f"[eval-steering] alpha={alpha} {i}/{len(records)} "
                f"id={rec.get('id')} correct={row['correct']} pred={_pred_for_log(row['pred'])} "
                f"n_new={nt} {fr}",
                flush=True,
            )
        if n_skip:
            print(f"[eval-steering] alpha={alpha} skipped={n_skip} resumed", flush=True)
        subset = [r for r in rows if float(r.get("alpha")) == float(alpha)]
        m = _alpha_metrics(subset)
        acc = m["acc"] if m["acc"] is not None else 0.0
        avg_tok = m["avg_gen_tokens"]
        tok_s = f" avg_gen_tokens={avg_tok:.1f}" if avg_tok is not None else ""
        print(f"[eval-steering] alpha={alpha} acc={acc:.3f}{tok_s}", flush=True)

    save_jsonl(rows, out_path)
    summary = {}
    for alpha in args.alphas:
        subset = [r for r in rows if float(r["alpha"]) == float(alpha)]
        summary[str(alpha)] = _alpha_metrics(subset)

    model_name = Path(args.model_path).name
    if model_name in OFFICIAL_MATH500_TARGETS and any(float(a) == 0.0 for a in args.alphas):
        key = "0.0" if "0.0" in summary else "0"
        a0 = summary.get(key)
        if a0 and a0.get("acc") is not None:
            target = OFFICIAL_MATH500_TARGETS[model_name]
            got = float(a0["acc"])
            print(
                f"[eval-steering] baseline vs Qwen3 report: {got:.1%} "
                f"(target non-thinking MATH-500 {target:.1%}, delta {100*(got-target):+.1f}pp)",
                flush=True,
            )

    meta = {
        "args": vars(args),
        "direction": args.direction,
        "hidden_state_index": layer,
        "summary": summary,
        "elapsed_sec": time.time() - t0,
    }
    meta_path = out_path.with_suffix(".summary.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[eval-steering] rows    -> {out_path}", flush=True)
    print(f"[eval-steering] summary -> {meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
