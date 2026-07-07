"""Batch runner — read JSONL of problems, compute χ/Φ/SPK, write enriched JSONL.

This is the high-level orchestrator. For each problem we:
  1. Format the prompt (with chat template if requested)
  2. Compute χ via susceptibility.compute_chi
  3. Compute Φ via coherence.compute_phi
  4. Compose SPK
  5. Write a record back

Memory:
  - We do not generate, only forward, so memory is bounded.
  - K perturbations + L layers are batched into one forward.
  - For Qwen3-4B with K=8 and T=2000, this fits in 24 GB at bf16.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import yaml

from src.spark.coherence import compute_phi
from src.spark.spark_index import compute_spark_index
from src.spark.susceptibility import compute_chi
from src.spark.types import ChiConfig, PhiConfig, SparkConfig
from src.utils.io import load_jsonl, save_jsonl


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _expand_env(value):
    if isinstance(value, str):
        return re.sub(r"\$\{(\w+)\}",
                      lambda m: os.environ.get(m.group(1), m.group(0)),
                      value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(x) for x in value]
    return value


def load_spark_config(yaml_path: str) -> SparkConfig:
    """Load configs/spark.yaml into a SparkConfig dataclass."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    chi_d = raw.get("chi", {}) or {}
    phi_d = raw.get("phi", {}) or {}
    return SparkConfig(
        chi=ChiConfig(
            K=int(chi_d.get("K", 8)),
            epsilon=float(chi_d.get("epsilon", 0.01)),
            layers=chi_d.get("layers", None),
            pool=str(chi_d.get("pool", "mean")),
            seed=int(chi_d.get("seed", 42)),
            use_attention_mask=bool(chi_d.get("use_attention_mask", True)),
        ),
        phi=PhiConfig(
            layers=phi_d.get("layers", None),
            pool=str(phi_d.get("pool", "mean")),
            use_attention_mask=bool(phi_d.get("use_attention_mask", True)),
        ),
        lambda_=float(raw.get("lambda", 1.0)),
        chi_aggregator=str(raw.get("chi_aggregator", "max")),
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_path: str, dtype: str = "bfloat16",
                              device_map: str = "auto"):
    """Load a HF causal LM and tokenizer.

    ``model_path`` may be:
      * a local directory (preferred — no network needed)
      * a HF/ModelScope id (requires VLLM_USE_MODELSCOPE or HF_ENDPOINT)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype, torch.bfloat16)

    print(f"[runner] loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    print(f"[runner] loading model from {model_path} (dtype={dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM = (
    "You are a careful reasoner. Think step by step, "
    "then put your final answer inside \\boxed{}."
)


def format_prompt(tokenizer, problem_text: str,
                  use_chat_template: bool = True,
                  system_prompt: str = DEFAULT_SYSTEM) -> str:
    """Apply the model's chat template (if requested) so that χ/Φ measure
    the hidden state of the same prompt the model would see during reasoning.
    """
    if not use_chat_template:
        return problem_text
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem_text},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        # Fall back to raw if tokenizer lacks a chat template.
        return problem_text


# ---------------------------------------------------------------------------
# Per-problem computation
# ---------------------------------------------------------------------------

def compute_one(model, tokenizer, prompt_text: str, cfg: SparkConfig) -> dict:
    """Compute χ + Φ + SPK for a single formatted prompt. Returns a dict
    that is JSON-serialisable."""
    chi_res = compute_chi(model, tokenizer, prompt_text, cfg.chi)
    phi_res = compute_phi(model, tokenizer, prompt_text, cfg.phi)

    chi_value = chi_res.chi_max if cfg.chi_aggregator == "max" else chi_res.chi_mean
    spk = compute_spark_index(chi_value, phi_res.phi, lambda_=cfg.lambda_)

    return {
        "chi_max": chi_res.chi_max,
        "chi_mean": chi_res.chi_mean,
        "chi_argmax_layer": chi_res.chi_argmax_layer,
        "chi_per_layer": {str(k): v for k, v in chi_res.chi_per_layer.items()},
        "phi": phi_res.phi,
        "spk": spk,
        "n_tokens": chi_res.n_tokens,
        "sigma": chi_res.sigma,
    }


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Local model path or HF/ModelScope id "
                         "(local path is strongly preferred).")
    ap.add_argument("--input", required=True,
                    help="JSONL of problems (each line must have 'id' and 'prompt').")
    ap.add_argument("--out", required=True,
                    help="JSONL output path.")
    ap.add_argument("--config", default="configs/spark.yaml")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N problems (sanity).")
    ap.add_argument("--no_chat_template", action="store_true",
                    help="Skip applying the model's chat template.")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--log_every", type=int, default=1,
                    help="Print progress every N newly processed problems.")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from <out>.partial if a previous run was interrupted.")
    args = ap.parse_args()

    cfg = load_spark_config(args.config)
    print(f"[runner] config: chi.K={cfg.chi.K}, chi.epsilon={cfg.chi.epsilon}, "
          f"chi.pool={cfg.chi.pool}, lambda={cfg.lambda_}")

    problems = load_jsonl(args.input)
    if args.limit is not None:
        problems = problems[: args.limit]
    print(f"[runner] {len(problems)} problems loaded from {args.input}", flush=True)

    out_path = Path(args.out)
    partial_path = Path(str(out_path) + ".partial")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = set()
    if args.resume and partial_path.exists():
        try:
            partial_records = load_jsonl(partial_path)
        except Exception as e:
            raise RuntimeError(
                f"Cannot resume because partial file is unreadable: {partial_path}: {e}"
            ) from e
        done_ids = {str(r["id"]) for r in partial_records if "id" in r and "error" not in r}
        print(f"[runner] resume enabled: {len(done_ids)} completed records found in {partial_path}",
              flush=True)
    elif partial_path.exists():
        partial_path.unlink()

    remaining = [p for p in problems if str(p.get("id")) not in done_ids]
    print(f"[runner] remaining problems: {len(remaining)}", flush=True)

    model, tokenizer = load_model_and_tokenizer(
        args.model, dtype=args.dtype, device_map=args.device_map
    )

    t0 = time.time()
    n_new_ok = 0
    n_new_error = 0
    log_every = max(1, int(args.log_every))

    with partial_path.open("a", encoding="utf-8") as f:
        for i, p in enumerate(remaining, start=1):
            pid = p.get("id", "?")
            prompt = format_prompt(
                tokenizer, p["prompt"],
                use_chat_template=(not args.no_chat_template),
            )
            try:
                metrics = compute_one(model, tokenizer, prompt, cfg)
                n_new_ok += 1
            except Exception as e:
                print(f"[runner] WARN: id={pid} failed: {type(e).__name__}: {e}",
                      flush=True)
                metrics = {"error": f"{type(e).__name__}: {e}"}
                n_new_error += 1

            rec = {**p, **metrics}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

            if i % log_every == 0 or i == len(remaining):
                dt = time.time() - t0
                total_done = len(done_ids) + i
                rate = i / max(dt, 1e-3)
                print(f"[runner] {total_done}/{len(problems)} done "
                      f"({i}/{len(remaining)} this run) | "
                      f"{rate:.3f} probs/sec | "
                      f"latest id={pid} | "
                      f"chi_max={metrics.get('chi_max', 'NA')} | "
                      f"phi={metrics.get('phi', 'NA')}",
                      flush=True)

    # Mark the domain complete only after all records have been written.
    shutil.move(str(partial_path), str(out_path))
    n_total_ok = len(done_ids) + n_new_ok
    print(f"[runner] done: ok={n_total_ok}, new_errors={n_new_error}, "
          f"total={len(problems)} -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
