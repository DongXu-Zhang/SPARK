"""Extract a Critical Reasoning Direction for SPARK-Steering.

For each active anchor problem, compare hidden states between:

    base   = target problem only
    active = SPARK-selected demonstrations + target problem

The mean difference h(active) - h(base) becomes the steering direction.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.steering.activation import (  # noqa: E402
    build_demo_prompt,
    collect_hidden_vector,
    load_hf_model_and_tokenizer,
    select_steering_layer,
)
from src.utils.io import load_jsonl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--demos", default="data/steering/Qwen3-4B__algorithmic__active_demos.jsonl")
    ap.add_argument("--anchors", default="data/steering/Qwen3-4B__algorithmic__active_anchors.jsonl")
    ap.add_argument("--out", default="data/steering/Qwen3-4B__algorithmic__critical_direction.pt")
    ap.add_argument("--layer", default="auto",
                    help="SPARK hidden-state index, e.g. 36. 'auto' uses select_steering_layer().")
    ap.add_argument("--min_layer", type=int, default=3,
                    help="When --layer auto, ignore chi peaks below this index (avoid layer-1 artifacts).")
    ap.add_argument("--pool", default="last", choices=["last", "mean"])
    ap.add_argument("--num_demos", type=int, default=4)
    ap.add_argument("--limit_anchors", type=int, default=32)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--max_length", type=int, default=None)
    ap.add_argument("--enable_thinking", action="store_true",
                    help="Enable Qwen3 thinking mode in chat template. Default is off to match calibration.")
    args = ap.parse_args()

    demos = load_jsonl(args.demos)
    anchors = load_jsonl(args.anchors)
    if args.limit_anchors is not None:
        anchors = anchors[: args.limit_anchors]
    if not demos:
        raise RuntimeError(f"No demos loaded from {args.demos}")
    if not anchors:
        raise RuntimeError(f"No anchors loaded from {args.anchors}")

    layer_meta = None
    if args.layer == "auto":
        layer, layer_meta = select_steering_layer(anchors, default=36, min_layer=args.min_layer)
        print(f"[extract-direction] auto layer={layer} meta={layer_meta}", flush=True)
    else:
        layer = int(args.layer)

    print(f"[extract-direction] loading model: {args.model_path}", flush=True)
    model, tokenizer = load_hf_model_and_tokenizer(
        args.model_path, dtype=args.dtype, device_map=args.device_map
    )

    deltas = []
    pair_meta = []
    for i, anchor in enumerate(anchors, start=1):
        demo_pool = [d for d in demos if d.get("id") != anchor.get("id")]
        active_problem_text = build_demo_prompt(
            target_prompt=anchor["prompt"],
            demos=demo_pool,
            k=args.num_demos,
        )
        base_problem_text = anchor["prompt"]

        h_active = collect_hidden_vector(
            model,
            tokenizer,
            active_problem_text,
            hidden_state_index=layer,
            pool=args.pool,
            use_chat_template=True,
            enable_thinking=args.enable_thinking,
            max_length=args.max_length,
        )
        h_base = collect_hidden_vector(
            model,
            tokenizer,
            base_problem_text,
            hidden_state_index=layer,
            pool=args.pool,
            use_chat_template=True,
            enable_thinking=args.enable_thinking,
            max_length=args.max_length,
        )
        delta = h_active - h_base
        deltas.append(delta)
        pair_meta.append({
            "anchor_id": anchor.get("id"),
            "delta_norm": float(delta.norm().item()),
            "anchor_d": anchor.get("d_value"),
            "anchor_chi_lc": anchor.get("chi_lc"),
            "anchor_spk": anchor.get("spk"),
        })
        print(
            f"[extract-direction] {i}/{len(anchors)} "
            f"id={anchor.get('id')} delta_norm={delta.norm().item():.4f}",
            flush=True,
        )

    mat = torch.stack(deltas, dim=0)
    vector = mat.mean(dim=0)
    vector = vector / vector.norm().clamp(min=1e-12)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "vector": vector.cpu(),
        "hidden_state_index": layer,
        "pool": args.pool,
        "num_demos": args.num_demos,
        "n_anchors": len(anchors),
        "mean_delta_norm": float(mat.norm(dim=1).mean().item()),
        "vector_norm_after_normalize": float(vector.norm().item()),
        "pair_meta": pair_meta,
        "layer_selection": layer_meta,
        "args": vars(args),
    }
    torch.save(payload, out_path)
    meta_path = out_path.with_suffix(".meta.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump({k: v for k, v in payload.items() if k != "vector"}, f, ensure_ascii=False, indent=2)
    print(f"[extract-direction] vector -> {out_path}", flush=True)
    print(f"[extract-direction] meta   -> {meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
