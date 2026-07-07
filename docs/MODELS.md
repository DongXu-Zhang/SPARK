# Calibration model pool

We use **a within-Qwen-family pool** to avoid API costs and ensure full
reproducibility. The pool spans ~50× in parameter count
(0.6B → 32B), giving a coherent difficulty gradient.

---

## 2× RTX 3090 default pool (7 models)

All weights are pulled from **ModelScope (魔塔)** — set
`MODELS_DIR=/your/big/disk/models` and run
`bash scripts/download_models.sh` once.

| # | Name in `configs/models.yaml` | ModelScope ID                                  | Size   | TP | VRAM       | Role               |
|---|-------------------------------|------------------------------------------------|--------|----|------------|--------------------|
| 1 | `Qwen3-0.6B`                  | `Qwen/Qwen3-0.6B`                              | 0.6B   | 1  | ~2 GB      | floor baseline     |
| 2 | `Qwen3-1.7B`                  | `Qwen/Qwen3-1.7B`                              | 1.7B   | 1  | ~5 GB      | weak               |
| 3 | `Qwen3-4B`                    | `Qwen/Qwen3-4B`                                | 4B     | 1  | ~11 GB     | mid-low (target)   |
| 4 | `Qwen3-8B`                    | `Qwen/Qwen3-8B`                                | 8B     | 1  | ~21 GB     | mid                |
| 5 | `Qwen3-14B-AWQ`               | `Qwen/Qwen3-14B-AWQ`                           | 14B/4b | 1  | ~14 GB     | mid-high           |
| 6 | `DSR1-Distill-Qwen-7B`        | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`      | 7B     | 1  | ~19 GB     | reasoning-distilled|
| 7 | `Qwen3-32B-AWQ`               | `Qwen/Qwen3-32B-AWQ`                           | 32B/4b | 2  | ~26 GB     | strong (quantized) |

After download:

```
$MODELS_DIR/
├── Qwen3-0.6B/         ~1.2 GB
├── Qwen3-1.7B/         ~3.5 GB
├── Qwen3-4B/           ~8 GB
├── Qwen3-8B/           ~16 GB
├── Qwen3-14B-AWQ/      ~9 GB
├── DSR1-Distill-Qwen-7B/  ~14 GB
└── Qwen3-32B-AWQ/      ~18 GB
                       ────
                        ~70 GB total
```

**Time estimate (4500 problems, single A6000-equivalent throughput):**
~5-8 hours per small model, ~12-18 h for 32B-AWQ → ~2-3 days end-to-end.
Using `tee` logs in `logs/calib_*.log` for each step.

---

## 4× RTX 3090 extended pool (+2 models)

Append the following to your run by setting `POOL=4card`:

```bash
POOL=4card bash scripts/download_models.sh
POOL=4card bash scripts/run_calibration.sh
```

| # | Name                          | ModelScope ID                                  | Size | TP | VRAM        | Role                |
|---|-------------------------------|------------------------------------------------|------|----|-------------|---------------------|
| 8 | `Qwen3-32B`                   | `Qwen/Qwen3-32B`                               | 32B  | 4  | ~80 GB      | strong (no quant)   |
| 9 | `DSR1-Distill-Qwen-32B`       | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`    | 32B  | 4  | ~80 GB      | reasoning strong    |

These add ~130 GB on disk. Total disk for 4-card pool: ~200 GB.

These two are the most informative models: AWQ quantization can sometimes
soften the difficulty signal at the high end, and adding the bf16 32B model
sharpens `d_empirical` near the capability frontier.

---

## How the pool maps to `d_empirical`

For each problem the empirical difficulty is

```
d_empirical = 1 - mean(correct_i across all models in the pool)
```

A problem with d_empirical = 0 → all 7 models solved it → very easy.
A problem with d_empirical = 1 → all 7 models failed → at the frontier.
Most signal lives in the middle band where the 4B / 8B / 14B models start
diverging.

---

## Per-model launch flags

The default `gpu_memory_utilization` is 0.85 in `configs/models.yaml`. If
you observe out-of-memory during weight load, drop to 0.80 first; if it
happens during KV cache allocation, drop `max_model_len` to 2048.

For the 32B-AWQ on 2×3090, KV cache is the bottleneck. Empirically:

- `--gpu-memory-utilization 0.90` works at `max_model_len=4096` only with
  short prompts; safer to keep `max_model_len=3072`.
- If you need more output budget for R1-Distill, raise `max_tokens_override`
  in `configs/models.yaml` and lower `max_model_len`.

---

## Adding / swapping a model

1. Add an entry to `configs/models.yaml`.
2. For manual debugging, run `vllm serve` with the same `local_path` / tensor-parallel
   settings as in `configs/models.yaml` (see vLLM docs).
3. Add the name to the relevant `MODELS=(...)` array in
   `scripts/run_calibration.sh`.

Keep model names short and stable — they end up as JSON keys in
`model_pass_rate` for every problem in the released dataset.
