# SPARK-Steering Next Experiments

This document records the executable next-stage pipeline after the
length-control finding:

1. raw chi is length-sensitive;
2. chi_lc isolates length-controlled latent response;
3. steering should target hard under-activated algorithmic examples.

## Required Inputs

The following files must already exist on the server:

```text
data/raw/{symbolic,logical,algorithmic}.jsonl
data/calibrated_full/Qwen3-4B__{symbolic,logical,algorithmic}.jsonl
data/spark_full/Qwen3-4B__{symbolic,logical,algorithmic}.jsonl
```

## 1. Export Length-Controlled Records

```bash
python scripts/export_length_controlled_records.py \
  --model Qwen3-4B \
  --domains symbolic logical algorithmic \
  --raw_dir data/raw \
  --calibration_dir data/calibrated_full \
  --spark_dir data/spark_full \
  --out_dir data/analysis \
  --score_field chi_max \
  --d_field d_structural
```

Outputs:

```text
data/analysis/Qwen3-4B__algorithmic__length_controlled.jsonl
data/analysis/Qwen3-4B__algorithmic__length_controlled.meta.json
data/analysis/Qwen3-4B__length_controlled.summary.json
```

Each row adds:

```text
chi_lc
chi_ratio_len
chi_expected_for_length
lc_pred_log_chi
lc_log_tokens
lc_log_chi
```

## 2. Matched-Length Frontier Test

```bash
python scripts/matched_length_frontier_test.py \
  --model Qwen3-4B \
  --domain algorithmic \
  --analysis_dir data/analysis \
  --out_dir figures/matched_length_Qwen3-4B \
  --easy_d_min 0.30 \
  --easy_d_max 0.55 \
  --hard_d_min 0.55 \
  --hard_d_max 0.80 \
  --token_tol_frac 0.20 \
  --max_pairs 200
```

Purpose: compare hard/frontier samples against easier same-length samples.

Outputs:

```text
figures/matched_length_Qwen3-4B/matched_length_report__Qwen3-4B__algorithmic.md
figures/matched_length_Qwen3-4B/matched_pairs__Qwen3-4B__algorithmic.csv
```

Key metric:

```text
mean hard-easy chi_lc
fraction hard lower chi_lc
```

## 3. Select Steering Sets

```bash
python scripts/select_steering_sets.py \
  --model Qwen3-4B \
  --domain algorithmic \
  --analysis_dir data/analysis \
  --out_dir data/steering \
  --active_d_min 0.30 \
  --active_d_max 0.55 \
  --target_d_min 0.55 \
  --target_d_max 0.80 \
  --n_demos 32 \
  --n_anchors 64 \
  --n_targets 160 \
  --n_test 240
```

Outputs:

```text
data/steering/Qwen3-4B__algorithmic__active_demos.jsonl
data/steering/Qwen3-4B__algorithmic__active_anchors.jsonl
data/steering/Qwen3-4B__algorithmic__underactivated_targets.jsonl
data/steering/Qwen3-4B__algorithmic__heldout_test.jsonl
data/steering/Qwen3-4B__algorithmic__selection_report.md
```

## 4. Extract Critical Reasoning Direction

Requires GPU and the local HF model path.

```bash
export CUDA_VISIBLE_DEVICES=6,7

nohup python -u scripts/extract_critical_direction.py \
  --model_path /home/zhangdongxu/AAAI2027/models/Qwen3-4B \
  --demos data/steering/Qwen3-4B__algorithmic__active_demos.jsonl \
  --anchors data/steering/Qwen3-4B__algorithmic__active_anchors.jsonl \
  --out data/steering/Qwen3-4B__algorithmic__critical_direction.pt \
  --layer auto \
  --pool last \
  --num_demos 4 \
  --limit_anchors 32 \
  --dtype bfloat16 \
  --device_map auto \
  > logs/extract_critical_direction_Qwen3-4B_algorithmic.log 2>&1 &
```

Outputs:

```text
data/steering/Qwen3-4B__algorithmic__critical_direction.pt
data/steering/Qwen3-4B__algorithmic__critical_direction.meta.json
```

## 5. Evaluate Steering on Held-Out Targets

Start small first:

```bash
export CUDA_VISIBLE_DEVICES=6,7

nohup python -u scripts/eval_spark_steering.py \
  --model_path /home/zhangdongxu/AAAI2027/models/Qwen3-4B \
  --direction data/steering/Qwen3-4B__algorithmic__critical_direction.pt \
  --input data/steering/Qwen3-4B__algorithmic__heldout_test.jsonl \
  --out data/steering/Qwen3-4B__algorithmic__steering_eval_limit30.jsonl \
  --alphas 0.0 0.25 0.5 1.0 \
  --limit 30 \
  --max_new_tokens 4096 \
  --repetition_penalty 1.05 \
  --dtype bfloat16 \
  --device_map auto \
  --position all \
  --concise_prompt \
  > logs/eval_spark_steering_Qwen3-4B_algorithmic_limit30.log 2>&1 &
```

Defaults in the script now match calibration-friendly settings: **`--max_new_tokens` defaults to 4096**, **`--repetition_penalty` defaults to 1.05** (same as `configs/default.yaml`) to suppress edge-list / tuple repetition loops, especially when `alpha>0`.

Each JSONL row includes **`n_gen_new_tokens`** and **`finish_reason`** (`stop` vs `length`) for quick truncation audits.

If the small run is stable, increase `--limit` and rerun with a larger batch.

Main metric:

```text
alpha=0.0 baseline accuracy
alpha>0 steered accuracy
```

Positive evidence:

```text
hard-bin accuracy improves for alpha>0
```

