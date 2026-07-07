# Environment setup

This project has two execution surfaces:

1. **Local / Windows** (problem generators only): pure Python, CPU-only.
2. **Linux GPU server** (calibration + SPARK + steering eval): vLLM / HF + 2-4 RTX 3090.

You can develop and sanity-test on (1), then `rsync` the repo to (2) for the
heavy steps.

---

## 1. Local development (Windows or Linux, CPU)

```bash
conda create -n frontier python=3.11 -y
conda activate frontier

# Just the parts needed for generators / sanity check
pip install numpy scipy pyyaml tqdm networkx

python scripts/sanity_check.py
```

Sanity check should print "All sanity checks passed." in < 1 minute and
write `data/sanity/{symbolic,logical,algorithmic}.jsonl`.

---

## 2. GPU server (Linux + CUDA 12.1+)

### 2.1 OS / driver prerequisites

- Ubuntu 22.04 or similar
- NVIDIA driver ≥ 535 (for CUDA 12.1)
- 2× or 4× RTX 3090 (24 GB each)
- Disk: ≥ **100 GB free** for the 2-card pool, **≥ 250 GB** for the 4-card pool

Verify drivers:
```bash
nvidia-smi                      # should list all GPUs
df -h                           # find a partition with enough free space
```

### 2.2 Python env

```bash
conda create -n frontier python=3.11 -y
conda activate frontier

pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python -c "import vllm, modelscope; print(vllm.__version__, modelscope.__version__)"
```

### 2.3 Pick a model storage directory and export `MODELS_DIR`

This is the single decision you need to make. Pick a directory on a disk
with enough free space, then:

```bash
# Examples — pick ONE that matches your server layout:
export MODELS_DIR=/data/models                    # dedicated /data partition
# export MODELS_DIR=$HOME/models                  # plain home dir
# export MODELS_DIR=/mnt/nvme/models              # fast NVMe drive
# export MODELS_DIR=/workspace/models             # cloud / docker layout

mkdir -p "$MODELS_DIR"

# Make it permanent for this server (optional but recommended):
echo "export MODELS_DIR=$MODELS_DIR" >> ~/.bashrc
```

> **No `HF_HOME` needed.** We pull weights from ModelScope (魔塔), not from
> HuggingFace. All weights live under `$MODELS_DIR` and the loader reads
> them from there directly.

### 2.4 Download all models from ModelScope

```bash
bash scripts/download_models.sh                   # 2-card pool (~70 GB)
# or:
POOL=4card bash scripts/download_models.sh        # 4-card pool (~200 GB)
# or, just one model:
ONLY=Qwen3-4B bash scripts/download_models.sh
```

The script is **resumable** — it skips a model whose `$MODELS_DIR/<name>/config.json`
already exists. Safe to re-run.

After it finishes, verify:
```bash
ls -lh "$MODELS_DIR"
# Expected: directories Qwen3-0.6B, Qwen3-1.7B, ..., Qwen3-32B-AWQ
```

### 2.5 Smoke test on GPU

After local sanity passes, `rsync` the project and run a single model on a
small subset to verify vLLM works:

```bash
# 30-problem sanity dataset (CPU only)
python scripts/sanity_check.py

# Smallest model first — Qwen3-0.6B fits easily on one card
python -m src.calibration.run_eval \
    --model Qwen3-0.6B \
    --input data/sanity/symbolic.jsonl \
    --out_dir data/calibrated \
    --limit 10
```

Watch the log: it should print
`[run_eval] loading model from /data/models/Qwen3-0.6B (TP=1)` —
confirming the loader picked up the local path.

If that works, proceed to the full pipeline:

```bash
bash scripts/run_full_pipeline.sh
```

---

## 3. Common issues

### CUDA OOM during model load

vLLM reserves KV cache after loading weights. If you OOM:

- Drop `gpu_memory_utilization` to 0.80 in `configs/models.yaml`.
- Reduce `max_model_len` from 4096 to 2048.
- For Qwen3-8B on a single 3090: keep ≤ 0.88; the tail of the 24 GB card
  is needed for the activation buffer.

### "model not found at /data/models/..."

You forgot to download. Either:
```bash
ONLY=<model-name> bash scripts/download_models.sh
```
or just run `bash scripts/download_models.sh` again to get everything.

### ModelScope download is slow / interrupted

`modelscope download` resumes by default — re-run the same command. If a
specific model keeps failing, log into ModelScope and accept any usage
agreement that appears on the model page (some Qwen variants gate downloads
behind a one-click acceptance).

### "trust_remote_code" warning

We pass `trust_remote_code=True` because Qwen3 ships custom modeling code.
This is required, not optional.

### R1-Distill output truncation

DeepSeek-R1-Distill emits long `<think>...</think>` blocks. We bumped
`max_tokens` to 3072 for those models in `configs/models.yaml`. If you still
see truncation, raise it to 4096 (and reduce `max_model_len` accordingly).

### Resuming a failed calibration

`run_calibration.sh` skips any model whose output JSONL is non-empty.
To force a re-run, delete the file and re-launch:

```bash
rm data/calibrated/Qwen3-8B.jsonl
bash scripts/run_calibration.sh
```
