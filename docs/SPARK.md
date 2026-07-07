# SPARK — Latent Susceptibility for Emergence Detection

`src/spark/` implements three things, all training-free and white-box:

1. **χ (Latent Susceptibility)** — input-perturbation response of hidden states
2. **Φ (Hierarchical Coherence)** — cross-layer cosine alignment
3. **SPK (composite)** — `log χ + λ · Φ`

For a single problem, you get back a per-layer χ spectrum and a Φ value
in O(K+1) forward passes (K=8 by default).

---

## API at a glance

```python
from src.spark import compute_chi, compute_phi, compute_spark_index
from src.spark import ChiConfig, PhiConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("/path/to/Qwen3-4B",
                                             torch_dtype="bfloat16",
                                             device_map="auto").eval()
tok   = AutoTokenizer.from_pretrained("/path/to/Qwen3-4B",
                                       trust_remote_code=True)

prompt = "Compute the sum of even integers from 1 to 100."

chi = compute_chi(model, tok, prompt, ChiConfig(K=8, epsilon=0.01))
print("chi_max:", chi.chi_max, "at layer", chi.chi_argmax_layer)

phi = compute_phi(model, tok, prompt, PhiConfig())
print("phi:", phi.phi)

spk = compute_spark_index(chi.chi_max, phi.phi, lambda_=1.0)
print("SPK:", spk)
```

---

## CLI usage (server)

```bash
cd ~/AAAI2027

python scripts/compute_spark.py \
    --model /home/zhangdongxu/AAAI2027/models/Qwen3-4B \
    --input data/sanity/symbolic.jsonl \
    --out   data/spark/Qwen3-4B__symbolic.jsonl \
    --config configs/spark.yaml \
    --limit 5
```

Output JSONL has the original problem fields **plus**:

| field | meaning |
|---|---|
| `chi_max` | max χ across probed layers |
| `chi_mean` | mean χ across probed layers |
| `chi_argmax_layer` | which layer hit the χ peak |
| `chi_per_layer` | full per-layer spectrum (dict, keys are stringified layer indices) |
| `phi` | mean cross-layer cosine |
| `spk` | `log(chi_max) + λ * phi` |
| `n_tokens` | non-pad token count |
| `sigma` | actual noise scale used (`ε * embedding_norm`) |

---

## Memory & speed

For Qwen3-4B at bf16 on a 24 GB RTX 3090:

| K | T (tokens) | hidden states size | ok? |
|---|---|---|---|
| 8 | 2000 | ~3.3 GB | ✅ |
| 8 | 4000 | ~6.6 GB | ✅ |
| 16 | 2000 | ~6.6 GB | ⚠️ tight |
| 8 | 8192 | ~13 GB | ⚠️ tight |

If OOM:
1. Drop `K` to 4 in `configs/spark.yaml` (still informative).
2. Lower the input length: truncate prompt during tokenisation (already
   `truncation=True` by default — set tokenizer `model_max_length` if needed).
3. Use `dtype=float16` on Ampere+ (slightly less stable than bf16 but smaller).

Per-problem time on Qwen3-4B (K=8, T~1500): ~1-3 seconds.
Full FRONTIER-1.5K with one model: ~30-60 minutes.

---

## What gets uploaded to the server

| Path | Why |
|---|---|
| `src/spark/` | core module |
| `scripts/compute_spark.py` | CLI entry |
| `configs/spark.yaml` | hyperparameters |
| `docs/SPARK.md` | this file |

`tests/test_spark_mock.py` is **CPU-only and does not need to be on the server** —
it's a Windows-side sanity check that the math is right.

---

## Verified locally (Windows, no GPU)

```bash
cd D:\zdx\thu\AAAI2027

# 1. import check
py -c "from src.spark import compute_chi, compute_phi, compute_spark_index; print('ok')"

# 2. mock unit tests (11 tests, ~30 ms)
py -m unittest tests.test_spark_mock -v

# 3. CLI argparse + yaml load
py scripts/compute_spark.py --help
py -c "from src.spark.runner import load_spark_config; print(load_spark_config('configs/spark.yaml'))"
```

---

## What runs on the server (after upload)

```bash
cd ~/AAAI2027
export MODELS_DIR=/home/zhangdongxu/AAAI2027/models

# Smoke test on 5 problems (about 15 seconds for Qwen3-4B)
mkdir -p data/spark
python scripts/compute_spark.py \
    --model $MODELS_DIR/Qwen3-4B \
    --input data/sanity/symbolic.jsonl \
    --out   data/spark/Qwen3-4B__symbolic_smoke.jsonl \
    --config configs/spark.yaml \
    --limit 5
```

Inspect the output:

```bash
python -c "
import json
recs = [json.loads(l) for l in open('data/spark/Qwen3-4B__symbolic_smoke.jsonl')]
for r in recs:
    print(f\"{r['id']}: chi_max={r['chi_max']:.3f} phi={r['phi']:.3f} spk={r['spk']:.3f}\")
"
```

If all 5 records have finite, positive `chi_max` (typically 0.1 - 100) and
`phi` in [0.5, 1.0], the pipeline is healthy.

---

## Theoretical references

- **Critical brain** : Beggs & Plenz 2003, "Neuronal avalanches in neocortical circuits"; Mora & Bialek 2011, "Are biological systems poised at criticality?"
- **Workspace ignition** : Dehaene 2014, *Consciousness and the Brain*
- **Edge of chaos in RNNs** : Bertschinger & Natschläger 2004
- **Susceptibility = Fisher information**: Amari 1998
