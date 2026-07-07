# FRONTIER 三域 SPARK-Steering 实验（Qwen3-4B）

在 **symbolic / logical / algorithmic** 上完整跑通：训练侧前置（χ_LC、选题、方向 \(v\)）→ held-out 评测（α 扫描）→ **准确率 + token 长度** 报告。

## 前置条件

以下文件应已存在（你当前仓库里 Qwen3-4B 三域均已具备）：

```text
data/raw/{symbolic,logical,algorithmic}.jsonl
data/calibrated_full/Qwen3-4B__{domain}.jsonl
data/spark_full/Qwen3-4B__{domain}.jsonl
```

若缺失校准或 SPARK：

```bash
cd /home/zhangdongxu/AAAI2027
export MODELS_DIR=/home/zhangdongxu/AAAI2027/models
MODEL=Qwen3-4B

python -m src.calibration.run_eval \
  --model "${MODEL}" --input data/raw/symbolic.jsonl \
  --out_dir data/calibrated_full --tag symbolic

python scripts/compute_spark.py \
  --model "${MODELS_DIR}/${MODEL}" \
  --input data/raw/symbolic.jsonl \
  --out data/spark_full/${MODEL}__symbolic.jsonl --resume
```

（`logical` / `algorithmic` 同理。）

## 快速冒烟（约 20 题 × 2 个 α）

```bash
export CUDA_VISIBLE_DEVICES=0   # 任选一张 GPU

TEST_LIMIT=20 ALPHAS="0.0 0.5" \
  bash scripts/run_frontier_domain_steering_train.sh symbolic
TEST_LIMIT=20 ALPHAS="0.0 0.5" \
  bash scripts/run_frontier_domain_steering_eval.sh symbolic
```

## 单域完整流程

```bash
# 1) 训练侧：选题 + 提取 critical direction（需 GPU，约十几分钟/域）
bash scripts/run_frontier_domain_steering_train.sh symbolic

# 2) 评测：held-out 240 题 × α=0,0.25,0.5,1.0（需 GPU，数小时/域）
bash scripts/run_frontier_domain_steering_eval.sh symbolic
```

产物：

```text
data/steering/Qwen3-4B__symbolic__critical_direction.pt
data/steering/Qwen3-4B__symbolic__heldout_test.jsonl
data/steering/Qwen3-4B__symbolic__steering_eval.jsonl
data/steering/Qwen3-4B__symbolic__steering_eval.summary.md
data/steering/Qwen3-4B__symbolic__steering_eval.frontier_report.md
```

## 三域一键（推荐 nohup）

```bash
export CUDA_VISIBLE_DEVICES=5
export MODELS_DIR=/home/zhangdongxu/AAAI2027/models

bash scripts/run_frontier_three_domains_nohup.sh
tail -f logs/frontier_three_domains__Qwen3-4B.nohup.log
```

分阶段恢复：

```bash
PHASE=prep  bash scripts/run_frontier_three_domains.sh   # 仅 χ_LC
PHASE=train bash scripts/run_frontier_three_domains.sh   # 三域选题 + 方向
PHASE=eval  bash scripts/run_frontier_three_domains.sh   # 三域评测 + 总报告
```

总报告：`data/steering/Qwen3-4B__frontier_three_domains.report.md`

## 评测设置

| 项 | 默认 |
|----|------|
| 模型 | Qwen3-4B（`models/Qwen3-4B`） |
| 评测集 | `heldout_test.jsonl`（240 题，困难/frontier 子集） |
| α | `0.0 0.25 0.5 1.0` |
| 解码 | **Greedy**（`GREEDY=1`，与校准 temperature=0 一致） |
| max_new_tokens | 4096 |
| algorithmic | 额外 `--concise_prompt`（抑制抄边表） |

采样评测（可选）：

```bash
GREEDY=0 TEMPERATURE=0.7 bash scripts/run_frontier_domain_steering_eval.sh logical
```

## 指标说明

`steering_eval.jsonl` 每行含：

| 字段 | 含义 |
|------|------|
| `correct` | 判分是否正确 |
| `n_prompt_tokens` | 输入 prompt token 数 |
| `n_gen_new_tokens` | 生成 token 数 |
| `n_total_tokens` | prompt + 生成 |
| `finish_reason` | `stop` / `length` |

`frontier_report.md` 给出各 α 的 **acc、Δacc vs α=0、平均/中位 gen tokens、总 tokens、length 截断比例**。

## algorithmic 已有实验

若只需复现 algorithmic 的 261 题 under-activated 全集：

```bash
bash scripts/run_steering_eval_261_nohup.sh
```

本管线默认三域统一用 **240 题 held-out**，便于横向对比。
