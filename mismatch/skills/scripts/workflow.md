# Relax 训推不一致检测工作流（Train-Inference Mismatch Debug Workflow）

## 背景

在 Relax 强化学习框架中，GRPO 训练通过 SGLang 推理引擎生成 response 并收集 `rollout_log_probs`，训练时（Megatron-LM）重新计算 `log_probs`。当两端对同一 response 的 logp 预测出现偏差时，PPO/GRPO 的 importance ratio 会偏离 1，导致训练不稳定、reward 下降。

本工作流提供一套**独立的测试脚本**，用于：
1. 从 Rollout 侧（SGLang）对给定 prompt 生成 response 并提取 logp
2. 从训练侧（Megatron-LM）提取**同一组 response token** 的 response-only logp
3. 从 HuggingFace Transformers 提取**同一组 response token** 的 logp 作为**纯净基准**
4. 从 vLLM 推理引擎提取**同一组 response token** 的 logp，隔离 vLLM 推理侧偏差
5. 离线对齐比对，量化逐 token 的训推差异，并隔离偏差来源

> **为什么先 Rollout 后 Train/HF？**  
> Rollout 生成 response token 是"真相来源"，训练侧和 HF 基准必须基于完全相同的 token 序列计算 logp，才能进行 apples-to-apples 比对。若先指定 target tokens，rollout 生成的可能不同，导致 token mismatch。

---

## 脚本分工

| 脚本 | 职责 | 输入 | 输出 |
|------|------|------|------|
| `test_rollout_logp.py` | Rollout 侧 logp 提取 | SGLang `/generate` endpoint + token IDs | `rollout_logp_result.json` |
| `test_train_logp.py` | 训练侧 logp 提取 | HuggingFace checkpoint + token IDs | `response_logp_rank_{rank}.pt` |
| `test_hf_logp.py` | HF 基准 logp 提取 | HuggingFace checkpoint + token IDs | `hf_logp_result.pt` |
| `test_vllm_logp.py` | vLLM 推理侧 logp 提取 | vLLM 模型路径 + token IDs | `vllm_logp_result.pt` |
| `compare_train_rollout_logp.py` | 离线比对分析 | `.pt` + `.json` | `compare_result.json` + 可选 CSV/ASCII 图 |

**设计原则**：采集与比对解耦，各端脚本只负责数据生产，`compare_train_rollout_logp.py` 负责所有分析逻辑，便于在不同环境/时间点复现比对。

**四端比对矩阵**：

| 比对组合 | 检测目标 |
|----------|---------|
| `test_train_logp.py` vs `test_hf_logp.py` | Megatron-LM / Bridge 实现是否有偏差 |
| `test_rollout_logp.py` vs `test_hf_logp.py` | SGLang 推理引擎是否有偏差 |
| `test_vllm_logp.py` vs `test_hf_logp.py` | vLLM 推理引擎是否有偏差 |
| `test_train_logp.py` vs `test_rollout_logp.py` | 端到端训推不一致（SGLang rollout） |
| `test_train_logp.py` vs `test_vllm_logp.py` | 端到端训推不一致（vLLM rollout） |
| `test_rollout_logp.py` vs `test_vllm_logp.py` | SGLang 与 vLLM 推理引擎差异 |

---

## 前置条件

### 1. 环境准备

```bash
# Python 依赖
pip install torch numpy httpx

# MindSpeed / Megatron-LM 环境（训练侧）
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export CUDA_DEVICE_MAX_CONNECTIONS=1

# SGLang 服务（Rollout 侧）
python3 -m sglang.launch_server \
    --model-path /mnt/sfs_turbo/models/Qwen3.5-9B/ \
    --page-size 1 \
    --attention-backend ascend \
    --disable-cuda-graph \
    --port 30000
```

### 2. 确认模型配置一致

训练侧（`test_train_logp.py`）和 SGLang 侧必须使用**同一套权重**，且推理参数（temperature、top-p 等）需与实际 rollout 配置对齐。

---

## 完整工作流

### Step 1: Rollout 侧采集 logp

```bash
python test_rollout_logp.py \
    --sglang-url http://localhost:30000/generate \
    --test-tokens "3710, 369, 279, 6511, 314, 9338, 30" \
    --max-new-tokens 50 \
    --temperature 0.0 \
    --output rollout_logp_result.json
```

**关键参数说明**：

| 参数 | 说明 |
|------|------|
| `--sglang-url` | SGLang `/generate` 端点地址 |
| `--test-tokens` | 作为 `input_ids` 传入 SGLang 的 token IDs（即 prompt），逗号分隔或 token 文件路径（`.txt`/`.json`/`.pt`） |
| `--max-new-tokens` | SGLang 生成的最大新 token 数。默认等于 `len(test-tokens)` |
| `--temperature` | 采样温度。建议设为 `0.0`（greedy），确保生成确定性结果便于比对 |
| `--return-prompt-logprob` | 可选，请求 SGLang 返回 prompt token 的 logp（需 SGLang 版本支持） |

**输出**：
- `rollout_logp_result.json`
- 关键字段：
  ```json
  {
      "generated_token_ids": [tok1, tok2, ...],
      "generated_logprobs": [logp1, logp2, ...],
      "prompt_logprobs": [logp1, ...],
      "text": "generated text",
      "meta_info": {...}
  }
  ```

**注意事项**：
- `test-tokens` 在这里只作为 **prompt**（input_ids），SGLang 会生成新的 response token
- 为便于比对，建议 `--temperature 0.0`（greedy），使生成结果确定且可复现
- 生成的 `generated_token_ids` 将在 Step 2 中用于构造训练侧的完整序列

---

### Step 2: 训练侧采集 logp

先读取 Rollout 生成的 response token，拼接成完整序列：

```bash
# 从 rollout 结果中提取 generated_token_ids 并拼接
# 例如：prompt = [12, 134, 45, 10, 89]，response = [100, 200, 300, 400, 500]
# 则完整 test-tokens = "12, 134, 45, 10, 89, 100, 200, 300, 400, 500"

torchrun --nproc_per_node=4 test_train_logp.py \
    --hf-checkpoint /mnt/sfs_turbo/models/Qwen3.5-9B/ \
    --tensor-model-parallel-size 4 \
    --micro-batch-size 1 \
    --seq-length 2048 \
    --test-tokens "3710, 369, 279, 6511, 314, 9338, 30, 271, 760, 6511, 314, 9338, 369, 11751, 13, 271, 3710, 369, 279, 6511, 314, 9564, 30, 271, 760, 6511, 314, 9564, 369, 19241, 13, 271, 3710, 369, 279, 6511, 314, 14898, 30, 271, 760, 6511, 314, 14898, 369, 21047, 13, 271, 3710, 369, 279, 6511, 314, 17163, 30, 271, 760" \
    --response-length 50 \
    --qkv-format bshd
```

**关键参数说明**：

| 参数 | 说明 |
|------|------|
| `--hf-checkpoint` | HuggingFace 模型路径（与 Relax 训练使用的一致） |
| `--tensor-model-parallel-size` | TP 大小，需与 SGLang 侧模型并行配置一致 |
| `--test-tokens` | 逗号分隔的 token IDs，或 token 文件路径（`.txt`/`.json`/`.pt`）；**完整序列**（prompt + rollout 生成的 response） |
| `--response-length` | response token 数量（最后 N 个 token）。不指定时默认除第一个 token 外全部作为 response |
| `--qkv-format` | `bshd` 或 `thd`，需与 Relax 训练配置一致 |

**输出**：
- `response_logp_rank_0.pt` / `response_logp_rank_1.pt` ...（每个 rank 一个）
- 文件内容为 dict：
  ```python
  {
      "response_tokens": Tensor([response_length]),   # response token ids（来自 rollout）
      "response_logp":  Tensor([response_length]),    # 逐 token logp
      "prompt_length": int,
      "response_length": int,
      "mean_logp": float,
      "input_token_ids": Tensor([total_length]),
  }
  ```

**注意事项**：
- `test_train_logp.py` 只提取 **response token** 的 logp（对齐 Relax 的 `get_responses()` 逻辑），prompt 部分不参与计算
- `--test-tokens` 的 response 部分必须与 Rollout 生成的 `generated_token_ids` **完全一致**，否则比对时会出现 token mismatch
- 若使用了 CP（Context Parallelism），需确保 CP size 与训练时一致

---

### Step 3: HF 基准采集 logp

使用 HuggingFace `transformers` 原生推理计算同一组 token 的 logp，作为**纯净基准**。

```bash
python test_hf_logp.py \
    --hf-checkpoint /mnt/sfs_turbo/models/Qwen3.5-9B/ \
    --test-tokens "3710, 369, 279, 6511, 314, 9338, 30, 271, 760, 6511, 314, 9338" \
    --response-length 5 \
    --bf16 \
    --output hf_logp_result.pt
```

**关键参数说明**：

| 参数 | 说明 |
|------|------|
| `--hf-checkpoint` | HuggingFace 模型路径（与训练侧一致） |
| `--test-tokens` | 逗号分隔的 token IDs，或 token 文件路径（`.txt`/`.json`/`.pt`）；**完整序列**（prompt + rollout 生成的 response） |
| `--response-length` | response token 数量（最后 N 个 token） |
| `--bf16` / `--fp16` | 使用混合精度推理（推荐与训练侧一致） |
| `--device` | 手动指定设备（`cuda:0`、`npu:0`、`cpu`） |

**输出**：
- `hf_logp_result.pt`
- 文件内容为 dict，格式与 `test_train_logp.py` 完全一致：
  ```python
  {
      "response_tokens": Tensor([response_length]),   # response token ids
      "response_logp":  Tensor([response_length]),    # 逐 token logp
      "prompt_length": int,
      "response_length": int,
      "mean_logp": float,
      "input_token_ids": Tensor([total_length]),
  }
  ```

**注意事项**：
- `test_hf_logp.py` 是**单卡脚本**，不需要 `torchrun`，不依赖 Megatron/MindSpeed
- 若 HF 结果与 Megatron 结果差异很大（`> 1e-4`），说明 Megatron-Bridge 或并行策略引入了偏差
- 若 HF 结果与 SGLang 结果差异很大，说明 SGLang 推理引擎引入了偏差

---

### Step 3b: vLLM 推理侧采集 logp

使用 vLLM 离线推理对同一组 token 计算 logp，隔离 **vLLM 推理引擎**的偏差。脚本将完整序列作为 `prompt_token_ids` 传入，通过 `prompt_logprobs` 读取每个 token 的 logp，切片逻辑与 `test_hf_logp.py` 完全一致。

```bash
# 通用模型（单卡 / vLLM 内部管理 TP，无需 torchrun）
python test_vllm_logp.py \
    --hf-checkpoint /mnt/sfs_turbo/models/Qwen3.5-9B/ \
    --test-tokens "3710, 369, 279, 6511, 314, 9338, 30, 271, 760" \
    --response-length 5 \
    --bf16 \
    --tensor-parallel-size 1 \
    --output vllm_logp_result.pt

# 对齐 offline.py 的 DeepSeek-V4 部署配置
python test_vllm_logp.py \
    --model /storage/models/DeepSeek-V4-Flash-BF16_hym_cut_layer \
    --test-tokens "3710, 369, 279, 6511, 314, 9338, 30, 271, 760" \
    --response-length 5 \
    --tensor-parallel-size 4 \
    --enable-return-routed-experts \
    --mamba-cache-dtype float32 \
    --enforce-eager \
    --gpu-memory-utilization 0.65 \
    --output vllm_logp_result.pt
```

**关键参数说明**：

| 参数 | 说明 |
|------|------|
| `--model` / `--hf-checkpoint` | 模型路径（vLLM `LLM(model=...)`）。`--hf-checkpoint` 为别名，便于与其他脚本互换 |
| `--test-tokens` | 逗号分隔的 token IDs，或 token 文件路径（`.txt`/`.json`/`.pt`）；**完整序列**（prompt + rollout 生成的 response） |
| `--response-length` | response token 数量（最后 N 个 token）。不指定时默认除第一个 token 外全部作为 response |
| `--bf16` / `--fp16` | 混合精度推理（推荐与训练侧一致） |
| `--tensor-parallel-size` | TP 大小。vLLM 内部管理，**无需 torchrun** |
| `--enforce-eager` | 关闭 CUDA graph，便于精确比对 / 调试 |
| `--enable-prefix-caching` | 默认关闭（保证干净比对路径） |
| `--mamba-cache-dtype` / `--enable-return-routed-experts` | DeepSeek-V4 等混合/MoE 模型专用（可选） |

**输出**：
- `vllm_logp_result.pt`
- 文件内容为 dict，格式与 `test_hf_logp.py` / `test_train_logp.py` 完全一致：
  ```python
  {
      "response_tokens": Tensor([response_length]),   # response token ids
      "response_logp":  Tensor([response_length]),    # 逐 token logp（ln）
      "prompt_length": int,
      "response_length": int,
      "mean_logp": float,
      "input_token_ids": Tensor([total_length]),
  }
  ```

**注意事项**：
- `test_vllm_logp.py` 通过 `prompt_logprobs` 读取**给定 token** 的 logp（不生成新 token），与 `test_hf_logp.py` 的 `model(input_ids)` 路径一一对应：`prompt_logprobs[i]` 即 `logits[i-1]` 对 `token[i]` 的预测
- vLLM 的 `prompt_logprobs` 为自然对数（ln），与 `F.log_softmax` 一致，无需转换
- 传 `prompt_token_ids` 时 vLLM **不添加 BOS**，与 HF `model(input_ids)` 行为一致
- vLLM 始终在返回的 logprob 字典中包含**实际 token**（即便不在 top-N 内），gather 不会漏取
- 若 vLLM 结果与 HF 差异很大（`> 1e-4`），说明 vLLM 推理引擎（attention 后端、KV cache、精度策略等）引入了偏差

---

### Step 4: 离线比对分析

```bash
# Train vs Rollout（端到端训推不一致）
python compare_train_rollout_logp.py \
    --train-logp response_logp_rank_0.pt \
    --rollout-logp rollout_logp_result.json \
    --output compare_train_vs_rollout.json

# Train vs HF（隔离 Megatron 偏差）
python compare_train_rollout_logp.py \
    --train-logp response_logp_rank_0.pt \
    --rollout-logp hf_logp_result.pt \
    --output compare_train_vs_hf.json

# Rollout vs HF（隔离 SGLang 偏差）
python compare_train_rollout_logp.py \
    --train-logp hf_logp_result.pt \
    --rollout-logp rollout_logp_result.json \
    --output compare_rollout_vs_hf.json

# vLLM vs HF（隔离 vLLM 偏差）
python compare_train_rollout_logp.py \
    --train-logp hf_logp_result.pt \
    --rollout-logp vllm_logp_result.pt \
    --output compare_vllm_vs_hf.json

# Train vs vLLM（端到端训推不一致，vLLM rollout）
python compare_train_rollout_logp.py \
    --train-logp response_logp_rank_0.pt \
    --rollout-logp vllm_logp_result.pt \
    --output compare_train_vs_vllm.json

# 多 rank 批量比对 + CSV 明细 + ASCII 差异图
python compare_train_rollout_logp.py \
    --train-logp "response_logp_rank_*.pt" \
    --rollout-logp rollout_logp_result.json \
    --output compare_result.json \
    --csv per_token_diff.csv \
    --plot
```

**关键参数说明**：

| 参数 | 说明 |
|------|------|
| `--train-logp` | 训练侧 `.pt` 文件路径，支持 glob 模式（如 `response_logp_rank_*.pt`） |
| `--rollout-logp` | Rollout 侧 JSON 文件路径 |
| `--output` | 比对结果 JSON 输出路径 |
| `--csv` | 可选，逐 token 差异 CSV 输出路径 |
| `--plot` | 可选，在终端打印 ASCII 差异柱状图 |
| `--token-offset` | 可选，当两端序列长度不一致时，手动指定 rollout 相对于 train 的偏移量 |

**输出**：
- `compare_result.json`：汇总指标
  ```json
  {
      "comparisons": [{
          "train_file": "response_logp_rank_0.pt",
          "rollout_file": "rollout_logp_result.json",
          "metrics": {
              "token_count": 5,
              "train_mean_logp": -0.823456,
              "rollout_mean_logp": -0.812345,
              "logp_mean_abs_diff": 0.011111,
              "logp_max_abs_diff": 0.045678,
              "logp_std_diff": 0.018234,
              "prob_mean_abs_diff": 0.004567,
              "logp_abs_diff_p50": 0.008000,
              "logp_abs_diff_p90": 0.025000,
              "logp_abs_diff_p95": 0.035000,
              "logp_abs_diff_p99": 0.045000
          },
          "per_token": [...]
      }],
      "summary": {
          "num_comparisons": 1,
          "mean_abs_diff": 0.011111
      }
  }
  ```
- `per_token_diff.csv`（若指定 `--csv`）：
  ```csv
  position,train_logp,rollout_logp,abs_diff,rel_diff,train_prob,rollout_prob,prob_abs_diff
  0,-0.800000,-0.790000,0.010000,0.012500,0.449329,0.453845,0.004516
  1,-0.850000,-0.860000,0.010000,0.011765,0.427415,0.423162,0.004253
  ...
  ```

---

## 结果解读

### 健康指标（参考阈值）

| 指标 | 说明 | 健康范围 | 异常信号 |
|------|------|---------|---------|
| `logp_mean_abs_diff` | 平均 logp 绝对差异 | `< 0.01` | `> 0.05` 需警惕，`|> 0.1` 严重 |
| `logp_max_abs_diff` | 最大单 token 差异 | `< 0.05` | `> 0.2` 可能存在特定 token 的系统性偏差 |
| `logp_std_diff` | 差异的标准差 | `< 0.02` | 过大说明差异分布不均匀 |
| `prob_mean_abs_diff` | 概率空间平均差异 | `< 0.005` | `> 0.02` 概率层面已出现显著偏离 |
| `logp_abs_diff_p99` | 99 分位数差异 | `< 0.05` | 尾部极端差异指示稀有 token 问题 |

### 常见差异来源

1. **温度缩放不一致**
   - Relax 训练侧在 `get_responses()` 中应用了 `rollout_temperature` 缩放：
     ```python
     if args.rollout_temperature != 1.0:
         logits = logits.div(args.rollout_temperature)
     ```
   - 若 SGLang 侧未使用相同温度，logp 会系统性偏移

2. **精度差异**
   - 训练侧使用 bf16/fp16 前向，SGLang 可能使用 fp16/bf16 或不同量化策略
   - 大规模模型（如 Qwen3.5-9B）中，精度差异在深层网络会累积

3. **并行策略差异**
   - TP/CP/PP 的切分和 all-gather 顺序不同可能导致浮点舍入差异
   - 需确保两端 TP size 一致，且使用相同的 attention 后端

4. **Router / MoE 行为差异**
   - MoE 模型的 top-k routing 在训练和推理时若使用不同算法（如 greedy vs sample），会导致不同 expert 选择，进而影响 logp

5. **KV Cache / Prefix Cache**
   - SGLang 的 prefix cache 可能导致 prompt 部分计算路径与训练侧不同
   - 这通常不影响 response token 的 logp，但需注意验证

---

## 典型排查流程

### Case 1: `logp_mean_abs_diff` 从 0.00x 逐渐增长到 0.x

**现象**：训练初期差异很小，随着训练步数增加差异逐渐放大。

**排查步骤**：

```bash
# 1. 固定一组测试 token，在训练不同阶段重复执行本工作流
# 2. 比较不同阶段的增长趋势
python compare_train_rollout_logp.py ...

# 3. 查看逐 token 差异分布（--plot）
# 若差异集中在尾部 token，可能是 KV cache 或长序列精度问题
# 若差异均匀分布，可能是权重同步或温度缩放问题

# 4. 检查温度配置
# 训练侧：查看 Relax 日志中的 rollout_temperature
# SGLang 侧：确认 sampling_params["temperature"] 与训练侧一致

# 5. 检查权重同步
# 确认 SGLang 已加载最新权重，且 weight_version 与训练步数匹配
```

### Case 2: 特定 token 出现极端差异

**现象**：`logp_max_abs_diff` 很大，但 `logp_mean_abs_diff` 正常。

**排查步骤**：

```bash
# 1. 查看 CSV 输出，定位差异最大的 position
# 2. 检查该 token 是否为：
#    - 特殊 token（BOS/EOS/PAD）
#    - 低频词（out-of-distribution）
#    - 多语言混合字符

# 3. 确认 tokenizer 在两端一致
#    训练侧使用 HuggingFace tokenizer
#    SGLang 使用内置 tokenizer，需确认 vocab 一致

# 4. 检查是否为 MoE routing 差异导致
#    查看 rollout_logp_result.json 中是否有 routed_experts 信息
```

### Case 3: 多 rank 间训练侧结果不一致

**现象**：`response_logp_rank_0.pt` 与 `response_logp_rank_1.pt` 的 logp 存在差异。

**排查步骤**：

```bash
# 1. 这是 TP/CP 并行下的正常现象，差异应非常小（< 1e-5）
# 2. 若差异较大，检查：
#    - 各 rank 是否加载了相同的权重
#    - TP group 的 all-gather 是否正确执行
#    - 是否存在 TP rank 间的随机性（如 dropout）
```

---

### Case 4: 需要定位到具体层/算子的训推不一致

**现象**：已知 Train 和 Rollout 的 logp 存在差异，但需要定位到**具体哪一层**开始出现差异。

**排查步骤**（分层注入比对）：

```bash
# Step A: 检查权重一致性（排除权重转换问题）
torchrun --nproc_per_node=1 check_weights.py \
    --hf-checkpoint /mnt/sfs_turbo/models/Qwen3.5-9B_clip/ \
    --tensor-model-parallel-size 1 \
    --bf16 \
    --skip-vision \
    --no-skip-fused
# 若 overall_max_diff > 1e-4，优先修复 Megatron-Bridge 权重转换问题

# Step B: 采集 HF 基准的逐层激活
torchrun --nproc_per_node=4 test_hf_logp.py \
    --hf-checkpoint /mnt/sfs_turbo/models/Qwen3.5-9B/ \
    --test-tokens "3710, 369, 279, 6511, 314, 9338, 30, 271, 760, 6511, 314, 9338" \
    --response-length 5 \
    --bf16 \
    --save-activations \
    --output hf_logp_result.pt
# 输出：hf_logp_result.pt + hf_logp_result.activations.pt

# Step C: 采集 Megatron 的逐层激活（建议单卡 TP=1 PP=1 CP=1）
torchrun --nproc_per_node=4 test_train_logp.py \
    --hf-checkpoint /mnt/sfs_turbo/models/Qwen3.5-9B/ \
    --tensor-model-parallel-size 4 \
    --micro-batch-size 1 \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --seq-length 2048 \
    --test-tokens "3710, 369, 279, 6511, 314, 9338, 30, 271, 760, 6511, 314, 9338" \
    --response-length 5 \
    --save-activations \
    --qkv-format bshd

# 输出：response_logp_rank_0.pt + megatron_activations_rank_0.pt

# Step D: 逐层激活比对
python compare_activations.py \
    --baseline hf_logp_result.activations.pt \
    --target megatron_activations_rank_0.pt \
    --plot \
    --output activation_compare.json

# 分析输出：
# - 若 embed_tokens 层差异很大 → Embedding/Position Embedding 问题
# - 若 layer_0 开始差异大 → Attention/MLP/Norm 实现差异
# - 若前几层一致、layer_N 开始发散 → 精度累积问题
# - 若 norm/lm_head 层差异大 → Output projection 或 vocab parallel 问题
```

**逐层比对输出解读**：

```
============================================================
  Layer-wise Activation Comparison
============================================================
  Baseline : hf_logp_result.activations.pt
  Target   : megatron_activations_rank_0.pt
  Matched  : 45 layers
  Threshold: 0.0001
------------------------------------------------------------
  ✓ embed_tokens                            max_diff=1.2e-07 mean_diff=3.4e-08
  ✓ layer_0                                max_diff=2.1e-07 mean_diff=5.6e-08
  ✓ layer_1                                max_diff=1.8e-07 mean_diff=4.2e-08
  ⚠️ layer_2                                max_diff=3.5e-04 mean_diff=8.2e-05   ← 首次发散
  ⚠️ layer_3                                max_diff=1.2e-03 mean_diff=2.1e-04
  ...
------------------------------------------------------------
  ⚠️ First layer exceeding threshold: layer_2
     → Inspect layers BEFORE this point for weight/embedding issues
     → Inspect THIS layer and AFTER for compute/activation issues
============================================================
```

**定位策略**：

| 首次发散位置 | 可能根因 |
|-------------|---------|
| `embed_tokens` | Token Embedding、Position Embedding、RoPE 初始化不一致 |
| `layer_0` | Attention 实现（QKV 投影、Softmax、Scale）、MLP 实现、RMSNorm/LayerNorm |
| `layer_N`（N>0） | 精度累积（bf16/fp16 舍入误差）、特定层的算子实现差异 |
| `norm` / `lm_head` | Output projection、Vocab Parallel、最终 Norm 实现 |

> **更深的定位（注入式二分）**：当上面逐层捕获找到首发散层后，要进一步定位到**层内具体算子**（尤其是 fused kernel 内部，如 NPU fused attn / fused RMSNorm / grouped MoE），用**注入式二分**方法论——见 skill `train-infer-mismatch-bisection`（`.claude/skills/train-infer-mismatch-bisection/SKILL.md`）。核心：`--dump-ref`/`--inject-ref`/`--force-ref`（任意子模块输入/输出注入）+ `--dump-fused-attn`/`--dump-attn-internals`（fused 内部 rotary/sparse/core dump）+ `compare-fused-attn`。GLM5 torchturbo vs HF 的完整定位链路在该 skill 里有工作样例。

---

## 快速开始（最小示例）

```bash
# 0. 启动 SGLang（单卡示例）
python3 -m sglang.launch_server \
    --model-path /mnt/sfs_turbo/models/Qwen3.5-9B/ \
    --port 30000

# 1. Rollout 侧采集（先生成 response token）
python test_rollout_logp.py \
    --sglang-url http://localhost:30000/generate \
    --test-tokens "12, 134, 45, 10, 89" \
    --max-new-tokens 3 \
    --temperature 0.0

# 2. 从 rollout 结果中获取 generated_token_ids，拼接成完整序列
# 假设 rollout 生成 [100, 200, 300]
# 则完整 test-tokens = "12, 134, 45, 10, 89, 100, 200, 300"

# 3. 训练侧采集（与 HF / vLLM 基准可并行执行）
torchrun --nproc_per_node=1 test_train_logp.py \
    --hf-checkpoint /mnt/sfs_turbo/models/Qwen3.5-9B/ \
    --test-tokens "12, 134, 45, 10, 89, 100, 200, 300" \
    --response-length 3

python test_hf_logp.py \
    --hf-checkpoint /mnt/sfs_turbo/models/Qwen3.5-9B/ \
    --test-tokens "12, 134, 45, 10, 89, 100, 200, 300" \
    --response-length 3 \
    --bf16

python test_vllm_logp.py \
    --hf-checkpoint /mnt/sfs_turbo/models/Qwen3.5-9B/ \
    --test-tokens "12, 134, 45, 10, 89, 100, 200, 300" \
    --response-length 3 \
    --bf16

# 4. 四端比对
python compare_train_rollout_logp.py \
    --train-logp response_logp_rank_0.pt \
    --rollout-logp rollout_logp_result.json \
    --output compare_train_vs_rollout.json

python compare_train_rollout_logp.py \
    --train-logp response_logp_rank_0.pt \
    --rollout-logp hf_logp_result.pt \
    --output compare_train_vs_hf.json

python compare_train_rollout_logp.py \
    --train-logp hf_logp_result.pt \
    --rollout-logp vllm_logp_result.pt \
    --output compare_vllm_vs_hf.json
```

---

## 文件清单

```
rltest/mismatch/skills/scripts/
├── test_train_logp.py              # 训练侧 logp 提取（Megatron-LM）
├── test_rollout_logp.py            # Rollout 侧 logp 提取（SGLang）
├── test_hf_logp.py                 # HF 基准 logp 提取（transformers 原生）
├── test_vllm_logp.py               # vLLM 推理侧 logp 提取（vLLM prompt_logprobs）
├── compare_train_rollout_logp.py   # 离线 logp 比对分析
├── compare_activations.py          # 逐层激活比对分析
├── check_weights.py                # 权重一致性检查
├── encode_prompt.py                # Prompt 文本 → Token IDs
└── workflow.md                     # 本文件
```
