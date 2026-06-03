# 训推不一致问题检测与修复方案

## 背景与问题描述
在 qwen3.5-9b 模型上进行 GRPO 训练时，模型训练到 70 步左右出现了 `logp_diff_abs` 显著增大（从 0.00x 增到 0.x）以及 Reward 快速下降的现象。这种现象是典型的**强化学习中的训推不一致问题（Train-Inference Mismatch）**。
在 RLHF/GRPO 流程中，生成阶段（Rollout）通常由推理框架（如 vLLM）完成，而训练阶段（Update）由训练框架（如 Megatron-LM）完成。由于底层算子、精度、分布式切分策略等差异，同一个 token 在推理引擎和训练引擎中产生对数值（Log Probabilities, logp）会发生偏移。

## 理论分析：训推不一致的常见根因
训推不一致通常由以下原因引起，特别是对于 MoE 模型（如 Qwen1.5-MoE 或 Qwen2-MoE，假定基于类似架构）：
1. **MoE Router 路由分歧**：由于训练框架与推理框架在计算 Router Logits 时存在微小的浮点数精度差异，可能导致同一个 Token 被分配给不同的专家（Expert）。随着层数加深，这种误差会呈指数级放大，直接导致输出的 Logits 和最终的 `logp` 完全不同。
2. **位置编码（RoPE）差异**：实现上的微小不同（例如是否对序列长度做特殊的 padding 处理、旋转矩阵的精度等）会导致 Attention 的结果产生偏差。
3. **算子与精度对齐（Precision & Kernels）**：推理框架（如 vLLM 中的 FlashInfer/FlashAttention）和训练框架（MindSpeed/Megatron 中的融合算子）在底层累加精度（FP32 vs FP16/BF16）上的差异。
4. **Attention Mask 与 Padding 策略**：训练过程中的 `thd` 格式（Packed Sequences）与推理时 Left-padding/Right-padding 的对齐存在差异，可能导致 Padding Token 影响有效 Token 的计算。

## 测试脚本设计：训练侧 Logp 提取 (`test_train_logp.py`)

为了定位问题，我们需要分别拿到**训练引擎**和**推理引擎**的确定性输出，对比它们在同一输入下的 Logits/Logp。当前的第一步是构建**训练侧的 `logp` 获取脚本**。

我们将参考 `test_megatron_bridge_relax.py` 实现以下流程：
1. **环境初始化**：使用 Megatron-Core 及分布式环境（支持 GPU/NPU）。
2. **模型加载**：通过 `AutoBridge` 构建模型并加载待测试的 HF/Megatron 权重。
3. **确定性输入**：摒弃随机生成的 Token，允许输入确定的 `input_ids` 序列。
4. **前向传播与 Logp 计算**：
   - 获取模型输出的 `logits`。
   - 使用 `log_softmax` 获取 `logp`。
   - 根据输入序列的 token_id 取出对应的 log_prob。
5. **结果落盘**：将输出的 `logp` 和 `logits` 保存为张量文件（或打印），以便后续与推理侧的结果进行逐 Token 比对。

> [!IMPORTANT]
> **User Review Required**
> 此前我们在排查 MoE 问题时提到过 Router 的精度问题，如果当前的 Qwen3.5-9b 是 MoE 版本，我们需要在脚本中特别加入对 MoE Router 选择的打印（或者先保证在单卡、不引入并行时的输出对齐）。
> 请确认本方案是否符合您的期望？如果同意，我将开始实现 `test_train_logp.py`。
