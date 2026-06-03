# 训练侧 Logp 提取脚本交付报告

按照计划，我已经为您开发了用于检测训推不一致的训练侧测试脚本，并存放于您的技能脚本目录中。

## 变更摘要
- [NEW] [test_train_logp.py](file:///d:/rl-test/rltest/mismatch/skills/scripts/test_train_logp.py)
  此脚本基于现有的 `test_megatron_bridge_relax.py` 实现，专门用于在 Megatron 训练侧获取确定性的输出 logits。

## 脚本设计说明

1. **去随机化**：我们将原来用于性能/功能测试的随机生成 prompt 的逻辑去除了，转而通过命令行参数 `--test-tokens "10,11,12..."` 提供确定性的 Token ID 序列。
2. **纯前向评估**：在 GRPO 中的 PPO 阶段计算 logp 实际上是由训练引擎负责的前向传播。脚本移除了 Loss 反向传播及优化器更新流程，仅在 `model.eval()` 和 `torch.no_grad()` 模式下调用 Pipeline 并行的前向传播 (`forward_only=True`)。
3. **Logits 与 Logp 计算**：在 Megatron 并行流程中，脚本通过截获输出的 Tensor（Logits），采用 `F.log_softmax(logits, dim=-1)` 获取针对完整词表的对数值。
4. **日志与持久化保存**：脚本将通过 `torch.save()` 将各个 Rank (尤其涉及 Tensor Parallelism) 截获的 Logits 固化为文件 `logits_rank_{rank}.pt`，方便后续脱机比较 vLLM（推理侧）同样序列输出的 Logits。

> [!TIP]
> **使用方法示例：**
> 针对 Qwen3.5 模型在 GPU 上的执行：
> ```bash
> torchrun --nproc_per_node=2 d:/rl-test/rltest/mismatch/skills/scripts/test_train_logp.py \
>     --hf-checkpoint /path/to/qwen3.5-9b \
>     --tensor-model-parallel-size 2 \
>     --micro-batch-size 1 --seq-length 128 \
>     --test-tokens "12, 134, 45, 10, 89"
> ```

如果您需要我进一步编写**推理引擎测（vLLM）的对齐测试脚本**或者编写自动比较这两种 logits 张量的工具（以便精准定位出现漂移的 Token 位置和 K3 KL 散度的变化），请随时告诉我！
