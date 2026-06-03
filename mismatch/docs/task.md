# 训推不一致：训练侧 Logp 测试脚本开发任务

- `[x]` 编写 `test_train_logp.py` 脚本，基于现有 `test_megatron_bridge_relax.py`
  - `[x]` 实现确定性输入生成或读取
  - `[x]` 提取前向传播 (Forward Pass) 的输出 Logits
  - `[x]` 基于 Logits 计算 Logp，并根据输入 Token ID 取出目标概率
  - `[x]` 保存或打印 Logp 结果用于比对
- `[x]` 验证脚本运行逻辑并进行调优
- `[x]` 汇总报告，完成第一阶段的脚本交付
