---
name: train-infer-mismatch-bisection
description: Localize numerical divergence between two implementations of the same model (custom fused-kernel engine vs HF eager, train vs inference, or two frameworks) down to the exact diverging sub-module/op. Uses input/output tensor injection + activation capture to bisect, instead of guessing. Invoke when logits/logp/activations differ across implementations and you need the precise op responsible — not just "they differ".
---

# 训推/双框架数值不一致的注入式二分定位

本 skill 把"两套同模型实现 forward 出来数值不一致"的问题，从"端到端 logp 差异"逐层下沉定位到**具体某个算子/子模块**。核心是**注入式二分**：用一方的中间张量替换另一方的，看输出/下游是否对齐，从而二分出首个发散边界。

工具链脚本位于 `mismatch/skills/scripts/`（`test_hf_logp.py` / `test_torchturbo3_logp.py` / `compare_layer_io.py`）。下文以 GLM5 (`GlmMoeDsaForCausalLM`, torchturbo v3 NPU fused 引擎 vs HF eager) 为工作样例，但方法论对任意"同模型双实现"通用。

## 何时使用

- 两个实现（自研 fused-kernel 引擎 vs HF / 训练侧 vs 推理侧 / 两框架）对**相同输入 token**算出 logp 或中间激活不同。
- 已用 `compare-logp` 量化了端到端差异，现在要找**根因算子**。
- 嫌疑涉及 fused kernel（NPU/CUDA fused attn、fused RMSNorm、grouped MoE），这些没有 `nn.Module` 边界，普通 hook 抓不到内部。

不适用：纯分布式通信精度（FSDP all-reduce、SP/EP 通信）——只能在真实多卡流程里 dump，单卡复现不了。

## 核心方法论：注入式二分

两个互补原语（PyTorch hook）：

| 原语 | 机制 | 回答的问题 |
|------|------|-----------|
| **输入注入** (`--inject-ref`) | `register_forward_pre_hook` 替换子模块输入为对方的 | "给定相同输入，本侧该子模块输出是否与对方一致？" → 隔离子模块实现 |
| **输出注入** (`--force-ref`) | `register_forward_hook` return 替换输出 | "强制本侧输出=对方，下游是否对齐？" → 二分上游/下游 |

配套：**捕获对比** (`--finegrain*`) 只挂 forward hook 抓输出，不做替换 → 找"首个发散的子模块"。流程是：捕获找首发散点 → 输入注入确认"发散在该子模块内" → 输出注入二分上下游 → fused 内部用 monkey-patch dump。

注入粒度阶梯：
- **Level 0**：decoder layer / final norm
- **Level 1**：self_attn / mlp / 每层 RMSNorm（隔离 fused attn / MoE / RMSNorm）
- **Level 2**：attention 内部（q_a_proj / q_a_layernorm / q_b_proj / kv_a_proj_with_mqa / kv_a_layernorm / o_proj）+ fused 核心 dump（rotary / sparse indices / core output）

## 工具链（脚本 + flag）

两端脚本输出格式一致，`compare_layer_io.py` 做离线对比。HF 侧 dump、torchturbo 侧 inject/force，配对使用。

| flag | 侧 | 作用 |
|------|----|------|
| `--save-activations` / `--finegrain` | 两端 | 捕获 embed/每层/norm/lm_head + 层内子模块输出 |
| `--finegrain-attn` `--finegrain-attn-layers` | 两端 | 捕获 self_attn 内部投影/indexer 输出（Level 2 capture） |
| `--dump-ref LAYERS:TARGET:DIR` | HF | dump 任意子模块 {input,output}，供注入 |
| `--inject-ref LAYERS:TARGET:DIR` | turbo | 输入注入（pre-hook 改输入），捕获本侧输出对比 |
| `--force-ref LAYERS:TARGET:DIR` | turbo | 输出注入（forward-hook return），看下游 logp 是否对齐 |
| `--dump-fused-attn DIR` | turbo | monkey-patch fused attn 子步骤+core op，dump rotary/sparse/core 中间量 |
| `--dump-attn-internals DIR` | HF | patch rotary 函数 + indexer，dump post-rotary Q/K + topk_indices |
| `compare-activations` | 离线 | 逐层 output 对比，标首发散层 |
| `compare-fused-attn` | 离线 | fused 内部 rotary/sparse 决定性对比 |

`TARGET ∈ {layer, input_layernorm, self_attn, post_attention_layernorm, mlp, norm, q_a_proj, q_a_layernorm, q_b_proj, kv_a_proj_with_mqa, kv_a_layernorm, o_proj}`（注入用；kv_b_proj/indexer 仅 capture）。

## 定位流程（决策树）

```
1. 端到端确认差异
   compare-logp hf_logp.pt v3_logp.pt          # 量化 max/mean abs diff

2. 捕获找首发散层
   两端 --finegrain --finegrain-layers all
   compare-activations hf.act.pt v3.act.pt     # 找首个 max>1e-5 的层

3. 捕获层内首发散子模块
   两端 --finegrain-attn --finegrain-attn-layers <首层>
   compare-activations                         # 投影/norm 哪个先发散

4. 输入注入确认"发散在该子模块内"
   HF:  --dump-ref "<i>:<target>:ref/"
   turbo: --inject-ref "<i>:<target>:ref/" --norm-ref-logp hf_logp.pt
   → 输出对齐 = 该子模块实现 OK，发散在上游
   → 输出不一致 = 坐实该子模块是发散源

5. 输出注入二分上下游
   turbo: --force-ref "<i>:<target>:ref/" --norm-ref-logp hf_logp.pt
   → 下游 logp 对齐 = 发散在该子模块或上游
   → 仍不一致 = 发散在下游

6. fused 核心（无模块边界）→ monkey-patch dump
   turbo: --dump-fused-attn ref/tt --dump-fused-attn-layers <i>
   HF:    --dump-attn-internals ref/hf --dump-attn-internals-layers <i>
   compare-fused-attn ref/tt ref/hf            # rotary / sparse 决定性判定
```

读法（fused 核心 4 嫌疑：rotary / MLA absorb / sparse indexer / core kernel）：
- `rotary q/k` 不一致 → rotary 函数不同（如 `apply_rotary_pos_emb` vs `_interleave`），坐实。
- `sparse indices` bit_id=False → 稀疏选择算法不同。
- 两者都一致 → 发散在 absorb(bf16 数值) 或 core kernel。

## fused kernel 结构陷阱（关键，否则会误判）

这些是调试中踩过的坑，**不是 bug 而是 fused 路径的本性**：

1. **fused 路径绕过 nn.Module 调用**——直接读 weight 做 einsum/算子。典型：MLA-absorb 直读 `kv_b_proj.weight`（不调 `kv_b_proj.forward`）；fused `npu_lightning_indexer` 替代 `self.indexer` 模块。→ 这些子模块的 forward hook **不触发**，capture 时该 key 缺失（`compare-activations` 标 "ONLY in one side"）。**缺失本身就是 fused-bypass 信号**，不是漏抓。

2. **关键字 vs 位置调用**——`self_attn` 常以 `self.self_attn(hidden_states=..., ...)` 关键字调用，pre-hook 的 `args` 为空。注入必须改 `kwargs['hidden_states']`，不能只改 `args[0]`。（脚本里的注入器已兼容两者。）

3. **并行计划会换掉子模块实例**——如 MoE 的 `_replace_moe_only` 把 `GlmMoeDsaMoE` 换成 `MoELayer`。→ hook 必须在 `engine.prepare_model` **之后**注册；换之前的 hook 是死的。

4. **逐专家 hook 不可能**——fused MoE (`GroupedMLP`/`GlmMoeDsaNaiveMoe`) 把专家存成 raw `nn.Parameter`（`weight1`/`weight2` + `gmm`），没有逐专家 `nn.Linear` 模块。只能到 `mlp`/`mlp.gate`/`mlp.shared_experts` 级。

5. **不同实现可能用不同 rotary 函数**——`apply_rotary_pos_emb`（半旋）vs `apply_rotary_pos_emb_interleave`（交错旋）数学不等价。对比 post-rotary Q/K 张量即可判定。

6. **RMSNorm 放大效应**——final norm 把上游微小 diff 放大成巨大输出 diff。看到 `norm` 层 diff 爆炸，根因在**上游**，不是 norm 本身（除非 input 注入后 norm 输出仍发散，才是 norm kernel 问题）。

7. **bf16 噪声判定**——值 ~0.08 时 bf16 ULP ~6e-4。abs diff > ULP×10 且 seq 很短（无累加）= 算法性差异，非精度噪声。看 abs diff，别看 rel（小值附近 rel 失真）。

## 工作样例：GLM5 torchturbo vs HF（已解决）

实际定位链路（供参照）：

1. `compare-logp`：端到端 logp 差异 ~0.06，确认有发散。
2. `--finegrain`：embed_tokens match，layer_0 起发散，逐层放大，`norm` 爆炸 → 根因在 layer_0，上游放大。
3. `--finegrain-attn`（layer_0）：`q_a_proj/q_a_layernorm/q_b_proj/kv_a_proj_with_mqa/kv_a_layernorm` 全 **max=0 一致**；`o_proj` **输出发散**（abs 1.78e-2，值 ±0.08 → ~22%，seq=5 无累加 → 算法性）；`kv_b_proj`/`indexer` 在 turbo 侧缺失（fused 绕过）。→ 发散在 fused 核心。
4. `--inject-ref o_proj`：注入 HF 的 core 输出 → o_proj 输出对齐 → **坐实唯一发散在 fused 核心**，o_proj/权重无误。
5. `--dump-fused-attn` + `--dump-attn-internals` + `compare-fused-attn`：rotary / sparse 决定性对比。关键线索：turbo 用 `apply_rotary_pos_emb`（非 interleave），HF 用 `apply_rotary_pos_emb_interleave`——**不同 rotary 函数**。

修复方向（依 `compare-fused-attn` 结果选）：
- rotary 不一致 → 让 fused 路径用与 HF 一致的 rotary 函数（interleave）。
- sparse 不一致 → 对齐 indexer 选择算法。
- 两者一致 → absorb 数值或 core kernel，再对照 `core_attn_output` / `kv_b_proj` 输出。

## 注意

- 注入/捕获依赖 fused 路径已激活。若 dump 0 层，说明该路径未走（自诊断）。
- GLM5 torchturbo/HF 的源码位置、模块结构、hook 注入陷阱（kv_b_proj fused 绕过、MoE 换实例、不同 rotary 函数等）见归档参考 [`mismatch/skills/GLM5-framework-reference.md`](../../../mismatch/skills/GLM5-framework-reference.md)。查 fused 实现细节时按该文档的源码路径去读。
- 本 skill 的方法论通用；脚本 flag 是 GLM5/torchturbo 工具链的具体形态，换模型/框架时复用"注入二分 + fused dump"的思路即可。
