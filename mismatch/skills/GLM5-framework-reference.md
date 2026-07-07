# GLM5 框架源码位置与 hook 注入结构参考

> 归档参考文档。记录 GLM5 (`GlmMoeDsaForCausalLM`, MLA+DSA attention) torchturbo / HF 双实现的源码位置、模块结构与 hook 注入陷阱，供 mismatch 调试时查阅。
>
> 定位**方法论**（注入式二分）见 skill `train-infer-mismatch-bisection`（[`.claude/skills/train-infer-mismatch-bisection/SKILL.md`](../../.claude/skills/train-infer-mismatch-bisection/SKILL.md)）。本文档是该方法论针对 GLM5 的**具体结构事实**参考。

## 源码位置（均在 rltest 仓库外）

训推不一致调试涉及两套 GLM5 实现，源码树位于仓库外：

- **torchturbo v3 engine**：`D:\code\glm52\torchturbo\src\torchturbo\`
  - 模型实现：`models/glm_moe_dsa/modeling_glm_moe_dsa.py`
  - NPU fused patch（替换 RMSNorm / attn / MoE forward）：`models/glm_moe_dsa/model.py`
  - fused 算子实现：`models/glm_moe_dsa/npu.py`（`npu_fused_dsa_attn_forward` / `_npu_prepare_qkv_for_mla_absorb` / `_npu_compute_sparse_indices`）
- **HF baseline**：`D:\code\glm52\transformers\src\transformers\models\glm_moe_dsa\modeling_glm_moe_dsa.py`

测试脚本在仓库内：`mismatch/skills/scripts/`（`test_torchturbo3_logp.py` / `test_hf_logp.py` / `compare_layer_io.py`）。

## 模块结构（两侧属性名一致）

```
GlmMoeDsaForCausalLM
└── .model : GlmMoeDsaModel
    ├── .embed_tokens : nn.Embedding
    ├── .layers : ModuleList[GlmMoeDsaDecoderLayer]
    │   └── [i]
    │       ├── .input_layernorm        : GlmMoeDsaRMSNorm
    │       ├── .self_attn              : GlmMoeDsaAttention   (MLA + DSA indexer)
    │       │   ├── .q_a_proj           : nn.Linear            (默认 q_lora_rank=2048 时存在)
    │       │   ├── .q_a_layernorm      : GlmMoeDsaRMSNorm
    │       │   ├── .q_b_proj           : nn.Linear
    │       │   ├── .kv_a_proj_with_mqa : nn.Linear
    │       │   ├── .kv_a_layernorm     : GlmMoeDsaRMSNorm
    │       │   ├── .kv_b_proj          : nn.Linear            (turbo fused 路径下 forward 不触发)
    │       │   ├── .o_proj             : nn.Linear
    │       │   └── .indexer            : GlmMoeDsaIndexer | None   ("shared" 层为 None)
    │       ├── .post_attention_layernorm : GlmMoeDsaRMSNorm
    │       └── .mlp                    : GlmMoeDsaMLP (dense, 前 3 层) | GlmMoeDsaMoE (sparse)
    ├── .norm : GlmMoeDsaRMSNorm   (final norm)
    └── .rotary_emb
└── .lm_head : nn.Linear
```

要点：
- Attention 是低秩 MLA：query 是 `q_a_proj`+`q_b_proj`（**无 `q_proj`**——默认 `q_lora_rank=2048`），KV 是 `kv_a_proj_with_mqa`+`kv_b_proj`。
- `compare_layer_io.py` 的 `ATTN_SUBMODULES` 把 `q_proj` 列首位但 `hasattr` 会跳过；注入要直接对 `q_a_proj`/`q_b_proj`。

## Hook 注入陷阱（`--inject-ref`/`--force-ref`/`--dump-ref` 必读）

1. **torchturbo 的 `self_attn` 是关键字调用**——`self.self_attn(hidden_states=..., ...)`，pre-hook 的 `args` 为空。注入必须改 `kwargs['hidden_states']`，不能只改 `args[0]`。（脚本里的注入器已兼容两种。）

2. **torchturbo NPU fused attn 直读 `kv_b_proj.weight`**（MLA absorb：W_UK absorb 进 query），不调 `kv_b_proj.forward`。→ `kv_b_proj` 的 hook 不触发，capture 时该 key 在 turbo 侧缺失（`compare-activations` 标 "ONLY in HF"，是 fused-bypass 信号，非 bug）。

3. **torchturbo 并行计划换掉 `mlp` 实例**——`_replace_moe_only` 把 `GlmMoeDsaMoE` 换成 `MoELayer`；`GroupedMLP`/`GlmMoeDsaNaiveMoe` 把专家存成 raw `nn.Parameter`（`weight1`/`weight2` + `gmm`），**无逐专家 `nn.Linear` 模块**，逐专家 hook 不可能。→ hook 必须在 `engine.prepare_model` **之后**注册（换之前的 hook 是死的）。

4. **NPU fused `npu_rms_norm` 替换 `GlmMoeDsaRMSNorm.forward`**（class-level，`model.py:_pre_init_patches`）。`--no-fused-rmsnorm`（实验E）还原 eager fp32。所有 RMSNorm（`input_layernorm`/`post_attention_layernorm`/`q_a_layernorm`/`kv_a_layernorm`/`norm`）都受影响。

## 调查记录（2026-07-07，已解决）

定位链路（GLM5 torchturbo vs HF）：

1. `--inject-ref self_attn`：给定 HF 的 self_attn 输入，torchturbo 的 self_attn 输出仍发散 → 发散**在 self_attn 内**。
2. `--finegrain-attn` capture：所有投影+norm（`q_a_proj`/`q_a_layernorm`/`q_b_proj`/`kv_a_proj_with_mqa`/`kv_a_layernorm`）**max=0 一致**；`o_proj` 输出发散（abs 1.78e-2）；`kv_b_proj`/`indexer` 在 turbo 侧缺失（fused 绕过）→ 发散在 fused 核心。
3. `--inject-ref o_proj`：注入 HF 的 core 输出 → o_proj 输出对齐 → **坐实唯一发散在 fused 核心**，o_proj/权重无误。
4. 实验 F（`--dump-fused-attn` + `--dump-attn-internals` + `compare-fused-attn`）：
   - turbo 侧 patch `_npu_prepare_qkv_for_mla_absorb` / `_npu_compute_sparse_indices` / `torch_npu.npu_sparse_flash_attention`
   - HF 侧 patch `apply_rotary_pos_emb_interleave` + `GlmMoeDsaIndexer.forward`
   - **关键线索**：turbo 用 `apply_rotary_pos_emb`（非 interleave），HF 用 `apply_rotary_pos_emb_interleave`——**不同 rotary 函数**，头号嫌疑。
   - F 两个决定性测试：rotary post-rotary Q/K 对比；sparse `topk_indices` `torch.equal`。
