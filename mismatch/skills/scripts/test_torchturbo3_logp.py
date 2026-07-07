# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""
Extract response-only logp from torchturbo **v3** (src/torchturbo engine).

Sibling of test_hf_logp.py / test_train_logp.py / test_torchturbo2_logp.py. Output .pt
format is identical across all of them so they are cross-comparable:

    test_torchturbo2_logp.py  (torch_turbo v2)          ─┐
    test_torchturbo_logp.py   (src/torchturbo engine v3)─┼─> compare_layer_io.py compare-logp
    test_hf_logp.py           (HF baseline)              ─┘

v3 path — uses ONLY `torchturbo/src/torchturbo/` code (no torch_turbo v2 package,
no asystem_runtime/realhf), mirroring TorchTurboBackend.setup_model:

    import torchturbo  # triggers src __init__ + GLM5 model registration
    engine = TorchTurboEngine(EngineConfig(...))
    engine.prepare_model(path, torch_dtype, trust_remote_code, init_kwargs={config: hf_config})
        internally: TurboGlmMoeDsaForCausalLM._from_config (NPU fused patches)
                  -> FSDP2Backend.prepare_model (_replace_modules + _shard_model + setup_sp)
                  -> FSDP2Backend.load_model (DistParamLoader + WeightConverterMixin)
    model = engine.model
    model(input_ids=...) forward, hook captures lm_head logits, compute next-token logp.

Key v3 vs v2 difference (head suspect of training mismatch) via --mixed-precision:
    v3 src engine (_shard_model 写死): mixed_precision=bf16 -> param=bf16, reduce=fp32, cast_forward_inputs=True
    v2 torch_turbo: 无 mp_policy -> param=bf16, reduce=bf16
    （v3 mixed_precision 是整体枚举，不单独控制 reduce_dtype，故暴露 --mixed-precision 而非 --reduce-dtype）

Usage:
    # SP=1 EP=1
    python test_torchturbo_logp.py \\
        --path /path/to/glm5 \\
        --test-tokens "12,134,45,10,89,100,200" \\
        --response-length 3 --mixed-precision bf16 \\
        --save-activations --output v3_logp.pt

    # With SP (ulysses): seq_len auto-padded to sp_size multiple
    torchrun --nproc_per_node=16 test_torchturbo_logp.py \
        --path /storage/yzr02346555/gyy_Asystem/glm5_mini_bf16 --sp-size 2 --ep-size 16 \
        --mixed-precision bf16 \
        --test-tokens "12,134,45,10,89"  --output v3_logp.pt

    # Add backward + grad dump (to catch FSDP reduce_dtype differences)
    python test_torchturbo_logp.py ... --with-backward

Compare:
    python compare_layer_io.py compare-logp v2_logp.pt v3_logp.pt
    python compare_layer_io.py compare-activations v2_logp.activations.pt v3_logp.activations.pt
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# 让 compare_layer_io 可被 import（v3 脚本复用 v2 仓库的工具）
_HERE = Path(__file__).resolve().parent
_V2_DIR = _HERE.parent.parent / "torchturbo2" / "example" / "glm_moe_dsa"
if _V2_DIR.exists() and str(_V2_DIR) not in sys.path:
    sys.path.insert(0, str(_V2_DIR))
# 若两仓库未并列，可将 compare_layer_io.py 复制到 _HERE 并去掉上面三行

# 确保 src/torchturbo 可被 import（torchturbo 仓库的 src layout）
_REPO_ROOT = _HERE.parent.parent  # d:/code/glm52/torchturbo
_SRC = _REPO_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# --------------------------------------------------------------------------- #
# token 解析（与 test_hf_logp.py 一致，便于互换）
# --------------------------------------------------------------------------- #
def parse_test_tokens(arg: str) -> List[int]:
    value = arg.strip()
    if value and os.path.isfile(value):
        return _load_tokens_from_file(value)
    parts = [p.strip() for p in value.split(",") if p.strip()]
    try:
        return [int(p) for p in parts]
    except ValueError as e:
        raise ValueError(
            f"--test-tokens={arg!r} is neither a file nor a comma-separated int list."
        ) from e


def _load_tokens_from_file(path: str) -> List[int]:
    import json
    import re

    def _coerce(data) -> List[int]:
        if isinstance(data, dict):
            for k in ("tokens", "token_ids", "input_ids", "input_token_ids",
                      "response_tokens"):
                if k in data:
                    data = data[k]
                    break
        if hasattr(data, "flatten") and hasattr(data, "tolist"):
            data = data.flatten().tolist()
        if isinstance(data, (list, tuple)):
            return [int(t) for t in data]
        raise ValueError(f"Unsupported token content in {path!r}: {type(data)}")

    suffix = os.path.splitext(path)[1].lower()
    if suffix in (".pt", ".pth"):
        return _coerce(torch.load(path, map_location="cpu", weights_only=False))
    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            return _coerce(json.load(f))
    with open(path, encoding="utf-8") as f:
        nums = re.findall(r"-?\d+", f.read())
    return [int(n) for n in nums]


# --------------------------------------------------------------------------- #
# 设备抽象（NPU/CUDA）
# --------------------------------------------------------------------------- #
def _is_npu() -> bool:
    try:
        import torch_npu  # noqa: F401
        return hasattr(torch, "npu") and torch.npu.is_available()
    except ImportError:
        return False


def _acc_type() -> str:
    return "npu" if _is_npu() else "cuda"


def _set_device(local_rank: int) -> torch.device:
    acc = _acc_type()
    dev = torch.device(f"{acc}:{local_rank}")
    if acc == "npu":
        import torch_npu
        torch.npu.set_device(dev)
    else:
        torch.cuda.set_device(dev)
    return dev


def _sync():
    if _is_npu():
        torch.npu.synchronize()
    else:
        torch.cuda.synchronize()


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


# --------------------------------------------------------------------------- #
# activation hooks（输出格式与 test_hf_logp.py 兼容）
# --------------------------------------------------------------------------- #
def register_activation_hooks(model, activations: Dict[str, Any]):
    """hook embed_tokens / 每个 layer / norm / lm_head，存 {output, output_shape}。"""

    def make_hook(name):
        def hook(_m, inp, out):
            act = out[0] if isinstance(out, tuple) else out
            if not isinstance(act, torch.Tensor):
                return
            activations[name] = {
                "output": act.detach().cpu().float(),
                "output_shape": tuple(act.shape),
            }
        return hook

    hooks = []
    base = model.model if hasattr(model, "model") else model
    embed = getattr(base, "embed_tokens", None) or getattr(base, "word_embeddings", None)
    if embed is not None:
        hooks.append(embed.register_forward_hook(make_hook("embed_tokens")))
    if hasattr(base, "layers"):
        for i, layer in enumerate(base.layers):
            hooks.append(layer.register_forward_hook(make_hook(f"layer_{i}")))
    if hasattr(base, "norm"):
        hooks.append(base.norm.register_forward_hook(make_hook("norm")))
    if hasattr(model, "lm_head"):
        hooks.append(model.lm_head.register_forward_hook(make_hook("lm_head")))
    logger.info(f"[_rank {_rank()}] activation hooks: {len(hooks)} registered")
    return hooks


# --------------------------------------------------------------------------- #
# 细粒度 hook：定位层内首次发散点（与 test_hf_logp.py 同名 key，便于 compare-activations 配对）
# --------------------------------------------------------------------------- #
def register_finegrained_hooks(model, activations: Dict[str, Any], fg_spec: str):
    """在选定 decoder layer 的子模块上挂 hook，外加 final norm 的输入。

    对每个选定层 i 捕获：
      layer_{i}_input_layernorm         - pre-attention RMSNorm 输出
      layer_{i}_self_attn               - 注意力输出（MLA/DSA，NPU 上是 npu_fused_dsa_attn_forward）
      layer_{i}_post_attention_layernorm - post-attention RMSNorm 输出
      layer_{i}_mlp                     - MLP/MoE 输出
    以及始终捕获：
      norm_input                        - 进入 final norm 的 hidden（让 RMSNorm 放大效应单独成行）

    key 用 `layer_{i}_{submodule}` 形式，compare_layer_io._act_layer_idx 会把它
    排在父层 layer_{i} 之后。与 test_hf_logp.py 输出同名，compare-activations 按名配对。

    Args:
        fg_spec: "0", "0,1,2", 或 "all"。
    """
    def make_hook(name):
        def hook(_m, _inp, out):
            act = out[0] if isinstance(out, tuple) else out
            if not isinstance(act, torch.Tensor):
                return
            activations[name] = {
                "output": act.detach().cpu().float(),
                "output_shape": tuple(act.shape),
            }
        return hook

    def make_pre_hook(name):
        def pre(_m, args):
            if args and isinstance(args[0], torch.Tensor):
                t = args[0]
                activations[name] = {
                    "output": t.detach().cpu().float(),
                    "output_shape": tuple(t.shape),
                }
        return pre

    hooks = []
    base = model.model if hasattr(model, "model") else model
    layers = getattr(base, "layers", None)
    if layers is None:
        logger.warning("[_rank %s] finegrain: no .layers found, skipping", _rank())
        return hooks

    sel = set()
    for x in fg_spec.split(","):
        x = x.strip()
        if not x:
            continue
        if x == "all":
            sel.update(range(len(layers)))
        else:
            sel.add(int(x))
    sel = {i for i in sel if 0 <= i < len(layers)}

    submods = [
        ("input_layernorm", "input_layernorm"),
        ("self_attn", "self_attn"),
        ("post_attention_layernorm", "post_attention_layernorm"),
        ("mlp", "mlp"),
    ]
    for i in sorted(sel):
        layer = layers[i]
        for attr, suffix in submods:
            sub = getattr(layer, attr, None)
            if sub is not None:
                hooks.append(sub.register_forward_hook(make_hook(f"layer_{i}_{suffix}")))

    norm = getattr(base, "norm", None)
    if norm is not None:
        hooks.append(norm.register_forward_pre_hook(make_pre_hook("norm_input")))

    logger.info("[_rank %s] finegrain hooks: %d on layers %s + norm_input",
                _rank(), len(hooks), sorted(sel))
    return hooks


# --------------------------------------------------------------------------- #
# 通用注入（Level 0+1+2）：任意 decoder 子模块的 input/output 注入
#   --inject-ref LAYERS:TARGET:DIR  用 HF 输入覆盖子模块输入(pre-hook)，捕获本侧输出对比
#   --force-ref  LAYERS:TARGET:DIR  用 HF 输出覆盖子模块输出(forward-hook return)，看下游是否对齐
# 此前注入只到 layer / final norm；这里下沉到 self_attn / mlp / 每层两个 RMSNorm，
# 以及 self_attn 内部投影/ norm / o_proj（Level 2，定位 self_attn 内首次发散点）。
# --------------------------------------------------------------------------- #
# attention 内部子模块（capture 全覆盖；inject 仅单 tensor 位置输入者）。
ATTN_INTERNAL_TARGETS = ("q_a_proj", "q_a_layernorm", "q_b_proj",
                         "kv_a_proj_with_mqa", "kv_a_layernorm", "kv_b_proj",
                         "o_proj", "indexer")
# 可注入：单 tensor 位置输入，pre-hook 替 args[0] 即可。排除：
#   kv_b_proj —— torchturbo NPU fused 路径直读 weight(MLA absorb)，forward 不触发，注入无效
#   indexer   —— 多入参(hidden_states, q_resid, ...)，只替 args[0] 不一致
_INJECTABLE_ATTN_INTERNAL = ("q_a_proj", "q_a_layernorm", "q_b_proj",
                             "kv_a_proj_with_mqa", "kv_a_layernorm", "o_proj")
_INJECT_TARGETS = (("layer", "input_layernorm", "self_attn",
                    "post_attention_layernorm", "mlp", "norm")
                   + _INJECTABLE_ATTN_INTERNAL)


def _resolve_inject_target(model, layer_idx: int, target: str):
    """target -> nn.Module。target='norm' 用 final norm，忽略 layer_idx。
    HF 与 torchturbo 的 GLM5 结构一致：.model.layers[i].{input_layernorm,self_attn,
    post_attention_layernorm,mlp} + .model.norm；attention 内部 self_attn.{q_a_proj,
    q_a_layernorm,q_b_proj,kv_a_proj_with_mqa,kv_a_layernorm,kv_b_proj,o_proj,indexer}。"""
    base = model.model if hasattr(model, "model") else model
    if target == "norm":
        mod = getattr(base, "norm", None)
        if mod is None:
            raise AttributeError("target=norm 但 model 无 .norm")
        return mod
    layers = getattr(base, "layers", None)
    if not layers or not (0 <= layer_idx < len(layers)):
        raise AttributeError(f"target={target} 但无 layers[{layer_idx}]")
    layer = layers[layer_idx]
    if target == "layer":
        return layer
    if target in ATTN_INTERNAL_TARGETS:
        attn = getattr(layer, "self_attn", None)
        mod = getattr(attn, target, None) if attn is not None else None
        if mod is None:
            raise AttributeError(f"layer {layer_idx} self_attn 无属性 {target}")
        return mod
    mod = getattr(layer, target, None)
    if mod is None:
        raise AttributeError(f"layer {layer_idx} 无属性 {target}")
    return mod


def _inject_ref_filename(target: str, layer_idx: int) -> str:
    """与 HF 侧 --dump-ref 约定一致。target=layer 复用旧 --dump-layer-ref 的 layer_{i}.pt
    （内容格式 {input,output,input_dtype} 也一致，向后兼容）。"""
    if target == "norm":
        return "norm.pt"
    if target == "layer":
        return f"layer_{layer_idx}.pt"
    return f"layer_{layer_idx}_{target}.pt"


def _parse_ref_spec(spec: str):
    """'LAYERS:TARGET:DIR' -> (layers_part, target, ref_dir:Path)。"""
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"spec 需 'LAYERS:TARGET:DIR'，得到 {spec!r}")
    layers_part, target, ref_dir = (p.strip() for p in parts)
    if target not in _INJECT_TARGETS:
        raise ValueError(f"target={target!r} 不在支持列表 {_INJECT_TARGETS}")
    return layers_part, target, Path(ref_dir)


def _pad_to_sp(t: torch.Tensor, pad_to: int) -> torch.Tensor:
    """SP/EP 下 torchturbo seq 被 pad 到 sp_size 倍数；把 HF ref 对齐 pad_to。
    子模块输入/输出均为 [1, seq, hidden]，逐行计算，pad 位不影响真实 token。"""
    if t is None or t.dim() < 2 or pad_to <= 0:
        return t
    seq = t.shape[1]
    if seq > pad_to:
        return t[:, :pad_to, :].contiguous()
    if seq < pad_to:
        return F.pad(t, (0, 0, 0, pad_to - seq), value=0)
    return t


def _apply_ref_injection(model, spec: str, mode: str, device, pad_to: int,
                         captured: Dict[int, torch.Tensor]):
    """注册 input 或 output 注入 hook。

    mode='input' : pre-hook 用 HF 输入覆盖；forward-hook 捕获本侧输出(写入 captured)
    mode='output': forward-hook 用 HF 输出覆盖本侧输出(return 替换)；同时把被替换前的
                   本侧输出写入 captured(供报告)
    返回 (handles, ref_outputs, sel_layers, target, ref_real_len)。

    关键：self_attn 在 torchturbo 里是关键字调用(self.self_attn(hidden_states=..., ...))，
    pre-hook 拿到的 args 为空，必须改 kwargs['hidden_states']；其它 target 位置调用改 args[0]。
    """
    layers_part, target, ref_dir = _parse_ref_spec(spec)
    base = model.model if hasattr(model, "model") else model
    layers = getattr(base, "layers", None) or []
    sel = set()
    if target == "norm":
        sel.add(0)  # norm 无层概念，单次
    else:
        for x in layers_part.split(","):
            x = x.strip()
            if not x:
                continue
            if x == "all":
                sel.update(range(len(layers)))
            else:
                sel.add(int(x))
        sel = {i for i in sel if 0 <= i < len(layers)}

    handles, ref_outputs = [], {}
    ref_real_len = None
    for i in sorted(sel):
        ref_file = ref_dir / _inject_ref_filename(target, i)
        if not ref_file.exists():
            logger.warning("[_rank %s] %s-ref: %s 不存在,跳过 %s layer %d",
                           _rank(), mode, ref_file, target, i)
            continue
        lr = torch.load(ref_file, map_location=device, weights_only=False)
        ref_in = lr.get("input")
        ref_out = lr.get("output")
        if ref_real_len is None and ref_in is not None and ref_in.dim() >= 2:
            ref_real_len = ref_in.shape[1]
        mod = _resolve_inject_target(model, i, target)
        ref_in = _pad_to_sp(ref_in.to(device), pad_to) if ref_in is not None else None
        ref_out = _pad_to_sp(ref_out.to(device), pad_to) if ref_out is not None else None
        ref_outputs[i] = ref_out

        if mode == "input" and ref_in is not None:
            def _mk_pre(ri):
                def pre(_m, a, kwargs):
                    # self_attn 关键字调用(args 空) -> 改 kwargs；其余位置调用 -> 改 args[0]
                    if "hidden_states" in kwargs:
                        kwargs = dict(kwargs)
                        kwargs["hidden_states"] = ri
                        return a, kwargs
                    if a and isinstance(a[0], torch.Tensor):
                        return (ri,) + tuple(a[1:]), kwargs
                    return a, kwargs
                return pre

            def _mk_cap(idx):
                def post(_m, _inp, out):
                    act = out[0] if isinstance(out, tuple) else out
                    if isinstance(act, torch.Tensor):
                        captured[idx] = act.detach()
                return post

            handles.append(mod.register_forward_pre_hook(_mk_pre(ref_in), with_kwargs=True))
            handles.append(mod.register_forward_hook(_mk_cap(i)))
        elif mode == "output" and ref_out is not None:
            def _mk_force(idx, ro):
                def post(_m, _inp, out):
                    act = out[0] if isinstance(out, tuple) else out
                    if isinstance(act, torch.Tensor):
                        captured[idx] = act.detach()  # 被替换前的本侧输出
                    # return 非 None -> 替换输出；tuple 保留 out[1:]（如 layer 的 topk_indices）
                    if isinstance(out, tuple):
                        return (ro,) + tuple(out[1:])
                    return ro
                return post

            handles.append(mod.register_forward_hook(_mk_force(i, ref_out)))
    logger.info("[_rank %s] %s-ref: target=%s layers=%s (ref_dir=%s, %d 命中)",
                _rank(), mode, target, sorted(ref_outputs), ref_dir, len(ref_outputs))
    return handles, ref_outputs, sorted(ref_outputs), target, ref_real_len


# --------------------------------------------------------------------------- #
# attention 内部 capture hook（--finegrain-attn）：定位 self_attn 内首次发散点
# 与 register_finegrained_hooks 同写一个 activations dict，key: layer_{i}_self_attn_{sub}，
# compare-activations 按名配对。索引器输出是 int32 topk indices，.float() 后 abs diff=0 即同路由。
# --------------------------------------------------------------------------- #
def register_attn_internal_hooks(model, activations: Dict[str, Any], fg_spec: str):
    """在选定层 self_attn 的内部子模块上挂 capture hook，捕输出 + shape。

    捕获 q_a_proj/q_a_layernorm/q_b_proj/kv_a_proj_with_mqa/kv_a_layernorm/kv_b_proj/
    o_proj/indexer。

    注意：torchturbo NPU fused 路径下 kv_b_proj.forward 不触发(MLA absorb 直读 weight)，
    故 layer_{i}_self_attn_kv_b_proj 在 torchturbo 侧缺失——compare-activations 会标
    "ONLY in HF"，正是 MLA-absorb 不对称的信号(非 bug)。indexer 在 "shared" 层为 None，
    自动跳过。
    """
    def make_hook(name):
        def hook(_m, _inp, out):
            act = out[0] if isinstance(out, tuple) else out
            if not isinstance(act, torch.Tensor):
                return
            activations[name] = {
                "output": act.detach().cpu().float(),
                "output_shape": tuple(act.shape),
            }
        return hook

    hooks = []
    base = model.model if hasattr(model, "model") else model
    layers = getattr(base, "layers", None)
    if layers is None:
        logger.warning("[_rank %s] finegrain-attn: no .layers found, skipping", _rank())
        return hooks
    sel = set()
    for x in fg_spec.split(","):
        x = x.strip()
        if not x:
            continue
        if x == "all":
            sel.update(range(len(layers)))
        else:
            sel.add(int(x))
    sel = {i for i in sel if 0 <= i < len(layers)}
    for i in sorted(sel):
        attn = getattr(layers[i], "self_attn", None)
        if attn is None:
            continue
        for sub in ATTN_INTERNAL_TARGETS:
            mod = getattr(attn, sub, None)
            if mod is not None:
                hooks.append(mod.register_forward_hook(
                    make_hook(f"layer_{i}_self_attn_{sub}")))
    logger.info("[_rank %s] finegrain-attn hooks: %d on layers %s",
                _rank(), len(hooks), sorted(sel))
    return hooks


# --------------------------------------------------------------------------- #
# next-token logp 提取（与 test_hf_logp.py 完全一致）
# --------------------------------------------------------------------------- #
def extract_next_token_logp(logits: torch.Tensor,
                             input_ids: torch.Tensor,
                             response_length: int) -> Dict[str, Any]:
    """logits: [1, T, V] 或 [T, V]; input_ids: [T]。返回 logp dict。"""
    if logits.dim() == 3:
        logits = logits.squeeze(0)
    logits = logits.float()  # [T, V]

    target_tokens = input_ids.flatten()
    total_length = target_tokens.numel()
    if response_length is None or response_length <= 0:
        prompt_length = 1
        response_length = total_length - 1
    else:
        prompt_length = total_length - response_length

    start_idx = prompt_length - 1
    end_idx = total_length - 1
    if start_idx < 0 or end_idx <= start_idx:
        raise ValueError(f"bad split: prompt={prompt_length} total={total_length}")

    response_logits = logits[start_idx:end_idx]                  # [R, V]
    response_target = target_tokens[prompt_length:total_length].to(logits.device)
    log_probs = F.log_softmax(response_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, response_target.unsqueeze(-1)).squeeze(-1)

    return {
        "response_tokens": response_target.cpu(),
        "response_logp": token_log_probs.detach().cpu(),
        "prompt_length": prompt_length,
        "response_length": response_length,
        "mean_logp": token_log_probs.mean().item(),
        "input_token_ids": target_tokens.cpu(),
    }


# --------------------------------------------------------------------------- #
# 对照实验：monkey-patch MixedPrecisionPolicy 构造，定位 forward 差异来源
# --------------------------------------------------------------------------- #
def _patch_shard_model_mp_policy(args):
    """monkey-patch FSDP MixedPrecisionPolicy 构造，按实验开关改 mp_policy。

    背景（实验A 报错已确认）：
      v3 模型参数是 **fp32 存储**（set_config_dtype_to_float32 + meta 建模 + 权重 upcast），
      forward 时靠 mp_policy param_dtype=bf16 cast 到 bf16。NPU fused kernel（npu_sparse_flash_attention）
      强制 q/k/v 为 bf16，故不能完全关掉 param_dtype cast（否则 forward 崩，即实验A）。
      v2 参数加载时直接 model.to(bf16)，全程 bf16，无 cast_forward_inputs。

    实验B (--no-cast-forward): param_dtype=bf16 保留（NPU 不崩），只关 cast_forward_inputs
        若 logp 与 v2 一致 -> 差异来自 cast_forward_inputs（每层输入被 cast bf16）
    实验C (--no-param-cast): param_dtype=None（参数 fp32 直接 forward，NPU 会崩，仅验证用）
    实验A (--no-mp-policy): mp_policy=None（参数 fp32，NPU 必崩，已确认 v3=fp32存储）

    原理：v3 _shard_model 里 `from torch.distributed.fsdp import MixedPrecisionPolicy`
    是函数内 import，每次执行重新解析，故 patch 模块属性即生效。
    """
    import torch.distributed.fsdp as fsdp_mod

    _orig_MMP = fsdp_mod.MixedPrecisionPolicy
    tag = []

    def _new_MMP(*a, **kw):
        # v3 调用形如 MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32, cast_forward_inputs=True)
        if getattr(args, "no_cast_forward", False):
            kw["cast_forward_inputs"] = False
            tag.append("B:no-cast-forward")
        if getattr(args, "no_param_cast", False):
            kw["param_dtype"] = None
            tag.append("C:no-param-cast")
        if getattr(args, "no_mp_policy", False):
            # 实验 A：返回 None 让 mp_policy 为 None（但参数 fp32 会导致 NPU 崩，仅供确认机制）
            tag.append("A:no-mp-policy")
            return None
        return _orig_MMP(*a, **kw)

    fsdp_mod.MixedPrecisionPolicy = _new_MMP
    logger.info("[_rank %s] mp_policy patch 已装: %s", _rank(),
                tag or ["(默认 v3 mp_policy)"])


def _patch_moe_dtype_v2():
    """monkey-patch topk_softmax_with_capacity 的 sigmoid 分支用 v2 dtype 行为。

    v3 vs v2 已确认的代码级差异（GLM5 每层 MoE 走 sigmoid 分支）：
      v3: sigmoid 全程 fp32, topk_masked_gates = zeros_like(logits, dtype=fp32)
      v2: sigmoid 中途 type_as(logits) 转回 bf16, topk_masked_gates = zeros_like(logits)

    若 --v2-moe-dtype 后 logp 对齐 v2 -> 差异来自 MoE 路由 dtype。
    """
    import torch_turbo.distributed.moe.utils as moe_utils

    _orig = moe_utils.topk_softmax_with_capacity
    _group_limited_topk = moe_utils.group_limited_topk

    def _compute_topk(scores, topk, num_groups, group_topk, shape):
        num_tokens, num_experts = shape
        if group_topk:
            return _group_limited_topk(scores=scores, topk=topk, num_tokens=num_tokens,
                                       num_experts=num_experts, num_groups=num_groups,
                                       group_topk=group_topk)
        return torch.topk(scores, k=topk, dim=1)

    def _v2_dtype_wrapper(logits, topk, use_pre_softmax=False, num_groups=None,
                          group_topk=None, scaling_factor=None, score_function="softmax",
                          expert_bias=None, use_hash_routing=False, hash_table=None,
                          input_ids=None):
        # 非 sigmoid 分支调原 v3（GLM5 不走）
        if score_function != "sigmoid":
            return _orig(logits, topk, use_pre_softmax, num_groups, group_topk,
                         scaling_factor, score_function, expert_bias,
                         use_hash_routing, hash_table, input_ids)
        # v2 sigmoid 逻辑：中途 type_as(logits) 转回 bf16
        scores = torch.sigmoid(logits.float()).type_as(logits)
        if expert_bias is not None:
            scores_for_routing = scores + expert_bias
            _, top_indices = _compute_topk(scores_for_routing, topk, num_groups, group_topk, logits.shape)
            scores = torch.gather(scores, dim=1, index=top_indices).type_as(logits)
        else:
            scores, top_indices = _compute_topk(scores, topk, num_groups, group_topk, logits.shape)
        probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else scores
        if scaling_factor:
            probs = probs * scaling_factor
        topk_masked_gates = torch.zeros_like(logits).scatter(1, top_indices, probs)
        topk_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()
        return topk_masked_gates, topk_map, topk_map.sum(dim=0)

    moe_utils.topk_softmax_with_capacity = _v2_dtype_wrapper
    print("[_rank %s] MoE sigmoid dtype 已 patch 为 v2 行为(bf16)", _rank())


# --------------------------------------------------------------------------- #
# 实验E：还原 eager fp32 RMSNorm（验证 NPU fused RMSNorm 是否为发散根因）
# --------------------------------------------------------------------------- #
def _restore_eager_rmsnorm():
    """把 GlmMoeDsaRMSNorm.forward 还原回 eager fp32 实现，对齐 HF baseline。

    背景（细粒度 hook 已证实）：
      embed_tokens 完全一致，但 layer_0_input_layernorm 就发散 -> 首因是 NPU fused
      npu_rms_norm（_pre_init_patches 在 [model.py] 替换了 GlmMoeDsaRMSNorm.forward）。
      HF eager 实现：variance/rsqrt 在 fp32，最后 weight*x 回 input dtype。

    若 --no-fused-rmsnorm 后 layer_0_input_layernorm diff -> ~0 且 logp 对齐 HF
    -> ⑥ NPU fused RMSNorm 坐实为发散根因，且是修复目标（换 eager 或修 NPU kernel）。

    在 build_v3_model 之后调用（此时 _pre_init_patches 已替换 forward）。
    """
    import torchturbo.models.glm_moe_dsa.modeling_glm_moe_dsa as glm5_model

    def _eager_rms_norm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hs = hidden_states.to(torch.float32)
        variance = hs.pow(2).mean(-1, keepdim=True)
        hs = hs * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hs.to(input_dtype)

    glm5_model.GlmMoeDsaRMSNorm.forward = _eager_rms_norm_forward
    print("[_rank %s] RMSNorm 已还原为 eager fp32 实现（NPU fused 已关闭）", _rank())


# --------------------------------------------------------------------------- #
# 实验F：dump NPU fused DSA attention 内部中间量（定位 self_attn 内发散源）
# 已确认：投影+norm 全 match，o_proj 输出发散，发散在 fused 核心。
# 本实验 monkey-patch npu_fused_dsa_attn_forward 的三个子步骤 + core op，dump：
#   q_nope_absorbed   MLA absorb 结果(Q @ W_UK)         —— absorb 嫌疑
#   q_pe_postrotary   query rope(已 rotary)              —— rotary 嫌疑(对 HF interleave)
#   k_pe_postrotary   key rope(已 rotary)                —— rotary 嫌疑
#   k_lat / v_lat     压缩 KV(未 up-projection)
#   topk_indices       npu_lightning_indexer 稀疏选择     —— indexer 嫌疑(对 HF indexer)
#   indexer_query_index/key_index/weights               —— 闪电 indexer 中间量
#   core_attn_output  npu_sparse_flash_attention 输出    —— core kernel 嫌疑
# 落盘到 out_dir/layer_{i}.pt，与 test_hf_logp.py --dump-attn-internals 配对，
# 用 compare_layer_io.py compare-fused-attn 对比(rotary/sparse 决定性)。
# --------------------------------------------------------------------------- #
def _patch_fused_attn_dump(model, out_dir: str, layers_spec: str):
    """monkey-patch fused attn 子步骤 + core op，捕获中间量。返回 (save, restore)。
    在 build_v3_model 之后、forward 之前调用；forward 后先 save() 再 restore()。

    三个 patch 均经模块属性赋值(npu_fused_dsa_attn_forward 内是 bare name 引用 +
    torch_npu 属性引用，运行时查找，故 patch 模块属性即生效)。
    """
    import torchturbo.models.glm_moe_dsa.npu as npu_mod
    import torch_npu

    base = model.model if hasattr(model, "model") else model
    layers = getattr(base, "layers", None) or []
    id2idx = {id(layers[i].self_attn): i for i in range(len(layers))}
    sel = set()
    for x in layers_spec.split(","):
        x = x.strip()
        if not x:
            continue
        if x == "all":
            sel.update(range(len(layers)))
        else:
            sel.add(int(x))
    sel = {i for i in sel if 0 <= i < len(layers)}

    captured: Dict[int, Dict[str, Any]] = {}
    prepare_all: List[int] = []       # 每次 prepare 的 layer idx(全层,顺序)
    core_all: List[Any] = []          # 每次 core 的 attn_output(全层,顺序)

    _orig_prepare = npu_mod._npu_prepare_qkv_for_mla_absorb
    _orig_sparse = npu_mod._npu_compute_sparse_indices
    _orig_core = torch_npu.npu_sparse_flash_attention

    def _wrap_prepare(self, hidden_states, position_embeddings):
        out = _orig_prepare(self, hidden_states, position_embeddings)
        idx = id2idx.get(id(self))
        prepare_all.append(idx)
        if idx is not None and idx in sel:
            q_nope_absorbed, q_pe, k_lat, v_lat, k_pe, q_resid, kv_b_weight = out
            d = captured.setdefault(idx, {})
            d["q_nope_absorbed"] = q_nope_absorbed.detach().cpu()
            d["q_pe_postrotary"] = q_pe.detach().cpu()
            d["k_pe_postrotary"] = k_pe.detach().cpu()
            d["k_lat"] = k_lat.detach().cpu()
            d["v_lat"] = v_lat.detach().cpu()
            d["q_resid"] = q_resid.detach().cpu() if q_resid is not None else None
            # kv_b_weight: dump 一份供对照 absorb(W_UV = kv_b_weight[:, -v_head_dim:])
            d["kv_b_weight"] = kv_b_weight.detach().cpu()
        return out

    def _wrap_sparse(self, hidden_states, q_resid, position_embeddings, attention_mask,
                     sp_enabled, sp_group, sp_size):
        out = _orig_sparse(self, hidden_states, q_resid, position_embeddings, attention_mask,
                           sp_enabled, sp_group, sp_size)
        idx = id2idx.get(id(self))
        if idx is not None and idx in sel:
            topk_indices, query_index, key_index, weights_indexer = out
            d = captured.setdefault(idx, {})
            d["topk_indices"] = topk_indices.detach().cpu()
            d["indexer_query_index"] = query_index.detach().cpu()
            d["indexer_key_index"] = key_index.detach().cpu()
            d["indexer_weights"] = weights_indexer.detach().cpu()
        return out

    def _wrap_core(query, key, value, sparse_indices, scale_value, **kwargs):
        out = _orig_core(query, key, value, sparse_indices, scale_value, **kwargs)
        # out = (attn_output, softmax_max, softmax_sum, ...)
        try:
            core_all.append(out[0].detach().cpu())
        except Exception:
            core_all.append(None)
        return out

    npu_mod._npu_prepare_qkv_for_mla_absorb = _wrap_prepare
    npu_mod._npu_compute_sparse_indices = _wrap_sparse
    torch_npu.npu_sparse_flash_attention = _wrap_core
    logger.info("[_rank %s] 实验F fused-attn dump patch 已装: layers=%s -> %s",
                _rank(), sorted(sel), out_dir)

    def save():
        # 关联 core_attn_output: prepare_all[i] == i(层顺序), core_all[i] 对应第 i 次 core
        for i, idx in enumerate(prepare_all):
            if idx in sel and i < len(core_all) and core_all[i] is not None:
                captured.setdefault(idx, {})["core_attn_output"] = core_all[i]
        out_dir_p = Path(out_dir)
        out_dir_p.mkdir(parents=True, exist_ok=True)
        for idx, d in captured.items():
            torch.save(d, out_dir_p / f"layer_{idx}.pt")
        logger.info("[_rank %s] 实验F dumped %d layers to %s",
                    _rank(), len(captured), out_dir_p.resolve())
        return captured

    def restore():
        npu_mod._npu_prepare_qkv_for_mla_absorb = _orig_prepare
        npu_mod._npu_compute_sparse_indices = _orig_sparse
        torch_npu.npu_sparse_flash_attention = _orig_core

    return save, restore


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# 权重 dump（定位 v2/v3 加载后权重值是否一致）
# --------------------------------------------------------------------------- #
def dump_model_weights(model, out_dir: str, tag: str = ""):
    """dump 加载后的关键权重/buffer 到文件，便于 v2/v3 直接对比数值。

    FSDP2 下参数可能 sharded，用 full_tensor() 取完整值。
    收集：expert weight1/weight2、router(gate) weight、expert_bias、
          embed_tokens、lm_head、q_proj/kv_a_proj（首层 attention）
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {}

    def _full(p):
        if hasattr(p, "full_tensor"):
            try:
                return p.full_tensor().detach().cpu().contiguous()
            except Exception:
                return p.detach().cpu().contiguous()
        return p.detach().cpu().contiguous()

    for name, p in model.named_parameters():
        if not any(k in name for k in (
            "layers.0.", "layers.1.", "embed_tokens", "word_embeddings",
            "lm_head", "experts", "router", "gate")):
            continue
        try:
            state[f"param.{name}"] = _full(p)
        except Exception as e:
            logger.warning(f"dump param {name} failed: {e}")

    for name, buf in model.named_buffers():
        if not any(k in name for k in ("expert_bias", "e_score_correction_bias",
                                       "inv_freq", "layers.0.", "layers.1.")):
            continue
        try:
            state[f"buffer.{name}"] = buf.detach().cpu().contiguous()
        except Exception as e:
            logger.warning(f"dump buffer {name} failed: {e}")

    out_file = out_dir / f"weights_{tag or 'dump'}.pt"
    torch.save(state, out_file)
    logger.info(f"[rank {_rank()}] dumped {len(state)} tensors to {out_file}")
    print(f"\n=== Weights dump summary ({tag}) ===")
    for k in sorted(state.keys())[:40]:
        t = state[k]
        print(f"  {k:60s} dtype={str(t.dtype):16s} shape={list(t.shape)} "
              f"min={t.float().min().item():.6e} max={t.float().max().item():.6e} "
              f"mean={t.float().mean().item():.6e}")
    print(f"  ... (total {len(state)} tensors)")
    return out_file


# v3 模型构建（纯 src/torchturbo 引擎 API，对齐 TorchTurboBackend.setup_model）
# --------------------------------------------------------------------------- #
def build_v3_model(args, device: torch.device):
    """用 src/torchturbo 的 TorchTurboEngine 构建并行模型。

    严格对齐 Asystem-HybridEngine 的 torchturbo_backend.py::setup_model 流程：
      1. TorchTurboEngine(EngineConfig(...))
      2. engine.prepare_model(path, torch_dtype, trust_remote_code, init_kwargs)
         —— 注意是 prepare_model，不是 build_model（v3 引擎无 build_model）
         init_kwargs 传预加载的 hf_config 并设 _attn_implementation
      3. self.model = engine.model
      4. engine.backend._sync_distributed_state（SP/DP/EP 由 backend 暴露）

    v3 _shard_model 写死：mixed_precision=bf16 -> param=bf16, reduce=fp32, cast_forward_inputs=True
    （v2 torch_turbo 无 mp_policy -> reduce=bf16，这是头号嫌疑）
    mixed_precision 是整体枚举，不单独控制 reduce_dtype，故只暴露 --mixed-precision。
    """
    import torchturbo  # noqa: F401  触发 src __init__ + GLM5 注册
    from torchturbo import TorchTurboEngine, EngineConfig, MixedPrecisionPolicy
    from torchturbo.engine import DistributedBackend
    from torchturbo.engine.backends.fsdp2 import FSDPConfig
    from torchturbo.parallel import FSDPParallelPlan
    from transformers import AutoConfig

    # mixed_precision 决定 FSDP param/reduce dtype（v3 写死 bf16->reduce fp32）
    mp = {"bf16": MixedPrecisionPolicy.BF16, "fp16": MixedPrecisionPolicy.FP16,
          "fp32": MixedPrecisionPolicy.FP32}[args.mixed_precision]

    config = EngineConfig(
        backend=DistributedBackend.FSDP2,
        mixed_precision=mp,
        backend_config=FSDPConfig(reshard_after_forward=True),
        parallel_plan=FSDPParallelPlan(sp_size=args.sp_size, ep_size=args.ep_size),
        extra={"ep_size": args.ep_size, "sp_size": args.sp_size},
    )
    engine = TorchTurboEngine(config)

    # ── 对照实验：monkey-patch _shard_model 改 mp_policy，定位 forward 差异来源 ──
    # v3 默认: param=bf16, reduce=fp32, cast_forward_inputs=True
    # v2 默认: 无 mp_policy（param/reduce 沿用 bf16，无 cast_forward_inputs）
    if getattr(args, "no_mp_policy", False) or getattr(args, "no_cast_forward", False) \
       or getattr(args, "no_param_cast", False):
        _patch_shard_model_mp_policy(args)

    # ── 实验D：MoE 路由 dtype patch 为 v2（sigmoid 分支中途转 bf16）──
    if getattr(args, "v2_moe_dtype", False):
        _patch_moe_dtype_v2()

    # 预加载 hf_config 并设 attn_implementation（对齐 torchturbo_backend:481-487）
    hf_config = AutoConfig.from_pretrained(args.path, trust_remote_code=True)
    if args.attn_impl:
        hf_config._attn_implementation = args.attn_impl
    init_kwargs = {"config": hf_config}

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    # prepare_model 内部完成: meta建模 -> MoE替换 -> FSDP2分片 -> setup_sp -> 权重加载
    engine.prepare_model(
        args.path,
        torch_dtype=dtype,
        trust_remote_code=True,
        init_kwargs=init_kwargs,
    )
    model = engine.model
    model.eval()
    # reduce_dtype 由 mixed_precision 决定：bf16 -> fp32（v3 引擎写死）
    reduce_dt = "fp32" if args.mixed_precision == "bf16" else args.mixed_precision
    logger.info(f"[_rank {_rank()}] v3 engine model built "
                f"(mixed_precision={mp.value}, reduce_dtype={reduce_dt})")
    return engine, model


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract response-only logp from torchturbo v3 (src/torchturbo engine)")
    parser.add_argument("--path", required=True, help="模型权重路径")
    parser.add_argument("--test-tokens", type=str, default="10,11,12,13,14,15,16,17",
                        help="逗号分隔 token id 或文件路径")
    parser.add_argument("--response-length", type=int, default=None,
                        help="response 长度，默认除首 token 外全为 response")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--mixed-precision", default="bf16",
                        choices=["bf16", "fp16", "fp32"],
                        help="v3 mixed_precision 枚举。bf16->reduce=fp32(写死)，fp16->reduce=fp16")
    parser.add_argument("--attn-impl", default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"],
                        help="attention 实现，经 init_kwargs 传给 v3 engine")
    # ── 对照实验开关：定位 forward logp 差异来自 mp_policy 哪一项 ──
    # 背景：v3 参数 fp32 存储，forward 靠 param_dtype=bf16 cast；v2 全程 bf16 不 cast
    parser.add_argument("--no-cast-forward", action="store_true",
                        help="实验B(推荐): 保留 param_dtype=bf16(NPU不崩)，只关 cast_forward_inputs。"
                             "若 logp=v2 -> 差异来自每层输入被 cast bf16")
    parser.add_argument("--no-param-cast", action="store_true",
                        help="实验C: param_dtype=None(参数fp32直接forward，NPU会崩，仅验证机制)")
    parser.add_argument("--no-mp-policy", action="store_true",
                        help="实验A: mp_policy=None(参数fp32，NPU必崩，已确认v3=fp32存储，勿用)")
    parser.add_argument("--v2-moe-dtype", action="store_true",
                        help="实验D: MoE sigmoid 路由 dtype patch 为 v2 行为(bf16)。"
                             "若 logp=v2 -> 差异来自 topk_softmax_with_capacity 路由 dtype")
    parser.add_argument("--no-fused-rmsnorm", action="store_true",
                        help="实验E(推荐): 还原 eager fp32 RMSNorm(关 NPU fused npu_rms_norm)。"
                             "细粒度 hook 已证 embed_tokens 一致但 layer_0_input_layernorm 发散,"
                             "首因是 NPU fused RMSNorm。若开启后该 diff -> ~0 且 logp 对齐 HF -> 坐实根因。")
    parser.add_argument("--sp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="v3_logp_result.pt")
    parser.add_argument("--save-activations", action="store_true")
    parser.add_argument("--activations-output", default=None)
    parser.add_argument("--with-backward", action="store_true",
                        help="额外跑 backward + dump 关键参数梯度")
    parser.add_argument("--dump-weights", type=str, default=None,
                        help="dump 加载后权重到指定目录(定位 v2/v3 权重值差异)")
    parser.add_argument("--finegrain", action="store_true",
                        help="在选定层子模块上挂 hook(input_layernorm/self_attn/"
                             "post_attention_layernorm/mlp)+final norm 输入,"
                             "定位层内首次发散点。隐含 --save-activations。"
                             "与 test_hf_logp.py 输出同名 key,compare-activations 按名配对。")
    parser.add_argument("--finegrain-layers", type=str, default="all",
                        help="--finegrain 的层索引,逗号分隔或 'all'。默认 0(layer_0)。")
    parser.add_argument("--finegrain-attn", action="store_true",
                        help="在选定层 self_attn 内部子模块上挂 capture hook:"
                             "q_a_proj/q_a_layernorm/q_b_proj/kv_a_proj_with_mqa/"
                             "kv_a_layernorm/kv_b_proj/o_proj/indexer。定位 self_attn 内"
                             "首次发散点(已确认 self_attn 发散后用)。隐含 --save-activations。"
                             "key=layer_{i}_self_attn_{sub},与 test_hf_logp.py 同名,"
                             "compare-activations 按名配对。注意 kv_b_proj 在 NPU fused "
                             "路径下 forward 不触发,本侧缺失(MLA-absorb 不对称信号)。")
    parser.add_argument("--finegrain-attn-layers", type=str, default="0",
                        help="--finegrain-attn 的层索引,逗号分隔或 'all'。默认 0。")
    parser.add_argument("--dump-fused-attn", type=str, default=None,
                        help="实验F: 目录。monkey-patch NPU fused DSA attn 的子步骤+core op,"
                             "dump 中间量(q_nope_absorbed/q_pe_postrotary/k_pe_postrotary/"
                             "k_lat/topk_indices/core_attn_output 等)到 layer_{i}.pt,"
                             "定位 self_attn 内发散源(rotary/MLA absorb/sparse indexer/core kernel)。"
                             "与 test_hf_logp.py --dump-attn-internals 配对,用 "
                             "compare_layer_io.py compare-fused-attn 对比。在 build 后 forward 前生效。")
    parser.add_argument("--dump-fused-attn-layers", type=str, default="0",
                        help="--dump-fused-attn 的层索引,逗号分隔或 'all'。默认 0。")
    parser.add_argument("--inject-norm-ref", type=str, default=None,
                        help="路径: 读 HF 脚本 --dump-norm-ref 产出的 .pt,"
                             "走到 final norm 时用其中的 norm_input 覆盖自己的输入,"
                             "验证'给定相同输入,torchturbo 的 norm 输出是否与 HF 一致'。"
                             "需配合 --norm-ref-logp 给出 HF 的 logp .pt 做下游对比。")
    parser.add_argument("--norm-ref-logp", type=str, default=None,
                        help="HF 侧的 logp .pt(test_hf_logp.py --output),"
                             "用于 --inject-norm-ref 注入后对比下游 logp 是否对齐。")
    parser.add_argument("--dump-norm-out", type=str, default=None,
                        help="路径: dump 注入后的 norm 输出 + logp + 对比结果到 .pt。"
                             "若指定,脚本会在末尾打印 norm 输出与 HF ref 的逐项 diff。")
    parser.add_argument("--inject-layer-ref", type=str, default=None,
                        help="目录: 读 HF 脚本 --dump-layer-ref 产出的 layer_{i}.pt,"
                             "走到指定层时用 HF 的输入覆盖自己的输入。格式: "
                             "'LAYERS:DIR' 或 'DIR'(默认 all)。逐层定位"
                             "'相同输入下,哪一层开始输出不一致'。")
    parser.add_argument("--inject-ref", type=str, default=None,
                        help="通用 input 注入(推荐): 'LAYERS:TARGET:DIR'。走到选定子模块时"
                             "用 HF --dump-ref 产出的输入覆盖自己的输入,捕获本侧输出对比。"
                             "TARGET ∈ {layer,input_layernorm,self_attn,"
                             "post_attention_layernorm,mlp,norm}。"
                             "self_attn/mlp/norm 直接隔离 NPU fused attn/MoE/RMSNorm——"
                             "'相同输入下,该子模块输出是否与 HF 一致'。"
                             "target=layer 等价于 --inject-layer-ref。")
    parser.add_argument("--force-ref", type=str, default=None,
                        help="通用 output 注入: 'LAYERS:TARGET:DIR'。用 HF 的输出强制覆盖"
                             "本侧子模块输出(forward-hook return),再看下游 logp 是否对齐——"
                             "二分'发散在该子模块或上游 vs 下游'。TARGET 同 --inject-ref。"
                             "注意:target=layer 时仅替换 hidden_states(out[0]),"
                             "topk_indices 等 out[1:] 仍用本侧。")
    args = parser.parse_args()

    # 分布式初始化
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if "RANK" in os.environ and not dist.is_initialized():
        backend = "hccl" if _is_npu() else "nccl"
        dist.init_process_group(backend=backend)
        device = _set_device(local_rank)
    else:
        device = torch.device(f"{_acc_type()}:0") if _is_npu() else (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
    rank = _rank()

    torch.manual_seed(args.seed)
    if _is_npu():
        torch.npu.manual_seed_all(args.seed)

    token_ids = parse_test_tokens(args.test_tokens)
    logger.info(f"[rank {rank}] input tokens ({len(token_ids)}): {token_ids}")
    if len(token_ids) < 2:
        logger.error("need >= 2 tokens (1 prompt + 1 response)")
        return 1

    # 构建模型（v3 src 引擎）
    engine, model = build_v3_model(args, device)

    # 实验E：还原 eager fp32 RMSNorm（build 之后，forward 之前）
    if getattr(args, "no_fused_rmsnorm", False):
        _restore_eager_rmsnorm()

    # dump 加载后权重（定位 v2/v3 权重值差异）
    if args.dump_weights:
        dump_model_weights(model, args.dump_weights, tag=f"v3_rank{rank}")

    # 实验F：装 fused-attn dump patch（build 之后，forward 之前）
    _fused_save = _fused_restore = None
    if args.dump_fused_attn:
        _fused_save, _fused_restore = _patch_fused_attn_dump(
            model, args.dump_fused_attn, args.dump_fused_attn_layers)

    # SP ulysses: pad seq_len 到 sp_size 倍数
    sp = max(args.sp_size, 1)
    total = len(token_ids)
    pad_to = ((total + sp - 1) // sp) * sp
    pad_len = pad_to - total
    input_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    if pad_len > 0:
        input_ids = F.pad(input_ids, (0, pad_len), value=0)
        logger.info(f"[rank {rank}] SP pad: {total} -> {pad_to} (pad {pad_len} zeros)")

    # hook 捕 lm_head logits + embed input_ids
    captured = {}

    def _cap_logits(_m, _inp, out):
        o = out[0] if isinstance(out, tuple) else out
        captured["logits"] = o.detach()

    def _cap_input_ids(_m, inp, _out):
        captured["input_ids"] = inp[0].detach()

    h1 = model.lm_head.register_forward_hook(_cap_logits)
    base = model.model if hasattr(model, "model") else model
    embed = getattr(base, "embed_tokens", None) or getattr(base, "word_embeddings", None)
    h2 = embed.register_forward_hook(_cap_input_ids) if embed is not None else None

    activations: Dict[str, Any] = {}
    save_act = args.save_activations or args.finegrain or args.finegrain_attn
    ahooks = register_activation_hooks(model, activations) if save_act else []
    fghooks = register_finegrained_hooks(
        model, activations, args.finegrain_layers
    ) if args.finegrain else []
    atn_hooks = register_attn_internal_hooks(
        model, activations, args.finegrain_attn_layers
    ) if args.finegrain_attn else []

    # ── norm 注入实验：走到 final norm 时用 HF 的输入覆盖 ──
    norm_out_captured = {}
    inject_handles = []
    ref_real_len = total  # 真实 token 数(HF ref 的 seq 长度)
    if args.inject_norm_ref:
        ref = torch.load(args.inject_norm_ref, map_location=device,
                         weights_only=False)
        ref_input = ref["norm_input"].to(device)
        ref_real_len = ref_input.shape[1]
        # SP/EP 下 torchturbo 的 seq 可能被 pad 到 sp_size 倍数；把 HF ref 输入
        # 末尾补零对齐 pad_to。RMSNorm 是逐行计算,pad 位不影响真实 token 的输出。
        if ref_input.shape[1] != pad_to:
            if ref_input.shape[1] > pad_to:
                logger.warning("[rank %s] inject: ref seq (%d) > pad_to (%d), 截断",
                               rank, ref_input.shape[1], pad_to)
                ref_input = ref_input[:, :pad_to, :]
            else:
                pad_n = pad_to - ref_input.shape[1]
                ref_input = F.pad(ref_input, (0, 0, 0, pad_n), value=0)
                logger.info(f"[rank {rank}] inject: ref seq {ref_real_len} -> "
                            f"pad {pad_to}(补 {pad_n} 零对齐 SP)")
        base = model.model if hasattr(model, "model") else model
        norm = getattr(base, "norm", None)
        if norm is None:
            logger.error("[rank %s] --inject-norm-ref: no .norm on model, abort", rank)
            return 1
        logger.info(f"[rank {rank}] inject-norm-ref: 用 HF 的 norm 输入覆盖"
                    f"(shape={tuple(ref_input.shape)}, dtype={ref_input.dtype})")

        def _inject_pre(_m, a, kwargs):
            # 返回新输入替换原输入(PyTorch pre-hook 协议)
            new_args = (ref_input,) + tuple(a[1:])
            return new_args, kwargs

        def _capture_post(_m, _inp, out):
            if isinstance(out, torch.Tensor):
                norm_out_captured["output"] = out.detach()

        # with_kwargs=True 才能拦截 kwargs（FSDP2/transformers 常用 kw 传参）
        inject_handles.append(
            norm.register_forward_pre_hook(_inject_pre, with_kwargs=True))
        inject_handles.append(norm.register_forward_hook(_capture_post))

    # ── 逐层注入实验：走到指定 decoder layer 时用 HF 的输入覆盖 ──
    layer_out_captured: Dict[int, torch.Tensor] = {}
    layer_ref_inputs: Dict[int, torch.Tensor] = {}   # 注入的 HF 输入(用于对比)
    layer_ref_outputs: Dict[int, torch.Tensor] = {}  # HF ref 输出(用于对比)
    if args.inject_layer_ref:
        spec = args.inject_layer_ref
        if ":" in spec:
            layers_part, ref_dir = spec.split(":", 1)
        else:
            layers_part, ref_dir = "all", spec
        base = model.model if hasattr(model, "model") else model
        layers = getattr(base, "layers", None)
        if layers is None:
            logger.error("[rank %s] --inject-layer-ref: no .layers, abort", rank)
            return 1
        sel = set()
        for x in layers_part.split(","):
            x = x.strip()
            if x == "all":
                sel.update(range(len(layers)))
            elif x:
                sel.add(int(x))
        ref_dir_p = Path(ref_dir)
        for i in sorted(j for j in sel if 0 <= j < len(layers)):
            ref_file = ref_dir_p / f"layer_{i}.pt"
            if not ref_file.exists():
                logger.warning("[rank %s] inject-layer: %s 不存在,跳过 layer %d",
                               rank, ref_file, i)
                continue
            lr = torch.load(ref_file, map_location=device, weights_only=False)
            li = lr["input"].to(device)
            if "output" in lr and lr["output"] is not None:
                layer_ref_outputs[i] = lr["output"].to(device)
            # SP pad 对齐(逐层 RMSNorm/attn 仍逐行,pad 位不影响真实 token)
            if li.shape[1] != pad_to:
                if li.shape[1] > pad_to:
                    li = li[:, :pad_to, :]
                else:
                    li = F.pad(li, (0, 0, 0, pad_to - li.shape[1]), value=0)
            layer_ref_inputs[i] = li
            ref_real_len = lr["input"].shape[1]  # HF 真实 seq 长
            _li = li  # 闭包绑定

            def _mk_lpre(ref_in):
                def _pre(_m, a, kwargs):
                    new_args = (ref_in,) + tuple(a[1:])
                    return new_args, kwargs
                return _pre

            def _mk_lpost(idx):
                def _post(_m, _inp, out):
                    act = out[0] if isinstance(out, tuple) else out
                    if isinstance(act, torch.Tensor):
                        layer_out_captured[idx] = act.detach()
                return _post

            inject_handles.append(
                layers[i].register_forward_pre_hook(_mk_lpre(_li), with_kwargs=True))
            inject_handles.append(layers[i].register_forward_hook(_mk_lpost(i)))
        if layer_ref_inputs:
            logger.info(f"[rank {rank}] inject-layer-ref: 注入层 "
                        f"{sorted(layer_ref_inputs)} (ref_dir={ref_dir_p})")

    # ── 通用注入(Level 0+1)：任意子模块 input/output 注入 ──
    # --inject-ref: 用 HF 输入覆盖子模块输入,捕获本侧输出对比(隔离子模块实现)
    # --force-ref : 用 HF 输出强制覆盖子模块输出,看下游 logp 是否对齐(二分上下游)
    inject_ref_captured: Dict[int, torch.Tensor] = {}
    _inj = None  # (sel, target, ref_outputs, ref_real_len)
    if args.inject_ref:
        _h, _ref_outs, _sel, _tgt, _rrl = _apply_ref_injection(
            model, args.inject_ref, "input", device, pad_to, inject_ref_captured)
        inject_handles.extend(_h)
        _inj = (_sel, _tgt, _ref_outs, _rrl)

    force_ref_captured: Dict[int, torch.Tensor] = {}
    _frc = None  # (sel, target, ref_outputs, ref_real_len)
    if args.force_ref:
        _h, _ref_outs, _sel, _tgt, _rrl = _apply_ref_injection(
            model, args.force_ref, "output", device, pad_to, force_ref_captured)
        inject_handles.extend(_h)
        _frc = (_sel, _tgt, _ref_outs, _rrl)

    # forward
    # use_cache=False 必须：npu_fused_dsa_attn_forward 断言 not use_cache（训练专用路径）
    # output_attentions=False 同理（fused attn 不支持）
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=None,
                        use_cache=False, output_attentions=False)
    logits = captured.get("logits")
    if logits is None:  # fallback
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    used_ids = captured.get("input_ids", input_ids).squeeze(0)[:total]

    # 提取 next-token logp（仅用真实 token，去掉 pad）
    result = extract_next_token_logp(logits, used_ids, args.response_length)

    # backward（可选）
    if args.with_backward:
        model.train()
        model.zero_grad(set_to_none=True)
        labels = input_ids.clone()
        out = model(input_ids=input_ids, labels=labels,
                    use_cache=False, output_attentions=False)
        loss = out.loss if hasattr(out, "loss") else out[0].float().mean()
        loss.backward()
        _sync()
        # 收集梯度（FSDP2: 用 .grad，sharded 时 full_tensor）
        total_gn = 0.0
        param_grads = {}
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            try:
                g = p.grad
                if hasattr(g, "full_tensor"):
                    g = g.full_tensor()
                gn = float(g.float().norm().item())
            except Exception:
                continue
            total_gn += gn ** 2
            if any(k in name for k in ("embed", "layers.0", "lm_head")):
                param_grads[name] = {"grad_norm": gn, "grad_mean": float(g.mean()),
                                     "grad_min": float(g.min()), "grad_max": float(g.max()),
                                     "grad_shape": tuple(g.shape)}
        result["loss"] = float(loss.item())
        result["grad_norm"] = total_gn ** 0.5
        result["param_grads"] = param_grads
        logger.info(f"[rank {rank}] backward done: loss={loss.item():.6f} "
                    f"grad_norm={result['grad_norm']:.6f}")

    for h in [h1, h2] + ahooks + fghooks + atn_hooks + inject_handles:
        if h is not None:
            h.remove()

    # 实验F：落盘 fused-attn 中间量并还原 patch（forward 后）
    if _fused_save is not None:
        _fused_save()
    if _fused_restore is not None:
        _fused_restore()

    # rank 0 保存
    if rank == 0:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if save_act and activations:
            act_path = Path(args.activations_output or
                            str(out_path.with_suffix(".activations.pt")))
            result["activations"] = activations
            torch.save(result, out_path)
            torch.save(activations, act_path)
            logger.info(f"[rank {rank}] activations saved to {act_path} "
                        f"({len(activations)} layers)")
        else:
            torch.save(result, out_path)
        logger.info(f"[rank {rank}] logp saved to {out_path.resolve()}")

        # ── norm 注入实验对比：相同输入下,norm 输出 + 下游 logp 是否对齐 HF ──
        if args.inject_norm_ref:
            ref = torch.load(args.inject_norm_ref, map_location="cpu",
                             weights_only=False)
            print("\n" + "=" * 60)
            print("  NORM INJECT 实验：HF norm 输入 -> torchturbo")
            print("=" * 60)
            # [1] norm 输出对比（核心：相同输入 -> 输出是否一致）
            if "norm_output" in ref and norm_out_captured.get("output") is not None:
                tt_out = norm_out_captured["output"].float().cpu()
                hf_out = ref["norm_output"].float().cpu()
                # 只比真实 token(去掉 SP pad 位)
                tt_out = tt_out[:, :ref_real_len, :]
                hf_out = hf_out[:, :ref_real_len, :]
                d = (tt_out - hf_out).abs()
                denom = hf_out.abs().clamp_min(1e-12)
                print(f"  [1] norm OUTPUT (相同输入,对比 HF norm 输出,真实 {ref_real_len} token)")
                print(f"      shape  : {list(tt_out.shape)}")
                print(f"      max_abs: {d.max().item():.6e}")
                print(f"      mean_abs: {d.mean().item():.6e}")
                print(f"      max_rel: {(d/denom).max().item():.6e}")
                print(f"      bit_id : {torch.equal(tt_out, hf_out)}")
                if d.max().item() < 1e-5:
                    print("      => norm 输出一致：相同输入下 NPU RMSNorm 行为与 HF 相同")
                else:
                    print("      => norm 输出不一致：相同输入下 NPU RMSNorm kernel 行为不同！")
            else:
                print("  [1] 跳过 norm 输出对比（缺少 ref norm_output 或未捕获本侧输出）")

            # [2] 下游 logp 对比（注入输入后,lm_head 输出是否对齐 HF）
            if args.norm_ref_logp:
                hf = torch.load(args.norm_ref_logp, map_location="cpu",
                                weights_only=False)
                hf_lp = hf.get("response_logp")
                tt_lp = result.get("response_logp")
                if hf_lp is not None and tt_lp is not None:
                    hf_lp = hf_lp.float().flatten()
                    tt_lp = tt_lp.float().flatten()
                    n = min(hf_lp.numel(), tt_lp.numel())
                    hf_lp, tt_lp = hf_lp[:n], tt_lp[:n]
                    dl = (tt_lp - hf_lp).abs()
                    print(f"  [2] 下游 logp (注入后 vs HF,前 {n} 个 token)")
                    print(f"      HF  mean_logp : {hf.get('mean_logp','?')}")
                    print(f"      v3  mean_logp : {result.get('mean_logp','?')}")
                    print(f"      max_abs_diff : {dl.max().item():.6e}")
                    print(f"      mean_abs_diff: {dl.mean().item():.6e}")
                    if dl.max().item() < 1e-5:
                        print("      => logp 对齐：norm 注入后下游完全一致")
                    else:
                        print("      => logp 仍有差异：norm 之外(attention/MLA-absorb)还有发散源")
                else:
                    print("  [2] 跳过 logp 对比（缺 response_logp）")
            else:
                print("  [2] 跳过下游 logp 对比（未给 --norm-ref-logp）")

            # 可选 dump
            if args.dump_norm_out:
                dp = Path(args.dump_norm_out)
                dp.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "norm_output_injected": norm_out_captured.get("output"),
                    "response_logp_injected": result.get("response_logp"),
                    "ref_norm_input": ref.get("norm_input"),
                    "ref_norm_output": ref.get("norm_output"),
                }, dp)
                logger.info(f"[rank {rank}] norm-inject dump saved to {dp.resolve()}")
            print("=" * 60)

        # ── 逐层注入实验对比：相同输入下,每层输出是否对齐 HF ──
        if args.inject_layer_ref and layer_out_captured:
            print("\n" + "=" * 60)
            print("  LAYER INJECT 实验：HF layer 输入 -> torchturbo (逐层)")
            print("=" * 60)
            print(f"  {'layer':>5} {'max_abs':>12} {'mean_abs':>12} "
                  f"{'max_rel':>12} {'bit_id':>7}  verdict")
            first_div_layer = None
            for i in sorted(layer_out_captured):
                tt = layer_out_captured[i].float().cpu()[:, :ref_real_len, :]
                if i not in layer_ref_outputs:
                    print(f"  {i:5d}  (无 HF ref 输出,跳过)")
                    continue
                hf = layer_ref_outputs[i].float().cpu()[:, :ref_real_len, :]
                if tt.shape != hf.shape:
                    print(f"  {i:5d}  SHAPE MISMATCH tt={list(tt.shape)} hf={list(hf.shape)}")
                    continue
                d = (tt - hf).abs()
                denom = hf.abs().clamp_min(1e-12)
                max_abs = d.max().item()
                mean_abs = d.mean().item()
                max_rel = (d / denom).max().item()
                bit = torch.equal(tt, hf)
                verdict = "一致" if max_abs < 1e-5 else "<< 不一致"
                if max_abs >= 1e-5 and first_div_layer is None:
                    first_div_layer = i
                print(f"  {i:5d} {max_abs:12.3e} {mean_abs:12.3e} "
                      f"{max_rel:12.3e} {str(bit):>7}  {verdict}")
            print("-" * 60)
            if first_div_layer is None:
                print("  => 所有注入层在相同输入下输出均一致：这些层的实现与 HF 对齐")
            else:
                print(f"  => 首个'相同输入下输出不一致'的层: layer_{first_div_layer}")
                print(f"     -> 该层的实现(NPU fused attn/RMSNorm/MoE)与 HF 不同,")
                print(f"        且差异不来自上游(输入已被 mock 成 HF 的)。")
                if first_div_layer > 0:
                    print(f"     -> 检查 layer_{first_div_layer} 内部: "
                          f"self_attn(MLA-absorb)/post_attn_layernorm/mlp。")
            # 下游 logp 对比(若给了 ref logp)
            if args.norm_ref_logp:
                hf = torch.load(args.norm_ref_logp, map_location="cpu",
                                weights_only=False)
                hf_lp = hf.get("response_logp")
                tt_lp = result.get("response_logp")
                if hf_lp is not None and tt_lp is not None:
                    hf_lp = hf_lp.float().flatten()
                    tt_lp = tt_lp.float().flatten()
                    n = min(hf_lp.numel(), tt_lp.numel())
                    dl = (tt_lp[:n] - hf_lp[:n]).abs()
                    print(f"  下游 logp(注入层后): max_abs={dl.max().item():.3e} "
                          f"mean_abs={dl.mean().item():.3e}")
            print("=" * 60)

        # ── 通用 input 注入对比：相同输入下,子模块输出是否对齐 HF ──
        if _inj is not None:
            _sel, _tgt, _ref_outs, _rrl = _inj
            print("\n" + "=" * 60)
            print(f"  INJECT-REF 实验：HF {_tgt} 输入 -> torchturbo (input 注入)")
            print("=" * 60)
            print(f"  {'layer':>5} {'max_abs':>12} {'mean_abs':>12} "
                  f"{'max_rel':>12} {'bit_id':>7}  verdict")
            _first_div = None
            for _i in sorted(inject_ref_captured):
                _tt = inject_ref_captured[_i].float().cpu()
                if _i not in _ref_outs or _ref_outs[_i] is None:
                    print(f"  {_i:5d}  (无 HF ref 输出,跳过)")
                    continue
                _hf = _ref_outs[_i].float().cpu()
                # 去掉 SP pad 位,只比真实 token
                if _rrl and _tt.dim() == 3:
                    _tt = _tt[:, :_rrl, :]
                    _hf = _hf[:, :_rrl, :]
                if _tt.shape != _hf.shape:
                    print(f"  {_i:5d}  SHAPE MISMATCH tt={list(_tt.shape)} "
                          f"hf={list(_hf.shape)}")
                    continue
                _d = (_tt - _hf).abs()
                _denom = _hf.abs().clamp_min(1e-12)
                _max_abs = _d.max().item()
                _mean_abs = _d.mean().item()
                _max_rel = (_d / _denom).max().item()
                _bit = torch.equal(_tt, _hf)
                _verdict = "一致" if _max_abs < 1e-5 else "<< 不一致"
                if _max_abs >= 1e-5 and _first_div is None:
                    _first_div = _i
                print(f"  {_i:5d} {_max_abs:12.3e} {_mean_abs:12.3e} "
                      f"{_max_rel:12.3e} {str(_bit):>7}  {_verdict}")
            print("-" * 60)
            if _first_div is None:
                print(f"  => 所有注入层在相同输入下 {_tgt} 输出均一致：实现与 HF 对齐")
            else:
                print(f"  => 首个'相同输入下输出不一致': layer_{_first_div} 的 {_tgt}")
                print(f"     -> 差异不来自上游(输入已 mock 成 HF)，"
                      f"{_tgt} 实现与 HF 不同")
            # 下游 logp 对比(若给了 ref logp)
            if args.norm_ref_logp:
                _hf = torch.load(args.norm_ref_logp, map_location="cpu",
                                 weights_only=False)
                _hf_lp = _hf.get("response_logp")
                _tt_lp = result.get("response_logp")
                if _hf_lp is not None and _tt_lp is not None:
                    _hf_lp = _hf_lp.float().flatten()
                    _tt_lp = _tt_lp.float().flatten()
                    _n = min(_hf_lp.numel(), _tt_lp.numel())
                    _dl = (_tt_lp[:_n] - _hf_lp[:_n]).abs()
                    print(f"  下游 logp(注入 {_tgt} 输入后): "
                          f"max_abs={_dl.max().item():.3e} "
                          f"mean_abs={_dl.mean().item():.3e}")
            print("=" * 60)

        # ── 通用 output 注入对比：强制子模块输出=HF 后,下游 logp 是否对齐 ──
        if _frc is not None:
            _sel, _tgt, _ref_outs, _rrl = _frc
            print("\n" + "=" * 60)
            print(f"  FORCE-REF 实验：强制 torchturbo {_tgt} 输出 = HF (output 注入)")
            print("=" * 60)
            for _i in sorted(force_ref_captured):
                _tt = force_ref_captured[_i].float().cpu()
                if _i in _ref_outs and _ref_outs[_i] is not None:
                    _hf = _ref_outs[_i].float().cpu()
                    if _rrl and _tt.dim() == 3:
                        _tt, _hf = _tt[:, :_rrl, :], _hf[:, :_rrl, :]
                    if _tt.shape == _hf.shape:
                        _d = (_tt - _hf).abs()
                        print(f"  layer {_i:3d} {_tgt}: 本侧原输出 vs HF ref "
                              f"max_abs={_d.max().item():.3e} "
                              f"mean_abs={_d.mean().item():.3e}")
            if args.norm_ref_logp:
                _hf = torch.load(args.norm_ref_logp, map_location="cpu",
                                 weights_only=False)
                _hf_lp = _hf.get("response_logp")
                _tt_lp = result.get("response_logp")
                if _hf_lp is not None and _tt_lp is not None:
                    _hf_lp = _hf_lp.float().flatten()
                    _tt_lp = _tt_lp.float().flatten()
                    _n = min(_hf_lp.numel(), _tt_lp.numel())
                    _dl = (_tt_lp[:_n] - _hf_lp[:_n]).abs()
                    print(f"  下游 logp(强制 {_tgt}=HF 后): "
                          f"max_abs={_dl.max().item():.3e} "
                          f"mean_abs={_dl.mean().item():.3e}")
                    if _dl.max().item() < 1e-5:
                        print(f"  => logp 对齐：发散在 {_tgt} 或其上游(强制后即修复)")
                    else:
                        print(f"  => logp 仍有差异：发散在 {_tgt} 下游")
            else:
                print("  (未给 --norm-ref-logp,跳过下游 logp 对比)")
            print("=" * 60)

        print("\n" + "=" * 60)
        print("  TorchTurbo v3 (src engine) Logp Result")
        print("=" * 60)
        print(f"  Model            : {args.path}")
        print(f"  SP/EP            : {args.sp_size}/{args.ep_size}")
        print(f"  dtype            : {args.dtype}")
        print(f"  mixed_precision  : {args.mixed_precision} (bf16->reduce=fp32，v2 为 bf16 —— 头号嫌疑)")
        print(f"  Prompt length    : {result['prompt_length']}")
        print(f"  Response length  : {result['response_length']}")
        print(f"  Mean logp        : {result['mean_logp']:.6f}")
        print(f"  Min/Max logp     : {result['response_logp'].min():.6f} / "
              f"{result['response_logp'].max():.6f}")
        print("=" * 60)
        print("\n  Compare:")
        print(f"    python compare_layer_io.py compare-logp {args.output} <other>.pt")
        if args.save_activations:
            print(f"    python compare_layer_io.py compare-activations "
                  f"{args.activations_output or str(out_path.with_suffix('.activations.pt'))} "
                  f"<other>.activations.pt")

    if dist.is_initialized():
        dist.barrier()
    return 0


if __name__ == "__main__":
    sys.exit(main())
