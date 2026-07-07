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

    # dump 加载后权重（定位 v2/v3 权重值差异）
    if args.dump_weights:
        dump_model_weights(model, args.dump_weights, tag=f"v3_rank{rank}")

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
    ahooks = register_activation_hooks(model, activations) if args.save_activations else []

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

    for h in [h1, h2] + ahooks:
        if h is not None:
            h.remove()

    # rank 0 保存
    if rank == 0:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if args.save_activations and activations:
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
