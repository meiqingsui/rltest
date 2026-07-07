# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""
Extract response-only logp from torchturbo **v2** (torch_turbo: parallize_model).

Sibling of test_hf_logp.py / test_train_logp.py. Output .pt format is identical
so the three (plus test_torchturbo_logp.py for v3) are cross-comparable:

    test_torchturbo2_logp.py  (torch_turbo v2)  ─┐
    test_torchturbo_logp.py   (TorchTurbo v3)   ─┼─> compare_layer_io.py compare-logp
    test_hf_logp.py           (HF baseline)     ─┘

v2 path: build_fsdp_turbo_model(args, fsdp_kwargs, device) -> parallel model,
then model(input_ids=...) forward, hook captures lm_head logits, compute
next-token logp (NOT the PPO loss the backend may compute internally).

Usage:
    # SP=1 EP=1, single process (or torchrun --nproc_per_node=1)
    python test_torchturbo2_logp.py \\
        --path /path/to/glm5 \\
        --test-tokens "12,134,45,10,89,100,200" \\
        --response-length 3 --bf16 \\
        --save-activations --output v2_logp.pt

    # With SP (ulysses): seq_len must be divisible by sp_size (auto-padded)
    torchrun --nproc_per_node=16 test_torchturbo2_logp.py \
        --path /storage/yzr02346555/gyy_Asystem/glm5_mini_bf16  --sp-size 2 --ep-size 16 \
        --test-tokens "12,134,45,10,89" --output v2_logp.pt

    # Add backward + grad dump (to catch FSDP reduce_dtype differences)
    python test_torchturbo2_logp.py ... --with-backward

Compare:
    python compare_layer_io.py compare-logp v2_logp.pt v3_logp.pt
    python compare_layer_io.py compare-activations v2_logp.activations.pt v3_logp.activations.pt
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# 让 compare_layer_io 可被 import（同目录；v3 脚本通过 sys.path 复用）
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


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
    if hasattr(base, "embed_tokens"):
        hooks.append(base.embed_tokens.register_forward_hook(make_hook("embed_tokens")))
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

    # 参数
    for name, p in model.named_parameters():
        # 取首层 + 关键模块，避免 dump 全量过大
        if not any(k in name for k in (
            "layers.0.", "layers.1.", "embed_tokens", "word_embeddings",
            "lm_head", "experts", "router", "gate")):
            continue
        try:
            state[f"param.{name}"] = _full(p)
        except Exception as e:
            logger.warning(f"dump param {name} failed: {e}")

    # buffer（expert_bias / e_score_correction_bias / inv_freq 等）
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
    # 打印摘要：每个 tensor 的 dtype/shape/min/max/mean，便于快速肉眼对比
    logger.info(f"[rank {_rank()}] dumped {len(state)} tensors to {out_file}")
    print(f"\n=== Weights dump summary ({tag}) ===")
    for k in sorted(state.keys())[:40]:
        t = state[k]
        print(f"  {k:60s} dtype={str(t.dtype):16s} shape={list(t.shape)} "
              f"min={t.float().min().item():.6e} max={t.float().max().item():.6e} "
              f"mean={t.float().mean().item():.6e}")
    print(f"  ... (total {len(state)} tensors)")
    return out_file


# --------------------------------------------------------------------------- #
# v2 模型构建
# --------------------------------------------------------------------------- #
def build_v2_model(args, device: torch.device):
    """用 torch_turbo v2 的 build_fsdp_turbo_model 构建并行模型。"""
    from torch_turbo.distributed.parallize_model import build_fsdp_turbo_model
    from torch_turbo.distributed.parallel_state import initialize_model_parallel

    # build_fsdp_turbo_model -> init_mesh 需要 EP group 已建（内部断言
    # get_expert_model_parallel_group()），故先 initialize_model_parallel(ep_size)
    if dist.is_initialized():
        initialize_model_parallel(args.ep_size)

    # 关键：建 Ulysses SP group 并 set_ulysses_sequence_parallel_group
    # 否则 GLM5 的 npu_fused_dsa_attn_forward 内部 get_ulysses_sequence_parallel_group()
    # 返回 None -> sp_size=1，attention 实际未启用 SP（v3 经 setup_sp wrapper 注入，已启用）。
    # 必须用连续 ranks 分组（[0,1],[2,3]...）对齐 v3 _create_sp_group，否则 rank 映射不同仍发散。
    if dist.is_initialized() and args.sp_size > 1:
        from torch_turbo.distributed.parallel_state import (
            set_ulysses_sequence_parallel_group,
        )
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        sp_group = None
        for start in range(0, world_size, args.sp_size):
            ranks = list(range(start, start + args.sp_size))
            grp = dist.new_group(ranks)
            if rank in ranks:
                sp_group = grp
        set_ulysses_sequence_parallel_group(sp_group)
        logger.info(f"[_rank {_rank()}] v2 SP group built: sp_size={args.sp_size}, "
                    f"sp_rank={dist.get_rank(sp_group)}")

    # build_fsdp_turbo_model 期望 args 带 world_size/sp_size/ep_size 等字段
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    tt_args = SimpleNamespace(
        path=args.path,
        sp_size=args.sp_size,
        ep_size=args.ep_size,
        world_size=world_size,
        average_in_collective=True,
        sp_mode="ulysses",
        use_fused_swiglu=False,
        use_fused_rope=False,
        use_fused_rms_norm=False,
        use_fused_permute_unpermute=False,
        use_transformer_engine=False,
        moe_apply_probs_on_input=False,
    )
    # fsdp_kwargs：v2 无 mp_policy 抽象，全 bf16（对比 v3 的 reduce_dtype=fp32）
    # 注意：不能传 mp_policy=None，FSDP2 _init_mp_dtypes 会崩（访问 None.param_dtype）。
    # 不传 mp_policy key -> FSDP2 用默认（param/reduce 沿用 dtype=bf16），即 v2 真实行为。
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    fsdp_kwargs = {"reshard_after_forward": True}
    model = build_fsdp_turbo_model(tt_args, fsdp_kwargs, device)
    model = model.to(dtype=dtype) if hasattr(model, "to") else model
    model.eval()
    logger.info(f"[_rank {_rank()}] v2 model built (dtype={dtype})")
    return model


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract response-only logp from torchturbo v2 (torch_turbo)")
    parser.add_argument("--path", required=True, help="模型权重路径")
    parser.add_argument("--test-tokens", type=str, default="10,11,12,13,14,15,16,17",
                        help="逗号分隔 token id 或文件路径")
    parser.add_argument("--response-length", type=int, default=None,
                        help="response 长度，默认除首 token 外全为 response")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--sp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="v2_logp_result.pt")
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

    # 构建模型
    model = build_v2_model(args, device)

    # dump 加载后权重（定位 v2/v3 权重值差异）
    if args.dump_weights:
        dump_model_weights(model, args.dump_weights, tag=f"v2_rank{rank}")

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
                gn = float(g.norm().item())
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
        print("  TorchTurbo v2 Logp Result")
        print("=" * 60)
        print(f"  Model            : {args.path}")
        print(f"  SP/EP            : {args.sp_size}/{args.ep_size}")
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
