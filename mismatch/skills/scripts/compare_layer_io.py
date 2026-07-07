"""逐层比对工具：定位 torchturbo v2 与 v3(src 引擎) 训练数值差异。

设计
====
本工具不假设模型如何被分布式包装。它提供三类能力：

1. 库 API（插入真实训练流程，推荐用法）
   在 v2 训练脚本里 forward+backward 之后，在 v3 训练脚本里同样位置：

       from compare_layer_io import attach_io_hooks, save_io, snapshot_grads

       attach_io_hooks(model, "/tmp/io/v2")          # 注册 forward I/O hook
       out = model(...); loss = ...; loss.backward()
       save_io()                                       # 落盘 forward I/O
       snapshot_grads(model, "/tmp/io/v2")            # 落盘参数梯度

   这样能抓到 v2/v3 各自真实路径（FSDP2 mp_policy / SP / EP / weight_loader）
   产生的数值差异，正是单卡脚本复现不了的部分。

2. CLI dump（自包含，验证工具链 / native baseline）
       python compare_layer_io.py dump --path MODEL --out /tmp/io/native --num-layers 4

   用原生 GlmMoeDsaForCausalLM 跑一次 forward+backward，验证权重与输入一致。
   注意：原生 MoE 不走 torch_turbo 的 topk_softmax_with_capacity，因此 native
   模式下 v2/v3 输出应完全一致——它证明的是“权重+输入”基线，不是路由差异。

3. CLI compare（离线对比两个 dump 目录）
       python compare_layer_io.py compare /tmp/io/v2 /tmp/io/v3

   逐层、逐子模块、逐 tensor 给出 max_abs_diff / mean_abs_diff / max_rel_diff，
   标记首个超过阈值的发散点，并区分 forward I/O 差异与梯度差异。

能定位 / 不能定位
=================
能：模型实现差异（MoE 路由 dtype、cast_forward_inputs、buffer dtype）、
    MoE 内部量（router probs / routing_map / permuted_probs）、反向梯度。
不能：纯分布式通信精度（FSDP reduce_dtype all-reduce、SP/EP 通信）——
      这些只有在真实多卡流程里用库 API dump 才抓得到，单卡 CLI 复现不了。

依赖
====
safetensors、torch、transformers（仅 dump CLI 建 model 时）。compare 命令只需
safetensors + torch。两份 dump 必须分别在各仓库环境跑（import torch_turbo 路径
不同），再在任意环境 compare。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from safetensors.torch import load_file, save_file


# 每层要 hook 的子模块名（按 hasattr 探测，兼容原生 GlmMoeDsaMoE 与各类 MoELayer）
LAYER_SUBMODULES = [
    "input_layernorm",
    "self_attn",
    "post_attention_layernorm",
    "mlp",
]
ATTN_SUBMODULES = [
    "q_proj", "q_a_proj", "q_a_layernorm", "q_b_proj",
    "kv_a_proj_with_mqa", "kv_a_layernorm", "kv_b_proj",
    "o_proj", "indexer",
]
INDEXER_SUBMODULES = ["wq_b", "wk", "k_norm", "weights_proj"]
# MoE 内部子模块（探测式 hook，抓路由 probs / dispatch 等中间量）
MOE_SUBMODULES = ["router", "gate", "token_dispatcher", "experts",
                  "shared_experts", "shared_expert", "shared_expert_gate", "moe"]


# --------------------------------------------------------------------------- #
# tensor 提取与统计
# --------------------------------------------------------------------------- #
def _extract_tensors(value: Any, prefix: str = "") -> List[Tuple[str, Any]]:
    """把任意返回值拆成 (name, tensor|scalar) 列表，tensor 已 detach+cpu+clone。"""
    out: List[Tuple[str, Any]] = []
    if isinstance(value, torch.Tensor):
        out.append((prefix, value.detach().cpu().clone()))
    elif isinstance(value, (tuple, list)):
        for i, sub in enumerate(value):
            name = f"{prefix}_{i}" if prefix else str(i)
            out.extend(_extract_tensors(sub, name))
    elif isinstance(value, dict):
        for k, v in value.items():
            name = f"{prefix}_{k}" if prefix else str(k)
            out.extend(_extract_tensors(v, name))
    elif value is None:
        out.append((prefix, None))
    elif isinstance(value, (int, float, bool)):
        out.append((prefix, value))
    return out


def _forward_param_names(module: torch.nn.Module) -> List[str]:
    import inspect
    try:
        return list(inspect.signature(module.forward).parameters.keys())
    except (ValueError, TypeError):
        return []


def _tensor_stats(t: torch.Tensor) -> Dict[str, Any]:
    """记录原始 dtype 与统计量，便于对比 bf16/fp32 产生来源。"""
    tt = t.detach()
    flat = tt.float().flatten()
    finite = flat[torch.isfinite(flat)]
    return {
        "dtype": str(tt.dtype),
        "shape": list(tt.shape),
        "min": float(flat.min().item()) if flat.numel() else 0.0,
        "max": float(flat.max().item()) if flat.numel() else 0.0,
        "mean": float(flat.mean().item()) if flat.numel() else 0.0,
        "has_nan": bool(torch.isnan(tt).any().item()),
        "has_inf": bool(torch.isinf(tt).any().item()),
    }


# --------------------------------------------------------------------------- #
# IO Hook（库 API 核心）
# --------------------------------------------------------------------------- #
class LayerIOHook:
    """注册 forward pre/post hook，捕获每层及子模块的输入输出张量。

    forward 阶段攒在内存，save() 时落盘（float32 保精度 + meta.json 记 dtype）。
    """

    def __init__(self, out_dir: str, tag: str = ""):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.tag = tag
        self.handles: List[Any] = []
        self.module_io: Dict[str, Dict[str, List]] = {}
        self._meta: Dict[str, Dict[str, Any]] = {}

    # -- hook 工厂 --
    def _make_pre_hook(self, name: str):
        def pre_hook(module, args, kwargs):
            names = _forward_param_names(module)
            tensors: List[Tuple[str, Any]] = []
            for i, a in enumerate(args):
                pn = names[i] if i < len(names) else f"arg_{i}"
                tensors.extend(_extract_tensors(a, pn))
            for k, v in kwargs.items():
                tensors.extend(_extract_tensors(v, k))
            self.module_io.setdefault(name, {})["input"] = tensors
        return pre_hook

    def _make_post_hook(self, name: str):
        def post_hook(module, args, kwargs, output):
            tensors = _extract_tensors(output, "output")
            self.module_io.setdefault(name, {}).setdefault("input", [])
            self.module_io[name]["output"] = tensors
        return post_hook

    # -- 注册 --
    def _hook_module(self, module: torch.nn.Module, name: str):
        self.handles.append(module.register_forward_pre_hook(
            self._make_pre_hook(name), with_kwargs=True))
        self.handles.append(module.register_forward_hook(
            self._make_post_hook(name), with_kwargs=True))

    def _try_hook(self, parent: torch.nn.Module, attr: str, full_name: str) -> bool:
        if hasattr(parent, attr):
            self._hook_module(getattr(parent, attr), full_name)
            return True
        return False

    def register(self, model: torch.nn.Module):
        """在所有 decoder layer 及其关键子模块上注册 hook。"""
        layers = None
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "layers"):
            layers = model.layers
        if layers is None:
            print("[compare_layer_io] WARNING: no .layers found, hooking root only")
            self._hook_module(model, "root")
            return
        print(f"[compare_layer_io] hooking {len(layers)} layers")
        for idx, layer in enumerate(layers):
            self._hook_module(layer, f"layer_{idx}")
            for sub in LAYER_SUBMODULES:
                self._try_hook(layer, sub, f"layer_{idx}.{sub}")
            # attention 内部
            if hasattr(layer, "self_attn"):
                attn = layer.self_attn
                for sub in ATTN_SUBMODULES:
                    self._try_hook(attn, sub, f"layer_{idx}.self_attn.{sub}")
                if hasattr(attn, "indexer"):
                    idxr = attn.indexer
                    for sub in INDEXER_SUBMODULES:
                        self._try_hook(idxr, sub,
                                       f"layer_{idx}.self_attn.indexer.{sub}")
            # MoE 内部（路由 / dispatch / experts 输出）
            if hasattr(layer, "mlp"):
                mlp = layer.mlp
                for sub in MOE_SUBMODULES:
                    self._try_hook(mlp, sub, f"layer_{idx}.mlp.{sub}")

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    # -- 落盘 --
    def save(self):
        save_dict: Dict[str, torch.Tensor] = {}
        meta: Dict[str, Dict[str, Any]] = {}
        for mod_name, io in self.module_io.items():
            for kind in ("input", "output"):
                for tname, val in io.get(kind, []):
                    key = f"{mod_name}.{kind}.{tname}"
                    if isinstance(val, torch.Tensor):
                        save_dict[key] = val.to(torch.float32).contiguous()
                        meta[key] = _tensor_stats(val)
                    else:
                        meta[key] = {"scalar": str(val)}
        if save_dict:
            save_file(save_dict, str(self.out_dir / "io.safetensors"))
        with open(self.out_dir / "io_meta.json", "w") as f:
            json.dump({"tag": self.tag, "tensors": meta}, f, indent=2)
        print(f"[compare_layer_io] saved {len(save_dict)} tensors to {self.out_dir}")


def attach_io_hooks(model: torch.nn.Module, out_dir: str, tag: str = "") -> LayerIOHook:
    """库 API：注册 hook。forward 后调 save_io(hook) 落盘。"""
    hook = LayerIOHook(out_dir, tag)
    hook.register(model)
    # 暴露到模块级以便 save_io 取用
    _LAST_HOOK[0] = hook
    return hook


_LAST_HOOK: List[Optional[LayerIOHook]] = [None]


def save_io(hook: Optional[LayerIOHook] = None):
    """库 API：把已捕获的 forward I/O 落盘。"""
    hook = hook or _LAST_HOOK[0]
    if hook is None:
        raise RuntimeError("no active LayerIOHook; call attach_io_hooks first")
    hook.save()


def snapshot_grads(model: torch.nn.Module, out_dir: str, tag: str = ""):
    """库 API：把当前 .grad 落盘（含 dtype meta）。backward 之后调用。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_dict: Dict[str, torch.Tensor] = {}
    meta: Dict[str, Dict[str, Any]] = {}
    for name, p in model.named_parameters():
        g = p.grad
        if g is None:
            meta[name] = {"grad": None}
            continue
        key = f"grad.{name}"
        save_dict[key] = g.detach().to(torch.float32).cpu().contiguous()
        meta[key] = {**_tensor_stats(g), "param_dtype": str(p.dtype)}
    if save_dict:
        save_file(save_dict, str(out_dir / "grads.safetensors"))
    with open(out_dir / "grads_meta.json", "w") as f:
        json.dump({"tag": tag, "tensors": meta}, f, indent=2)
    print(f"[compare_layer_io] saved {len(save_dict)} grads to {out_dir}")


# --------------------------------------------------------------------------- #
# Compare
# --------------------------------------------------------------------------- #
DEFAULT_ABS_TOL = 1e-5
DEFAULT_REL_TOL = 1e-3


def _load_dump(d: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    io, meta = {}, {}
    io_file = d / "io.safetensors"
    if io_file.exists():
        io = load_file(str(io_file))
    grad_file = d / "grads.safetensors"
    if grad_file.exists():
        io.update(load_file(str(grad_file)))
    meta_file = d / "io_meta.json"
    if meta_file.exists():
        with open(meta_file) as f:
            meta.update(json.load(f).get("tensors", {}))
    gmeta_file = d / "grads_meta.json"
    if gmeta_file.exists():
        with open(gmeta_file) as f:
            meta.update(json.load(f).get("tensors", {}))
    return io, meta


def _layer_key(key: str) -> int:
    """'layer_2.mlp.router.output_0' -> 2；非 layer 键返回 -1。"""
    parts = key.split(".")
    if parts and parts[0].startswith("layer_"):
        try:
            return int(parts[0].split("_")[1])
        except (IndexError, ValueError):
            return -1
    return 9999  # 非 layer 键（如 logits）排到最后


def _diff(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float, float]:
    a = a.float().flatten()
    b = b.float().flatten()
    if a.shape != b.shape:
        return float("nan"), float("nan"), float("nan")
    # 把 nan/inf 替换为 0，避免污染统计
    mask = torch.isfinite(a) & torch.isfinite(b)
    a, b = a[mask], b[mask]
    d = (a - b).abs()
    max_abs = float(d.max().item()) if d.numel() else 0.0
    mean_abs = float(d.mean().item()) if d.numel() else 0.0
    denom = b.abs().clamp_min(1e-12)
    max_rel = float((d / denom).max().item()) if d.numel() else 0.0
    return max_abs, mean_abs, max_rel


def compare_dirs(d1: str, d2: str,
                 abs_tol: float = DEFAULT_ABS_TOL,
                 rel_tol: float = DEFAULT_REL_TOL) -> Dict[str, Any]:
    """对比两个 dump 目录，返回结构化结果并打印。

    返回 dict 含：
      - first_divergence: 首个超阈值的 key（按层序），None 表示全一致
      - summary: 每层 max_abs_diff
      - details: 逐 key 详情
    """
    d1, d2 = Path(d1), Path(d2)
    t1, m1 = _load_dump(d1)
    t2, m2 = _load_dump(d2)
    keys = sorted(set(t1) | set(t2), key=lambda k: (_layer_key(k), k))

    print(f"\n=== Compare {d1}  vs  {d2} ===")
    print(f"    tensors: {len(t1)} vs {len(t2)}  (union {len(keys)})")
    print(f"    abs_tol={abs_tol:.2e}  rel_tol={rel_tol:.2e}\n")

    first_div: Optional[str] = None
    summary: Dict[str, float] = {}
    details: List[Dict[str, Any]] = []
    diverged_count = 0

    # 按层分组打印
    cur_layer = None
    for k in keys:
        lk = _layer_key(k)
        if lk != cur_layer:
            cur_layer = lk
            hdr = f"[layer {lk}]" if 0 <= lk < 9999 else "[global]"
            print(f"\n{hdr}")

        if k in t1 and k in t2:
            max_abs, mean_abs, max_rel = _diff(t1[k], t2[k])
            diverged = (max_abs > abs_tol) or (max_rel > rel_tol)
            flag = "  << DIFF" if diverged else ""
            print(f"  {k:60s} max={max_abs:.3e} mean={mean_abs:.3e} "
                  f"rel={max_rel:.3e}{flag}")
            if diverged:
                diverged_count += 1
                if first_div is None:
                    first_div = k
                # 打印两端 dtype/统计，直击 bf16/fp32 来源
                s1 = m1.get(k, {})
                s2 = m2.get(k, {})
                print(f"      d1: dtype={s1.get('dtype','?')} "
                      f"min={s1.get('min',0):.3e} max={s1.get('max',0):.3e}")
                print(f"      d2: dtype={s2.get('dtype','?')} "
                      f"min={s2.get('min',0):.3e} max={s2.get('max',0):.3e}")
            summary[str(lk)] = max(summary.get(str(lk), 0.0), max_abs)
            details.append({"key": k, "max_abs": max_abs, "mean_abs": mean_abs,
                            "max_rel": max_rel, "diverged": diverged})
        elif k in t1:
            print(f"  {k:60s}  ONLY in d1")
            details.append({"key": k, "only": "d1"})
        else:
            print(f"  {k:60s}  ONLY in d2")
            details.append({"key": k, "only": "d2"})

    print("\n" + "=" * 60)
    if first_div:
        print(f">>> FIRST DIVERGENCE: {first_div}")
    else:
        print(">>> No divergence above tolerance (implementations match)")
    print(f"    diverged tensors: {diverged_count}/{len(keys)}")
    print("=" * 60)
    return {"first_divergence": first_div, "summary": summary, "details": details}


# --------------------------------------------------------------------------- #
# Compare logp / activations .pt（test_*_logp.py 输出格式）
# --------------------------------------------------------------------------- #
def _load_pt(path: str) -> Dict[str, Any]:
    """加载 test_*_logp.py 输出的 .pt（torch.save 的 dict）。"""
    return torch.load(path, map_location="cpu", weights_only=False)


def compare_logp(pt1: str, pt2: str,
                 abs_tol: float = 1e-5,
                 rel_tol: float = 1e-3) -> Dict[str, Any]:
    """对比两个 test_*_logp.py 的 .pt：response_logp 逐 token diff。

    期望 .pt 含 response_logp / response_tokens / mean_logp / input_token_ids。
    返回 first_divergence_token（首个超阈值 token index）。
    """
    d1, d2 = _load_pt(pt1), _load_pt(pt2)
    lp1 = d1.get("response_logp")
    lp2 = d2.get("response_logp")
    if lp1 is None or lp2 is None:
        print(f"[compare_logp] 缺 response_logp: d1={'有' if lp1 is not None else '无'} "
              f"d2={'有' if lp2 is not None else '无'}")
        return {"error": "missing response_logp"}

    lp1 = lp1.float().flatten()
    lp2 = lp2.float().flatten()
    n = min(lp1.numel(), lp2.numel())
    print(f"\n=== Compare logp: {pt1}  vs  {pt2} ===")
    print(f"    response_logp: {lp1.numel()} vs {lp2.numel()} (compare {n})")
    print(f"    d1 mean_logp={d1.get('mean_logp','?')}")
    print(f"    d2 mean_logp={d2.get('mean_logp','?')}")

    if lp1.numel() != lp2.numel():
        lp1, lp2 = lp1[:n], lp2[:n]
        print(f"    ⚠ 长度不同，截断到 {n}")

    a, b = lp1, lp2
    abs_d = (a - b).abs()
    rel_d = abs_d / b.abs().clamp_min(1e-12)
    max_abs = float(abs_d.max().item())
    mean_abs = float(abs_d.mean().item())
    max_rel = float(rel_d.max().item())

    print(f"    max_abs={max_abs:.3e}  mean_abs={mean_abs:.3e}  max_rel={max_rel:.3e}")

    # 找首个发散 token
    div_mask = (abs_d > abs_tol) | (rel_d > rel_tol)
    first_tok = int(div_mask.nonzero()[0].item()) if div_mask.any() else -1

    print(f"\n  逐 token (前 {min(20, n)} 个):")
    print(f"    {'idx':>4} {'d1_logp':>12} {'d2_logp':>12} {'abs_diff':>12} {'rel_diff':>12}")
    for i in range(min(20, n)):
        flag = "  << DIFF" if bool(div_mask[i]) else ""
        print(f"    {i:4d} {a[i].item():12.6f} {b[i].item():12.6f} "
              f"{abs_d[i].item():12.3e} {rel_d[i].item():12.3e}{flag}")

    print("\n" + "=" * 60)
    if first_tok >= 0:
        print(f">>> FIRST DIVERGENT TOKEN: idx={first_tok}  "
              f"d1={a[first_tok].item():.6f} d2={b[first_tok].item():.6f}")
    else:
        print(">>> No divergence above tolerance (logp match)")
    # response_tokens 一致性校验（输入是否对齐）
    rt1, rt2 = d1.get("response_tokens"), d2.get("response_tokens")
    if rt1 is not None and rt2 is not None:
        tok_match = bool(torch.equal(rt1.flatten()[:n], rt2.flatten()[:n]))
        print(f"    response_tokens match: {tok_match}  "
              f"(若 False 说明输入 token 不一致，logp 对比无意义)")
    print("=" * 60)
    return {"first_divergent_token": first_tok, "max_abs": max_abs,
            "mean_abs": mean_abs, "max_rel": max_rel}


def _act_layer_idx(name: str) -> int:
    """'layer_3' -> 3；'embed_tokens'/'norm'/'lm_head' 排序辅助。"""
    if name.startswith("layer_"):
        try:
            return int(name.split("_")[1])
        except (IndexError, ValueError):
            return -1
    return {"embed_tokens": -2, "norm": 10000, "lm_head": 10001}.get(name, 9999)


def compare_activations_pt(pt1: str, pt2: str,
                          abs_tol: float = DEFAULT_ABS_TOL,
                          rel_tol: float = DEFAULT_REL_TOL) -> Dict[str, Any]:
    """对比两个 .activations.pt（test_*_logp.py --save-activations 输出）。

    期望 {name: {"output": Tensor, "output_shape": tuple}} 或直接 {name: Tensor}。
    逐层 output tensor diff，标记首个发散层。
    """
    d1, d2 = _load_pt(pt1), _load_pt(pt2)
    names = sorted(set(d1) | set(d2), key=_act_layer_idx)
    print(f"\n=== Compare activations: {pt1}  vs  {pt2} ===")
    print(f"    layers: {len(d1)} vs {len(d2)} (union {len(names)})")
    print(f"    abs_tol={abs_tol:.2e}  rel_tol={rel_tol:.2e}\n")

    first_div = None
    diverged = 0
    for name in names:
        t1 = d1.get(name)
        t2 = d2.get(name)
        # 兼容 {"output": T} 和直接 T 两种格式
        if isinstance(t1, dict):
            t1 = t1.get("output")
        if isinstance(t2, dict):
            t2 = t2.get("output")
        if not isinstance(t1, torch.Tensor) or not isinstance(t2, torch.Tensor):
            only = "d1" if isinstance(t1, torch.Tensor) else ("d2" if isinstance(t2, torch.Tensor) else "none")
            print(f"  {name:20s}  skip (only {only})")
            continue
        max_abs, mean_abs, max_rel = _diff(t1, t2)
        is_div = (max_abs > abs_tol) or (max_rel > rel_tol)
        flag = "  << DIFF" if is_div else ""
        print(f"  {name:20s} shape={list(t1.shape)} max={max_abs:.3e} "
              f"mean={mean_abs:.3e} rel={max_rel:.3e}{flag}")
        if is_div:
            diverged += 1
            if first_div is None:
                first_div = name
                print(f"      d1: dtype={t1.dtype} min={t1.float().min():.3e} "
                      f"max={t1.float().max():.3e}")
                print(f"      d2: dtype={t2.dtype} min={t2.float().min():.3e} "
                      f"max={t2.float().max():.3e}")

    print("\n" + "=" * 60)
    if first_div:
        print(f">>> FIRST DIVERGENT LAYER: {first_div}")
    else:
        print(">>> No divergence above tolerance (activations match)")
    print(f"    diverged layers: {diverged}/{len(names)}")
    print("=" * 60)
    return {"first_divergent_layer": first_div, "diverged": diverged,
            "total": len(names)}


def compare_weights(pt1: str, pt2: str,
                    abs_tol: float = DEFAULT_ABS_TOL,
                    rel_tol: float = DEFAULT_REL_TOL) -> Dict[str, Any]:
    """对比两个权重 dump .pt（test_*_logp.py --dump-weights 输出）。

    逐 tensor 对比 param/buffer 的数值，标记首个不一致的权重。
    用于定位 v2/v3 加载后权重值是否一致。
    """
    d1, d2 = _load_pt(pt1), _load_pt(pt2)
    keys = sorted(set(d1) | set(d2))
    print(f"\n=== Compare weights: {pt1}  vs  {pt2} ===")
    print(f"    tensors: {len(d1)} vs {len(d2)} (union {len(keys)})")
    print(f"    abs_tol={abs_tol:.2e}  rel_tol={rel_tol:.2e}\n")

    first_div = None
    diverged = 0
    only_d1, only_d2 = [], []
    for k in keys:
        in1, in2 = k in d1, k in d2
        if not (in1 and in2):
            (only_d1 if in1 else only_d2).append(k)
            continue
        t1, t2 = d1[k], d2[k]
        if not (isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor)):
            continue
        if t1.shape != t2.shape:
            print(f"  {k:60s} SHAPE MISMATCH {list(t1.shape)} vs {list(t2.shape)}  << DIFF")
            if first_div is None:
                first_div = k
            diverged += 1
            continue
        max_abs, mean_abs, max_rel = _diff(t1, t2)
        is_div = (max_abs > abs_tol) or (max_rel > rel_tol)
        flag = "  << DIFF" if is_div else ""
        print(f"  {k:60s} dtype={str(t1.dtype):10s} max_abs={max_abs:.3e} "
              f"mean_abs={mean_abs:.3e} max_rel={max_rel:.3e}{flag}")
        if is_div:
            diverged += 1
            if first_div is None:
                first_div = k
                print(f"      d1: min={t1.float().min():.6e} max={t1.float().max():.6e} "
                      f"mean={t1.float().mean():.6e}")
                print(f"      d2: min={t2.float().min():.6e} max={t2.float().max():.6e} "
                      f"mean={t2.float().mean():.6e}")

    if only_d1:
        print(f"\n  ONLY in d1 ({len(only_d1)}): {only_d1[:10]}")
    if only_d2:
        print(f"\n  ONLY in d2 ({len(only_d2)}): {only_d2[:10]}")

    print("\n" + "=" * 60)
    if first_div:
        print(f">>> FIRST DIVERGENT WEIGHT: {first_div}")
    else:
        print(">>> All weights match (加载后权重值一致)")
    print(f"    diverged tensors: {diverged}/{len(keys)}")
    print("=" * 60)
    return {"first_divergent_weight": first_div, "diverged": diverged,
            "total": len(keys)}


# --------------------------------------------------------------------------- #
# CLI dump（自包含 native baseline）
# --------------------------------------------------------------------------- #
def _build_native_model(path: str, device: str, dtype: torch.dtype,
                        num_layers: Optional[int]):
    """加载原生 GlmMoeDsaForCausalLM（用当前环境 import 到的 torch_turbo）。"""
    import json as _json
    from torch_turbo.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaConfig, GlmMoeDsaForCausalLM)
    from safetensors.torch import load_file as _load_file

    cfg_path = Path(path) / "config.json"
    with open(cfg_path) as f:
        cfg_dict = _json.load(f)
    if num_layers is not None:
        cfg_dict["num_hidden_layers"] = num_layers
    config = GlmMoeDsaConfig(**cfg_dict)
    with torch.device("cpu"):
        model = GlmMoeDsaForCausalLM(config)
    model.to_empty(device=device)
    model = model.to(dtype=dtype)

    # 简化版权重加载：仅按 weight_map 逐 param copy
    idx = Path(path).glob("*safetensors.index.json")
    idx = list(idx)[0]
    with open(idx) as f:
        weight_map = _json.load(f)["weight_map"]
    loaded: Dict[str, dict] = {}

    def get_file(fn):
        if fn not in loaded:
            loaded[fn] = _load_file(str(Path(path) / fn), device="cpu")
        return loaded[fn]

    params = dict(model.named_parameters())
    bufs = dict(model.named_buffers())
    for name, p in params.items():
        if name not in weight_map:
            continue
        sd = get_file(weight_map[name])
        if name in sd and sd[name].shape == p.shape:
            p.data.copy_(sd[name].to(dtype).to(device))
    for name, b in bufs.items():
        if name not in weight_map:
            continue
        sd = get_file(weight_map[name])
        if name in sd and sd[name].shape == b.shape:
            b.copy_(sd[name].to(b.dtype).to(device))
    loaded.clear()
    print(f"[dump] loaded model, dtype={next(model.parameters()).dtype}")
    return model


def cmd_dump(args):
    torch.manual_seed(0)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}[args.dtype]
    model = _build_native_model(args.path, args.device, dtype, args.num_layers)
    model.eval()

    if args.replace_moe:
        print("[dump] --replace-moe: 替换为 MoELayer 需真实分布式 init，"
              "请在训练流程用库 API。此处跳过，跑 native。")

    hook = attach_io_hooks(model, args.out, tag=args.tag)

    print("[dump] forward + backward ...")
    input_ids = torch.randint(1, 1000, (1, args.seq_len), device=args.device)
    labels = input_ids.clone()
    out = model(input_ids, attention_mask=None, labels=labels)
    loss = out.loss if hasattr(out, "loss") else out[0].mean()
    print(f"[dump] loss = {loss.item():.6e}")
    if not args.no_backward:
        loss.backward()
    save_io(hook)
    if not args.no_backward:
        snapshot_grads(model, args.out, tag=args.tag)
    hook.remove()
    print(f"[dump] done -> {args.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pd = sub.add_parser("dump", help="自包含 native dump（验证工具链）")
    pd.add_argument("--path", required=True)
    pd.add_argument("--out", required=True)
    pd.add_argument("--device", default="cpu")
    pd.add_argument("--dtype", default="bfloat16",
                    choices=["float32", "float16", "bfloat16"])
    pd.add_argument("--num-layers", type=int, default=None,
                    help="覆盖 config.num_hidden_layers（调试用）")
    pd.add_argument("--seq-len", type=int, default=512)
    pd.add_argument("--tag", default="")
    pd.add_argument("--replace-moe", action="store_true",
                    help="（占位）替换为 MoELayer，需分布式环境，请改用库 API")
    pd.add_argument("--no-backward", action="store_true")
    pd.set_defaults(func=cmd_dump)

    pc = sub.add_parser("compare", help="对比两个 dump 目录")
    pc.add_argument("dir1")
    pc.add_argument("dir2")
    pc.add_argument("--abs-tol", type=float, default=DEFAULT_ABS_TOL)
    pc.add_argument("--rel-tol", type=float, default=DEFAULT_REL_TOL)
    pc.set_defaults(func=lambda a: compare_dirs(a.dir1, a.dir2,
                                                 a.abs_tol, a.rel_tol))

    pl = sub.add_parser("compare-logp",
                        help="对比两个 test_*_logp.py 的 .pt (response_logp)")
    pl.add_argument("pt1")
    pl.add_argument("pt2")
    pl.add_argument("--abs-tol", type=float, default=1e-5)
    pl.add_argument("--rel-tol", type=float, default=1e-3)
    pl.set_defaults(func=lambda a: compare_logp(a.pt1, a.pt2,
                                                a.abs_tol, a.rel_tol))

    pa = sub.add_parser("compare-activations",
                        help="对比两个 .activations.pt (逐层 output tensor)")
    pa.add_argument("pt1")
    pa.add_argument("pt2")
    pa.add_argument("--abs-tol", type=float, default=DEFAULT_ABS_TOL)
    pa.add_argument("--rel-tol", type=float, default=DEFAULT_REL_TOL)
    pa.set_defaults(func=lambda a: compare_activations_pt(a.pt1, a.pt2,
                                                          a.abs_tol, a.rel_tol))

    pw = sub.add_parser("compare-weights",
                        help="对比两个 --dump-weights 输出的权重 .pt")
    pw.add_argument("pt1")
    pw.add_argument("pt2")
    pw.add_argument("--abs-tol", type=float, default=DEFAULT_ABS_TOL)
    pw.add_argument("--rel-tol", type=float, default=DEFAULT_REL_TOL)
    pw.set_defaults(func=lambda a: compare_weights(a.pt1, a.pt2,
                                                   a.abs_tol, a.rel_tol))
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
