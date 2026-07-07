# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""
Script to extract response-only logp from HuggingFace Transformers (native HF inference).

This serves as a clean, third-party baseline for Train-Inference Mismatch debugging:
- test_train_logp.py  (Megatron-LM)  vs  test_hf_logp.py  → isolate Megatron-specific deviations
- test_rollout_logp.py (SGLang)      vs  test_hf_logp.py  → isolate SGLang-specific deviations

Input/output format is kept identical to test_train_logp.py for easy interchangeability.

pip install accelerate
Usage:
    # Single-GPU / single-process
    python test_hf_logp.py \
        --hf-checkpoint /mnt/sfs_turbo/models/Qwen3.5-9B/ \
        --test-tokens "12, 134, 45, 10, 89, 100, 200, 300" \
        --response-length 3 \
        --bf16 \
        --output hf_logp_result.pt

    # Multi-GPU via torchrun (each rank holds a full copy; data-parallel style)

    torchrun --nproc_per_node=1 test_hf_logp.py \
    --hf-checkpoint /storage/yzr02346555/gyy_Asystem/glm5_mini_bf16 \
    --test-tokens "12,134,45,10,89" \
    --response-length 4 \
    --bf16 \
    --save-activations \
    --output hf_logp_result.pt
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_test_tokens(arg: str) -> list[int]:
    """Parse --test-tokens: a comma-separated string OR a path to a token file.

    A value that points to an existing file is loaded from the file; otherwise
    it is parsed inline as a comma-separated token list. Supported file formats:
      - .txt/.csv : any integers in the text (one per line, or comma/space/
                    newline separated; optional brackets are fine)
      - .json     : a list of ints, or an object with a tokens/input_ids/
                    input_token_ids/response_tokens field
      - .pt/.pth  : a tensor/list of ints, or a saved result dict with
                    input_token_ids/response_tokens (round-trippable from the
                    other test_*_logp.py outputs)
    """
    import os
    value = arg.strip()
    if value and os.path.isfile(value):
        return _load_tokens_from_file(value)
    parts = [p.strip() for p in value.split(",") if p.strip()]
    try:
        return [int(p) for p in parts]
    except ValueError as e:
        raise ValueError(
            f"--test-tokens={arg!r} is neither an existing file nor a "
            f"comma-separated list of integers."
        ) from e


def _load_tokens_from_file(path: str) -> list[int]:
    import json
    import os
    import re

    def _coerce(data) -> list[int]:
        if isinstance(data, dict):
            for k in ("tokens", "token_ids", "input_ids", "input_token_ids", "response_tokens"):
                if k in data:
                    data = data[k]
                    break
            else:
                for v in data.values():
                    if isinstance(v, (list, tuple)) or hasattr(v, "flatten"):
                        data = v
                        break
        if hasattr(data, "flatten") and hasattr(data, "tolist"):  # torch/numpy tensor
            data = data.flatten().tolist()
        if isinstance(data, (list, tuple)):
            return [int(t) for t in data]
        raise ValueError(f"Unsupported token content in {path!r}: {type(data).__name__}")

    suffix = os.path.splitext(path)[1].lower()
    if suffix in (".pt", ".pth"):
        import torch
        return _coerce(torch.load(path, map_location="cpu", weights_only=False))
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return _coerce(json.load(f))
    with open(path, "r", encoding="utf-8") as f:
        nums = re.findall(r"-?\d+", f.read())
    if not nums:
        raise ValueError(f"No token IDs found in file {path!r}")
    return [int(n) for n in nums]


def _is_npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401
        return hasattr(torch, "npu") and torch.npu.is_available()
    except ImportError:
        return False


def get_device() -> torch.device:
    if _is_npu_available():
        return torch.device("npu:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def get_dist_backend() -> str:
    return "hccl" if _is_npu_available() else "nccl"


def init_distributed(tp_size: int = 1) -> int:
    """Initialize distributed process group when running under torchrun.

    Returns the current rank. If tp_size == 1 and no torchrun env is detected,
    returns 0 without initializing distributed.
    """
    if tp_size <= 1:
        # Check if we are under torchrun anyway (user may have omitted --tp-size)
        if "RANK" not in os.environ:
            return 0

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1 and not dist.is_initialized():
        backend = get_dist_backend()
        dist.init_process_group(backend=backend)
        logger.info(
            f"Distributed initialized: rank={rank}, local_rank={local_rank}, "
            f"world_size={world_size}, backend={backend}"
        )

    # Set device based on local_rank
    if _is_npu_available():
        import torch_npu
        torch.npu.set_device(f"npu:{local_rank}")
    elif torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return rank


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract response-only logp from HuggingFace Transformers (baseline)"
    )
    parser.add_argument(
        "--hf-checkpoint",
        type=str,
        required=True,
        help="Path to HuggingFace model checkpoint",
    )
    parser.add_argument(
        "--test-tokens",
        type=str,
        default="10,11,12,13,14,15,16,17",
        help="Comma-separated token IDs, or path to a token file (.txt/.json/.pt) (full sequence: prompt + response)",
    )
    parser.add_argument(
        "--response-length",
        type=int,
        default=None,
        help="Length of response tokens (last N). Default: all but first token",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 for model forward",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use float16 for model forward",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Trust remote code when loading HF model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cuda:0, npu:0, cpu). Auto-detected if omitted",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="hf_logp_result.pt",
        help="Output .pt file path",
    )
    parser.add_argument(
        "--save-activations",
        action="store_true",
        help="Save intermediate activations (embedding, each layer, norm, lm_head) for layer-wise comparison",
    )
    parser.add_argument(
        "--activations-output",
        type=str,
        default=None,
        help="Path to save activations .pt file. Default: <output>.activations.pt",
    )
    parser.add_argument(
        "--finegrain",
        action="store_true",
        help="Capture sub-module activations (input_layernorm/self_attn/"
             "post_attention_layernorm/mlp) on selected layers + final norm input, "
             "to pinpoint where divergence first appears inside a decoder layer. "
             "Implies --save-activations.",
    )
    parser.add_argument(
        "--finegrain-layers",
        type=str,
        default="all",
        help="Comma-separated decoder layer indices for --finegrain sub-module "
             "hooks, or 'all'. Default: 0 (layer_0).",
    )
    parser.add_argument(
        "--finegrain-attn",
        action="store_true",
        help="Capture self_attn internal sub-module outputs on selected layers: "
             "q_a_proj/q_a_layernorm/q_b_proj/kv_a_proj_with_mqa/kv_a_layernorm/"
             "kv_b_proj/o_proj/indexer. Pinpoints first divergence INSIDE self_attn "
             "(use after self_attn divergence is confirmed). Implies --save-activations. "
             "key=layer_{i}_self_attn_{sub}, paired by name with torchturbo via "
             "compare-activations. HF side fires for ALL projections incl. kv_b_proj.",
    )
    parser.add_argument(
        "--finegrain-attn-layers",
        type=str,
        default="0",
        help="Comma-separated decoder layer indices for --finegrain-attn, or 'all'. "
             "Default: 0.",
    )
    parser.add_argument(
        "--dump-attn-internals",
        type=str,
        default=None,
        help="实验F (HF 侧): 目录。patch apply_rotary_pos_emb_interleave + indexer, "
             "dump post-rotary q_rot/k_rot 与 topk_indices 到 layer_{i}.pt,与 "
             "test_torchturbo3_logp.py --dump-fused-attn 配对,用 "
             "compare_layer_io.py compare-fused-attn 对比 rotary/sparse。注意 HF 用 "
             "apply_rotary_pos_emb_interleave,torchturbo 用 apply_rotary_pos_emb(非 interleave)。",
    )
    parser.add_argument(
        "--dump-attn-internals-layers",
        type=str,
        default="0",
        help="Comma-separated decoder layer indices for --dump-attn-internals, or 'all'. "
             "Default: 0.",
    )
    parser.add_argument(
        "--dump-norm-ref",
        type=str,
        default=None,
        help="Dump final-norm input+output to .pt as the 'reference input' for "
             "--inject-norm-ref in test_torchturbo3_logp.py. Enables the mock "
             "experiment: torchturbo's norm input is replaced with HF's, then "
             "outputs + downstream logp are compared.",
    )
    parser.add_argument(
        "--dump-layer-ref",
        type=str,
        default=None,
        help="Dump per-layer input+output for --inject-layer-ref. Format: "
             "'LAYER_OUTDIR' or 'LAYERS:OUTDIR' (e.g. '0,1,2:layer_ref' or "
             "'all:layer_ref'). Saves layer_{i}.pt with {input, output} for each "
             "selected decoder layer — the mock-input reference for torchturbo.",
    )
    parser.add_argument(
        "--dump-ref",
        type=str,
        default=None,
        help="通用 ref dump (推荐): 'LAYERS:TARGET:DIR'。对选定子模块捕获 input+output "
             "落盘,供 test_torchturbo3_logp.py --inject-ref / --force-ref 注入。"
             "TARGET ∈ {layer,input_layernorm,self_attn,post_attention_layernorm,mlp,norm}。"
             "target=layer 写 layer_{i}.pt(等价 --dump-layer-ref);target=norm 写 norm.pt;"
             "其余写 layer_{i}_{target}.pt。每文件含 {input, output, input_dtype}。",
    )
    parser.add_argument(
        "--tensor-model-parallel-size",
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallelism size. When > 1, launch via torchrun (e.g. torchrun --nproc_per_node=2 test_hf_logp.py --tp-size 2)",
    )
    return parser.parse_args()


def load_model(args: argparse.Namespace, rank: int = 0):
    """Load HF causal LM and move to target device."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading tokenizer from {args.hf_checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_checkpoint,
        trust_remote_code=args.trust_remote_code,
    )

    dtype = torch.float32
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16

    # Determine target device
    if args.device:
        device = torch.device(args.device)
    elif dist.is_initialized():
        local_rank = get_local_rank()
        acc = "npu" if _is_npu_available() else "cuda"
        device = torch.device(f"{acc}:{local_rank}")
    else:
        device = get_device()

    logger.info(f"Loading model from {args.hf_checkpoint} (dtype={dtype}, device={device})")

    model = AutoModelForCausalLM.from_pretrained(
        args.hf_checkpoint,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        device_map=None,
    )
    model = model.to(device)
    model.eval()
    logger.info(f"[Rank {rank}] Model loaded on {device}")
    return model, tokenizer


def register_activation_hooks(model, activations: dict):
    """Register forward hooks on key layers to save intermediate activations.

    Hooks are placed at:
      - embed_tokens (input embedding)
      - each transformer layer (layer_0, layer_1, ...)
      - final norm
      - lm_head (output projection)

    Saved data per layer:
      {
          "output": Tensor detached to CPU (full activation),
          "input_shape": tuple,
          "output_shape": tuple,
      }
    """

    def make_hook(name):
        def hook(module, input, output):
            # Transformer layers return tuples (hidden_states, ...)
            if isinstance(output, tuple):
                act = output[0]
            else:
                act = output

            activations[name] = {
                "input_shape": tuple(input[0].shape) if input and hasattr(input[0], "shape") else None,
                "output_shape": tuple(act.shape) if hasattr(act, "shape") else None,
                "output": act.detach().cpu().float(),
            }

        return hook

    hooks = []
    base_model = model.model if hasattr(model, "model") else model

    # Embedding
    if hasattr(base_model, "embed_tokens"):
        h = base_model.embed_tokens.register_forward_hook(make_hook("embed_tokens"))
        hooks.append(h)
        logger.info("Hook registered: embed_tokens")

    # Transformer layers
    if hasattr(base_model, "layers"):
        for i, layer in enumerate(base_model.layers):
            h = layer.register_forward_hook(make_hook(f"layer_{i}"))
            hooks.append(h)
        logger.info(f"Hooks registered: {len(base_model.layers)} transformer layers")

    # Final norm
    if hasattr(base_model, "norm"):
        h = base_model.norm.register_forward_hook(make_hook("norm"))
        hooks.append(h)
        logger.info("Hook registered: norm")

    # LM Head
    if hasattr(model, "lm_head"):
        h = model.lm_head.register_forward_hook(make_hook("lm_head"))
        hooks.append(h)
        logger.info("Hook registered: lm_head")

    return hooks


def remove_hooks(hooks: list):
    for h in hooks:
        h.remove()


def register_finegrained_hooks(model, activations: dict, fg_spec: str) -> list:
    """Register sub-module hooks on selected decoder layers + final norm input.

    Captures (in addition to the coarse per-layer outputs from
    register_activation_hooks), for each selected layer i:
      layer_{i}_input_layernorm        - pre-attention RMSNorm output
      layer_{i}_self_attn              - attention output (MLA/DSA)
      layer_{i}_post_attention_layernorm - post-attention RMSNorm output
      layer_{i}_mlp                    - MLP / MoE output
    and always:
      norm_input                       - hidden states entering the final norm
                                        (= last layer output; makes the RMSNorm
                                        amplification visible as its own line)

    Key names use the `layer_{i}_{submodule}` form so compare_layer_io's
    _act_layer_idx sorts them right after their parent layer_{i}. Both this
    script and test_torchturbo3_logp.py emit identical key names, so
    compare-activations pairs them by name.

    Args:
        fg_spec: "0", "0,1,2", or "all".
    """
    def make_hook(name):
        def hook(module, inp, out):
            act = out[0] if isinstance(out, tuple) else out
            if not isinstance(act, torch.Tensor):
                return
            activations[name] = {
                "output": act.detach().cpu().float(),
                "output_shape": tuple(act.shape),
            }
        return hook

    def make_pre_hook(name):
        def pre(module, args):
            if args and isinstance(args[0], torch.Tensor):
                t = args[0]
                activations[name] = {
                    "output": t.detach().cpu().float(),
                    "output_shape": tuple(t.shape),
                }
        return pre

    hooks = []
    base_model = model.model if hasattr(model, "model") else model
    layers = getattr(base_model, "layers", None)
    if layers is None:
        logger.warning("finegrain: no .layers found on model, skipping")
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

    norm = getattr(base_model, "norm", None)
    if norm is not None:
        hooks.append(norm.register_forward_pre_hook(make_pre_hook("norm_input")))

    logger.info(f"finegrain hooks: {len(hooks)} on layers {sorted(sel)} + norm_input")
    return hooks


# --------------------------------------------------------------------------- #
# 通用 ref dump：对任意 decoder 子模块捕获 input+output，供 torchturbo 注入
#   --dump-ref LAYERS:TARGET:DIR
# 文件名约定与 test_torchturbo3_logp.py --inject-ref/--force-ref 一致：
#   target=layer -> layer_{i}.pt     (兼容旧 --dump-layer-ref，内容格式同)
#   target=norm  -> norm.pt           (注意：与旧 --dump-norm-ref 的单文件格式不同)
#   其余         -> layer_{i}_{target}.pt
# 每文件含 {input, output, input_dtype}。
# --------------------------------------------------------------------------- #
# attention 内部子模块（capture 全覆盖；inject 仅单 tensor 位置输入者）。
ATTN_INTERNAL_TARGETS = ("q_a_proj", "q_a_layernorm", "q_b_proj",
                         "kv_a_proj_with_mqa", "kv_a_layernorm", "kv_b_proj",
                         "o_proj", "indexer")
# 可注入：单 tensor 位置输入。排除 kv_b_proj(turbo fused 直读 weight,forward 不触发)
# 与 indexer(多入参 hidden_states+q_resid,只替 args[0] 不一致)。
_INJECTABLE_ATTN_INTERNAL = ("q_a_proj", "q_a_layernorm", "q_b_proj",
                             "kv_a_proj_with_mqa", "kv_a_layernorm", "o_proj")
_INJECT_TARGETS = (("layer", "input_layernorm", "self_attn",
                    "post_attention_layernorm", "mlp", "norm")
                   + _INJECTABLE_ATTN_INTERNAL)


def _resolve_inject_target(model, layer_idx: int, target: str):
    """target -> nn.Module。target='norm' 用 final norm，忽略 layer_idx。
    HF GLM5 结构：.model.layers[i].{input_layernorm,self_attn,
    post_attention_layernorm,mlp} + .model.norm（与 torchturbo 一致）；
    attention 内部 self_attn.{q_a_proj,q_a_layernorm,q_b_proj,kv_a_proj_with_mqa,
    kv_a_layernorm,kv_b_proj,o_proj,indexer}。"""
    base_model = model.model if hasattr(model, "model") else model
    if target == "norm":
        mod = getattr(base_model, "norm", None)
        if mod is None:
            raise AttributeError("target=norm 但 model 无 .norm")
        return mod
    layers = getattr(base_model, "layers", None)
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


def register_dump_ref_hooks(model, spec: str):
    """对选定 (layer, target) 捕获 input+output。返回 (handles, captured, sel, target)。

    captured: {layer_idx: {"input": Tensor, "output": Tensor}}（forward 后由调用方落盘）。
    input 取 args[0] 或 kwargs['hidden_states']（self_attn 关键字调用也能抓到）。
    output 取 out[0]（layer/self_attn 返回 tuple）或 out（单 tensor）。
    """
    layers_part, target, _ = _parse_ref_spec(spec)
    base_model = model.model if hasattr(model, "model") else model
    layers = getattr(base_model, "layers", None) or []
    sel = set()
    if target == "norm":
        sel.add(0)
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

    handles = []
    captured: dict = {}
    for i in sorted(sel):
        try:
            mod = _resolve_inject_target(model, i, target)
        except AttributeError as e:
            logger.warning(f"--dump-ref: {e}")
            continue

        def _mk_pre(idx):
            def pre(_m, args, kwargs):
                t = None
                if "hidden_states" in kwargs:
                    t = kwargs["hidden_states"]
                elif args and isinstance(args[0], torch.Tensor):
                    t = args[0]
                if isinstance(t, torch.Tensor):
                    captured.setdefault(idx, {})["input"] = t.detach()
            return pre

        def _mk_post(idx):
            def post(_m, _inp, out):
                act = out[0] if isinstance(out, tuple) else out
                if isinstance(act, torch.Tensor):
                    captured.setdefault(idx, {})["output"] = act.detach()
            return post

        handles.append(mod.register_forward_pre_hook(_mk_pre(i), with_kwargs=True))
        handles.append(mod.register_forward_hook(_mk_post(i)))
    logger.info(f"dump-ref hooks: target={target} layers={sorted(sel)} ({len(handles)} hooks)")
    return handles, captured, sorted(sel), target


def register_attn_internal_hooks(model, activations: dict, fg_spec: str):
    """在选定层 self_attn 内部子模块上挂 capture hook(--finegrain-attn)，
    定位 self_attn 内首次发散点。与 register_finegrained_hooks 同写 activations，
    key: layer_{i}_self_attn_{sub}，与 test_torchturbo3_logp.py 同名，compare-activations
    按名配对。

    捕获 q_a_proj/q_a_layernorm/q_b_proj/kv_a_proj_with_mqa/kv_a_layernorm/kv_b_proj/
    o_proj/indexer。indexer 在 "shared" 层为 None，自动跳过。
    HF 侧所有投影在 eager/sdpa/fa2 下都是正常 nn.Linear 调用，hook 均触发（含 kv_b_proj）。
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
    base_model = model.model if hasattr(model, "model") else model
    layers = getattr(base_model, "layers", None)
    if layers is None:
        logger.warning("finegrain-attn: no .layers found, skipping")
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
    logger.info(f"finegrain-attn hooks: {len(hooks)} on layers {sorted(sel)}")
    return hooks


# --------------------------------------------------------------------------- #
# 实验F (HF 侧): dump attention 内部中间量,与 torchturbo --dump-fused-attn 配对
#   - patch apply_rotary_pos_emb_interleave: 捕获 post-rotary q_rot/k_rot (rotary 嫌疑)
#   - patch GlmMoeDsaIndexer.forward: 捕获 topk_indices (sparse indexer 嫌疑)
# 落盘到 out_dir/layer_{i}.pt,用 compare_layer_io.py compare-fused-attn 对比。
# 注意:HF 用 apply_rotary_pos_emb_interleave,torchturbo 用 apply_rotary_pos_emb(非 interleave)
# —— 不同 rotary 函数,post-rotary 张量直接对比即可判定 rotary 是否为发散源。
# --------------------------------------------------------------------------- #
def _patch_hf_attn_dump(model, out_dir: str, layers_spec: str):
    """monkey-patch HF attention 的 rotary 函数 + indexer,捕获中间量。返回 (save, restore)。
    在 forward 前调用,forward 后先 save() 再 restore()。

    apply_rotary_pos_emb_interleave 仅在 attention forward(line 434)被调一次/层,顺序=层序,
    故按调用顺序索引即层 idx。indexer 按 self.layer_idx 索引(shared 层无 indexer,跳过)。
    """
    import transformers.models.glm_moe_dsa.modeling_glm_moe_dsa as M

    base_model = model.model if hasattr(model, "model") else model
    layers = getattr(base_model, "layers", None) or []
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

    captured: dict = {}
    rotary_calls: list = []  # 每次调用的 (q_rot, k_rot),顺序=层序

    _orig_rotary = M.apply_rotary_pos_emb_interleave
    _orig_indexer_fwd = M.GlmMoeDsaIndexer.forward

    def _wrap_rotary(q, k, cos, sin, *a, **kw):
        out = _orig_rotary(q, k, cos, sin, *a, **kw)
        q_rot, k_rot = out
        rotary_calls.append((q_rot.detach(), k_rot.detach()))
        return out

    def _wrap_indexer_fwd(self, hidden_states, q_resid, position_embeddings,
                          attention_mask, position_ids, past_key_values=None):
        out = _orig_indexer_fwd(self, hidden_states, q_resid, position_embeddings,
                                attention_mask, position_ids, past_key_values=past_key_values)
        idx = getattr(self, "layer_idx", None)
        if idx is not None and idx in sel and isinstance(out, torch.Tensor):
            captured.setdefault(idx, {})["topk_indices"] = out.detach().cpu()
        return out

    M.apply_rotary_pos_emb_interleave = _wrap_rotary
    M.GlmMoeDsaIndexer.forward = _wrap_indexer_fwd
    logger.info(f"实验F HF attn dump patch installed: layers={sorted(sel)} -> {out_dir}")

    def save():
        # 关联 rotary: 第 i 次调用 = layer i
        for i, (q_rot, k_rot) in enumerate(rotary_calls):
            if i in sel:
                d = captured.setdefault(i, {})
                d["q_rot_postrotary"] = q_rot.cpu()   # [B,H,S,qk_rope]
                d["k_rot_postrotary"] = k_rot.cpu()   # [B,1,S,qk_rope]
        out_dir_p = Path(out_dir)
        out_dir_p.mkdir(parents=True, exist_ok=True)
        for idx, d in captured.items():
            torch.save(d, out_dir_p / f"layer_{idx}.pt")
        logger.info(f"实验F HF dumped {len(captured)} layers to {out_dir_p.resolve()}")
        return captured

    def restore():
        M.apply_rotary_pos_emb_interleave = _orig_rotary
        M.GlmMoeDsaIndexer.forward = _orig_indexer_fwd

    return save, restore


def compute_response_logp(
    model,
    token_ids: list[int],
    response_length: int | None,
    save_activations: bool = False,
    rank: int = 0,
    fg_layers: str | None = None,
    fg_attn_layers: str | None = None,
    dump_norm_ref_path: str | None = None,
    dump_layer_ref_spec: str | None = None,
    dump_ref_spec: str | None = None,
    dump_attn_spec: str | None = None,
    dump_attn_layers: str = "0",
) -> dict:
    """Compute response-only logp, matching test_train_logp.py logic.

    For causal LM:
        logits[i] predicts token[i+1]
        response tokens are token_ids[prompt_length : total_length]
        response logits are logits[prompt_length-1 : total_length-1]
    """
    target_tokens = torch.tensor(token_ids, dtype=torch.long)
    total_length = len(target_tokens)

    if response_length is None or response_length <= 0:
        prompt_length = 1
        response_length = total_length - 1
    else:
        prompt_length = total_length - response_length

    device = next(model.parameters()).device
    input_ids = target_tokens.unsqueeze(0).to(device)  # [1, total_length]

    activations = {}
    hooks = []
    if save_activations:
        hooks = register_activation_hooks(model, activations)
    fg_hooks = []
    if fg_layers is not None:
        fg_hooks = register_finegrained_hooks(model, activations, fg_layers)
    hooks = hooks + fg_hooks
    if fg_attn_layers is not None:
        hooks = hooks + register_attn_internal_hooks(model, activations, fg_attn_layers)

    # norm-ref capture: grab final-norm input+output for --dump-norm-ref
    norm_ref = {}
    if dump_norm_ref_path is not None:
        base_model = model.model if hasattr(model, "model") else model
        norm = getattr(base_model, "norm", None)
        if norm is not None:
            def _nr_pre(_m, args):
                if args and isinstance(args[0], torch.Tensor):
                    norm_ref["input"] = args[0].detach()
            def _nr_post(_m, _inp, out):
                if isinstance(out, torch.Tensor):
                    norm_ref["output"] = out.detach()
            hooks.append(norm.register_forward_pre_hook(_nr_pre))
            hooks.append(norm.register_forward_hook(_nr_post))

    # layer-ref capture: grab per-layer input+output for --dump-layer-ref
    layer_ref = {}  # {i: {"input": t, "output": t}}
    layer_ref_dir = None
    if dump_layer_ref_spec is not None:
        spec = dump_layer_ref_spec
        if ":" in spec:
            layers_part, layer_ref_dir = spec.split(":", 1)
        else:
            layers_part, layer_ref_dir = "all", spec
        base_model = model.model if hasattr(model, "model") else model
        layers = getattr(base_model, "layers", None)
        if layers is None:
            logger.warning(f"[Rank {rank}] --dump-layer-ref: no .layers found, skipping")
        else:
            sel = set()
            for x in layers_part.split(","):
                x = x.strip()
                if x == "all":
                    sel.update(range(len(layers)))
                elif x:
                    sel.add(int(x))
            for i in sorted(j for j in sel if 0 <= j < len(layers)):
                def _mk_pre(idx):
                    def _pre(_m, args):
                        if args and isinstance(args[0], torch.Tensor):
                            layer_ref.setdefault(idx, {})["input"] = args[0].detach()
                    return _pre
                def _mk_post(idx):
                    def _post(_m, _inp, out):
                        act = out[0] if isinstance(out, tuple) else out
                        if isinstance(act, torch.Tensor):
                            layer_ref.setdefault(idx, {})["output"] = act.detach()
                    return _post
                hooks.append(layers[i].register_forward_pre_hook(_mk_pre(i)))
                hooks.append(layers[i].register_forward_hook(_mk_post(i)))
            logger.info(f"[Rank {rank}] layer-ref hooks on layers "
                        f"{sorted(layer_ref)} -> {layer_ref_dir}")

    # dump-ref capture: 通用子模块 input+output(供 torchturbo --inject-ref/--force-ref)
    dump_ref_captured: dict = {}
    dump_ref_info = None
    if dump_ref_spec is not None:
        _dr_handles, dump_ref_captured, _dr_sel, _dr_target = \
            register_dump_ref_hooks(model, dump_ref_spec)
        hooks = hooks + _dr_handles
        dump_ref_info = (_dr_sel, _dr_target)

    # 实验F: 装 HF attn dump patch(forward 前)
    _attn_save = _attn_restore = None
    if dump_attn_spec is not None:
        _attn_save, _attn_restore = _patch_hf_attn_dump(
            model, dump_attn_spec, dump_attn_layers)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits.float()  # [1, total_length, vocab_size]

    # 实验F: 落盘 + 还原(forward 后)
    if _attn_save is not None:
        _attn_save()
    if _attn_restore is not None:
        _attn_restore()

    if hooks:
        for h in hooks:
            h.remove()
        logger.info(f"[Rank {rank}] Captured {len(activations)} activation layers")

    if dump_norm_ref_path is not None and norm_ref:
        base_model = model.model if hasattr(model, "model") else model
        norm = getattr(base_model, "norm", None)
        weight = getattr(norm, "weight", None) if norm is not None else None
        eps = getattr(norm, "variance_epsilon", None) if norm is not None else None
        ref_path = Path(dump_norm_ref_path)
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "norm_input": norm_ref.get("input").detach().cpu().contiguous()
                             if norm_ref.get("input") is not None else None,
                "norm_output": norm_ref.get("output").detach().cpu().contiguous()
                               if norm_ref.get("output") is not None else None,
                "weight": weight.detach().cpu().contiguous() if weight is not None else None,
                "eps": eps,
                "input_dtype": str(norm_ref["input"].dtype) if norm_ref.get("input") is not None else None,
            },
            ref_path,
        )
        logger.info(f"[Rank {rank}] norm reference saved to {ref_path.resolve()}")

    # save per-layer ref
    if dump_layer_ref_spec is not None and layer_ref and layer_ref_dir:
        out_dir = Path(layer_ref_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, io in layer_ref.items():
            torch.save(
                {
                    "input": io["input"].detach().cpu().contiguous()
                             if io.get("input") is not None else None,
                    "output": io["output"].detach().cpu().contiguous()
                              if io.get("output") is not None else None,
                    "input_dtype": str(io["input"].dtype) if io.get("input") is not None else None,
                },
                out_dir / f"layer_{i}.pt",
            )
        logger.info(f"[Rank {rank}] layer ref saved ({len(layer_ref)} layers) "
                    f"to {out_dir.resolve()}")

    # save dump-ref (通用子模块 input+output，供 torchturbo --inject-ref/--force-ref)
    if dump_ref_spec is not None and dump_ref_info is not None:
        _dr_sel, _dr_target = dump_ref_info
        _, _, _dr_dir = _parse_ref_spec(dump_ref_spec)
        _dr_dir.mkdir(parents=True, exist_ok=True)
        for _i, _io in dump_ref_captured.items():
            torch.save(
                {
                    "input": _io["input"].detach().cpu().contiguous()
                             if _io.get("input") is not None else None,
                    "output": _io["output"].detach().cpu().contiguous()
                              if _io.get("output") is not None else None,
                    "input_dtype": str(_io["input"].dtype) if _io.get("input") is not None else None,
                },
                _dr_dir / _inject_ref_filename(_dr_target, _i),
            )
        logger.info(f"[Rank {rank}] dump-ref saved ({len(dump_ref_captured)} "
                    f"{_dr_target}) to {_dr_dir.resolve()}")

    # Squeeze batch dimension → [total_length, vocab_size]
    logits = logits.squeeze(0)

    # Extract response-aligned logits (same slicing as test_train_logp.py)
    start_idx = prompt_length - 1
    end_idx = total_length - 1

    if start_idx < 0 or end_idx <= start_idx:
        raise ValueError(
            f"Invalid prompt/response split: prompt_len={prompt_length}, "
            f"total_len={total_length}. Need at least 1 prompt + 1 response token."
        )

    response_logits = logits[start_idx:end_idx]              # [response_length, vocab_size]
    response_target = target_tokens[prompt_length:total_length].to(device)  # [response_length]

    log_probs = F.log_softmax(response_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, response_target.unsqueeze(-1)).squeeze(-1)

    result = {
        "response_tokens": response_target.cpu(),
        "response_logp": token_log_probs.cpu(),
        "prompt_length": prompt_length,
        "response_length": response_length,
        "mean_logp": token_log_probs.mean().item(),
        "input_token_ids": target_tokens.cpu(),
    }
    if activations:
        result["activations"] = activations
    return result


def main() -> int:
    args = parse_args()

    # Initialize distributed if running under torchrun or tp-size > 1
    rank = init_distributed(args.tensor_model_parallel_size)
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # Parse token IDs
    token_ids = parse_test_tokens(args.test_tokens)
    logger.info(f"[Rank {rank}] Input token IDs: {token_ids} (length={len(token_ids)})")

    if len(token_ids) < 2:
        logger.error("Need at least 2 tokens (1 prompt + 1 response)")
        return 1

    # Load model
    try:
        model, tokenizer = load_model(args, rank=rank)
    except Exception as e:
        logger.error(f"[Rank {rank}] Failed to load model: {e}")
        return 1

    # Compute logp
    try:
        result = compute_response_logp(
            model, token_ids, args.response_length,
            save_activations=args.save_activations or args.finegrain or args.finegrain_attn,
            rank=rank,
            fg_layers=args.finegrain_layers if args.finegrain else None,
            fg_attn_layers=args.finegrain_attn_layers if args.finegrain_attn else None,
            dump_norm_ref_path=args.dump_norm_ref,
            dump_layer_ref_spec=args.dump_layer_ref,
            dump_ref_spec=args.dump_ref,
            dump_attn_spec=args.dump_attn_internals,
            dump_attn_layers=args.dump_attn_internals_layers if args.dump_attn_internals else "0",
        )
    except Exception as e:
        logger.error(f"[Rank {rank}] Failed to compute logp: {e}")
        return 1

    # Save logp result (only rank 0 saves to avoid conflicts)
    if rank == 0:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, output_path)
        logger.info(f"[Rank {rank}] Result saved to {output_path.resolve()}")

        # Save activations separately (they can be large)
        if args.save_activations and "activations" in result:
            act_output = args.activations_output or str(output_path.with_suffix(".activations.pt"))
            act_path = Path(act_output)
            act_path.parent.mkdir(parents=True, exist_ok=True)
            # Pop activations from result to avoid duplicating in logp file
            activations = result.pop("activations")
            torch.save(activations, act_path)
            logger.info(f"[Rank {rank}] Activations saved to {act_path.resolve()} ({len(activations)} layers)")
            # Re-save result without activations
            torch.save(result, output_path)

            # Print activation layer summary
            print("\n  Activation layers captured:")
            for name, data in activations.items():
                out = data["output"]
                print(f"    {name:20s}  shape={tuple(out.shape)}  mean={out.mean():.6f}  std={out.std():.6f}")

        # Print summary
        print("\n" + "=" * 60)
        print("  HF Baseline Logp Result")
        print("=" * 60)
        print(f"  Model            : {args.hf_checkpoint}")
        print(f"  TP size          : {args.tensor_model_parallel_size}")
        print(f"  World size       : {world_size}")
        print(f"  Input tokens     : {token_ids}")
        print(f"  Prompt length    : {result['prompt_length']}")
        print(f"  Response length  : {result['response_length']}")
        print(f"  Response tokens  : {result['response_tokens'].tolist()}")
        print(f"  Mean logp        : {result['mean_logp']:.6f}")
        print(f"  Min  logp        : {result['response_logp'].min().item():.6f}")
        print(f"  Max  logp        : {result['response_logp'].max().item():.6f}")
        print("=" * 60)

        # Print ready-to-use compare command
        print("\n  Ready-to-use compare command:")
        print(f"    python compare_train_rollout_logp.py \\")
        print(f"        --train-logp {args.output} \\")
        print(f"        --rollout-logp rollout_logp_result.json \\")
        print(f"        --output compare_result.json")
        print("=" * 60)
    else:
        logger.info(f"[Rank {rank}] Computation completed; output written by rank 0 only")

    # Cleanup distributed
    if dist.is_initialized():
        dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    sys.exit(main())
