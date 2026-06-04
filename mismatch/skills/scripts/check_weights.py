# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""
Compare HF model weights with Megatron-Bridge converted weights.

This script loads the same checkpoint via:
  1. Raw HF state dict (model*.safetensors / pytorch_model*.bin)
  2. Megatron-Bridge (via AutoBridge + to_megatron_provider)

Then extracts and compares corresponding parameters to verify that
Megatron-Bridge weight conversion is numerically faithful.

For VLMs (e.g. Qwen3.5-VL), raw checkpoint loading preserves the original
key prefixes (``model.language_model.*`` / ``model.visual.*``) instead of
the stripped names produced by ``AutoModelForCausalLM``.

Usage:
    # Compare HF vs Megatron-Bridge weights
    python check_weights.py \
        --hf-checkpoint /mnt/sfs_turbo/models/Qwen3.5-9B/ \
        --tensor-model-parallel-size 1 \
        --bf16

    # Skip vision tower; compare only language model weights
    python check_weights.py \
        --hf-checkpoint /mnt/sfs_turbo/models/Qwen3.5-9B/ \
        --tensor-model-parallel-size 1 \
        --bf16 --skip-vision
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import mindspeed.megatron_adaptor  # noqa: F401
except ImportError:
    raise ValueError("mindspeed.megatron_adaptor not found")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare HF and Megatron-Bridge weights"
    )
    parser.add_argument(
        "--hf-checkpoint",
        type=str,
        required=True,
        help="Path to HuggingFace model checkpoint",
    )
    parser.add_argument(
        "--tensor-model-parallel-size",
        type=int,
        default=1,
        help="Tensor parallelism size for Megatron build (use 1 for full weight comparison)",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Trust remote code",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="weight_check.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Show top-K parameters with largest diff",
    )
    parser.add_argument(
        "--skip-vision",
        action="store_true",
        help="Skip vision model parameters and only compare language model weights",
    )
    parser.add_argument(
        "--skip-fused",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip fused projections that cannot be compared 1:1 (QKV, GDN in_proj, gate+up). "
             "Default: True. Use --no-skip-fused to compare them anyway.",
    )
    return parser.parse_args()


def _load_hf_state_dict_raw(checkpoint: str) -> dict[str, torch.Tensor]:
    """Load the original HF checkpoint keys from safetensors or pytorch bins.

    Using the raw checkpoint preserves VLM prefixes (e.g. ``model.language_model.*``
    or ``model.*``) exactly as stored, whereas ``AutoModelForCausalLM`` may strip
    or rename the language-model sub-module.
    """
    ckpt_path = Path(checkpoint)

    # Prefer safetensors (current HF default)
    safetensor_files = sorted(ckpt_path.glob("model*.safetensors"))
    if safetensor_files:
        try:
            from safetensors.torch import load_file
            state_dict: dict[str, torch.Tensor] = {}
            for f in safetensor_files:
                state_dict.update(load_file(str(f), device="cpu"))
            logger.info(f"Loaded raw HF state dict from {len(safetensor_files)} safetensors file(s)")
            return state_dict
        except Exception as e:
            logger.warning(f"Failed to load safetensors: {e}; falling back")

    # Fall back to pytorch_model.bin / model_x_of_y.bin
    bin_files = sorted(ckpt_path.glob("pytorch_model*.bin")) or sorted(ckpt_path.glob("model*.bin"))
    if bin_files:
        state_dict = {}
        for f in bin_files:
            state_dict.update(torch.load(str(f), map_location="cpu", weights_only=False))
        logger.info(f"Loaded raw HF state dict from {len(bin_files)} bin file(s)")
        return state_dict

    raise FileNotFoundError(
        f"No HF checkpoint found in {ckpt_path}. "
        "Expected model*.safetensors or pytorch_model*.bin"
    )


def load_hf_weights(checkpoint: str, trust_remote_code: bool = True) -> dict[str, torch.Tensor]:
    """Load HF model and return state_dict.

    For weight-name comparison we load the raw checkpoint files so that VLM
    prefixing (``model.language_model.*`` vs ``model.*``) is preserved exactly.
    If raw files are unavailable we fall back to ``AutoModelForCausalLM``.
    """
    logger.info(f"Loading HF weights from {checkpoint}")
    try:
        state_dict = _load_hf_state_dict_raw(checkpoint)
    except FileNotFoundError:
        logger.warning("Raw checkpoint files not found; falling back to AutoModelForCausalLM")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.float32,
            trust_remote_code=trust_remote_code,
            device_map="cpu",
        )
        state_dict = model.state_dict()

    state_dict = {k: v.float() for k, v in state_dict.items()}
    logger.info(f"HF weights loaded: {len(state_dict)} parameters")
    return state_dict


def _init_megatron_for_weight_check(checkpoint: str, tp_size: int, bf16: bool):
    """Initialize Megatron-Bridge model for weight extraction.

    Must be called under torchrun (all ranks participate).
    Returns (model, bridge) on all ranks.
    """
    import sys
    from pathlib import Path

    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))
    from test_train_logp import (
        parse_test_args, init_distributed, build_model_with_bridge,
        _patch_megatron_model, _patch_scatter_dtype_cast,
    )

    # Backup original argv and inject args for parse_test_args
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]]
    inject = [
        "--hf-checkpoint", checkpoint,
        "--tensor-model-parallel-size", str(tp_size),
        "--micro-batch-size", "1",
        "--seq-length", "2048",
        "--no-rope-fusion",
    ]
    if bf16:
        inject.append("--bf16")
    sys.argv.extend(inject)

    try:
        args = parse_test_args()
        init_distributed(args)
        model, bridge = build_model_with_bridge(args)

        # Load HF weights into Megatron model via bridge (all ranks must call this)
        with _patch_megatron_model(model):
            with _patch_scatter_dtype_cast():
                bridge.load_hf_weights(model)

        return model, bridge
    finally:
        sys.argv = original_argv


def _extract_megatron_state_dict(model) -> dict[str, torch.Tensor]:
    """Extract state dict from Megatron model (call on rank 0 only)."""
    from megatron.core.utils import unwrap_model
    unwrapped = unwrap_model(model)[0]
    return {
        name: param.detach().cpu().float()
        for name, param in unwrapped.named_parameters()
    }


import re


def normalize_param_name(name: str) -> str:
    """Normalize parameter names for cross-framework matching.

    Supports both standard Transformer (Qwen2) and Mamba/SSD (Qwen3.5) architectures,
    as well as multi-modal wrappers (vision_model.decoder.*, language_model.decoder.*).
    """
    n = name.lower()

    # Strip common framework prefixes so HF / Megatron names converge.
    # Megatron VL:  language_model.embedding.* / language_model.decoder.layers.*
    # Megatron VL:  vision_model.decoder.layers.* / vision_model.patch_embed.*
    # HF VLM:       model.language_model.* / model.visual.* / model.*
    # HF text-only: model.* / lm_head.*
    cleaned = n
    for prefix_pat in [
        r"^model\.language_model\.",
        r"^language_model\.",
        r"^model\.visual\.",
        r"^vision_model\.",
        r"^model\.",
        r"^decoder\.",
    ]:
        cleaned = re.sub(prefix_pat, "", cleaned)

    # Extract layer index from remaining path (language model uses layers.N,
    # vision model HF uses blocks.N, Megatron vision uses layers.N)
    layer_match = re.search(r'(?:layers|blocks)\.(\d+)', cleaned)
    layer_idx = layer_match.group(1) if layer_match else None

    # --- Global / non-layer params ---
    if layer_idx is None:
        # Distinguish vision embeddings from text embeddings.
        if "patch_embed" in cleaned:
            return "global_vision_patch_embed"
        if "pos_embed" in cleaned:
            return "global_vision_pos_embed"
        if any(k in cleaned for k in ("word_embeddings", "embed_tokens", "tok_embeddings")):
            return "global_embed"
        if any(k in cleaned for k in ("lm_head", "output_layer")):
            return "global_lm_head"
        if "norm" in cleaned:
            # Distinguish vision merger norm from final text norm by path context
            if "merger" in cleaned or "patch_norm" in cleaned:
                return "global_vision_merger_norm"
            return "global_final_norm"
        return f"global_{cleaned.replace('.', '_')}"

    # --- Layer-specific params ---
    prefix = f"layer_{layer_idx}"
    # Work with the substring after layers.N. or blocks.N.
    layer_body = re.sub(rf'^(?:layers|blocks)\.{layer_idx}\.', '', cleaned)

    # Attention QKV (various naming conventions)
    # Megatron fused: self_attention.linear_qkv, self_attention.in_proj
    # HF split:      self_attn.q_proj/k_proj/v_proj, linear_attn.in_proj_qkv
    # Vision:        attn.qkv (HF) -> self_attention.linear_qkv (MG)
    if any(k in layer_body for k in ("q_proj", "k_proj", "v_proj", "query_key_value",
                                     "in_proj_qkv", "linear_qkv", "in_proj.weight",
                                     "attn.qkv")):
        # Distinguish GDN fused in_proj from standard QKV by checking path
        if "linear_attn" in layer_body and "in_proj.weight" in layer_body:
            return f"{prefix}_mamba_in_proj"  # fused qkv+z+b+a
        if "self_attention.in_proj.weight" in layer_body:
            return f"{prefix}_mamba_in_proj"
        return f"{prefix}_attn_qkv"

    # Attention output projection
    # Vision:        attn.proj (HF) -> self_attention.linear_proj (MG)
    if any(k in layer_body for k in ("o_proj", "out_proj", "linear_proj", "attn.proj")):
        return f"{prefix}_attn_proj"

    # MLP gate + up (fused or split)
    if any(k in layer_body for k in ("gate_proj", "up_proj", "linear_fc1")):
        return f"{prefix}_mlp_fc1"

    # MLP down
    if any(k in layer_body for k in ("down_proj", "linear_fc2")):
        return f"{prefix}_mlp_fc2"

    # Expert / MoE router
    if "router" in layer_body or "gate.weight" in layer_body:
        return f"{prefix}_mlp_router"

    # Layer norms
    # Vision HF uses norm1/norm2; text uses input_layernorm/post_attention_layernorm.
    # Megatron may store input norm inside self_attention.linear_qkv.layer_norm_weight
    # and post-attn norm inside mlp.linear_fc1.layer_norm_weight.
    if any(k in layer_body for k in ("input_layernorm", "input_norm", "norm1")) \
            or "linear_qkv.layer_norm_weight" in layer_body:
        return f"{prefix}_norm_input"
    if any(k in layer_body for k in ("post_attention_layernorm", "post_attention_norm",
                                      "pre_mlp_layernorm", "norm2")) \
            or "mlp.linear_fc1.layer_norm_weight" in layer_body:
        return f"{prefix}_norm_post_attn"
    if "out_norm" in layer_body:
        return f"{prefix}_mamba_out_norm"
    if "q_layernorm" in layer_body or "q_norm" in layer_body:
        return f"{prefix}_attn_q_norm"
    if "k_layernorm" in layer_body or "k_norm" in layer_body:
        return f"{prefix}_attn_k_norm"

    # Mamba / Linear Attention / SSD specific params
    if "dt_bias" in layer_body or "dt_proj" in layer_body:
        return f"{prefix}_mamba_dt"
    if "a_log" in layer_body or "a_norm" in layer_body:
        return f"{prefix}_mamba_a"
    if "conv1d" in layer_body or "x_conv" in layer_body:
        return f"{prefix}_mamba_conv"
    if "in_proj_z" in layer_body or "z_proj" in layer_body:
        return f"{prefix}_mamba_z"
    if "in_proj_b" in layer_body or "b_proj" in layer_body:
        return f"{prefix}_mamba_b"
    if "in_proj_a" in layer_body:
        return f"{prefix}_mamba_a_proj"
    if "time_mip" in layer_body:
        return f"{prefix}_mamba_time"
    if "d_log" in layer_body:
        return f"{prefix}_mamba_d"

    # Fallback
    remainder = re.sub(r'^(?:layers|blocks)\.\d+\.', '', cleaned)
    remainder = remainder.replace('.', '_')
    return f"{prefix}_{remainder}"


def _detect_architecture(params: dict[str, torch.Tensor], label: str) -> dict:
    """Detect model architecture from parameter names for diagnostics."""
    names = list(params.keys())
    has_vision = any("vision_model" in n or "model.visual" in n for n in names)
    has_mamba = any("linear_attn" in n or "conv1d" in n or "dt_bias" in n for n in names)
    has_standard_attn = any("self_attn" in n or "self_attention" in n for n in names)
    has_lm_prefix = any("language_model" in n for n in names)
    has_hf_lm_prefix = any("model.language_model" in n for n in names)
    has_hf_text_prefix = any(n.startswith("model.layers.") or n.startswith("model.embed_tokens") for n in names)

    llm_layers = set(
        re.search(r'(?:layers|blocks)\.(\d+)', n).group(1)
        for n in names
        if re.search(r'(?:layers|blocks)\.(\d+)', n)
    )

    vision_count = sum(1 for n in names if "vision_model" in n or "model.visual" in n or "visual." in n)
    text_count = sum(1 for n in names if "vision_model" not in n and "model.visual" not in n)

    return {
        "label": label,
        "param_count": len(params),
        "num_layers": len(llm_layers),
        "has_vision": has_vision,
        "vision_params": vision_count,
        "text_params": text_count,
        "has_mamba": has_mamba,
        "has_standard_attn": has_standard_attn,
        "has_lm_prefix": has_lm_prefix,
        "has_hf_lm_prefix": has_hf_lm_prefix,
        "has_hf_text_prefix": has_hf_text_prefix,
    }


def compare_param(hf_name: str, hf_tensor: torch.Tensor, mg_name: str, mg_tensor: torch.Tensor) -> dict:
    """Compare two parameter tensors."""
    h = hf_tensor.numpy()
    m = mg_tensor.numpy()

    # Handle shape mismatches (e.g. fused vs split projections)
    if h.shape != m.shape:
        # Log but still try to compare if shapes are close
        logger.warning(f"Shape mismatch: HF {hf_name} {h.shape} vs Megatron {mg_name} {m.shape}")
        # For now, skip detailed comparison on shape mismatch
        return {
            "hf_name": hf_name,
            "megatron_name": mg_name,
            "hf_shape": list(h.shape),
            "megatron_shape": list(m.shape),
            "shape_mismatch": True,
            "mean_abs_diff": None,
            "max_abs_diff": None,
        }

    diff = h - m
    abs_diff = np.abs(diff)

    return {
        "hf_name": hf_name,
        "megatron_name": mg_name,
        "hf_shape": list(h.shape),
        "megatron_shape": list(m.shape),
        "shape_mismatch": False,
        "mean_abs_diff": float(abs_diff.mean()),
        "max_abs_diff": float(abs_diff.max()),
        "std_diff": float(diff.std()),
        "rel_diff": float(abs_diff.mean() / (np.abs(h).mean() + 1e-12)),
        "hf_mean": float(h.mean()),
        "mg_mean": float(m.mean()),
    }


def main() -> int:
    args = parse_args()

    # Step 1: All ranks initialize Megatron (required for model build + weight load)
    try:
        model, _ = _init_megatron_for_weight_check(
            args.hf_checkpoint, args.tensor_model_parallel_size, args.bf16
        )
    except Exception as e:
        logger.error(f"Failed to initialize Megatron: {e}")
        import traceback
        traceback.print_exc()
        return 1

    import torch.distributed as dist
    rank = dist.get_rank()

    # Step 2: Rank 0 extracts Megatron state dict + loads HF model + compares
    results = []
    unmatched_hf = []
    unmatched_mg = []
    skipped_fused_hf = []
    skipped_fused_mg = []
    hf_state = None
    mg_state = None

    if rank == 0:
        try:
            mg_state = _extract_megatron_state_dict(model)
            logger.info(f"Megatron params extracted: {len(mg_state)}")
        except Exception as e:
            logger.error(f"Failed to extract Megatron state dict: {e}")

        try:
            hf_state = load_hf_weights(args.hf_checkpoint, args.trust_remote_code)
        except Exception as e:
            logger.error(f"Failed to load HF weights: {e}")

        if hf_state and mg_state:
            # Optionally drop vision-only parameters.
            def _is_vision(name: str) -> bool:
                return any(p in name for p in ("vision_model.", "model.visual.", ".visual."))

            skip_fused = args.skip_fused

            hf_keys = list(hf_state.keys())
            mg_keys = list(mg_state.keys())
            if args.skip_vision:
                hf_keys = [k for k in hf_keys if not _is_vision(k)]
                mg_keys = [k for k in mg_keys if not _is_vision(k)]

            # Optionally skip fused projections that cannot be compared 1:1.
            # These are recorded separately rather than as shape-mismatch warnings.
            FUSED_SUFFIXES = (
                "attn_qkv",      # HF: q/k/v split  → MG: fused linear_qkv
                "mamba_in_proj", # HF: qkv/z/b/a split → MG: fused in_proj
                "mlp_fc1",       # HF: gate_proj + up_proj → MG: fused linear_fc1
            )

            def _is_fused(norm_name: str) -> bool:
                return any(norm_name.endswith(s) for s in FUSED_SUFFIXES)

            skipped_fused_hf = []
            skipped_fused_mg = []

            hf_norm = {normalize_param_name(k): k for k in hf_keys}
            mg_norm = {normalize_param_name(k): k for k in mg_keys}

            for norm_name, hf_name in hf_norm.items():
                if skip_fused and _is_fused(norm_name):
                    skipped_fused_hf.append(hf_name)
                    continue
                if norm_name in mg_norm:
                    mg_name = mg_norm[norm_name]
                    if skip_fused and _is_fused(norm_name):
                        skipped_fused_mg.append(mg_name)
                        continue
                    result = compare_param(hf_name, hf_state[hf_name], mg_name, mg_state[mg_name])
                    results.append(result)
                else:
                    unmatched_hf.append(hf_name)

            for norm_name, mg_name in mg_norm.items():
                if skip_fused and _is_fused(norm_name):
                    if mg_name not in skipped_fused_mg:
                        skipped_fused_mg.append(mg_name)
                    continue
                if norm_name not in hf_norm:
                    unmatched_mg.append(mg_name)

            # Sort by max_abs_diff descending
            results.sort(
                key=lambda x: x.get("max_abs_diff", 0) if x.get("max_abs_diff") is not None else 0,
                reverse=True,
            )

    # Step 3: Rank 0 prints + saves results
    if rank == 0:
        # Architecture diagnostics
        if hf_state and mg_state:
            hf_arch = _detect_architecture(hf_state, "HF")
            mg_arch = _detect_architecture(mg_state, "Megatron")
            print("\n" + "=" * 80)
            print("  Architecture Detection")
            print("=" * 80)
            for arch in (hf_arch, mg_arch):
                print(f"  {arch['label']:10s}: {arch['param_count']:4d} params, "
                      f"{arch['num_layers']:3d} layers, "
                      f"text={arch['text_params']:4d}, vision={arch['vision_params']:4d}, "
                      f"mamba={arch['has_mamba']}, std_attn={arch['has_standard_attn']}")
            print("  Naming hints:")
            print(f"    HF    has_hf_lm_prefix={hf_arch['has_hf_lm_prefix']}, "
                  f"has_hf_text_prefix={hf_arch['has_hf_text_prefix']}")
            print(f"    MG    has_lm_prefix={mg_arch['has_lm_prefix']}")
            if hf_arch['has_hf_lm_prefix'] != hf_arch['has_hf_text_prefix']:
                print("  ℹ️  HF checkpoint uses mixed naming; normalization will handle both.")
            if hf_arch['has_mamba'] and not mg_arch['has_mamba']:
                print("  ⚠️ HF uses Mamba/SSD but Megatron uses standard Attention!")
                print("     → Weight mapping may be incomplete; consider activation comparison instead.")
            if hf_arch['vision_params'] == 0 and mg_arch['vision_params'] > 0:
                print("  ⚠️ HF state dict has NO vision params but Megatron DOES.")
                print("     → This usually means HF was loaded via AutoModelForCausalLM.")
                print("     → Raw checkpoint loading should now preserve vision weights.")
            print("=" * 80)

        print("\n" + "=" * 80)
        print("  Weight Consistency Check: HF vs Megatron-Bridge")
        print("=" * 80)
        print(f"  HF params        : {len(hf_state) if hf_state else 0}")
        print(f"  Megatron params  : {len(mg_state) if mg_state else 0}")
        print(f"  skip_vision      : {args.skip_vision}")
        print(f"  skip_fused       : {args.skip_fused}")
        print(f"  Matched          : {len(results)}")
        print(f"  Unmatched (HF)   : {len(unmatched_hf)}")
        print(f"  Unmatched (MG)   : {len(unmatched_mg)}")
        if skipped_fused_hf:
            print(f"  Skipped fused HF : {len(skipped_fused_hf)}")
        if skipped_fused_mg:
            print(f"  Skipped fused MG : {len(skipped_fused_mg)}")

        if unmatched_hf:
            print(f"\n  Unmatched HF params (first 10):")
            for name in unmatched_hf[:10]:
                print(f"    {name}")
        if unmatched_mg:
            print(f"\n  Unmatched Megatron params (first 10):")
            for name in unmatched_mg[:10]:
                print(f"    {name}")
        if skipped_fused_hf:
            print(f"\n  Skipped fused HF params (first 10):")
            for name in skipped_fused_hf[:10]:
                print(f"    {name}")

        print("-" * 80)

        # Top-K differences
        print(f"\n  Top-{args.top_k} parameters with largest max abs diff:")
        for i, r in enumerate(results[:args.top_k]):
            if r.get("shape_mismatch"):
                print(f"  {i+1:2d}. {r['hf_name']:50s} SHAPE MISMATCH {r['hf_shape']} vs {r['megatron_shape']}")
            else:
                print(
                    f"  {i+1:2d}. {r['hf_name']:50s} "
                    f"max_diff={r['max_abs_diff']:.6e} "
                    f"mean_diff={r['mean_abs_diff']:.6e} "
                    f"rel_diff={r['rel_diff']:.6e}"
                )

        # Overall stats
        valid_results = [r for r in results if not r.get("shape_mismatch") and r.get("max_abs_diff") is not None]
        overall_max = None
        overall_mean = None
        if valid_results:
            overall_max = max(r["max_abs_diff"] for r in valid_results)
            overall_mean = np.mean([r["mean_abs_diff"] for r in valid_results])
            print("-" * 80)
            print(f"  Overall max diff : {overall_max:.6e}")
            print(f"  Overall mean diff: {overall_mean:.6e}")
            if overall_max < 1e-6:
                print(f"  ✓ Weights are numerically identical (max diff < 1e-6)")
            elif overall_max < 1e-4:
                print(f"  ⚠️ Small weight differences (max diff < 1e-4), likely fp16/bf16 rounding")
            else:
                print(f"  ❌ Significant weight differences detected!")
        print("=" * 80)

        # Save JSON
        output = {
            "hf_checkpoint": args.hf_checkpoint,
            "tensor_model_parallel_size": args.tensor_model_parallel_size,
            "bf16": args.bf16,
            "skip_vision": args.skip_vision,
            "skip_fused": args.skip_fused,
            "matched_count": len(results),
            "unmatched_hf": unmatched_hf,
            "unmatched_mg": unmatched_mg,
            "skipped_fused_hf": skipped_fused_hf,
            "skipped_fused_mg": skipped_fused_mg,
            "overall_max_diff": overall_max,
            "overall_mean_diff": overall_mean,
            "top_differences": results[:args.top_k],
        }

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        logger.info(f"Weight check result saved to {output_path.resolve()}")

    # Step 4: All ranks cleanup
    if dist.is_initialized():
        dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    sys.exit(main())
