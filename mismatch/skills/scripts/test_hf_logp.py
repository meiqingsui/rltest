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


def compute_response_logp(
    model,
    token_ids: list[int],
    response_length: int | None,
    save_activations: bool = False,
    rank: int = 0,
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

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits.float()  # [1, total_length, vocab_size]

    if hooks:
        remove_hooks(hooks)
        logger.info(f"[Rank {rank}] Captured {len(activations)} activation layers")

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
            save_activations=args.save_activations,
            rank=rank,
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
