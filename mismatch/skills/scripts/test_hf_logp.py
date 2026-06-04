# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""
Script to extract response-only logp from HuggingFace Transformers (native HF inference).

This serves as a clean, third-party baseline for Train-Inference Mismatch debugging:
- test_train_logp.py  (Megatron-LM)  vs  test_hf_logp.py  → isolate Megatron-specific deviations
- test_rollout_logp.py (SGLang)      vs  test_hf_logp.py  → isolate SGLang-specific deviations

Input/output format is kept identical to test_train_logp.py for easy interchangeability.

Usage:
    python test_hf_logp.py \
        --hf-checkpoint /mnt/sfs_turbo/models/Qwen3.5-9B/ \
        --test-tokens "12, 134, 45, 10, 89, 100, 200, 300" \
        --response-length 3 \
        --bf16 \
        --output hf_logp_result.pt
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


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
        help="Comma-separated token IDs (full sequence: prompt + response)",
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
    return parser.parse_args()


def load_model(args: argparse.Namespace):
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

    logger.info(f"Loading model from {args.hf_checkpoint} (dtype={dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_checkpoint,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        device_map=None,  # we handle device placement manually
    )

    device = torch.device(args.device) if args.device else get_device()
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded on {device}")
    return model, tokenizer


def compute_response_logp(
    model,
    token_ids: list[int],
    response_length: int | None,
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

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits.float()  # [1, total_length, vocab_size]

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
    return result


def main() -> int:
    args = parse_args()

    # Parse token IDs
    token_ids = [int(x.strip()) for x in args.test_tokens.split(",")]
    logger.info(f"Input token IDs: {token_ids} (length={len(token_ids)})")

    if len(token_ids) < 2:
        logger.error("Need at least 2 tokens (1 prompt + 1 response)")
        return 1

    # Load model
    try:
        model, tokenizer = load_model(args)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1

    # Compute logp
    try:
        result = compute_response_logp(model, token_ids, args.response_length)
    except Exception as e:
        logger.error(f"Failed to compute logp: {e}")
        return 1

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)
    logger.info(f"Result saved to {output_path.resolve()}")

    # Print summary
    print("\n" + "=" * 60)
    print("  HF Baseline Logp Result")
    print("=" * 60)
    print(f"  Model            : {args.hf_checkpoint}")
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
