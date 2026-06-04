# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""
Encode a text prompt into token IDs using a HuggingFace tokenizer.

Useful for preparing --test-tokens arguments for test_train_logp.py / test_rollout_logp.py.

Usage:
    # Basic: encode a prompt
    python encode_prompt.py \
        --model-path /mnt/sfs_turbo/models/Qwen3.5-9B/ \
        --prompt "What is the capital of France?"

    # Without special tokens (default, aligns with test_rollout_logp.py input_ids)
    python encode_prompt.py \
        --model-path /mnt/sfs_turbo/models/Qwen3.5-9B/ \
        --prompt "What is the capital of France?" \
        --no-special-tokens

    # With special tokens (includes BOS/EOS)
    python encode_prompt.py \
        --model-path /mnt/sfs_turbo/models/Qwen3.5-9B/ \
        --prompt "What is the capital of France?" \
        --add-special-tokens

    # Save to file
    python encode_prompt.py \
        --model-path /mnt/sfs_turbo/models/Qwen3.5-9B/ \
        --prompt "What is the capital of France?" \
        --output prompt_tokens.txt
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode text prompt to token IDs")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to HuggingFace model (used for tokenizer loading)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt to encode",
    )
    parser.add_argument(
        "--add-special-tokens",
        action="store_true",
        help="Add special tokens (BOS/EOS) during encoding (default: False)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Trust remote code for tokenizer loading",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional file to save token IDs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logger.info(f"Loading tokenizer from {args.model_path}")
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError("transformers is required. Install: pip install transformers") from e

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )

    # Encode
    add_special_tokens = args.add_special_tokens
    token_ids = tokenizer.encode(args.prompt, add_special_tokens=add_special_tokens)
    tokens_str = ", ".join(str(tid) for tid in token_ids)

    # Decode each token individually for inspection
    decoded_tokens = []
    for tid in token_ids:
        try:
            decoded = tokenizer.decode([tid], skip_special_tokens=False)
            decoded_tokens.append((tid, repr(decoded)))
        except Exception:
            decoded_tokens.append((tid, "<decode_error>"))

    # Print summary
    print("\n" + "=" * 60)
    print("  Prompt Tokenization Result")
    print("=" * 60)
    print(f"  Model path       : {args.model_path}")
    print(f"  Prompt           : {args.prompt[:80]}")
    print(f"  Add special tok  : {add_special_tokens}")
    print(f"  Token count      : {len(token_ids)}")
    print("-" * 60)
    print(f"  Token IDs        : {tokens_str}")
    print("-" * 60)
    print("  Per-token breakdown:")
    for i, (tid, decoded) in enumerate(decoded_tokens):
        print(f"    {i:3d}: {tid:8d} -> {decoded}")
    print("=" * 60)

    # Print ready-to-use command examples
    print("\n  Ready-to-use commands:")
    print(f"    test_rollout_logp.py:")
    print(f"      --test-tokens \"{tokens_str}\"")
    print(f"    test_train_logp.py:")
    print(f"      --test-tokens \"{tokens_str}\" --response-length <N>")
    print("=" * 60)

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(f"# Model: {args.model_path}\n")
            f.write(f"# Prompt: {args.prompt}\n")
            f.write(f"# Add special tokens: {add_special_tokens}\n")
            f.write(f"# Token count: {len(token_ids)}\n")
            f.write(f"token_ids = [{tokens_str}]\n")
            f.write("\n# Per-token breakdown\n")
            for i, (tid, decoded) in enumerate(decoded_tokens):
                f.write(f"# {i:3d}: {tid:8d} -> {decoded}\n")
        logger.info(f"Token IDs saved to {output_path.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
