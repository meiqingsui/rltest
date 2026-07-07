# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""
Script to extract logp values from SGLang rollout service to debug Train-Inference Mismatch.

This script sends a sequence of token IDs to an SGLang inference endpoint and retrieves
log probabilities. Results are saved as JSON for offline comparison via compare_train_rollout_logp.py.

Usage:
    export ASCEND_RT_VISIBLE_DEVICES=12,13,14,15
    python3 -m sglang.launch_server --model-path /mnt/sfs_turbo/models/Qwen3.5-9B/ --page-size 1 --attention-backend ascend --disable-cuda-graph --port 30000
    # Basic: send tokens as prompt and get rollout logp for generated tokens
    python test_rollout_logp.py \
        --sglang-url http://localhost:30000/generate \
        --test-tokens "12, 134, 45, 10, 89" \
        --max-new-tokens 5 \
        --temperature 0.0 \
        --output rollout_logp_result.json

    # Request prompt logprobs (SGLang extension, max_new_tokens=0)
    python test_rollout_logp.py \
        --sglang-url http://localhost:30000/generate \
        --test-tokens "12, 134, 45, 10, 89" \
        --max-new-tokens 0 \
        --return-prompt-logprob \
        --output rollout_logp_result.json

    # Compare with training result (use compare_train_rollout_logp.py)
    python compare_train_rollout_logp.py \
        --train-logp response_logp_rank_0.pt \
        --rollout-logp rollout_logp_result.json \
        --output compare_result.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract logp from SGLang rollout for train-inference mismatch debugging"
    )
    parser.add_argument(
        "--sglang-url",
        type=str,
        required=True,
        help="SGLang /generate endpoint URL, e.g. http://localhost:30000/generate",
    )
    parser.add_argument(
        "--test-tokens",
        type=str,
        default="10,11,12,13,14,15,16,17",
        help="Comma-separated list of token IDs to send as input_ids, or path to a token file (.txt/.json/.pt)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Max new tokens to generate. Default = len(test-tokens)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = greedy)",
    )
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--top-k", type=int, default=-1, help="Top-k sampling")
    parser.add_argument(
        "--return-prompt-logprob",
        action="store_true",
        help="Request prompt token logprobs (SGLang extension)",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Ignore EOS token and force generation of max_new_tokens tokens",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="rollout_logp_result.json",
        help="Output JSON file path",
    )
    return parser.parse_args()


def _send_request(url: str, payload: dict) -> dict:
    """Send POST request to SGLang /generate and return JSON response."""
    try:
        import httpx
    except ImportError as e:
        raise ImportError("httpx is required. Install: pip install httpx") from e

    logger.info(f"POST {url}")
    logger.info(f"Payload sampling_params: {json.dumps(payload.get('sampling_params', {}), indent=2)}")
    logger.info(f"Payload input_ids length: {len(payload.get('input_ids', []))}")

    with httpx.Client(timeout=httpx.Timeout(120.0)) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"SGLang returned error: {data['error']}")

    return data


def extract_rollout_logp(response: dict) -> dict:
    """Parse SGLang /generate response and extract logp information.

    SGLang returns:
        {
            "text": "generated text",
            "meta_info": {
                "output_token_logprobs": [[logp, token_id], ...],
                "prompt_logprobs": [logp, ...],  # optional
                "prompt_tokens": int,
                "completion_tokens": int,
                "cached_tokens": int,
                "finish_reason": {...},
                ...
            }
        }
    """
    result = {
        "text": response.get("text", ""),
        "meta_info": {},
        "output_token_logprobs": [],
        "prompt_logprobs": [],
        "generated_token_ids": [],
        "generated_logprobs": [],
    }

    meta_info = response.get("meta_info", {})

    # output_token_logprobs: list of [logp, token_id] for generated tokens
    if "output_token_logprobs" in meta_info:
        result["output_token_logprobs"] = meta_info["output_token_logprobs"]
        result["generated_logprobs"] = [item[0] for item in meta_info["output_token_logprobs"]]
        result["generated_token_ids"] = [item[1] for item in meta_info["output_token_logprobs"]]
        logger.info(f"Got {len(result['output_token_logprobs'])} output token logprobs")

    # prompt logprobs if available (SGLang extension)
    if "prompt_logprobs" in meta_info:
        result["prompt_logprobs"] = meta_info["prompt_logprobs"]
        logger.info(f"Got {len(result['prompt_logprobs'])} prompt token logprobs")

    # Copy remaining meta_info (excluding heavy fields already extracted)
    skip_keys = {"output_token_logprobs", "prompt_logprobs", "routed_experts"}
    for k, v in meta_info.items():
        if k not in skip_keys:
            result["meta_info"][k] = v

    return result


def main() -> int:
    args = parse_args()

    # Parse token IDs (same format as test_train_logp.py)
    token_ids = parse_test_tokens(args.test_tokens)
    logger.info(f"Input token IDs: {token_ids} (length={len(token_ids)})")

    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else len(token_ids)
    logger.info(f"max_new_tokens: {max_new_tokens}")

    # Build SGLang /generate payload
    # Reference: relax/engine/rollout/sglang_rollout.py::generate()
    payload = {
        "input_ids": token_ids,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        },
        "return_logprob": True,
    }

    if args.ignore_eos:
        payload["sampling_params"]["ignore_eos"] = True

    if args.return_prompt_logprob:
        payload["return_prompt_logprob"] = True

    # Send request to SGLang
    try:
        response = _send_request(args.sglang_url, payload)
    except Exception as e:
        logger.error(f"Request to SGLang failed: {e}")
        return 1

    # Extract logp from response
    result = extract_rollout_logp(response)
    result["request"] = {
        "sglang_url": args.sglang_url,
        "input_token_ids": token_ids,
        "max_new_tokens": max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }

    # Save result to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Result saved to {output_path.resolve()}")

    # Print summary
    print("\n" + "=" * 60)
    print("  Rollout Logp Test Result")
    print("=" * 60)
    print(f"  SGLang URL       : {args.sglang_url}")
    print(f"  Input token IDs  : {token_ids}")
    print(f"  max_new_tokens   : {max_new_tokens}")
    print(f"  Generated text   : {result['text'][:200] if result['text'] else '(empty)'}")
    print(f"  Generated tokens : {result['generated_token_ids']}")
    print(f"  Output logprobs  : {len(result['output_token_logprobs'])} tokens")
    print(f"  Prompt logprobs  : {len(result['prompt_logprobs'])} tokens")
    if result["generated_logprobs"]:
        mean_lp = sum(result["generated_logprobs"]) / len(result["generated_logprobs"])
        print(f"  Mean gen logp    : {mean_lp:.6f}")
        print(f"  Min  gen logp    : {min(result['generated_logprobs']):.6f}")
        print(f"  Max  gen logp    : {max(result['generated_logprobs']):.6f}")
    if len(result["generated_logprobs"]) != max_new_tokens:
        print(f"  ⚠️  Token count mismatch: expected {max_new_tokens}, got {len(result['generated_logprobs'])}")
        print(f"     Hint: model may have hit EOS early. Re-run with --ignore-eos to force full generation.")
    if result["prompt_logprobs"]:
        mean_plp = sum(result["prompt_logprobs"]) / len(result["prompt_logprobs"])
        print(f"  Mean prompt logp : {mean_plp:.6f}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
