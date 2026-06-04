# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""
Standalone comparison script for train-side vs rollout-side response logp.

Reads:
    - Train output: response_logp_rank_{rank}.pt (from test_train_logp.py)
    - Rollout output: rollout_logp_result.json (from test_rollout_logp.py)

Computes per-token absolute/relative differences and aggregates mismatch metrics.

Usage:
    # Basic comparison
    python compare_train_rollout_logp.py \
        --train-logp response_logp_rank_0.pt \
        --rollout-logp rollout_logp_result.json \
        --output compare_result.json

    # Compare multiple train ranks
    python compare_train_rollout_logp.py \
        --train-logp "response_logp_rank_*.pt" \
        --rollout-logp rollout_logp_result.json \
        --output compare_result.json

    # With CSV dump and console plot
    python compare_train_rollout_logp.py \
        --train-logp response_logp_rank_0.pt \
        --rollout-logp rollout_logp_result.json \
        --output compare_result.json \
        --csv compare.csv \
        --plot
"""

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare train-side and rollout-side response logp for mismatch analysis"
    )
    parser.add_argument(
        "--train-logp",
        type=str,
        required=True,
        help="Path to train-side .pt file (supports glob like 'response_logp_rank_*.pt')",
    )
    parser.add_argument(
        "--rollout-logp",
        type=str,
        required=True,
        help="Path to rollout-side JSON file (e.g. rollout_logp_result.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="compare_result.json",
        help="Output JSON file for comparison result",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional CSV output for per-token comparison",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Print ASCII plot of per-token logp difference",
    )
    parser.add_argument(
        "--token-offset",
        type=int,
        default=0,
        help="Offset to align rollout tokens with train tokens (default 0)",
    )
    return parser.parse_args()


def load_train_data(train_path: str) -> dict | None:
    """Load train-side logp data from .pt file."""
    path = Path(train_path)
    if not path.exists():
        logger.warning(f"Train logp file not found: {train_path}")
        return None

    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        logger.warning(f"Failed to load train logp {train_path}: {e}")
        return None

    # Handle both old format (plain tensor) and new format (dict)
    if isinstance(data, torch.Tensor):
        logger.info(f"Loaded train logp from {train_path} (legacy tensor format, len={data.numel()})")
        return {
            "response_tokens": None,
            "response_logp": data,
            "prompt_length": 0,
            "response_length": data.numel(),
            "mean_logp": data.mean().item(),
        }

    if isinstance(data, dict):
        logger.info(
            f"Loaded train logp from {train_path} "
            f"(prompt_len={data.get('prompt_length')}, response_len={data.get('response_length')})"
        )
        return data

    logger.warning(f"Unknown train data type: {type(data)}")
    return None


def load_rollout_data(rollout_path: str) -> dict | None:
    """Load rollout-side logp data from JSON file."""
    path = Path(rollout_path)
    if not path.exists():
        logger.warning(f"Rollout logp file not found: {rollout_path}")
        return None

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load rollout logp {rollout_path}: {e}")
        return None

    generated_token_ids = data.get("generated_token_ids", [])
    generated_logprobs = data.get("generated_logprobs", [])
    prompt_logprobs = data.get("prompt_logprobs", [])

    logger.info(
        f"Loaded rollout logp from {rollout_path} "
        f"(gen_tokens={len(generated_token_ids)}, prompt_logprobs={len(prompt_logprobs)})"
    )
    return {
        "generated_token_ids": generated_token_ids,
        "generated_logprobs": generated_logprobs,
        "prompt_logprobs": prompt_logprobs,
        "request": data.get("request", {}),
    }


def align_sequences(
    train_data: dict,
    rollout_data: dict,
    token_offset: int = 0,
) -> tuple[list[int], list[float], list[int], list[float], int]:
    """Align train and rollout token sequences for comparison.

    Returns:
        (aligned_train_tokens, aligned_train_logp, aligned_rollout_tokens, aligned_rollout_logp, offset_used)
    """
    train_tokens = train_data.get("response_tokens")
    train_logp = train_data["response_logp"]
    if isinstance(train_logp, torch.Tensor):
        train_logp = train_logp.tolist()
    if train_tokens is not None and isinstance(train_tokens, torch.Tensor):
        train_tokens = train_tokens.tolist()

    rollout_tokens = rollout_data["generated_token_ids"]
    rollout_logp = rollout_data["generated_logprobs"]

    # Default: no token ids available on train side
    if train_tokens is None:
        logger.warning(
            "Train-side response_tokens not found (legacy format). "
            "Alignment will be done purely by position."
        )
        train_tokens = list(range(len(train_logp)))

    # Apply offset
    if token_offset != 0:
        logger.info(f"Applying token offset={token_offset} to rollout sequence")
        if token_offset > 0:
            rollout_tokens = rollout_tokens[token_offset:]
            rollout_logp = rollout_logp[token_offset:]
        else:
            train_tokens = train_tokens[-token_offset:]
            train_logp = train_logp[-token_offset:]

    # Truncate to common length
    common_len = min(len(train_logp), len(rollout_logp))
    if common_len == 0:
        raise ValueError("No overlapping tokens after alignment.")

    if len(train_logp) != len(rollout_logp):
        logger.warning(
            f"Length mismatch: train={len(train_logp)}, rollout={len(rollout_logp)}. "
            f"Truncating to common_len={common_len}."
        )

    train_tokens = train_tokens[:common_len]
    train_logp = train_logp[:common_len]
    rollout_tokens = rollout_tokens[:common_len]
    rollout_logp = rollout_logp[:common_len]

    # Token mismatch warning
    mismatched_positions = [i for i, (t, r) in enumerate(zip(train_tokens, rollout_tokens)) if t != r]
    if mismatched_positions:
        logger.warning(
            f"Token mismatch at {len(mismatched_positions)} positions: "
            f"{mismatched_positions[:10]}{'...' if len(mismatched_positions) > 10 else ''}"
        )

    return train_tokens, train_logp, rollout_tokens, rollout_logp, token_offset


def compute_metrics(train_logp: list[float], rollout_logp: list[float]) -> dict:
    """Compute mismatch metrics between aligned logp sequences."""
    train_arr = np.array(train_logp, dtype=np.float64)
    rollout_arr = np.array(rollout_logp, dtype=np.float64)

    diff = train_arr - rollout_arr
    abs_diff = np.abs(diff)
    train_prob = np.exp(train_arr)
    rollout_prob = np.exp(rollout_arr)
    prob_abs_diff = np.abs(train_prob - rollout_prob)

    metrics = {
        "token_count": len(train_logp),
        "train_mean_logp": float(train_arr.mean()),
        "rollout_mean_logp": float(rollout_arr.mean()),
        "logp_mean_abs_diff": float(abs_diff.mean()),
        "logp_max_abs_diff": float(abs_diff.max()),
        "logp_min_abs_diff": float(abs_diff.min()),
        "logp_std_diff": float(diff.std()),
        "prob_mean_abs_diff": float(prob_abs_diff.mean()),
        "prob_max_abs_diff": float(prob_abs_diff.max()),
    }

    # Per-token breakdown
    per_token = []
    for i in range(len(train_logp)):
        per_token.append(
            {
                "position": i,
                "train_logp": float(train_arr[i]),
                "rollout_logp": float(rollout_arr[i]),
                "abs_diff": float(abs_diff[i]),
                "rel_diff": float(abs_diff[i] / (abs(train_arr[i]) + 1e-12)),
                "train_prob": float(train_prob[i]),
                "rollout_prob": float(rollout_prob[i]),
                "prob_abs_diff": float(prob_abs_diff[i]),
            }
        )

    # Quantiles
    for q in [0.5, 0.9, 0.95, 0.99]:
        metrics[f"logp_abs_diff_p{int(q*100)}"] = float(np.quantile(abs_diff, q))

    return metrics, per_token


def plot_ascii(train_logp: list[float], rollout_logp: list[float], diff: list[float], width: int = 60):
    """Print ASCII bar chart of per-token logp difference."""
    if not diff:
        return

    max_abs = max(abs(d) for d in diff)
    if max_abs == 0:
        max_abs = 1e-9

    print("\n  Per-token logp difference (train - rollout):")
    print("  " + "-" * width)
    for i, d in enumerate(diff):
        bar_len = int(width * abs(d) / max_abs)
        bar = "█" * bar_len
        sign = "+" if d >= 0 else "-"
        print(f"  tok{i:3d} {sign}{abs(d):.6f} |{bar}")
    print("  " + "-" * width)


def main() -> int:
    args = parse_args()

    # Load rollout data first (single source)
    rollout_data = load_rollout_data(args.rollout_logp)
    if rollout_data is None:
        logger.error("Failed to load rollout data. Exiting.")
        return 1

    # Resolve train logp paths (may be glob)
    train_paths = sorted(glob.glob(args.train_logp))
    if not train_paths:
        train_paths = [args.train_logp]

    all_results = []
    for train_path in train_paths:
        train_data = load_train_data(train_path)
        if train_data is None:
            continue

        try:
            train_tokens, train_logp, rollout_tokens, rollout_logp, offset = align_sequences(
                train_data, rollout_data, args.token_offset
            )
        except ValueError as e:
            logger.error(f"Alignment failed for {train_path}: {e}")
            continue

        metrics, per_token = compute_metrics(train_logp, rollout_logp)

        result = {
            "train_file": str(train_path),
            "rollout_file": str(args.rollout_logp),
            "alignment_offset": offset,
            "metrics": metrics,
            "per_token": per_token,
        }
        all_results.append(result)

        # Print summary
        print("\n" + "=" * 60)
        print(f"  Comparison: {Path(train_path).name} vs {Path(args.rollout_logp).name}")
        print("=" * 60)
        print(f"  Aligned tokens      : {metrics['token_count']}")
        print(f"  Train mean logp     : {metrics['train_mean_logp']:.6f}")
        print(f"  Rollout mean logp   : {metrics['rollout_mean_logp']:.6f}")
        print(f"  LogP mean abs diff  : {metrics['logp_mean_abs_diff']:.6f}")
        print(f"  LogP max abs diff   : {metrics['logp_max_abs_diff']:.6f}")
        print(f"  LogP std diff       : {metrics['logp_std_diff']:.6f}")
        print(f"  Prob mean abs diff  : {metrics['prob_mean_abs_diff']:.6f}")
        print(f"  Prob max abs diff   : {metrics['prob_max_abs_diff']:.6f}")
        print("-" * 60)
        for q in [0.5, 0.9, 0.95, 0.99]:
            print(f"  LogP abs diff p{int(q*100):02d} : {metrics[f'logp_abs_diff_p{int(q*100)}']:.6f}")
        print("=" * 60)

        if args.plot:
            diff = [t - r for t, r in zip(train_logp, rollout_logp)]
            plot_ascii(train_logp, rollout_logp, diff)

        # Optional CSV
        if args.csv:
            csv_path = Path(args.csv)
            if len(train_paths) > 1:
                csv_path = csv_path.parent / f"{csv_path.stem}_{Path(train_path).stem}{csv_path.suffix}"
            try:
                import csv

                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=per_token[0].keys())
                    writer.writeheader()
                    writer.writerows(per_token)
                logger.info(f"Per-token CSV saved to {csv_path}")
            except Exception as e:
                logger.warning(f"Failed to write CSV: {e}")

    # Save aggregated JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "comparisons": all_results,
                "summary": {
                    "num_comparisons": len(all_results),
                    "mean_abs_diff": float(np.mean([r["metrics"]["logp_mean_abs_diff"] for r in all_results])) if all_results else 0.0,
                },
            },
            f,
            indent=2,
            default=str,
        )
    logger.info(f"Comparison result saved to {output_path.resolve()}")

    return 0 if all_results else 1


if __name__ == "__main__":
    sys.exit(main())
