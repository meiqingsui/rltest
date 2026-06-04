# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""
Layer-wise activation comparison between HF baseline and Megatron/SGLang.

Reads two .pt activation dumps and computes per-module / per-token differences
to pinpoint exactly where the outputs start diverging.

Usage:
    # Compare HF vs Megatron
    python compare_activations.py \
        --baseline hf_logp_result.activations.pt \
        --target megatron_activations_rank_0.pt \
        --output activation_compare.json

    # With ASCII plot of per-layer diff
    python compare_activations.py \
        --baseline hf_logp_result.activations.pt \
        --target megatron_activations_rank_0.pt \
        --plot
"""

import argparse
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
        description="Compare layer-wise activations between two model runs"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Path to baseline activations .pt file (e.g. HF)",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Path to target activations .pt file (e.g. Megatron)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="activation_compare.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Print ASCII bar chart of per-layer max abs diff",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-4,
        help="Warn if max abs diff exceeds this threshold for any layer",
    )
    return parser.parse_args()


def load_activations(path: str) -> dict:
    """Load activations .pt file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Activations file not found: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict of activations, got {type(data)}")
    return data


def normalize_module_name(name: str) -> str:
    """Normalize module names for cross-framework matching.

    Examples:
        HF:        "layer_0", "layer_1", "norm", "lm_head"
        Megatron:  "decoder.layers.0", "decoder.layers.1", "decoder.final_layernorm", "output_layer"
    """
    n = name.lower().replace("decoder.", "").replace("transformer.", "")
    n = n.replace("final_layernorm", "norm")
    n = n.replace("output_layer", "lm_head")
    n = n.replace("layers.", "layer_")
    n = n.replace("word_embeddings", "embed_tokens")
    n = n.replace("embedding.word_embeddings", "embed_tokens")
    return n


def match_layers(baseline: dict, target: dict) -> list[tuple[str, str]]:
    """Match layers between baseline and target by normalized name.

    Returns list of (baseline_name, target_name) tuples.
    """
    baseline_norm = {normalize_module_name(k): k for k in baseline.keys()}
    target_norm = {normalize_module_name(k): k for k in target.keys()}

    matched = []
    for norm_name, base_name in baseline_norm.items():
        if norm_name in target_norm:
            matched.append((base_name, target_norm[norm_name]))
        else:
            logger.warning(f"Layer '{base_name}' (normalized: '{norm_name}') not found in target")

    for norm_name, tgt_name in target_norm.items():
        if norm_name not in baseline_norm:
            logger.warning(f"Layer '{tgt_name}' (normalized: '{norm_name}') not found in baseline")

    # Sort by normalized name for consistent ordering
    matched.sort(key=lambda x: normalize_module_name(x[0]))
    return matched


def compute_activation_diff(baseline_tensor: torch.Tensor, target_tensor: torch.Tensor) -> dict:
    """Compute diff metrics between two activation tensors."""
    b = baseline_tensor.float().numpy()
    t = target_tensor.float().numpy()

    # Handle shape mismatches (e.g. TP sharding causes hidden dim mismatch)
    if b.shape != t.shape:
        # Try to align by minimum dimensions
        min_shape = tuple(min(bd, td) for bd, td in zip(b.shape, t.shape))
        slices = tuple(slice(0, ms) for ms in min_shape)
        b = b[slices]
        t = t[slices]
        logger.warning(f"Shape mismatch, aligned to {min_shape} for comparison")

    diff = b - t
    abs_diff = np.abs(diff)

    return {
        "baseline_shape": list(baseline_tensor.shape),
        "target_shape": list(target_tensor.shape),
        "aligned_shape": list(b.shape),
        "mean_baseline": float(b.mean()),
        "mean_target": float(t.mean()),
        "mean_abs_diff": float(abs_diff.mean()),
        "max_abs_diff": float(abs_diff.max()),
        "min_abs_diff": float(abs_diff.min()),
        "std_diff": float(diff.std()),
        "rel_diff": float(abs_diff.mean() / (np.abs(b).mean() + 1e-12)),
    }


def plot_ascii_layer_diff(layer_diffs: list[tuple[str, float]], width: int = 60):
    """Print ASCII bar chart of per-layer max abs diff."""
    if not layer_diffs:
        return

    max_val = max(d for _, d in layer_diffs)
    if max_val == 0:
        max_val = 1e-12

    print("\n  Per-layer max abs diff (baseline - target):")
    print("  " + "-" * width)
    for name, diff in layer_diffs:
        bar_len = int(width * min(diff / max_val, 1.0))
        bar = "█" * bar_len
        print(f"  {name:40s} {diff:.6f} |{bar}")
    print("  " + "-" * width)


def main() -> int:
    args = parse_args()

    # Load activations
    try:
        baseline = load_activations(args.baseline)
        target = load_activations(args.target)
    except Exception as e:
        logger.error(f"Failed to load activations: {e}")
        return 1

    logger.info(f"Baseline: {len(baseline)} modules from {args.baseline}")
    logger.info(f"Target:   {len(target)} modules from {args.target}")

    # Match layers
    matched = match_layers(baseline, target)
    if not matched:
        logger.error("No matching layers found between baseline and target")
        return 1

    logger.info(f"Matched {len(matched)} layers for comparison")

    # Compute diffs
    results = []
    layer_diffs = []
    first_large_diff_layer = None

    for base_name, tgt_name in matched:
        base_data = baseline[base_name]
        tgt_data = target[tgt_name]

        base_out = base_data["output"] if isinstance(base_data, dict) else base_data
        tgt_out = tgt_data["output"] if isinstance(tgt_data, dict) else tgt_data

        metrics = compute_activation_diff(base_out, tgt_out)
        metrics["baseline_name"] = base_name
        metrics["target_name"] = tgt_name
        results.append(metrics)

        max_diff = metrics["max_abs_diff"]
        layer_diffs.append((base_name, max_diff))

        if max_diff > args.threshold and first_large_diff_layer is None:
            first_large_diff_layer = base_name

    # Print summary
    print("\n" + "=" * 80)
    print("  Layer-wise Activation Comparison")
    print("=" * 80)
    print(f"  Baseline : {args.baseline}")
    print(f"  Target   : {args.target}")
    print(f"  Matched  : {len(matched)} layers")
    print(f"  Threshold: {args.threshold}")
    print("-" * 80)

    for metrics in results:
        status = "⚠️" if metrics["max_abs_diff"] > args.threshold else "✓"
        print(
            f"  {status} {metrics['baseline_name']:40s} "
            f"max_diff={metrics['max_abs_diff']:.6e} "
            f"mean_diff={metrics['mean_abs_diff']:.6e} "
            f"rel_diff={metrics['rel_diff']:.6e}"
        )

    print("-" * 80)
    overall_max = max(r["max_abs_diff"] for r in results)
    overall_mean = np.mean([r["mean_abs_diff"] for r in results])
    print(f"  Overall max diff : {overall_max:.6e}")
    print(f"  Overall mean diff: {overall_mean:.6e}")

    if first_large_diff_layer:
        print(f"  ⚠️ First layer exceeding threshold: {first_large_diff_layer}")
        print(f"     → Inspect layers BEFORE this point for weight/embedding issues")
        print(f"     → Inspect THIS layer and AFTER for compute/activation issues")
    else:
        print(f"  ✓ All layers within threshold")
    print("=" * 80)

    if args.plot:
        plot_ascii_layer_diff(layer_diffs)

    # Save JSON
    output = {
        "baseline_file": args.baseline,
        "target_file": args.target,
        "threshold": args.threshold,
        "first_large_diff_layer": first_large_diff_layer,
        "overall_max_diff": overall_max,
        "overall_mean_diff": overall_mean,
        "layer_results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Comparison result saved to {output_path.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
