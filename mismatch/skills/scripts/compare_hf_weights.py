#!/usr/bin/env python3
"""Compare two HuggingFace checkpoints tensor-by-tensor.

Usage:
    python compare_hf_weights.py /path/to/checkpoint_a /path/to/checkpoint_b
    python compare_hf_weights.py /path/to/checkpoint_a /path/to/checkpoint_b --tol 1e-5
    python compare_hf_weights.py /path/to/checkpoint_a /path/to/checkpoint_b --keys model.layers.0

Outputs:
    - Keys only in A / only in B
    - Shape / dtype mismatches
    - Numerical differences (max abs diff, mean abs diff, relative diff)
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open


def get_key_to_filename_map(checkpoint_dir: Path) -> dict:
    """Read model.safetensors.index.json or scan directory."""
    index_file = checkpoint_dir / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("weight_map", {})

    # Fallback: scan all .safetensors files
    key_map = {}
    for st_file in sorted(checkpoint_dir.glob("*.safetensors")):
        with safe_open(str(st_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                key_map[key] = st_file.name
    return key_map


def load_tensor(checkpoint_dir: Path, key: str, key_map: dict) -> torch.Tensor | None:
    """Load a single tensor from checkpoint on demand.

    Returns None if the key is recorded in index but not actually present in the file.
    """
    filename = key_map.get(key)
    if filename is None:
        return None

    file_path = checkpoint_dir / filename
    if not file_path.exists():
        return None

    try:
        with safe_open(str(file_path), framework="pt", device="cpu") as f:
            return f.get_tensor(key)
    except Exception:
        return None


def compare_checkpoints(dir_a: Path, dir_b: Path, tol: float = 1e-6, key_filter: str = None):
    """Compare two HF checkpoints and print differences."""
    print(f"\n{'=' * 60}")
    print(f"Comparing:")
    print(f"  A: {dir_a}")
    print(f"  B: {dir_b}")
    print(f"  Tolerance (atol): {tol}")
    if key_filter:
        print(f"  Key filter: {key_filter}")
    print(f"{'=' * 60}\n")

    map_a = get_key_to_filename_map(dir_a)
    map_b = get_key_to_filename_map(dir_b)

    keys_a = set(map_a.keys())
    keys_b = set(map_b.keys())

    if key_filter:
        keys_a = {k for k in keys_a if key_filter in k}
        keys_b = {k for k in keys_b if key_filter in k}

    only_in_a = sorted(keys_a - keys_b)
    only_in_b = sorted(keys_b - keys_a)
    common = sorted(keys_a & keys_b)

    print(f"Keys in A: {len(keys_a)}")
    print(f"Keys in B: {len(keys_b)}")
    print(f"Common keys: {len(common)}")

    if only_in_a:
        print(f"\n[Only in A] ({len(only_in_a)} keys):")
        for k in only_in_a[:20]:
            print(f"  - {k}")
        if len(only_in_a) > 20:
            print(f"  ... and {len(only_in_a) - 20} more")

    if only_in_b:
        print(f"\n[Only in B] ({len(only_in_b)} keys):")
        for k in only_in_b[:20]:
            print(f"  - {k}")
        if len(only_in_b) > 20:
            print(f"  ... and {len(only_in_b) - 20} more")

    # Compare common keys
    shape_mismatch = []
    dtype_mismatch = []
    close_but_not_exact = []
    exact_match = 0
    large_diff = []
    failed_to_load = []  # key recorded in index but missing from actual file

    print(f"\n[Comparing {len(common)} common keys...]")
    for idx, key in enumerate(common):
        if idx % 100 == 0 and idx > 0:
            print(f"  ... {idx}/{len(common)} done")

        t_a = load_tensor(dir_a, key, map_a)
        t_b = load_tensor(dir_b, key, map_b)

        if t_a is None or t_b is None:
            failed_to_load.append((key, t_a is None, t_b is None))
            continue

        if t_a.shape != t_b.shape:
            shape_mismatch.append((key, tuple(t_a.shape), tuple(t_b.shape)))
            continue

        if t_a.dtype != t_b.dtype:
            dtype_mismatch.append((key, t_a.dtype, t_b.dtype))
            continue

        diff = (t_a - t_b).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        # Compute relative diff (avoid div by zero)
        t_a_max = t_a.abs().max().item()
        rel_diff = max_diff / (abs(t_a_max) + 1e-12)

        if max_diff == 0.0:
            exact_match += 1
        elif max_diff <= tol:
            close_but_not_exact.append((key, max_diff, mean_diff, rel_diff))
        else:
            large_diff.append((key, max_diff, mean_diff, rel_diff))

    print(f"\n{'=' * 60}")
    print(f"Exact match:      {exact_match}")
    print(f"Within tol:       {len(close_but_not_exact)}")
    print(f"Shape mismatch:   {len(shape_mismatch)}")
    print(f"Dtype mismatch:   {len(dtype_mismatch)}")
    print(f"Large diff:       {len(large_diff)}")
    print(f"Failed to load:   {len(failed_to_load)}")
    print(f"{'=' * 60}")

    if shape_mismatch:
        print(f"\n[Shape mismatch] ({len(shape_mismatch)} keys):")
        for key, s_a, s_b in shape_mismatch[:20]:
            print(f"  {key}: A={s_a}, B={s_b}")
        if len(shape_mismatch) > 20:
            print(f"  ... and {len(shape_mismatch) - 20} more")

    if dtype_mismatch:
        print(f"\n[Dtype mismatch] ({len(dtype_mismatch)} keys):")
        for key, d_a, d_b in dtype_mismatch[:20]:
            print(f"  {key}: A={d_a}, B={d_b}")

    if failed_to_load:
        print(f"\n[Failed to load] ({len(failed_to_load)} keys):")
        for key, a_missing, b_missing in failed_to_load[:20]:
            where = []
            if a_missing:
                where.append("A")
            if b_missing:
                where.append("B")
            print(f"  {key}: missing from {','.join(where)}")
        if len(failed_to_load) > 20:
            print(f"  ... and {len(failed_to_load) - 20} more")

    if close_but_not_exact:
        print(f"\n[Within tolerance but not exact] ({len(close_but_not_exact)} keys):")
        # Sort by max diff descending
        close_but_not_exact.sort(key=lambda x: x[1], reverse=True)
        for key, max_d, mean_d, rel_d in close_but_not_exact[:20]:
            print(f"  {key}: max_diff={max_d:.6e}, mean_diff={mean_d:.6e}, rel_diff={rel_d:.6e}")
        if len(close_but_not_exact) > 20:
            print(f"  ... and {len(close_but_not_exact) - 20} more")

    if large_diff:
        print(f"\n[Large difference (> tol)] ({len(large_diff)} keys):")
        large_diff.sort(key=lambda x: x[1], reverse=True)
        for key, max_d, mean_d, rel_d in large_diff[:30]:
            print(f"  {key}: max_diff={max_d:.6e}, mean_diff={mean_d:.6e}, rel_diff={rel_d:.6e}")
        if len(large_diff) > 30:
            print(f"  ... and {len(large_diff) - 30} more")

    print()
    return (
        len(large_diff) == 0
        and len(shape_mismatch) == 0
        and len(dtype_mismatch) == 0
        and len(failed_to_load) == 0
    )


def main():
    parser = argparse.ArgumentParser(description="Compare two HF checkpoints tensor-by-tensor")
    parser.add_argument("checkpoint_a", type=Path, help="Path to first HF checkpoint")
    parser.add_argument("checkpoint_b", type=Path, help="Path to second HF checkpoint")
    parser.add_argument("--tol", type=float, default=1e-6, help="Absolute tolerance for numerical comparison")
    parser.add_argument("--keys", type=str, default=None, help="Only compare keys containing this substring")
    args = parser.parse_args()

    if not args.checkpoint_a.exists():
        print(f"Error: checkpoint A does not exist: {args.checkpoint_a}")
        sys.exit(1)
    if not args.checkpoint_b.exists():
        print(f"Error: checkpoint B does not exist: {args.checkpoint_b}")
        sys.exit(1)

    ok = compare_checkpoints(args.checkpoint_a, args.checkpoint_b, tol=args.tol, key_filter=args.keys)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
