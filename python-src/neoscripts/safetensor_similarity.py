#!/usr/bin/env python3

# Simple script to calculate the cosine similarity between safetensors files (or sets of safetensors files).

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import click
import safetensors
import torch
from torch import Tensor
from tqdm import tqdm

# Only these dtypes are supported by CPU backend of PyTorch
safe_dtypes = [
    torch.float32,
    torch.float64,
    torch.float,
    torch.double,
]


def get_tensor_locations(
    file_sets: Tuple[Tuple[str, ...], ...],
) -> List[Dict[str, Tuple[Tuple[int, ...], str]]]:
    """Scan all file sets and return a mapping of tensor key -> (shape, file_path) for each set.

    For sharded models, each tensor is guaranteed to appear in only one shard per set.
    Returns a list of dicts, one per file set.
    """
    metadata_list: List[Dict[str, Tuple[Tuple[int, ...], str]]] = []
    for file_set in file_sets:
        tensor_map: Dict[str, Tuple[Tuple[int, ...], str]] = {}
        for file in file_set:
            with safetensors.safe_open(file, framework="pt") as f:
                for key in f.keys():
                    if key not in tensor_map:
                        shape = tuple(f.get_slice(key).get_shape())
                        tensor_map[key] = (shape, file)
        metadata_list.append(tensor_map)
    return metadata_list


def get_common_tensors(
    metadata_list: List[Dict[str, Tuple[Tuple[int, ...], str]]],
) -> Optional[Tuple[str, ...]]:
    """Get a list of tensor keys that are common across all file sets with the same shape."""
    if not metadata_list:
        return None

    common_keys: Optional[set] = None
    for tensor_map in metadata_list:
        keys = set(tensor_map.keys())
        if common_keys is None:
            common_keys = keys
        else:
            common_keys.intersection_update(keys)

    if not common_keys:
        return None

    # Filter by matching shapes across all sets
    result: List[str] = []
    for key in common_keys:
        shapes = [metadata_list[idx][key][0] for idx in range(len(metadata_list))]
        if all(s == shapes[0] for s in shapes):
            result.append(key)

    return tuple(result) if result else None


@torch.inference_mode()
def cosine_similarity(a: Tensor, b: Tensor) -> float:
    """Compute cosine similarity between two tensors (using FP32)."""
    a_flat = a.flatten().to(torch.float32)
    b_flat = b.flatten().to(torch.float32)

    # Handle NaN/Inf by replacing with zero
    a_flat = torch.nan_to_num(a_flat)
    b_flat = torch.nan_to_num(b_flat)

    dot = torch.dot(a_flat, b_flat)
    norm_a = torch.linalg.norm(a_flat)
    norm_b = torch.linalg.norm(b_flat)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return (dot / (norm_a * norm_b)).item()


def _load_tensors_from_file(file_path: str, keys: List[str]) -> Dict[str, Tensor]:
    """Load multiple tensors from a single safetensors file in one open call."""
    result: Dict[str, Tensor] = {}
    with safetensors.safe_open(file_path, framework="pt", device="cpu") as f:
        for key in keys:
            result[key] = f.get_tensor(key).to(torch.float32)
    return result


def _preload_tensors(
    common_tensors: Tuple[str, ...],
    metadata_list: List[Dict[str, Tuple[Tuple[int, ...], str]]],
) -> List[Dict[str, Tensor]]:
    """Pre-load all common tensors for each file set.

    Opens each safetensors file at most once per set.
    Returns a list of dicts: tensor_cache[set_idx][key] = Tensor
    """
    num_sets = len(metadata_list)
    tensor_cache: List[Dict[str, Tensor]] = [{} for _ in range(num_sets)]

    for set_idx, tensor_map in enumerate(metadata_list):
        # Group keys by file_path so each file is opened only once
        file_to_keys: Dict[str, List[str]] = {}
        for key in common_tensors:
            file_path = tensor_map[key][1]
            file_to_keys.setdefault(file_path, []).append(key)

        # Load all keys per file in one shot
        for file_path, keys in file_to_keys.items():
            loaded = _load_tensors_from_file(file_path, keys)
            tensor_cache[set_idx].update(loaded)

    return tensor_cache


def _compute_one_pair(
    key: str,
    tensor_i: Tensor,
    tensor_j: Tensor,
) -> Tuple[str, float]:
    """Compute cosine similarity for a single tensor pair.

    Standalone function (not a closure) so it works cleanly with ThreadPoolExecutor.
    """
    return key, cosine_similarity(tensor_i, tensor_j)


def calculate_similarities(
    common_tensors: Tuple[str, ...],
    metadata_list: List[Dict[str, Tuple[Tuple[int, ...], str]]],
    num_sets: int,
    num_workers: int = 0,
) -> Dict[Tuple[int, int], Dict]:
    """Calculate pairwise cosine similarities for all pairs of file sets.

    Returns a dict: (i, j) -> {
        "overall": float,  # mean cosine similarity across all tensors
        "per_tensor": {tensor_name: similarity, ...},
        "tensor_count": int,
    }
    """
    # Pre-load all tensors (each file opened only once per set)
    print("Loading tensors into memory...")
    tensor_cache = _preload_tensors(common_tensors, metadata_list)

    # Determine number of threads
    if num_workers <= 0:
        num_workers = min(os.cpu_count() or 4, 16)

    similarities: Dict[Tuple[int, int], Dict] = {}

    for i in range(num_sets):
        for j in range(i + 1, num_sets):
            per_tensor: Dict[str, float] = {}

            # PyTorch CPU ops release the GIL, so threads give good parallelism
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {}
                for key in common_tensors:
                    future = executor.submit(
                        _compute_one_pair,
                        key,
                        tensor_cache[i][key],
                        tensor_cache[j][key],
                    )
                    futures[future] = key

                t = tqdm(
                    as_completed(futures),
                    total=len(common_tensors),
                    desc=f"Set {i} vs Set {j}",
                    unit="tensor",
                    leave=False,
                )
                for future in t:
                    key, sim = future.result()
                    per_tensor[key] = sim

            total_sim = sum(per_tensor.values())
            avg_sim = total_sim / len(common_tensors) if common_tensors else 0.0
            similarities[(i, j)] = {
                "overall": avg_sim,
                "per_tensor": per_tensor,
                "tensor_count": len(common_tensors),
            }

    return similarities


def format_file_sets_info(file_sets: Tuple[Tuple[str, ...], ...]) -> str:
    lines = []
    for idx, file_set in enumerate(file_sets):
        if len(file_set) == 1:
            lines.append(f"  Set {idx}: {file_set[0]}")
        else:
            shard_str = ", ".join(file_set)
            lines.append(f"  Set {idx} ({len(file_set)} shards): {shard_str}")
    return "\n".join(lines)


def print_results(
    similarities: Dict[Tuple[int, int], Dict],
    file_sets: Tuple[Tuple[str, ...], ...],
    common_tensors: Tuple[str, ...],
    per_tensor: bool = False,
) -> None:
    """Print similarity results to stdout."""
    num_sets = len(file_sets)

    print(f"\nComparing {num_sets} model(s):")
    print(format_file_sets_info(file_sets))
    print(f"\nCommon tensors found: {len(common_tensors)}")

    if per_tensor:
        print()

    for (i, j), result in similarities.items():
        name_i = f"Set {i}"
        name_j = f"Set {j}"
        print(f"\n{name_i} vs {name_j}:")
        print(f"  Overall cosine similarity: {result['overall']:.6f}")
        print(f"  Number of tensors: {result['tensor_count']}")

        if per_tensor and result["tensor_count"] > 0:
            # Sort by similarity (ascending) to show most different first
            sorted_tensors = sorted(result["per_tensor"].items(), key=lambda x: x[1])
            print(f"  Per-tensor similarities (most different first):")
            for key, sim in sorted_tensors:
                print(f"    {key}: {sim:.6f}")

    # Overall summary across all pairs
    if len(similarities) > 0:
        print(f"\n--- Summary ---")
        for (i, j), result in similarities.items():
            print(f"  [{i}] vs [{j}]: {result['overall']:.6f}")


@click.command()
@click.argument(
    "files",
    nargs=-1,
    required=True,
)
@click.option(
    "-s",
    "--set-size",
    type=int,
    default=1,
    show_default=True,
    help="Number of files per set (for comparing sharded models). " "Example: if each model has 2 shards, use -s 2.",
)
@click.option(
    "--per-tensor",
    is_flag=True,
    default=False,
    help="Show per-tensor similarity details.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Output JSON file with detailed results.",
)
@click.option(
    "-w",
    "--num-workers",
    type=int,
    default=0,
    show_default=True,
    help="Number of worker threads for parallel computation. " "0 = auto (min(cpu_count, 16)).",
)
def __main__(
    files: Tuple[str, ...],
    set_size: int,
    per_tensor: bool,
    output: Optional[str],
    num_workers: int,
) -> None:
    """Calculate cosine similarity between safetensors files (or sets of safetensors files).

    Example usage:

        Compare two standalone models:

            safetensor_similarity.py model_a.safetensors model_b.safetensors

        Compare multiple models (all pairwise):

            safetensor_similarity.py a.safetensors b.safetensors c.safetensors

        Compare sharded models (2 shards per model):

            safetensor_similarity.py -s 2 model_a-00001.safetensors model_a-00002.safetensors model_b-00001.safetensors model_b-00002.safetensors
    """
    # Validate arguments
    if len(files) < 2:
        print("Error: At least 2 files are required.", file=sys.stderr)
        sys.exit(1)

    if len(files) % set_size != 0:
        print(
            f"Error: Number of files ({len(files)}) must be divisible by set size ({set_size}).",
            file=sys.stderr,
        )
        sys.exit(1)

    num_sets = len(files) // set_size
    if num_sets < 2:
        print(
            f"Error: At least 2 sets are required (got {num_sets} set(s)).",
            file=sys.stderr,
        )
        sys.exit(1)

    # Build file sets
    file_sets: Tuple[Tuple[str, ...], ...] = tuple(tuple(files[i * set_size : (i + 1) * set_size]) for i in range(num_sets))

    # Scan metadata
    print("Scanning tensors...")
    metadata_list = get_tensor_locations(file_sets)

    if not metadata_list:
        print("Error: No tensors found.", file=sys.stderr)
        sys.exit(1)

    # Find common tensors
    common_tensors = get_common_tensors(metadata_list)
    if common_tensors is None or len(common_tensors) == 0:
        print(
            "Error: No common tensors found (with matching shapes) across all file sets.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Calculate similarities
    print(f"Found {len(common_tensors)} common tensor(s). Computing similarities...\n")
    similarities = calculate_similarities(common_tensors, metadata_list, num_sets, num_workers)

    # Print results
    print_results(similarities, file_sets, common_tensors, per_tensor)

    # Save to JSON if requested
    if output:
        json_output: Dict = {
            "file_sets": [list(fs) for fs in file_sets],
            "common_tensors": list(common_tensors),
            "pairwise_results": {},
        }
        for (i, j), result in similarities.items():
            key = f"set_{i}_vs_set_{j}"
            json_output["pairwise_results"][key] = {
                "overall_similarity": round(result["overall"], 8),
                "tensor_count": result["tensor_count"],
                "per_tensor_similarities": {k: round(v, 8) for k, v in result["per_tensor"].items()},
            }
        with open(output, "w") as f:
            json.dump(json_output, f, indent=2)
        print(f"\nResults saved to: {output}")


if __name__ == "__main__":
    __main__()
