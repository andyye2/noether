# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Build auditable nested DrivAerML training subsets from the official split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from noether.data.datasets.cfd.caeml.drivaerml.split import DrivAerMLDefaultSplitIDs

DEFAULT_SIZES = (8, 16, 32, 64, 128, 256, 400)


def build_nested_manifest(seed: int, sizes: tuple[int, ...]) -> dict[str, Any]:
    """Create nested design-ID prefixes matching Noether's ShuffleWrapper.

    Args:
        seed: NumPy random-generator seed used by ``ShuffleWrapper``.
        sizes: Requested prefix lengths from the 400-run training split.

    Returns:
        JSON-serializable split manifest with validation checks.

    Raises:
        ValueError: If sizes are empty, duplicated, or outside 1..400.
    """
    if not sizes or len(set(sizes)) != len(sizes):
        raise ValueError("sizes must be non-empty and unique")
    sizes = tuple(sorted(sizes))
    if sizes[0] < 1 or sizes[-1] > 400:
        raise ValueError(f"sizes must be in [1, 400], got {sizes}")

    splits = DrivAerMLDefaultSplitIDs()
    shuffled_indices = np.arange(len(splits.train), dtype=int)
    np.random.default_rng(seed=seed).shuffle(shuffled_indices)
    ordered_train_ids = [splits.train[int(index)] for index in shuffled_indices]
    subsets = {str(size): ordered_train_ids[:size] for size in sizes}

    for smaller, larger in zip(sizes, sizes[1:], strict=False):
        if subsets[str(larger)][:smaller] != subsets[str(smaller)]:
            raise AssertionError(f"n={smaller} is not a prefix of n={larger}")
    if set(splits.train) & set(splits.val) or set(splits.train) & set(splits.test):
        raise AssertionError("Official DrivAerML splits overlap")

    available_ids = sorted(set(splits.train) | set(splits.val) | set(splits.test))
    unavailable_ids = sorted(set(range(1, 501)) - set(available_ids))
    return {
        "schema_version": 1,
        "dataset": "DrivAerML subsampled_10x",
        "seed": seed,
        "sampling_algorithm": "numpy.random.default_rng(seed).shuffle(arange(400))",
        "official_split_counts": {
            "train": len(splits.train),
            "val": len(splits.val),
            "test": len(splits.test),
            "available": len(available_ids),
        },
        "unavailable_run_ids": unavailable_ids,
        "val_run_ids": splits.val,
        "test_run_ids": splits.test,
        "ordered_train_run_ids": ordered_train_ids,
        "nested_train_run_ids": subsets,
    }


def main() -> None:
    """Write one deterministic nested-split manifest as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_nested_manifest(args.seed, tuple(args.sizes))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
