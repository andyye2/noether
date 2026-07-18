# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Materialize preregistered, hashed nested DrivAerML train manifests.

The manifest records both positional indices used by ``SubsetWrapper`` and the
corresponding DrivAerML design IDs.  Its SHA256 covers the canonical JSON payload
before the ``manifest_sha256`` field is attached.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from noether.data.datasets.cfd.caeml.drivaerml.split import DrivAerMLDefaultSplitIDs

DEFAULT_SIZES = (25, 50, 100, 200, 400)
DEFAULT_SEEDS = (1103, 2207, 3301, 4409, 5519, 6619, 7723, 8837)


def canonical_bytes(payload: dict[str, Any]) -> bytes:
    """Return the stable UTF-8 JSON representation used for hashing."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def build_manifest(seed: int, sizes: tuple[int, ...] = DEFAULT_SIZES) -> dict[str, Any]:
    """Build and validate one nested acquisition ladder.

    Args:
        seed: PCG64 seed used to permute the 400 base-dataset positions.
        sizes: Strictly increasing prefix sizes.

    Returns:
        JSON-serializable manifest with a content hash.
    """
    if tuple(sorted(set(sizes))) != sizes or not sizes:
        raise ValueError(f"sizes must be non-empty, unique, and sorted; got {sizes}")

    split = DrivAerMLDefaultSplitIDs()
    if sizes[0] < 1 or sizes[-1] > len(split.train):
        raise ValueError(f"sizes must lie in [1, {len(split.train)}]; got {sizes}")

    ordered_indices = np.arange(len(split.train), dtype=np.int64)
    np.random.default_rng(seed).shuffle(ordered_indices)
    base_indices = ordered_indices.tolist()
    run_ids = [split.train[index] for index in base_indices]
    subsets = {
        str(size): {
            "base_dataset_indices": base_indices[:size],
            "run_ids": run_ids[:size],
        }
        for size in sizes
    }

    for size in sizes:
        cell = subsets[str(size)]
        resolved = [split.train[index] for index in cell["base_dataset_indices"]]
        if resolved != cell["run_ids"]:
            raise AssertionError(f"base-index/run-ID mismatch at N={size}")
    for smaller, larger in zip(sizes, sizes[1:], strict=False):
        if subsets[str(larger)]["run_ids"][:smaller] != subsets[str(smaller)]["run_ids"]:
            raise AssertionError(f"N={smaller} is not a prefix of N={larger}")

    payload: dict[str, Any] = {
        "schema_version": 2,
        "dataset": "DrivAerML subsampled_10x",
        "split_implementation": ("noether.data.datasets.cfd.caeml.drivaerml.split.DrivAerMLDefaultSplitIDs"),
        "generator": "numpy.random.Generator(PCG64(seed)).shuffle(arange(400))",
        "seed": seed,
        "sizes": list(sizes),
        "official_split_counts": {
            "train": len(split.train),
            "val": len(split.val),
            "test": len(split.test),
            "hidden_test": len(split.hidden_test),
        },
        "official_val_run_ids": split.val,
        "official_test_run_ids": split.test,
        "official_hidden_test_run_ids": split.hidden_test,
        "ordered_base_dataset_indices": base_indices,
        "ordered_train_run_ids": run_ids,
        "subsets": subsets,
    }
    payload["manifest_sha256"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return payload


def verify_manifest(manifest: dict[str, Any]) -> None:
    """Validate the complete deterministic v2 manifest contract."""
    if not isinstance(manifest, dict):
        raise TypeError("manifest root must be a mapping")
    claimed_hash = manifest.get("manifest_sha256")
    if (
        not isinstance(claimed_hash, str)
        or len(claimed_hash) != 64
        or any(character not in "0123456789abcdef" for character in claimed_hash)
    ):
        raise ValueError("manifest_sha256 must be a lowercase SHA256")
    unhashed = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    actual_hash = hashlib.sha256(canonical_bytes(unhashed)).hexdigest()
    if claimed_hash != actual_hash:
        raise ValueError(f"manifest SHA256 mismatch: claimed={claimed_hash}, actual={actual_hash}")

    if manifest.get("schema_version") != 2:
        raise ValueError("manifest schema_version must be 2")
    expected_literals = {
        "dataset": "DrivAerML subsampled_10x",
        "split_implementation": "noether.data.datasets.cfd.caeml.drivaerml.split.DrivAerMLDefaultSplitIDs",
        "generator": "numpy.random.Generator(PCG64(seed)).shuffle(arange(400))",
    }
    for key, expected in expected_literals.items():
        if manifest.get(key) != expected:
            raise ValueError(f"manifest {key} must be {expected!r}")

    split = DrivAerMLDefaultSplitIDs()
    expected_counts = {
        "train": len(split.train),
        "val": len(split.val),
        "test": len(split.test),
        "hidden_test": len(split.hidden_test),
    }
    if manifest.get("official_split_counts") != expected_counts:
        raise ValueError("manifest official split counts differ from the current official split")
    expected_split_ids = {
        "official_val_run_ids": split.val,
        "official_test_run_ids": split.test,
        "official_hidden_test_run_ids": split.hidden_test,
    }
    for key, expected in expected_split_ids.items():
        if manifest.get(key) != expected:
            raise ValueError(f"manifest {key} differs from the current official split")

    seed = manifest.get("seed")
    if type(seed) is not int:
        raise ValueError("manifest seed must be an integer")
    sizes_raw = manifest.get("sizes")
    if not isinstance(sizes_raw, list) or any(type(size) is not int for size in sizes_raw):
        raise ValueError("manifest sizes must be an integer list")
    sizes = tuple(sizes_raw)
    if not sizes or tuple(sorted(set(sizes))) != sizes:
        raise ValueError("manifest sizes must be non-empty, unique, and sorted")
    if sizes[0] < 1 or sizes[-1] > len(split.train):
        raise ValueError(f"manifest sizes must lie in [1, {len(split.train)}]")

    expected_indices_array = np.arange(len(split.train), dtype=np.int64)
    np.random.default_rng(seed).shuffle(expected_indices_array)
    expected_indices = expected_indices_array.tolist()
    expected_run_ids = [split.train[index] for index in expected_indices]
    if manifest.get("ordered_base_dataset_indices") != expected_indices:
        raise ValueError("manifest ordered indices do not match the recorded PCG64 seed")
    if manifest.get("ordered_train_run_ids") != expected_run_ids:
        raise ValueError("manifest ordered train IDs do not match the deterministic index order")

    subsets = manifest.get("subsets")
    expected_subset_keys = {str(size) for size in sizes}
    if not isinstance(subsets, dict) or set(subsets) != expected_subset_keys:
        raise ValueError("manifest subset keys must exactly match sizes")
    for size in sizes:
        cell = subsets[str(size)]
        if not isinstance(cell, dict) or set(cell) != {"base_dataset_indices", "run_ids"}:
            raise ValueError(f"manifest subset N={size} has an invalid schema")
        if cell["base_dataset_indices"] != expected_indices[:size]:
            raise ValueError(f"manifest subset N={size} is not the deterministic index prefix")
        if cell["run_ids"] != expected_run_ids[:size]:
            raise ValueError(f"manifest subset N={size} is not the deterministic run-ID prefix")

    optional_split_aliases = {
        "val_run_ids": split.val,
        "test_run_ids": split.test,
    }
    for key, expected in optional_split_aliases.items():
        if key in manifest and manifest[key] != expected:
            raise ValueError(f"manifest {key} differs from the current official split")
    if "nested_train_run_ids" in manifest:
        nested = manifest["nested_train_run_ids"]
        if not isinstance(nested, dict) or set(nested) != expected_subset_keys:
            raise ValueError("manifest nested_train_run_ids keys must exactly match sizes")
        for size in sizes:
            if nested[str(size)] != expected_run_ids[:size]:
                raise ValueError(f"manifest nested train IDs for N={size} are not the deterministic prefix")


def main() -> None:
    """Write one JSON file per seed and verify each file after round-trip."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    args = parser.parse_args()

    sizes = tuple(args.sizes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        manifest = build_manifest(seed=seed, sizes=sizes)
        output_path = args.output_dir / f"drivaerml_nested_seed{seed}.json"
        output_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        verify_manifest(json.loads(output_path.read_text(encoding="utf-8")))
        print(f"{output_path} {manifest['manifest_sha256']}")


if __name__ == "__main__":
    main()
