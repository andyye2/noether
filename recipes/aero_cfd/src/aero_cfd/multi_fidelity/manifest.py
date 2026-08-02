# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Deterministic nested training manifests and their consumer-side validation.

A manifest freezes one acquisition ladder: a PCG64 permutation of the 400
official DrivAerML training positions, cut into nested prefixes.  Because the
permutation is reproducible from the seed alone, :func:`verify_manifest`
recomputes it rather than trusting the file.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from noether.data.datasets.cfd.caeml.drivaerml.split import DrivAerMLDefaultSplitIDs

from .integrity import GitState, canonical_json_bytes, is_git_commit, is_sha256

DEFAULT_SIZES = (25, 50, 100, 200, 400)
DEFAULT_SEEDS = (1103, 2207, 3301, 4409, 5519, 6619, 7723, 8837)
MANIFEST_SCHEMA_VERSION = 2
GENERATOR = "numpy.random.Generator(PCG64(seed)).shuffle(arange(400))"
SPLIT_IMPLEMENTATION = "noether.data.datasets.cfd.caeml.drivaerml.split.DrivAerMLDefaultSplitIDs"
DATASET = "DrivAerML subsampled_10x"


def _ordered_positions(seed: int, train_count: int) -> list[int]:
    """Return the deterministic PCG64 permutation of base-dataset positions."""
    positions = np.arange(train_count, dtype=np.int64)
    np.random.default_rng(seed).shuffle(positions)
    return [int(position) for position in positions]


def build_manifest(seed: int, sizes: tuple[int, ...] = DEFAULT_SIZES) -> dict[str, Any]:
    """Build and validate one nested acquisition ladder.

    Args:
        seed: PCG64 seed used to permute the 400 base-dataset positions.
        sizes: Strictly increasing prefix sizes.

    Returns:
        JSON-serializable manifest with a canonical-payload content hash.

    Raises:
        ValueError: If the sizes are empty, unsorted, duplicated, or outside
            the official training split.
    """
    if not sizes or tuple(sorted(set(sizes))) != sizes:
        raise ValueError(f"sizes must be non-empty, unique, and sorted; got {sizes}")
    split = DrivAerMLDefaultSplitIDs()
    if sizes[0] < 1 or sizes[-1] > len(split.train):
        raise ValueError(f"sizes must lie in [1, {len(split.train)}]; got {sizes}")

    base_indices = _ordered_positions(seed, len(split.train))
    run_ids = [split.train[index] for index in base_indices]
    payload: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": DATASET,
        "split_implementation": SPLIT_IMPLEMENTATION,
        "generator": GENERATOR,
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
        "subsets": {
            str(size): {"base_dataset_indices": base_indices[:size], "run_ids": run_ids[:size]} for size in sizes
        },
    }
    payload["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return payload


def build_study_manifest(
    seed: int,
    sizes: tuple[int, ...],
    *,
    study_id: str,
    protocol_sha256: str,
    git_state: GitState,
) -> dict[str, Any]:
    """Build one hashed manifest accepted by the training and statistics tools.

    The study fields bind the ladder to a preregistration and to the exact
    implementation state, and the aliases keep the statistics tool free of
    schema-version branches.

    Args:
        seed: PCG64 subset seed of this replicate.
        sizes: Nested prefix sizes.
        study_id: Study identifier from the protocol.
        protocol_sha256: Raw-file SHA256 of the protocol.
        git_state: Implementation state that produced the manifest.

    Returns:
        JSON-serializable manifest whose ``manifest_sha256`` covers every
        field above.
    """
    manifest = build_manifest(seed=seed, sizes=sizes)
    manifest.pop("manifest_sha256")
    manifest["study_id"] = study_id
    manifest["protocol_sha256"] = protocol_sha256
    manifest["implementation_git_commit"] = git_state.commit
    manifest["implementation_git_dirty"] = git_state.dirty
    manifest["val_run_ids"] = list(manifest["official_val_run_ids"])
    manifest["test_run_ids"] = list(manifest["official_test_run_ids"])
    manifest["nested_train_run_ids"] = {str(size): list(manifest["subsets"][str(size)]["run_ids"]) for size in sizes}
    manifest["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    verify_manifest(manifest)
    return manifest


def verify_manifest(manifest: dict[str, Any]) -> None:
    """Validate the complete deterministic manifest contract.

    Args:
        manifest: Parsed manifest mapping.

    Raises:
        TypeError: If the root is not a mapping.
        ValueError: If any field is missing, malformed, or disagrees with the
            permutation recomputed from the recorded seed.
    """
    if not isinstance(manifest, dict):
        raise TypeError("manifest root must be a mapping")
    claimed_hash = manifest.get("manifest_sha256")
    if not is_sha256(claimed_hash):
        raise ValueError("manifest_sha256 must be a lowercase SHA256")
    unhashed = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    actual_hash = hashlib.sha256(canonical_json_bytes(unhashed)).hexdigest()
    if claimed_hash != actual_hash:
        raise ValueError(f"manifest SHA256 mismatch: claimed={claimed_hash}, actual={actual_hash}")

    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"manifest schema_version must be {MANIFEST_SCHEMA_VERSION}")
    for key, expected_literal in (
        ("dataset", DATASET),
        ("split_implementation", SPLIT_IMPLEMENTATION),
        ("generator", GENERATOR),
    ):
        if manifest.get(key) != expected_literal:
            raise ValueError(f"manifest {key} must be {expected_literal!r}")

    split = DrivAerMLDefaultSplitIDs()
    expected_counts = {
        "train": len(split.train),
        "val": len(split.val),
        "test": len(split.test),
        "hidden_test": len(split.hidden_test),
    }
    if manifest.get("official_split_counts") != expected_counts:
        raise ValueError("manifest official split counts differ from the current official split")
    for key, expected_ids in (
        ("official_val_run_ids", split.val),
        ("official_test_run_ids", split.test),
        ("official_hidden_test_run_ids", split.hidden_test),
        ("val_run_ids", split.val),
        ("test_run_ids", split.test),
    ):
        if key in manifest and manifest[key] != expected_ids:
            raise ValueError(f"manifest {key} differs from the current official split")
    for key in ("official_val_run_ids", "official_test_run_ids", "official_hidden_test_run_ids"):
        if key not in manifest:
            raise ValueError(f"manifest must define {key}")

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

    expected_indices = _ordered_positions(seed, len(split.train))
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

    if "nested_train_run_ids" in manifest:
        nested = manifest["nested_train_run_ids"]
        if not isinstance(nested, dict) or set(nested) != expected_subset_keys:
            raise ValueError("manifest nested_train_run_ids keys must exactly match sizes")
        for size in sizes:
            if nested[str(size)] != expected_run_ids[:size]:
                raise ValueError(f"manifest nested train IDs for N={size} are not the deterministic prefix")


@dataclass(frozen=True)
class ManifestCell:
    """One exact nested subset drawn from a validated manifest.

    Attributes:
        path: Resolved manifest path.
        base_dataset_indices: Positions consumed by ``SubsetWrapper``.
        run_ids: DrivAerML design IDs the positions resolve to.
        raw_file_sha256: SHA256 of the manifest bytes; binds the statistics.
        payload_sha256: SHA256 of the canonical payload; names the run.
        seed: Subset seed recorded in the manifest.
        study_id: Study identifier recorded in the manifest.
        protocol_sha256: Protocol SHA256 recorded in the manifest.
        implementation_commit: Implementation commit recorded in the manifest.
        implementation_dirty: Implementation dirty flag recorded in the manifest.
    """

    path: Path
    base_dataset_indices: tuple[int, ...]
    run_ids: tuple[int, ...]
    raw_file_sha256: str
    payload_sha256: str
    seed: int
    study_id: str
    protocol_sha256: str
    implementation_commit: str
    implementation_dirty: bool

    def require_implementation(self, runtime: GitState) -> None:
        """Ensure execution uses the implementation frozen into the manifest.

        Args:
            runtime: Implementation state of the running process.

        Raises:
            ValueError: If the commit or dirty state differs.
        """
        if self.implementation_commit != runtime.commit:
            raise ValueError(
                "manifest/implementation commit mismatch: "
                f"manifest={self.implementation_commit}, runtime={runtime.commit}"
            )
        if self.implementation_dirty is not runtime.dirty:
            raise ValueError(
                "manifest/implementation dirty-state mismatch: "
                f"manifest={self.implementation_dirty}, runtime={runtime.dirty}"
            )


def load_manifest_cell(
    manifest_path: Path,
    sample_size: int,
    *,
    expected_protocol_sha256: str | None = None,
    expected_study_id: str | None = None,
) -> ManifestCell:
    """Validate a frozen manifest and return one exact nested subset cell.

    Args:
        manifest_path: Path of the materialized manifest.
        sample_size: Requested training-subset size.
        expected_protocol_sha256: Protocol SHA256 the manifest must repeat.
        expected_study_id: Study identifier the manifest must repeat.

    Returns:
        The validated :class:`ManifestCell`.

    Raises:
        TypeError: If the manifest root is not a mapping.
        ValueError: If the manifest is inconsistent, lacks the requested
            subset, or overlaps the official validation/test splits.
    """
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("manifest root must be a mapping")
    verify_manifest(manifest)

    study_id = manifest.get("study_id")
    protocol_sha256 = manifest.get("protocol_sha256")
    commit = manifest.get("implementation_git_commit")
    dirty = manifest.get("implementation_git_dirty")
    if not isinstance(study_id, str) or not study_id:
        raise ValueError("manifest must define a non-empty study_id")
    if not is_sha256(protocol_sha256):
        raise ValueError("manifest must define a lowercase protocol_sha256")
    if expected_protocol_sha256 is not None and protocol_sha256 != expected_protocol_sha256:
        raise ValueError(
            f"manifest/protocol SHA256 mismatch: manifest={protocol_sha256}, protocol={expected_protocol_sha256}"
        )
    if expected_study_id is not None and study_id != expected_study_id:
        raise ValueError(f"manifest/protocol study_id mismatch: manifest={study_id}, protocol={expected_study_id}")
    if not is_git_commit(commit):
        raise ValueError("manifest must define a lowercase 40-character implementation_git_commit")
    if not isinstance(dirty, bool):
        raise ValueError("manifest must define boolean implementation_git_dirty")

    subsets = manifest["subsets"]
    if not isinstance(subsets.get(str(sample_size)), dict):
        raise ValueError(f"manifest has no subset cell for N={sample_size}")
    cell = subsets[str(sample_size)]
    indices = [int(value) for value in cell["base_dataset_indices"]]
    run_ids = [int(value) for value in cell["run_ids"]]
    if len(indices) != sample_size or len(run_ids) != sample_size:
        raise ValueError(f"manifest N={sample_size} lengths are indices={len(indices)}, run_ids={len(run_ids)}")

    official = DrivAerMLDefaultSplitIDs()
    if set(run_ids) & (set(official.val) | set(official.test)):
        raise ValueError("manifest training subset overlaps validation/test")

    unhashed = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return ManifestCell(
        path=manifest_path.resolve(),
        base_dataset_indices=tuple(indices),
        run_ids=tuple(run_ids),
        raw_file_sha256=hashlib.sha256(raw).hexdigest(),
        payload_sha256=hashlib.sha256(canonical_json_bytes(unhashed)).hexdigest(),
        seed=int(manifest["seed"]),
        study_id=study_id,
        protocol_sha256=protocol_sha256,
        implementation_commit=commit,
        implementation_dirty=dirty,
    )
