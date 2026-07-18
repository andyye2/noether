# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Compute leakage-safe normalization statistics for a frozen train subset.

The tool accepts only an exact ``nested_train_run_ids[N]`` entry from an
audited manifest. It never scans validation/test target files. Position
normalization uses fixed, user-declared physical envelope bounds and therefore
does not inspect coordinates from any split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import yaml

from aero_cfd.datasets.transfer_drivaerml import transform_loaded_drivaerml_field
from noether.data.datasets.cfd.caeml.drivaerml.split import DrivAerMLDefaultSplitIDs
from noether.data.stats import RunningMoments
from research.multi_fidelity.tools.materialize_nested_manifests import verify_manifest

CoordinateFrame = Literal["native", "shapenet"]
FROZEN_PROTOCOL_STATUS = "frozen_before_first_target_job"


@dataclass(frozen=True)
class FieldSpec:
    """On-disk and normalization metadata for one DrivAerML target field."""

    filename: str
    components: int
    log_scale: bool = False


FIELD_SPECS: dict[str, FieldSpec] = {
    "surface_pressure": FieldSpec("surface_pressure.pt", 1),
    "surface_friction": FieldSpec("surface_wallshearstress.pt", 3),
    # Match CAEML_FILEMAP and the DrivAerML preset: this is total-pressure
    # coefficient, not the separately stored raw cell pressure.
    "volume_pressure": FieldSpec("volume_cell_totalpcoeff.pt", 1),
    "volume_velocity": FieldSpec("volume_cell_velocity.pt", 3),
    "volume_vorticity": FieldSpec("volume_cell_vorticity.pt", 3, log_scale=True),
}


def _read_structured_file(path: Path, raw: bytes) -> dict[str, Any]:
    """Decode a JSON or YAML mapping while preserving raw bytes for hashing."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        value = json.loads(raw.decode("utf-8"))
    elif suffix in {".yaml", ".yml"}:
        value = yaml.safe_load(raw.decode("utf-8"))
    else:
        raise ValueError(f"Manifest must be JSON or YAML, got {path}")
    if not isinstance(value, dict):
        raise TypeError(f"Manifest root must be a mapping, got {type(value).__name__}")
    return value


def _validate_study_manifest(manifest: dict[str, Any]) -> None:
    """Validate the complete hashed v2 manifest against the official split."""
    verify_manifest(manifest)
    if manifest.get("schema_version") != 2:
        raise ValueError("Manifest schema_version must be 2")
    claimed_payload_sha256 = manifest.get("manifest_sha256")
    unhashed = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    canonical_payload = json.dumps(
        unhashed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    actual_payload_sha256 = hashlib.sha256(canonical_payload).hexdigest()
    if claimed_payload_sha256 != actual_payload_sha256:
        raise ValueError(
            f"Manifest payload SHA256 mismatch: claimed={claimed_payload_sha256}, actual={actual_payload_sha256}"
        )

    implementation_commit = manifest.get("implementation_git_commit")
    if (
        not isinstance(implementation_commit, str)
        or len(implementation_commit) != 40
        or any(character not in "0123456789abcdef" for character in implementation_commit)
    ):
        raise ValueError("Manifest must contain a lowercase 40-character implementation_git_commit")
    if not isinstance(manifest.get("implementation_git_dirty"), bool):
        raise ValueError("Manifest must contain a boolean implementation_git_dirty")

    official = DrivAerMLDefaultSplitIDs()
    split_fields = {
        "official_val_run_ids": official.val,
        "official_test_run_ids": official.test,
        "official_hidden_test_run_ids": official.hidden_test,
        "val_run_ids": official.val,
        "test_run_ids": official.test,
    }
    for key, expected_ids in split_fields.items():
        if manifest.get(key) != expected_ids:
            raise ValueError(f"Manifest {key} differs from the current official split")

    ordered_indices = manifest.get("ordered_base_dataset_indices")
    ordered_run_ids = manifest.get("ordered_train_run_ids")
    if not isinstance(ordered_indices, list) or not isinstance(ordered_run_ids, list):
        raise ValueError("Manifest must contain ordered base indices and train run IDs")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in ordered_indices):
        raise ValueError("Manifest ordered base indices must be integers")
    if sorted(ordered_indices) != list(range(len(official.train))):
        raise ValueError("Manifest ordered base indices must permute the official train positions")
    resolved_ordered = [official.train[index] for index in ordered_indices]
    if ordered_run_ids != resolved_ordered:
        raise ValueError("Manifest ordered train run IDs do not resolve from base indices")

    sizes = manifest.get("sizes")
    subsets = manifest.get("subsets")
    nested = manifest.get("nested_train_run_ids")
    if not isinstance(sizes, list) or sizes != sorted(set(sizes)) or not sizes:
        raise ValueError("Manifest sizes must be a non-empty sorted unique list")
    if not isinstance(subsets, dict) or not isinstance(nested, dict):
        raise ValueError("Manifest must contain subsets and nested_train_run_ids mappings")
    for size in sizes:
        cell = subsets.get(str(size))
        alias = nested.get(str(size))
        if not isinstance(cell, dict) or not isinstance(alias, list):
            raise ValueError(f"Manifest has no complete subset cell for N={size}")
        indices = cell.get("base_dataset_indices")
        run_ids = cell.get("run_ids")
        if not isinstance(indices, list) or not isinstance(run_ids, list):
            raise ValueError(f"Manifest subset N={size} must contain index and run-ID lists")
        if len(indices) != size or len(run_ids) != size or len(set(indices)) != size:
            raise ValueError(f"Manifest subset N={size} has invalid lengths or duplicate indices")
        if run_ids != [official.train[index] for index in indices]:
            raise ValueError(f"Manifest subset N={size} index/run-ID mapping is invalid")
        if alias != run_ids or run_ids != ordered_run_ids[:size]:
            raise ValueError(f"Manifest subset N={size} is not the exact ordered prefix")


def validate_frozen_protocol_for_statistics(protocol_path: Path, manifest_path: Path) -> None:
    """Bind target-label scanning to an explicitly frozen protocol and manifest."""
    protocol_raw = protocol_path.read_bytes()
    protocol = yaml.safe_load(protocol_raw.decode("utf-8"))
    if not isinstance(protocol, dict):
        raise TypeError("protocol root must be a mapping")
    if protocol.get("status") != FROZEN_PROTOCOL_STATUS:
        raise ValueError(
            f"statistics execution requires protocol status {FROZEN_PROTOCOL_STATUS!r}; got {protocol.get('status')!r}"
        )
    study_id = protocol.get("study_id")
    if not isinstance(study_id, str) or not study_id:
        raise ValueError("protocol must contain a non-empty study_id")

    manifest_raw = manifest_path.read_bytes()
    manifest = _read_structured_file(manifest_path, manifest_raw)
    _validate_study_manifest(manifest)
    protocol_sha256 = hashlib.sha256(protocol_raw).hexdigest()
    if manifest.get("protocol_sha256") != protocol_sha256:
        raise ValueError("manifest/protocol SHA256 mismatch before statistics scan")
    if manifest.get("study_id") != study_id:
        raise ValueError("manifest/protocol study_id mismatch before statistics scan")


def load_frozen_train_subset(manifest_path: Path, n: int) -> tuple[list[int], dict[str, Any]]:
    """Load and validate exactly one nested training prefix.

    Args:
        manifest_path: Frozen split manifest produced by
            ``build_nested_splits.py``.
        n: Requested training-subset size.

    Returns:
        Selected run IDs and immutable provenance metadata.

    Raises:
        ValueError: If the manifest does not contain an exact, nested, disjoint
            prefix of length ``n``.
    """
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    raw = manifest_path.read_bytes()
    manifest = _read_structured_file(manifest_path, raw)
    _validate_study_manifest(manifest)
    protocol_sha256 = manifest.get("protocol_sha256")
    if (
        not isinstance(protocol_sha256, str)
        or len(protocol_sha256) != 64
        or any(character not in "0123456789abcdef" for character in protocol_sha256)
    ):
        raise ValueError("Manifest must contain a lowercase protocol_sha256")
    study_id = manifest.get("study_id")
    if not isinstance(study_id, str) or not study_id:
        raise ValueError("Manifest must contain a non-empty study_id")
    nested = manifest.get("nested_train_run_ids")
    ordered = manifest.get("ordered_train_run_ids")
    if not isinstance(nested, dict) or not isinstance(ordered, list):
        raise ValueError("Manifest must contain nested_train_run_ids and ordered_train_run_ids")
    selected_raw = nested.get(str(n), nested.get(n))
    if not isinstance(selected_raw, list):
        raise ValueError(f"Manifest has no frozen nested train subset for N={n}")

    selected = [int(run_id) for run_id in selected_raw]
    ordered_ids = [int(run_id) for run_id in ordered]
    if len(selected) != n:
        raise ValueError(f"N={n} entry contains {len(selected)} IDs")
    if len(set(selected)) != n:
        raise ValueError(f"N={n} entry contains duplicate IDs")
    if selected != ordered_ids[:n]:
        raise ValueError(f"N={n} entry is not the exact ordered training prefix")

    val_ids = {int(run_id) for run_id in manifest.get("val_run_ids", [])}
    test_ids = {int(run_id) for run_id in manifest.get("test_run_ids", [])}
    leaked = set(selected) & (val_ids | test_ids)
    if leaked:
        raise ValueError(f"Training subset overlaps validation/test IDs: {sorted(leaked)}")

    canonical_ids = json.dumps(selected, separators=(",", ":")).encode("ascii")
    provenance = {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "manifest_payload_sha256": manifest.get("manifest_sha256"),
        "manifest_schema_version": manifest.get("schema_version"),
        "manifest_seed": manifest.get("seed"),
        "study_id": study_id,
        "protocol_sha256": protocol_sha256,
        "implementation_git_commit": manifest.get("implementation_git_commit"),
        "implementation_git_dirty": manifest.get("implementation_git_dirty"),
        "selected_run_ids_sha256": hashlib.sha256(canonical_ids).hexdigest(),
        "selection_rule": f"nested_train_run_ids[{n}] == ordered_train_run_ids[:{n}]",
        "val_target_files_read": False,
        "test_target_files_read": False,
    }
    return selected, provenance


def _as_feature_matrix(tensor: torch.Tensor, field: str, components: int) -> torch.Tensor:
    """Return a point-major feature matrix with an exact component count."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{field} did not deserialize as a torch.Tensor")
    if components == 1:
        if tensor.ndim == 1:
            return tensor.unsqueeze(1)
        if tensor.ndim >= 2 and tensor.shape[-1] == 1:
            return tensor.reshape(-1, 1)
    elif tensor.ndim >= 2 and tensor.shape[-1] == components:
        return tensor.reshape(-1, components)
    raise ValueError(f"{field} expects {components} component(s), got shape {tuple(tensor.shape)}")


def _tensor_or_scalar_to_json(value: torch.Tensor | float) -> list[float] | float:
    """Convert RunningMoments output to JSON-safe Python values."""
    if isinstance(value, torch.Tensor):
        return [float(item) for item in value.detach().cpu().reshape(-1).tolist()]
    return float(value)


def compute_subset_statistics(
    *,
    root: Path,
    run_ids: list[int],
    fields: list[str],
    coordinate_frame: CoordinateFrame,
    chunk_rows: int,
) -> tuple[dict[str, Any], dict[str, list[float] | float]]:
    """Stream selected train targets into per-component sample moments.

    Args:
        root: DrivAerML directory containing ``run_<ID>`` subdirectories.
        run_ids: Already-validated frozen training IDs.
        fields: Target fields to scan.
        coordinate_frame: Frame in which vector statistics are produced.
        chunk_rows: Maximum number of point rows converted to float64 at once.

    Returns:
        Detailed field audit and flat Noether normalizer-stat entries.

    Raises:
        FileNotFoundError: If any selected training target is absent.
        ValueError: If tensors have invalid shape or non-finite values.
    """
    if chunk_rows < 1:
        raise ValueError(f"chunk_rows must be positive, got {chunk_rows}")
    unknown = sorted(set(fields) - set(FIELD_SPECS))
    if unknown:
        raise ValueError(f"Unknown fields: {unknown}; choices are {sorted(FIELD_SPECS)}")
    if not fields:
        raise ValueError("fields must not be empty")
    if len(set(fields)) != len(fields):
        raise ValueError("fields must not contain duplicates")

    moments = {field: RunningMoments(log_scale=FIELD_SPECS[field].log_scale) for field in fields}
    bytes_read = dict.fromkeys(fields, 0)
    files_read = dict.fromkeys(fields, 0)

    for run_id in run_ids:
        run_dir = root / f"run_{run_id}"
        for field in fields:
            spec = FIELD_SPECS[field]
            path = run_dir / spec.filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing selected-train target: {path}")
            tensor = torch.load(path, map_location="cpu", weights_only=True)
            matrix = _as_feature_matrix(tensor, field, spec.components)
            if matrix.shape[0] == 0:
                raise ValueError(f"Empty tensor in {path}")
            for chunk in matrix.split(chunk_rows, dim=0):
                aligned = transform_loaded_drivaerml_field(
                    chunk,
                    spec.filename,
                    coordinate_frame=coordinate_frame,
                )
                if not torch.isfinite(aligned).all():
                    raise ValueError(f"Non-finite values in {path}")
                moments[field].push_tensor(aligned, dim=1)
            bytes_read[field] += path.stat().st_size
            files_read[field] += 1
            del tensor, matrix

    details: dict[str, Any] = {}
    flat_stats: dict[str, list[float] | float] = {}
    for field in fields:
        stat = moments[field]
        spec = FIELD_SPECS[field]
        prefix = f"{field}_logscale" if spec.log_scale else field
        mean = _tensor_or_scalar_to_json(stat.mean)
        std = _tensor_or_scalar_to_json(stat.std)
        minimum = _tensor_or_scalar_to_json(stat.min)
        maximum = _tensor_or_scalar_to_json(stat.max)
        details[field] = {
            "filename": spec.filename,
            "components": spec.components,
            "transform": "sign(x)*log1p(abs(x))" if spec.log_scale else "identity",
            "coordinate_frame": coordinate_frame,
            "count_per_component": stat.count,
            "files_read": files_read[field],
            "bytes_read": bytes_read[field],
            "mean": mean,
            "std": std,
            "min": minimum,
            "max": maximum,
            "std_ddof": 1,
        }
        flat_stats[f"{prefix}_mean"] = mean
        flat_stats[f"{prefix}_std"] = std
    return details, flat_stats


def compute_nested_subset_statistics(
    *,
    root: Path,
    run_ids: list[int],
    snapshot_sizes: list[int],
    fields: list[str],
    coordinate_frame: CoordinateFrame,
    chunk_rows: int,
) -> dict[int, tuple[dict[str, Any], dict[str, list[float] | float]]]:
    """Scan one nested ladder once and snapshot moments at requested prefixes."""
    sizes = tuple(snapshot_sizes)
    if not sizes or tuple(sorted(set(sizes))) != sizes:
        raise ValueError("snapshot_sizes must be non-empty, unique, and sorted")
    if sizes[0] < 1 or sizes[-1] != len(run_ids):
        raise ValueError("snapshot_sizes must end at the supplied run_ids length")
    if chunk_rows < 1:
        raise ValueError(f"chunk_rows must be positive, got {chunk_rows}")
    unknown = sorted(set(fields) - set(FIELD_SPECS))
    if unknown:
        raise ValueError(f"Unknown fields: {unknown}; choices are {sorted(FIELD_SPECS)}")
    if not fields or len(set(fields)) != len(fields):
        raise ValueError("fields must be non-empty and contain no duplicates")

    moments = {field: RunningMoments(log_scale=FIELD_SPECS[field].log_scale) for field in fields}
    bytes_read = dict.fromkeys(fields, 0)
    files_read = dict.fromkeys(fields, 0)
    requested = set(sizes)
    snapshots: dict[int, tuple[dict[str, Any], dict[str, list[float] | float]]] = {}

    for prefix_size, run_id in enumerate(run_ids, start=1):
        run_dir = root / f"run_{run_id}"
        for field in fields:
            spec = FIELD_SPECS[field]
            path = run_dir / spec.filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing selected-train target: {path}")
            tensor = torch.load(path, map_location="cpu", weights_only=True)
            matrix = _as_feature_matrix(tensor, field, spec.components)
            if matrix.shape[0] == 0:
                raise ValueError(f"Empty tensor in {path}")
            for chunk in matrix.split(chunk_rows, dim=0):
                aligned = transform_loaded_drivaerml_field(
                    chunk,
                    spec.filename,
                    coordinate_frame=coordinate_frame,
                )
                if not torch.isfinite(aligned).all():
                    raise ValueError(f"Non-finite values in {path}")
                moments[field].push_tensor(aligned, dim=1)
            bytes_read[field] += path.stat().st_size
            files_read[field] += 1
            del tensor, matrix

        if prefix_size in requested:
            details: dict[str, Any] = {}
            flat_stats: dict[str, list[float] | float] = {}
            for field in fields:
                stat = moments[field]
                spec = FIELD_SPECS[field]
                prefix = f"{field}_logscale" if spec.log_scale else field
                mean = _tensor_or_scalar_to_json(stat.mean)
                std = _tensor_or_scalar_to_json(stat.std)
                details[field] = {
                    "filename": spec.filename,
                    "components": spec.components,
                    "transform": "sign(x)*log1p(abs(x))" if spec.log_scale else "identity",
                    "coordinate_frame": coordinate_frame,
                    "count_per_component": stat.count,
                    "files_read": files_read[field],
                    "bytes_read": bytes_read[field],
                    "mean": mean,
                    "std": std,
                    "min": _tensor_or_scalar_to_json(stat.min),
                    "max": _tensor_or_scalar_to_json(stat.max),
                    "std_ddof": 1,
                }
                flat_stats[f"{prefix}_mean"] = mean
                flat_stats[f"{prefix}_std"] = std
            snapshots[prefix_size] = (details, flat_stats)
    return snapshots


def build_statistics_artifact(
    *,
    root: Path,
    manifest_path: Path,
    n: int,
    fields: list[str],
    coordinate_frame: CoordinateFrame,
    chunk_rows: int,
    position_min: float,
    position_max: float,
) -> tuple[dict[str, Any], dict[str, list[float] | float]]:
    """Validate selection and build detailed plus flat statistics artifacts."""
    if not position_min < position_max:
        raise ValueError("position_min must be strictly less than position_max")
    run_ids, provenance = load_frozen_train_subset(manifest_path, n)
    details, flat_stats = compute_subset_statistics(
        root=root,
        run_ids=run_ids,
        fields=fields,
        coordinate_frame=coordinate_frame,
        chunk_rows=chunk_rows,
    )
    # Match the existing DrivAerML PositionNormalizer convention: one scalar
    # envelope bound is broadcast across x/y/z. No position file is scanned.
    flat_stats["raw_pos_min"] = [position_min]
    flat_stats["raw_pos_max"] = [position_max]
    artifact = {
        "schema_version": 1,
        "study_id": provenance["study_id"],
        "protocol_sha256": provenance["protocol_sha256"],
        "dataset": "DrivAerML subsampled_10x",
        "dataset_root": str(root.resolve()),
        "train_subset_size": n,
        "train_run_ids": run_ids,
        "coordinate_frame": coordinate_frame,
        "provenance": provenance,
        "leakage_guard": {
            "target_splits_read": ["train"],
            "selection_is_exact_frozen_prefix": True,
            "position_files_read": False,
            "position_policy": "fixed physical envelope bounds supplied on command line",
            "position_bound_semantics": "single scalar min/max broadcast across all three axes",
        },
        "position_bounds": {"raw_pos_min": [position_min], "raw_pos_max": [position_max]},
        "field_statistics": details,
        "normalizer_stats": flat_stats,
    }
    return artifact, flat_stats


def build_nested_statistics_artifacts(
    *,
    root: Path,
    manifest_path: Path,
    sizes: list[int],
    fields: list[str],
    coordinate_frame: CoordinateFrame,
    chunk_rows: int,
    position_min: float,
    position_max: float,
) -> dict[int, tuple[dict[str, Any], dict[str, list[float] | float]]]:
    """Build all requested nested-prefix artifacts with one target-data scan."""
    if not position_min < position_max:
        raise ValueError("position_min must be strictly less than position_max")
    ordered_sizes = tuple(sizes)
    if not ordered_sizes or tuple(sorted(set(ordered_sizes))) != ordered_sizes:
        raise ValueError("sizes must be non-empty, unique, and sorted")

    selections: dict[int, tuple[list[int], dict[str, Any]]] = {}
    for size in ordered_sizes:
        selections[size] = load_frozen_train_subset(manifest_path, size)
    largest_run_ids = selections[ordered_sizes[-1]][0]
    snapshots = compute_nested_subset_statistics(
        root=root,
        run_ids=largest_run_ids,
        snapshot_sizes=list(ordered_sizes),
        fields=fields,
        coordinate_frame=coordinate_frame,
        chunk_rows=chunk_rows,
    )

    products: dict[int, tuple[dict[str, Any], dict[str, list[float] | float]]] = {}
    for size in ordered_sizes:
        run_ids, provenance = selections[size]
        details, flat_stats = snapshots[size]
        flat_stats["raw_pos_min"] = [position_min]
        flat_stats["raw_pos_max"] = [position_max]
        artifact = {
            "schema_version": 1,
            "study_id": provenance["study_id"],
            "protocol_sha256": provenance["protocol_sha256"],
            "dataset": "DrivAerML subsampled_10x",
            "dataset_root": str(root.resolve()),
            "train_subset_size": size,
            "train_run_ids": run_ids,
            "coordinate_frame": coordinate_frame,
            "provenance": provenance,
            "leakage_guard": {
                "target_splits_read": ["train"],
                "selection_is_exact_frozen_prefix": True,
                "position_files_read": False,
                "position_policy": "fixed physical envelope bounds supplied on command line",
                "position_bound_semantics": "single scalar min/max broadcast across all three axes",
            },
            "position_bounds": {"raw_pos_min": [position_min], "raw_pos_max": [position_max]},
            "field_statistics": details,
            "normalizer_stats": flat_stats,
        }
        products[size] = (artifact, flat_stats)
    return products


def _write_structured_file(path: Path, value: dict[str, Any]) -> None:
    """Atomically replace a JSON/YAML mapping after a durable complete write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    elif path.suffix.lower() in {".yaml", ".yml"}:
        encoded = yaml.safe_dump(value, sort_keys=False).encode("utf-8")
    else:
        raise ValueError(f"Output must end in .json, .yaml, or .yml, got {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as file:
            file.write(encoded)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for strict subset statistics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Directory containing DrivAerML run_<ID> folders")
    parser.add_argument("--manifest", type=Path, required=True, help="Frozen nested split manifest (JSON/YAML)")
    parser.add_argument("--protocol", type=Path, required=True, help="Explicitly frozen experiment protocol")
    parser.add_argument("--n", type=int, nargs="+", required=True, help="Sorted nested training subset sizes")
    parser.add_argument(
        "--fields",
        nargs="+",
        choices=sorted(FIELD_SPECS),
        default=list(FIELD_SPECS),
        help="Target fields whose train-only moments will be computed",
    )
    parser.add_argument("--coordinate-frame", choices=("native", "shapenet"), default="shapenet")
    parser.add_argument("--chunk-rows", type=int, default=1_000_000)
    parser.add_argument(
        "--position-min",
        type=float,
        default=-40.0,
        help="Fixed physical envelope minimum, broadcast to xyz; positions are not scanned",
    )
    parser.add_argument(
        "--position-max",
        type=float,
        default=80.0,
        help="Fixed physical envelope maximum, broadcast to xyz; positions are not scanned",
    )
    parser.add_argument("--output", type=Path, nargs="+", required=True, help="One detailed artifact per --n value")
    parser.add_argument(
        "--flat-stats-output",
        type=Path,
        nargs="+",
        help="Optional Noether-compatible flat statistics file (.json/.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    """Compute and persist one or more nested train-subset statistics artifacts."""
    args = parse_args()
    validate_frozen_protocol_for_statistics(args.protocol, args.manifest)
    if len(args.n) != len(args.output):
        raise ValueError("--n and --output must contain the same number of values")
    if args.flat_stats_output is not None and len(args.n) != len(args.flat_stats_output):
        raise ValueError("--flat-stats-output must contain one path per --n value")
    products = build_nested_statistics_artifacts(
        root=args.root,
        manifest_path=args.manifest,
        sizes=args.n,
        fields=args.fields,
        coordinate_frame=args.coordinate_frame,
        chunk_rows=args.chunk_rows,
        position_min=args.position_min,
        position_max=args.position_max,
    )
    for index, size in enumerate(args.n):
        artifact, flat_stats = products[size]
        _write_structured_file(args.output[index], artifact)
        if args.flat_stats_output is not None:
            _write_structured_file(args.flat_stats_output[index], flat_stats)
    largest_artifact = products[args.n[-1]][0]
    print(
        json.dumps(
            {
                "outputs": [str(path) for path in args.output],
                "flat_stats_outputs": (
                    [str(path) for path in args.flat_stats_output] if args.flat_stats_output else None
                ),
                "manifest_sha256": largest_artifact["provenance"]["manifest_sha256"],
                "train_subset_sizes": args.n,
                "fields": args.fields,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
