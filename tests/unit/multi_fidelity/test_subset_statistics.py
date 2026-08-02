# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Tests for coordinate-consistent, leakage-safe transfer support."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from aero_cfd.datasets.transfer_drivaerml import (
    drivaer_to_shapenet_frame,
    transform_loaded_drivaerml_field,
)
from aero_cfd.multi_fidelity.integrity import canonical_json_bytes as canonical_bytes
from aero_cfd.multi_fidelity.manifest import build_manifest
from research.multi_fidelity.tools.compute_subset_statistics import (
    build_nested_statistics_artifacts,
    build_statistics_artifact,
    load_frozen_train_subset,
    validate_frozen_protocol_for_statistics,
)


def test_coordinate_transform_covers_vectors_but_not_scalars() -> None:
    """Every frame-dependent Cartesian field uses the same cyclic mapping."""
    native = torch.tensor([[1.0, 2.0, 3.0]])
    expected = torch.tensor([[2.0, 3.0, 1.0]])
    assert torch.equal(drivaer_to_shapenet_frame(native), expected)

    vector_filenames = (
        "surface_position_vtp.pt",
        "surface_wallshearstress.pt",
        "surface_normal_vtp.pt",
        "volume_cell_position.pt",
        "volume_cell_velocity.pt",
        "volume_cell_vorticity.pt",
    )
    for filename in vector_filenames:
        transformed = transform_loaded_drivaerml_field(native, filename, coordinate_frame="shapenet")
        assert torch.equal(transformed, expected)

    pressure = torch.tensor([7.0])
    unchanged = transform_loaded_drivaerml_field(
        pressure,
        "surface_pressure.pt",
        coordinate_frame="shapenet",
    )
    assert unchanged is pressure
    assert transform_loaded_drivaerml_field(native, "volume_cell_velocity.pt", coordinate_frame="native") is native


def _write_manifest(path: Path, sizes: tuple[int, ...]) -> tuple[bytes, list[int]]:
    manifest = build_manifest(seed=9, sizes=sizes)
    manifest.pop("manifest_sha256")
    manifest["study_id"] = "unit-test-study"
    manifest["protocol_sha256"] = "a" * 64
    manifest["implementation_git_commit"] = "b" * 40
    manifest["implementation_git_dirty"] = True
    manifest["val_run_ids"] = list(manifest["official_val_run_ids"])
    manifest["test_run_ids"] = list(manifest["official_test_run_ids"])
    manifest["nested_train_run_ids"] = {str(size): list(manifest["subsets"][str(size)]["run_ids"]) for size in sizes}
    manifest["manifest_sha256"] = hashlib.sha256(canonical_bytes(manifest)).hexdigest()
    raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(raw)
    return raw, list(manifest["nested_train_run_ids"][str(sizes[-1])])


def _write_selected_targets(root: Path, run_id: int, offset: float) -> None:
    run_dir = root / f"run_{run_id}"
    run_dir.mkdir(parents=True)
    torch.save(torch.tensor([1.0 + offset, 3.0 + offset]), run_dir / "surface_pressure.pt")
    torch.save(
        torch.tensor([[1.0 + offset, 2.0 + offset, 3.0 + offset]]),
        run_dir / "volume_cell_velocity.pt",
    )
    torch.save(
        torch.tensor([[1.0 + offset, -2.0 - offset, 3.0 + offset]]),
        run_dir / "volume_cell_vorticity.pt",
    )


def test_statistics_use_exact_train_prefix_and_fixed_position_bounds(tmp_path: Path) -> None:
    """Statistics succeed without any validation/test directory or position file."""
    manifest_path = tmp_path / "manifest.json"
    raw_manifest, run_ids = _write_manifest(manifest_path, (2,))
    data_root = tmp_path / "data"
    _write_selected_targets(data_root, run_ids[0], 0.0)
    _write_selected_targets(data_root, run_ids[1], 3.0)

    artifact, flat = build_statistics_artifact(
        root=data_root,
        manifest_path=manifest_path,
        n=2,
        fields=["surface_pressure", "volume_velocity", "volume_vorticity"],
        coordinate_frame="shapenet",
        chunk_rows=1,
        position_min=-40.0,
        position_max=80.0,
    )

    assert artifact["provenance"]["manifest_sha256"] == hashlib.sha256(raw_manifest).hexdigest()
    assert artifact["study_id"] == "unit-test-study"
    assert artifact["protocol_sha256"] == "a" * 64
    assert artifact["train_run_ids"] == run_ids
    assert artifact["leakage_guard"]["target_splits_read"] == ["train"]
    assert artifact["leakage_guard"]["position_files_read"] is False
    assert flat["raw_pos_min"] == [-40.0]
    assert flat["raw_pos_max"] == [80.0]
    assert flat["surface_pressure_mean"] == pytest.approx([3.5])
    # Native velocity rows [1,2,3] and [4,5,6] become [2,3,1] and [5,6,4].
    assert flat["volume_velocity_mean"] == pytest.approx([3.5, 4.5, 2.5])
    assert "volume_vorticity_logscale_mean" in flat


def test_manifest_rejects_validation_id_in_train_prefix(tmp_path: Path) -> None:
    """A non-training ID cannot be smuggled into a hashed frozen prefix."""
    manifest_path = tmp_path / "leaked.json"
    _write_manifest(manifest_path, (1,))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["nested_train_run_ids"]["1"] = [manifest["official_val_run_ids"][0]]
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = hashlib.sha256(canonical_bytes(manifest)).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="deterministic prefix"):
        load_frozen_train_subset(manifest_path, 1)


def test_nested_statistics_scan_each_target_file_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """All requested prefix snapshots reuse one streaming pass over the largest prefix."""
    manifest_path = tmp_path / "manifest.json"
    _, run_ids = _write_manifest(manifest_path, (1, 2))
    data_root = tmp_path / "data"
    _write_selected_targets(data_root, run_ids[0], 0.0)
    _write_selected_targets(data_root, run_ids[1], 3.0)
    original_load = torch.load
    loaded_paths: list[Path] = []

    def tracking_load(path: Path, *args, **kwargs):
        loaded_paths.append(Path(path))
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", tracking_load)
    products = build_nested_statistics_artifacts(
        root=data_root,
        manifest_path=manifest_path,
        sizes=[1, 2],
        fields=["surface_pressure"],
        coordinate_frame="shapenet",
        chunk_rows=1,
        position_min=-40.0,
        position_max=80.0,
    )

    assert len(loaded_paths) == 2
    assert products[1][1]["surface_pressure_mean"] == pytest.approx([2.0])
    assert products[2][1]["surface_pressure_mean"] == pytest.approx([3.5])
    assert products[1][0]["field_statistics"]["surface_pressure"]["files_read"] == 1
    assert products[2][0]["field_statistics"]["surface_pressure"]["files_read"] == 2


def test_manifest_seed_must_reproduce_the_recorded_permutation(tmp_path: Path) -> None:
    """Rehashing an arbitrary permutation under another seed is rejected."""
    manifest_path = tmp_path / "wrong-seed.json"
    _write_manifest(manifest_path, (1, 2))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["seed"] = 10
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = hashlib.sha256(canonical_bytes(manifest)).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="recorded PCG64 seed"):
        load_frozen_train_subset(manifest_path, 2)


def test_statistics_cli_binding_accepts_frozen_protocol(protocol_path: Path, manifest_path: Path) -> None:
    """The frozen repository protocol and a materialized manifest are bound."""
    validate_frozen_protocol_for_statistics(protocol_path, manifest_path)


def test_statistics_cli_binding_rejects_draft_protocol(
    tmp_path: Path,
    protocol_path: Path,
    manifest_path: Path,
) -> None:
    """Target-label statistics cannot start from a draft protocol."""
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    protocol["status"] = "preregistration_draft"
    draft_protocol = tmp_path / "draft_protocol.yaml"
    draft_protocol.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="statistics execution requires protocol status"):
        validate_frozen_protocol_for_statistics(draft_protocol, manifest_path)
