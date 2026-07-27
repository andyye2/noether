# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Tests for the exploratory car-scale-aligned position normalization option."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from aero_cfd.presets.drivaerml_transfer import DrivAerMLTransferCommonPreset
from recipes.aero_cfd.scripts import eval_drivaerml_transfer_frozen
from recipes.aero_cfd.scripts.eval_drivaerml_transfer_frozen import build_eval_config
from recipes.aero_cfd.scripts.run_drivaerml_transfer_strict import (
    build_experiment_config,
    load_manifest_cell,
    sha256_file,
    write_training_provenance_sidecar,
)

REPO_ROOT = Path(__file__).parents[4]
PROTOCOL = REPO_ROOT / "research/multi_fidelity/experiment_protocol.yaml"
MANIFEST = REPO_ROOT / "research/multi_fidelity/evidence/manifests/drivaerml_nested_seed1103.json"
PROTOCOL_SHA256 = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
STUDY_ID = "shapenet_ab_upt_to_drivaerml_hf_efficiency"


@pytest.fixture
def clean_evaluator_worktree(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report a clean evaluator worktree so these tests are worktree-independent.

    ``build_eval_config`` refuses to run from a dirty implementation worktree.
    That release guard is orthogonal to the geometry-rendering binding under
    test here, so it is stubbed out rather than requiring a committed tree.
    """
    monkeypatch.setattr(
        eval_drivaerml_transfer_frozen,
        "_implementation_git_state",
        lambda repo_root: ("0" * 40, False),
    )


def _write_common_stats(path: Path, manifest_raw_sha: str) -> None:
    """Write a minimal detailed train-only artifact for construction tests."""
    artifact = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "train_subset_size": 25,
        "coordinate_frame": "shapenet",
        "provenance": {"manifest_sha256": manifest_raw_sha},
        "leakage_guard": {"target_splits_read": ["train"]},
        "normalizer_stats": {
            "raw_pos_min": [-40.0],
            "raw_pos_max": [80.0],
            "surface_pressure_mean": [-200.0],
            "surface_pressure_std": [250.0],
            "volume_velocity_mean": [0.0, 0.0, 16.0],
            "volume_velocity_std": [8.0, 7.0, 16.0],
        },
    }
    path.write_text(json.dumps(artifact), encoding="utf-8")


def _scratch_args(
    stats_path: Path,
    tmp_path: Path,
    position_scale: float = 1000.0,
    supernode_radius: float = 9.0,
) -> argparse.Namespace:
    """Return the complete Namespace consumed by the training config builder."""
    return argparse.Namespace(
        dataset_root=Path("/scratch/andyye2/data/drivaerml_subsampled_10x"),
        protocol=PROTOCOL,
        manifest=MANIFEST,
        target_statistics=stats_path,
        output_path=tmp_path / "outputs",
        task="common",
        strategy="scratch",
        sample_size=25,
        budget="compute_matched",
        smoke_updates=10,
        coordinate_frame="shapenet",
        position_scale=position_scale,
        supernode_radius=supernode_radius,
        replicate=0,
        model_seed=7103,
        eval_point_seed=4242,
        run_id=None,
        source_output_path=Path("/home/feng/Projects/ABUPT/outputs"),
        source_run_id="2026-04-25_7d0mv",
        source_stage_name="train",
        source_model_name="ab_upt",
        source_model_info=None,
        source_checkpoint_tag="latest",
        expected_source_sha256=None,
        learning_rate=5e-5,
        end_learning_rate=1e-6,
        weight_decay=0.05,
        body_lr_multiplier=None,
        decoder_lr_multiplier=None,
        effective_batch_size=1,
        precision="float16",
        accelerator="gpu",
        num_workers=8,
        dry_run=True,
    )


def _preset(stats_path: Path, manifest_raw_sha: str, position_scale: float = 1000.0) -> DrivAerMLTransferCommonPreset:
    """Build one common-field preset bound to the construction-test artifact."""
    return DrivAerMLTransferCommonPreset(
        statistics_artifact=stats_path,
        coordinate_frame="shapenet",
        expected_manifest_sha256=manifest_raw_sha,
        expected_train_subset_size=25,
        position_scale=position_scale,
    )


def test_default_position_scale_is_frozen_confirmatory_value(tmp_path: Path) -> None:
    """Omitting position_scale reproduces the frozen scale-1000 normalizers."""
    raw_sha = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    stats_path = tmp_path / "stats.json"
    _write_common_stats(stats_path, raw_sha)
    preset = DrivAerMLTransferCommonPreset(
        statistics_artifact=stats_path,
        coordinate_frame="shapenet",
        expected_manifest_sha256=raw_sha,
        expected_train_subset_size=25,
    )
    normalizers = preset.build_normalizers()
    assert preset.position_scale == 1000.0
    assert float(normalizers["surface_position"][0].scale) == 1000.0
    assert float(normalizers["volume_position"][0].scale) == 1000.0


def test_position_scale_flows_into_both_position_normalizers(tmp_path: Path) -> None:
    """A custom scale reaches surface and volume position normalizers only."""
    raw_sha = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    stats_path = tmp_path / "stats.json"
    _write_common_stats(stats_path, raw_sha)
    normalizers = _preset(stats_path, raw_sha, position_scale=8000.0).build_normalizers()
    assert float(normalizers["surface_position"][0].scale) == 8000.0
    assert float(normalizers["volume_position"][0].scale) == 8000.0
    # Field statistics are untouched by the geometric rescale:
    assert normalizers["surface_pressure"][0].mean.tolist() == [-200.0]
    assert normalizers["volume_velocity"][0].std.tolist() == [8.0, 7.0, 16.0]


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), True])
def test_invalid_position_scale_is_rejected(tmp_path: Path, value: float) -> None:
    """Non-finite or non-positive scales cannot configure a dataset."""
    raw_sha = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    stats_path = tmp_path / "stats.json"
    _write_common_stats(stats_path, raw_sha)
    with pytest.raises(ValueError, match="finite and positive"):
        _preset(stats_path, raw_sha, position_scale=value)


@pytest.mark.parametrize("value", [10000.1, 12000.0, 100000.0])
def test_position_scale_must_not_exceed_rope_wavelength(tmp_path: Path, value: float) -> None:
    """Scales above the RoPE max wavelength would wrap the coarsest band."""
    raw_sha = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    stats_path = tmp_path / "stats.json"
    _write_common_stats(stats_path, raw_sha)
    with pytest.raises(ValueError, match="RoPE/sincos maximum wavelength"):
        _preset(stats_path, raw_sha, position_scale=value)


def test_position_scale_may_equal_rope_wavelength(tmp_path: Path) -> None:
    """A scale of exactly one full coarse period is the audited upper bound."""
    raw_sha = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    stats_path = tmp_path / "stats.json"
    _write_common_stats(stats_path, raw_sha)
    normalizers = _preset(stats_path, raw_sha, position_scale=10000.0).build_normalizers()
    assert float(normalizers["surface_position"][0].scale) == 10000.0
    assert float(normalizers["volume_position"][0].scale) == 10000.0


def test_runner_records_geometry_rendering_in_config_audit_and_sidecar(tmp_path: Path) -> None:
    """The runner binds scale and radius into datasets, model, run ID, and sidecar."""
    raw_sha = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    stats_path = tmp_path / "stats.json"
    _write_common_stats(stats_path, raw_sha)
    args = _scratch_args(stats_path, tmp_path, position_scale=10000.0, supernode_radius=1.0)
    manifest_cell = load_manifest_cell(MANIFEST, 25)
    config, audit = build_experiment_config(args, manifest_cell)

    for split in ("train", "val"):
        normalizers = config.datasets[split].dataset_normalizers
        assert float(normalizers["surface_position"][0].scale) == 10000.0
        assert float(normalizers["volume_position"][0].scale) == 10000.0
    assert config.model.supernode_pooling_config.radius == 1.0
    assert audit["position_scale"] == 10000.0
    assert audit["supernode_radius"] == 1.0
    assert audit["run_id"].endswith("-ps10000-sr1")

    checkpoint_dir = args.output_path / audit["run_id"] / "train/checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "ab_upt_cp=latest_model.th").write_bytes(b"latest-target-weights")
    git_state = {
        "commit": manifest_cell["implementation_git_commit"],
        "dirty": manifest_cell["implementation_git_dirty"],
        "status_sha256": "1" * 64,
    }
    sidecar = write_training_provenance_sidecar(args, config, audit, git_state)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["position_scale"] == 10000.0
    assert payload["supernode_radius"] == 1.0


def test_runner_default_keeps_run_id_and_sidecar_value(tmp_path: Path) -> None:
    """Default runs keep historical run IDs and record the frozen rendering."""
    raw_sha = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    stats_path = tmp_path / "stats.json"
    _write_common_stats(stats_path, raw_sha)
    args = _scratch_args(stats_path, tmp_path)
    manifest_cell = load_manifest_cell(MANIFEST, 25)
    config, audit = build_experiment_config(args, manifest_cell)

    assert audit["position_scale"] == 1000.0
    assert audit["supernode_radius"] == 9.0
    assert "-ps" not in audit["run_id"]
    assert "-sr" not in audit["run_id"]
    assert config.model.supernode_pooling_config.radius == 9.0
    normalizers = config.datasets["train"].dataset_normalizers
    assert float(normalizers["surface_position"][0].scale) == 1000.0

    checkpoint_dir = args.output_path / audit["run_id"] / "train/checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "ab_upt_cp=latest_model.th").write_bytes(b"latest-target-weights")
    git_state = {
        "commit": manifest_cell["implementation_git_commit"],
        "dirty": manifest_cell["implementation_git_dirty"],
        "status_sha256": "1" * 64,
    }
    sidecar = write_training_provenance_sidecar(args, config, audit, git_state)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["position_scale"] == 1000.0
    assert payload["supernode_radius"] == 9.0


def _eval_args(
    tmp_path: Path,
    *,
    sidecar_position_scale: float | None,
    cli_position_scale: float,
    sidecar_supernode_radius: float | None = None,
    cli_supernode_radius: float = 9.0,
) -> argparse.Namespace:
    """Create a checksum-complete training result and matching evaluation arguments."""
    manifest_cell = load_manifest_cell(MANIFEST, 25)
    statistics = tmp_path / "stats.json"
    _write_common_stats(statistics, manifest_cell["raw_file_sha256"])
    target_root = tmp_path / "target"
    checkpoint = target_root / "run-a/train/checkpoints/ab_upt_cp=latest_model.th"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"construction-only-checkpoint")
    checkpoint_sha = sha256_file(checkpoint)
    sidecar_payload = {
        "schema_version": 1,
        "kind": "drivaerml_transfer_training_provenance",
        "study_id": STUDY_ID,
        "protocol_path": str(PROTOCOL.resolve()),
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_git_commit": manifest_cell["implementation_git_commit"],
        "implementation_git_dirty": manifest_cell["implementation_git_dirty"],
        "implementation_git_status_sha256": "1" * 64,
        "manifest_path": str(MANIFEST.resolve()),
        "manifest_raw_sha256": manifest_cell["raw_file_sha256"],
        "manifest_payload_sha256": manifest_cell["payload_sha256"],
        "target_statistics_path": str(statistics.resolve()),
        "target_statistics_sha256": sha256_file(statistics),
        "source_checkpoint": None,
        "source_checkpoint_sha256": None,
        "task": "common",
        "strategy": "scratch",
        "method": "S",
        "replicate": 0,
        "train_sample_size": 25,
        "coordinate_frame": "shapenet",
        "budget": "compute_matched",
        "subset_seed": 1103,
        "model_seed": 7103,
        "data_seed": None,
        "training_pipeline_seed": None,
        "data_loader_seed": 7103,
        "validation_seed": 4242,
        "evaluation_seed": 4242,
        "resolved_config_sha256": "2" * 64,
        "run_id": "run-a",
        "stage_name": "train",
        "model_checkpoints_sha256": {checkpoint.name: checkpoint_sha},
    }
    if sidecar_position_scale is not None:
        sidecar_payload["position_scale"] = sidecar_position_scale
    if sidecar_supernode_radius is not None:
        sidecar_payload["supernode_radius"] = sidecar_supernode_radius
    sidecar = checkpoint.parents[1] / "training_provenance.json"
    sidecar.write_text(json.dumps(sidecar_payload), encoding="utf-8")
    return argparse.Namespace(
        dataset_root=Path("/scratch/andyye2/data/drivaerml_subsampled_10x"),
        protocol=PROTOCOL,
        manifest=MANIFEST,
        target_statistics=statistics,
        task="common",
        coordinate_frame="shapenet",
        position_scale=cli_position_scale,
        supernode_radius=cli_supernode_radius,
        sample_size=25,
        method="S",
        replicate=0,
        budget="compute_matched",
        evaluation_split="val",
        confirm_test_release=False,
        target_output_path=target_root,
        target_run_id="run-a",
        target_stage_name="train",
        target_model_name="ab_upt",
        target_checkpoint_tag="latest",
        expected_target_sha256=checkpoint_sha,
        eval_output_path=tmp_path / "eval",
        eval_run_id=None,
        output_csv=tmp_path / "metrics.csv",
        num_geometry_points=16384,
        num_geometry_supernodes=1024,
        num_surface_queries=16384,
        num_volume_queries=16384,
        eval_point_seed=4242,
        precision="float16",
        accelerator="gpu",
        num_workers=8,
        dry_run=True,
    )


def test_eval_accepts_legacy_sidecar_at_default_rendering(tmp_path: Path, clean_evaluator_worktree: None) -> None:
    """Sidecars predating these keys evaluate only at the frozen 1000.0 / 9.0."""
    args = _eval_args(tmp_path, sidecar_position_scale=None, cli_position_scale=1000.0)
    config, audit = build_eval_config(args)
    assert audit["position_scale"] == 1000.0
    assert audit["supernode_radius"] == 9.0
    normalizers = config.datasets["val"].dataset_normalizers
    assert float(normalizers["surface_position"][0].scale) == 1000.0
    assert config.model.supernode_pooling_config.radius == 9.0


def test_eval_matches_recorded_geometry_rendering(tmp_path: Path, clean_evaluator_worktree: None) -> None:
    """A recorded exploratory rendering must be requested explicitly and is reused."""
    args = _eval_args(
        tmp_path,
        sidecar_position_scale=10000.0,
        cli_position_scale=10000.0,
        sidecar_supernode_radius=1.0,
        cli_supernode_radius=1.0,
    )
    config, audit = build_eval_config(args)
    assert audit["position_scale"] == 10000.0
    assert audit["supernode_radius"] == 1.0
    normalizers = config.datasets["val"].dataset_normalizers
    assert float(normalizers["surface_position"][0].scale) == 10000.0
    assert float(normalizers["volume_position"][0].scale) == 10000.0
    assert config.model.supernode_pooling_config.radius == 1.0


@pytest.mark.parametrize(
    ("sidecar_position_scale", "cli_position_scale", "key"),
    [
        (10000.0, 1000.0, "position_scale"),
        (None, 10000.0, "position_scale"),
        (1000.0, 10000.0, "position_scale"),
    ],
)
def test_eval_rejects_position_scale_mismatch(
    tmp_path: Path,
    clean_evaluator_worktree: None,
    sidecar_position_scale: float | None,
    cli_position_scale: float,
    key: str,
) -> None:
    """Evaluating a checkpoint at a different geometric scale is refused."""
    args = _eval_args(
        tmp_path,
        sidecar_position_scale=sidecar_position_scale,
        cli_position_scale=cli_position_scale,
    )
    with pytest.raises(ValueError, match=f"training provenance mismatch for {key}"):
        build_eval_config(args)


@pytest.mark.parametrize(
    ("sidecar_supernode_radius", "cli_supernode_radius"),
    [(1.0, 9.0), (None, 1.0), (9.0, 1.0)],
)
def test_eval_rejects_supernode_radius_mismatch(
    tmp_path: Path,
    clean_evaluator_worktree: None,
    sidecar_supernode_radius: float | None,
    cli_supernode_radius: float,
) -> None:
    """Evaluating a checkpoint under a different message graph is refused."""
    args = _eval_args(
        tmp_path,
        sidecar_position_scale=1000.0,
        cli_position_scale=1000.0,
        sidecar_supernode_radius=sidecar_supernode_radius,
        cli_supernode_radius=cli_supernode_radius,
    )
    with pytest.raises(ValueError, match="training provenance mismatch for supernode_radius"):
        build_eval_config(args)
