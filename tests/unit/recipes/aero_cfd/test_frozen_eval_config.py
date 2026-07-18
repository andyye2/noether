# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Construction tests for the frozen DrivAerML evaluation entry point."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from recipes.aero_cfd.scripts.eval_drivaerml_transfer_frozen import build_eval_config
from recipes.aero_cfd.scripts.run_drivaerml_transfer_strict import load_manifest_cell, sha256_file

REPO_ROOT = Path(__file__).parents[4]
PROTOCOL = REPO_ROOT / "research/multi_fidelity/experiment_protocol.yaml"
MANIFEST = REPO_ROOT / "research/multi_fidelity/evidence/manifests/drivaerml_nested_seed1103.json"
PROTOCOL_SHA256 = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
STUDY_ID = "shapenet_ab_upt_to_drivaerml_hf_efficiency"


def _eval_args(tmp_path: Path) -> argparse.Namespace:
    """Create a checksum-complete training result and matching evaluation arguments."""
    manifest_cell = load_manifest_cell(MANIFEST, 25)
    statistics = tmp_path / "stats.json"
    statistics.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "study_id": STUDY_ID,
                "protocol_sha256": PROTOCOL_SHA256,
                "train_subset_size": 25,
                "coordinate_frame": "shapenet",
                "provenance": {"manifest_sha256": manifest_cell["raw_file_sha256"]},
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
        ),
        encoding="utf-8",
    )
    target_root = tmp_path / "target"
    checkpoint = target_root / "run-a/train/checkpoints/ab_upt_cp=latest_model.th"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"construction-only-checkpoint")
    checkpoint_sha = sha256_file(checkpoint)
    sidecar = checkpoint.parents[1] / "training_provenance.json"
    sidecar.write_text(
        json.dumps(
            {
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
        ),
        encoding="utf-8",
    )
    return argparse.Namespace(
        dataset_root=Path("/scratch/andyye2/data/drivaerml_subsampled_10x"),
        protocol=PROTOCOL,
        manifest=MANIFEST,
        target_statistics=statistics,
        task="common",
        coordinate_frame="shapenet",
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


def test_frozen_eval_is_test_only_and_json_serializable(tmp_path: Path) -> None:
    """An explicitly released test evaluation reconstructs test only."""
    args = _eval_args(tmp_path)
    args.evaluation_split = "test"
    args.confirm_test_release = True
    config, audit = build_eval_config(args)

    assert set(config.datasets) == {"test"}
    assert config.trainer.max_epochs == 0
    assert audit["evaluation_split"] == "test"
    assert audit["test_design_count"] == 50
    assert audit["surface_queries_per_design"] == 16384
    assert audit["training_pipeline_seed"] is None
    assert audit["data_loader_seed"] == 7103
    assert audit["training_provenance_sidecar_sha256"] == sha256_file(
        args.target_output_path / "run-a/train/training_provenance.json"
    )
    assert audit["target_checkpoint_sha256"] == args.expected_target_sha256
    dumped = config.model_dump(mode="json", exclude_computed_fields=True)
    assert set(dumped["datasets"]) == {"test"}


@pytest.mark.parametrize(
    ("field", "false_value"),
    [
        ("method", "P-FT"),
        ("replicate", 1),
        ("sample_size", 50),
        ("coordinate_frame", "native"),
        ("budget", "fixed_epoch"),
    ],
)
def test_frozen_eval_rejects_false_cli_labels(
    tmp_path: Path,
    field: str,
    false_value: str | int,
) -> None:
    """Metrics labels cannot disagree with the sidecar that created the checkpoint."""
    args = _eval_args(tmp_path)
    setattr(args, field, false_value)

    with pytest.raises(ValueError, match="training provenance mismatch"):
        build_eval_config(args)


def test_frozen_validation_export_is_val_only(tmp_path: Path) -> None:
    """The futility gate can export validation metrics without touching test."""
    args = _eval_args(tmp_path)
    args.evaluation_split = "val"
    config, audit = build_eval_config(args)

    assert set(config.datasets) == {"val"}
    assert audit["evaluation_split"] == "val"
    assert audit["evaluation_design_count"] == 34
    assert audit["test_design_count"] is None


def test_test_split_requires_confirmation_in_eval_runner(tmp_path: Path) -> None:
    """Direct eval-runner use cannot bypass the coordinated test release gate."""
    args = _eval_args(tmp_path)
    args.evaluation_split = "test"

    with pytest.raises(ValueError, match="explicit --confirm-test-release"):
        build_eval_config(args)


def test_scratch_sidecar_rejects_recorded_source_checkpoint(tmp_path: Path) -> None:
    """Scratch metrics cannot be relabeled from provenance that records pretraining."""
    args = _eval_args(tmp_path)
    sidecar_path = args.target_output_path / "run-a/train/training_provenance.json"
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    payload["source_checkpoint"] = "/frozen/source/ab_upt_cp=latest_model.th"
    payload["source_checkpoint_sha256"] = "0" * 64
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="scratch training provenance must not record"):
        build_eval_config(args)


def test_transfer_sidecar_requires_protocol_primary_source_sha(tmp_path: Path) -> None:
    """Transfer metrics require the source checkpoint frozen by the protocol."""
    args = _eval_args(tmp_path)
    args.method = "P-FT"
    sidecar_path = args.target_output_path / "run-a/train/training_provenance.json"
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    payload["strategy"] = "finetune"
    payload["method"] = "P-FT"
    payload["source_checkpoint"] = "/frozen/source/ab_upt_cp=latest_model.th"
    payload["source_checkpoint_sha256"] = "0" * 64
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must bind the protocol primary source checkpoint"):
        build_eval_config(args)
