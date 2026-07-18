# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Construction tests for the strict DrivAerML transfer runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from pathlib import Path

import pytest
import yaml

from research.multi_fidelity.tools import materialize_study_manifests
from research.multi_fidelity.tools.generate_training_commands import Cell, command_for_cell
from recipes.aero_cfd.scripts.run_drivaerml_transfer_strict import (
    FROZEN_PROTOCOL_STATUS,
    KNOWN_SOURCE_SHA256,
    build_experiment_config,
    build_lr_modifiers,
    load_manifest_cell,
    load_protocol_binding,
    sha256_file,
    require_frozen_protocol_for_execution,
    source_checkpoint_path,
    write_training_provenance_sidecar,
)

REPO_ROOT = Path(__file__).parents[4]
PROTOCOL = REPO_ROOT / "research/multi_fidelity/experiment_protocol.yaml"
MANIFEST = REPO_ROOT / "research/multi_fidelity/evidence/manifests/drivaerml_nested_seed1103.json"
PROTOCOL_SHA256 = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
STUDY_ID = "shapenet_ab_upt_to_drivaerml_hf_efficiency"


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


def _scratch_args(stats_path: Path, tmp_path: Path) -> argparse.Namespace:
    """Return the complete Namespace consumed by the config builder."""
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


def test_strict_config_uses_train_val_only_and_exact_updates(tmp_path: Path) -> None:
    """Confirmatory construction has no test dataset or test callback."""
    raw_sha = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    stats_path = tmp_path / "stats.json"
    _write_common_stats(stats_path, raw_sha)
    manifest_cell = load_manifest_cell(MANIFEST, 25)
    config, audit = build_experiment_config(_scratch_args(stats_path, tmp_path), manifest_cell)

    assert set(config.datasets) == {"train", "val"}
    assert audit["test_dataset_in_training_config"] is False
    assert config.trainer.max_epochs is None
    assert config.trainer.max_updates == 40_000
    callback_dataset_keys = {
        callback.dataset_key for callback in config.trainer.callbacks if hasattr(callback, "dataset_key")
    }
    assert callback_dataset_keys == {"val"}
    assert config.datasets["train"].pipeline.seed is None
    assert config.datasets["val"].pipeline.seed == 4242
    assert config.datasets["train"].dataset_wrappers[0].indices == manifest_cell["base_dataset_indices"]


def test_statistics_manifest_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    """A stats cache from another acquisition ladder cannot configure a run."""
    stats_path = tmp_path / "wrong.json"
    _write_common_stats(stats_path, "0" * 64)
    manifest_cell = load_manifest_cell(MANIFEST, 25)
    try:
        build_experiment_config(_scratch_args(stats_path, tmp_path), manifest_cell)
    except ValueError as error:
        assert "statistics/manifest SHA256 mismatch" in str(error)
    else:
        raise AssertionError("mismatched statistics artifact was accepted")


def test_success_sidecar_binds_config_and_all_model_checkpoints(tmp_path: Path) -> None:
    """A completed run publishes one atomic, checksum-complete provenance sidecar."""
    manifest_sha = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    stats_path = tmp_path / "stats.json"
    _write_common_stats(stats_path, manifest_sha)
    args = _scratch_args(stats_path, tmp_path)
    manifest_cell = load_manifest_cell(MANIFEST, 25)
    config, audit = build_experiment_config(args, manifest_cell)
    checkpoint_dir = args.output_path / audit["run_id"] / "train/checkpoints"
    checkpoint_dir.mkdir(parents=True)
    latest = checkpoint_dir / "ab_upt_cp=latest_model.th"
    best = checkpoint_dir / "ab_upt_cp=best_model.loss.val.total_model.th"
    latest.write_bytes(b"latest-target-weights")
    best.write_bytes(b"best-target-weights")
    git_state = {
        "commit": manifest_cell["implementation_git_commit"],
        "dirty": manifest_cell["implementation_git_dirty"],
        "status_sha256": "1" * 64,
    }

    sidecar = write_training_provenance_sidecar(args, config, audit, git_state)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))

    assert sidecar.name == "training_provenance.json"
    assert payload["protocol_sha256"] == PROTOCOL_SHA256
    assert payload["manifest_raw_sha256"] == manifest_sha
    assert payload["target_statistics_sha256"] == sha256_file(stats_path)
    assert payload["method"] == "S"
    assert payload["replicate"] == 0
    assert payload["data_seed"] is None
    assert payload["training_pipeline_seed"] is None
    assert payload["data_loader_seed"] == 7103
    assert payload["validation_seed"] == 4242
    assert payload["model_checkpoints_sha256"] == {
        best.name: sha256_file(best),
        latest.name: sha256_file(latest),
    }


def test_replicate_cannot_claim_another_model_seed(tmp_path: Path) -> None:
    """The protocol replicate label determines the model seed exactly."""
    manifest_sha = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    stats_path = tmp_path / "stats.json"
    _write_common_stats(stats_path, manifest_sha)
    args = _scratch_args(stats_path, tmp_path)
    args.model_seed = 9999
    manifest_cell = load_manifest_cell(MANIFEST, 25)
    try:
        build_experiment_config(args, manifest_cell)
    except ValueError as error:
        assert "requires model seed" in str(error)
    else:
        raise AssertionError("a false replicate/model-seed label was accepted")


def test_generated_command_uses_remote_protocol_and_uv_project() -> None:
    """Remote commands honor an external production manifest root."""
    repo_root = Path("/scratch/andyye2/ABUPT/noether")
    manifest_root = Path("/scratch/andyye2/ABUPT/frozen_manifests")
    command = command_for_cell(
        Cell(
            replicate=0,
            subset_seed=1103,
            model_seed=7103,
            n=25,
            task="common",
            strategy="scratch",
            budget="smoke",
        ),
        repo_root=repo_root,
        manifest_root=manifest_root,
        protocol_path=repo_root / "research/multi_fidelity/experiment_protocol.yaml",
        dataset_root=Path("/scratch/andyye2/data/drivaerml_subsampled_10x"),
        output_path=Path("/scratch/andyye2/ABUPT/outputs/mf"),
        stats_root=Path("/scratch/andyye2/ABUPT/stats"),
        source_output_path=Path("/scratch/andyye2/ABUPT/outputs"),
    )
    tokens = shlex.split(command)

    assert tokens[tokens.index("--project") + 1] == str(repo_root)
    assert tokens[tokens.index("--protocol") + 1] == str(repo_root / "research/multi_fidelity/experiment_protocol.yaml")
    assert tokens[tokens.index("--replicate") + 1] == "0"
    assert tokens[tokens.index("--manifest") + 1] == str(manifest_root / "drivaerml_nested_seed1103.json")
    assert "--point-seed" not in tokens


def test_protocol_primary_source_matches_audited_constant(tmp_path: Path) -> None:
    """Protocol source identity is parsed and cannot drift from the audited constant."""
    binding = load_protocol_binding(PROTOCOL)
    assert binding["source_primary_checkpoint_tag"] == "latest"
    assert binding["source_primary_sha256"] == KNOWN_SOURCE_SHA256[("latest", None)]

    payload = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    payload["source"]["primary_sha256"] = "0" * 64
    mismatched = tmp_path / "mismatched_protocol.yaml"
    mismatched.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="disagrees with the audited constant"):
        load_protocol_binding(mismatched)


def test_expected_source_sha_cannot_override_protocol(tmp_path: Path) -> None:
    """A CLI checksum cannot silently substitute another confirmatory source."""
    args = _scratch_args(tmp_path / "unused_stats.json", tmp_path)
    args.expected_source_sha256 = "0" * 64
    with pytest.raises(ValueError, match="cannot override the protocol primary source"):
        source_checkpoint_path(args, load_protocol_binding(PROTOCOL))


def test_production_execution_accepts_frozen_protocol_and_rejects_draft() -> None:
    """The repository protocol is frozen while a draft binding remains blocked."""
    binding = load_protocol_binding(PROTOCOL)
    assert binding["status"] == FROZEN_PROTOCOL_STATUS
    require_frozen_protocol_for_execution(binding)

    draft_binding = {**binding, "status": "preregistration_draft"}
    with pytest.raises(ValueError, match="production execution is blocked"):
        require_frozen_protocol_for_execution(draft_binding)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_transfer_lr_multipliers_must_be_finite_and_positive(tmp_path: Path, value: float) -> None:
    """Invalid transfer LR ratios fail before optimizer construction."""
    args = _scratch_args(tmp_path / "unused.json", tmp_path)
    args.strategy = "finetune"
    args.body_lr_multiplier = value
    with pytest.raises(ValueError, match="finite and positive"):
        build_lr_modifiers(args)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("volume_velocity_std", [8.0, 7.0], "must have equal lengths"),
        ("surface_pressure_mean", [float("nan")], "finite numbers"),
    ],
)
def test_malformed_normalizer_statistics_are_rejected(
    tmp_path: Path,
    key: str,
    value: list[float],
    message: str,
) -> None:
    """Malformed or non-finite statistics cannot silently configure training."""
    manifest_sha = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    stats_path = tmp_path / "malformed.json"
    _write_common_stats(stats_path, manifest_sha)
    artifact = json.loads(stats_path.read_text(encoding="utf-8"))
    artifact["normalizer_stats"][key] = value
    stats_path.write_text(json.dumps(artifact), encoding="utf-8")

    manifest_cell = load_manifest_cell(MANIFEST, 25)
    with pytest.raises(ValueError, match=message):
        build_experiment_config(_scratch_args(stats_path, tmp_path), manifest_cell)


def test_production_manifest_materialization_rejects_draft_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No production manifest can be emitted before the explicit freeze."""
    draft_protocol = tmp_path / "draft_protocol.yaml"
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    protocol["status"] = "preregistration_draft"
    draft_protocol.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "materialize_study_manifests.py",
            "--output-dir",
            str(tmp_path / "external-manifests"),
            "--protocol",
            str(draft_protocol),
        ],
    )
    with pytest.raises(ValueError, match="production manifests require protocol status"):
        materialize_study_manifests.main()
