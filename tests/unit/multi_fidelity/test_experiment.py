# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Construction tests for the strict DrivAerML training experiment."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aero_cfd.multi_fidelity.experiment import (
    CHECKPOINT_POSITION_SCALE,
    GeometryRendering,
    SourceCheckpoint,
    TrainingRequest,
    build_experiment,
)
from aero_cfd.multi_fidelity.integrity import canonical_json_sha256, sha256_file, stable_sha256
from aero_cfd.multi_fidelity.manifest import load_manifest_cell
from aero_cfd.multi_fidelity.protocol import ProtocolBinding
from aero_cfd.multi_fidelity.provenance import (
    PROVENANCE_SCHEMA_VERSION,
    resolved_config_sha256,
    write_training_provenance,
)

from .conftest import FIXTURE_GIT_STATE, write_statistics_artifact

SOURCE_ROOT = Path("/scratch/andyye2/ABUPT/outputs")


def _request(
    tmp_path: Path,
    protocol_path: Path,
    manifest_path: Path,
    statistics_path: Path,
    **overrides: object,
) -> TrainingRequest:
    """Build a scratch request that the overrides can specialize."""
    arguments: dict[str, object] = {
        "dataset_root": Path("/scratch/andyye2/data/drivaerml_subsampled_10x"),
        "protocol_path": protocol_path,
        "manifest_path": manifest_path,
        "statistics_path": statistics_path,
        "output_path": tmp_path / "outputs",
        "task": "common",
        "strategy": "scratch",
        "sample_size": 25,
        "replicate": 0,
        "model_seed": 7103,
        "source": SourceCheckpoint(output_path=SOURCE_ROOT),
    }
    arguments.update(overrides)
    return TrainingRequest(**arguments)  # type: ignore[arg-type]


def test_config_uses_train_and_val_only_with_exact_updates(
    tmp_path: Path,
    protocol: ProtocolBinding,
    protocol_path: Path,
    manifest_path: Path,
    statistics_path: Path,
) -> None:
    """Confirmatory construction has no test dataset and a fixed update budget."""
    request = _request(tmp_path, protocol_path, manifest_path, statistics_path)
    cell = load_manifest_cell(manifest_path, 25)
    experiment = build_experiment(request, cell, protocol)

    assert set(experiment.config.datasets) == {"train", "val"}
    assert experiment.config.trainer.max_epochs is None
    assert experiment.config.trainer.max_updates == 40_000
    assert {
        callback.dataset_key for callback in experiment.config.trainer.callbacks if hasattr(callback, "dataset_key")
    } == {"val"}
    assert experiment.config.datasets["train"].pipeline.seed is None
    assert experiment.config.datasets["val"].pipeline.seed == 4242
    assert experiment.config.datasets["train"].dataset_wrappers[0].indices == list(cell.base_dataset_indices)
    assert experiment.audit["method"] == "S"
    assert experiment.audit["test_dataset_in_training_config"] is False


def test_geometry_rendering_reaches_normalizers_model_and_run_id(
    tmp_path: Path,
    protocol: ProtocolBinding,
    protocol_path: Path,
    manifest_path: Path,
    statistics_path: Path,
) -> None:
    """A non-default radius changes the graph, the run ID, and the audit."""
    request = _request(
        tmp_path,
        protocol_path,
        manifest_path,
        statistics_path,
        geometry=GeometryRendering(supernode_radius=0.1),
    )
    experiment = build_experiment(request, load_manifest_cell(manifest_path, 25), protocol)

    assert experiment.config.model.supernode_pooling_config.radius == 0.1
    for split in ("train", "val"):
        normalizers = experiment.config.datasets[split].dataset_normalizers
        assert float(normalizers["surface_position"][0].scale) == CHECKPOINT_POSITION_SCALE
    assert experiment.audit["run_id"].endswith("-sr0.1")
    assert experiment.audit["supernode_radius_position_fraction"] == pytest.approx(1e-4)
    # The fixed CFD envelope spans 120 raw metres, so 0.1/1000 of it is 12 mm.
    assert experiment.audit["supernode_radius_raw_units"] == pytest.approx(0.012)
    assert experiment.audit["position_span_raw_units"] == pytest.approx(120.0)


def test_default_rendering_keeps_the_historical_run_id(
    tmp_path: Path,
    protocol: ProtocolBinding,
    protocol_path: Path,
    manifest_path: Path,
    statistics_path: Path,
) -> None:
    """The frozen rendering is unsuffixed, so historical run IDs are stable."""
    request = _request(tmp_path, protocol_path, manifest_path, statistics_path)
    experiment = build_experiment(request, load_manifest_cell(manifest_path, 25), protocol)

    assert "-ps" not in experiment.audit["run_id"]
    assert "-sr" not in experiment.audit["run_id"]
    assert experiment.config.model.supernode_pooling_config.radius == 9.0


def test_transfer_cannot_rescale_the_pretrained_positional_basis(
    tmp_path: Path,
    protocol_path: Path,
    manifest_path: Path,
    statistics_path: Path,
) -> None:
    """Restoring a checkpoint under a different position scale is refused.

    The sincos and RoPE frequency buffers are restored from the checkpoint, so
    a rescaled position range would evaluate frozen bands at shifted phases.
    """
    with pytest.raises(ValueError, match="pretrained positional basis"):
        _request(
            tmp_path,
            protocol_path,
            manifest_path,
            statistics_path,
            strategy="finetune",
            geometry=GeometryRendering(position_scale=10_000.0, supernode_radius=1.0),
        )


def test_scratch_may_use_any_position_scale(
    tmp_path: Path,
    protocol_path: Path,
    manifest_path: Path,
    statistics_path: Path,
) -> None:
    """A scratch arm restores no checkpoint, so no basis constrains it."""
    request = _request(
        tmp_path,
        protocol_path,
        manifest_path,
        statistics_path,
        geometry=GeometryRendering(position_scale=10_000.0, supernode_radius=1.0),
    )
    assert request.geometry.position_scale == 10_000.0


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_geometry_is_rejected(value: float) -> None:
    """Non-finite or non-positive geometry parameters cannot configure a run."""
    with pytest.raises(ValueError, match="finite and positive"):
        GeometryRendering(supernode_radius=value)
    with pytest.raises(ValueError, match="finite and positive"):
        GeometryRendering(position_scale=value)


def test_replicate_label_determines_the_model_seed(
    tmp_path: Path,
    protocol: ProtocolBinding,
    protocol_path: Path,
    manifest_path: Path,
    statistics_path: Path,
) -> None:
    """A run cannot claim a replicate it does not use the seeds of."""
    request = _request(tmp_path, protocol_path, manifest_path, statistics_path, model_seed=9999)
    with pytest.raises(ValueError, match="requires model seed"):
        build_experiment(request, load_manifest_cell(manifest_path, 25), protocol)


def test_statistics_from_another_ladder_are_rejected(
    tmp_path: Path,
    protocol: ProtocolBinding,
    protocol_path: Path,
    manifest_path: Path,
) -> None:
    """A statistics artifact must be hashed against this exact manifest."""
    statistics = write_statistics_artifact(
        tmp_path / "wrong.json",
        protocol=protocol,
        manifest_raw_sha256="0" * 64,
        train_subset_size=25,
    )
    request = _request(tmp_path, protocol_path, manifest_path, statistics)
    with pytest.raises(ValueError, match="statistics/manifest SHA256 mismatch"):
        build_experiment(request, load_manifest_cell(manifest_path, 25), protocol)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("volume_velocity_std", [8.0, 7.0], "must have equal lengths"),
        ("surface_pressure_mean", [float("nan")], "finite numbers"),
    ],
)
def test_malformed_statistics_are_rejected(
    tmp_path: Path,
    protocol: ProtocolBinding,
    protocol_path: Path,
    manifest_path: Path,
    key: str,
    value: list[float],
    message: str,
) -> None:
    """Malformed statistics cannot silently configure normalizers."""
    statistics = write_statistics_artifact(
        tmp_path / "malformed.json",
        protocol=protocol,
        manifest_raw_sha256=sha256_file(manifest_path),
        train_subset_size=25,
        overrides={key: value},
    )
    request = _request(tmp_path, protocol_path, manifest_path, statistics)
    with pytest.raises(ValueError, match=message):
        build_experiment(request, load_manifest_cell(manifest_path, 25), protocol)


def test_sidecar_binds_the_config_and_every_checkpoint(
    tmp_path: Path,
    protocol: ProtocolBinding,
    protocol_path: Path,
    manifest_path: Path,
    statistics_path: Path,
) -> None:
    """A completed run publishes one atomic, checksum-complete sidecar."""
    request = _request(
        tmp_path,
        protocol_path,
        manifest_path,
        statistics_path,
        geometry=GeometryRendering(supernode_radius=0.1),
    )
    experiment = build_experiment(request, load_manifest_cell(manifest_path, 25), protocol)
    checkpoints = request.output_path / experiment.audit["run_id"] / "train/checkpoints"
    checkpoints.mkdir(parents=True)
    latest = checkpoints / "ab_upt_cp=latest_model.th"
    best = checkpoints / "ab_upt_cp=best_model.loss.val.total_model.th"
    latest.write_bytes(b"latest-target-weights")
    best.write_bytes(b"best-target-weights")

    sidecar = write_training_provenance(
        request,
        experiment.audit,
        experiment.config.model_dump(mode="python", exclude_computed_fields=True),
        FIXTURE_GIT_STATE,
    )
    payload = json.loads(sidecar.read_text(encoding="utf-8"))

    assert sidecar.name == "training_provenance.json"
    assert payload["schema_version"] == PROVENANCE_SCHEMA_VERSION
    assert payload["protocol_sha256"] == protocol.sha256
    assert payload["manifest_raw_sha256"] == sha256_file(manifest_path)
    assert payload["target_statistics_sha256"] == sha256_file(statistics_path)
    assert payload["method"] == "S"
    assert payload["position_scale"] == CHECKPOINT_POSITION_SCALE
    assert payload["supernode_radius"] == 0.1
    assert payload["expected_updates"] == 40_000
    assert payload["data_loader_seed"] == 7103
    assert payload["model_checkpoints_sha256"] == {
        best.name: sha256_file(best),
        latest.name: sha256_file(latest),
    }


def test_resolved_config_hash_does_not_depend_on_set_iteration_order(
    tmp_path: Path,
    protocol: ProtocolBinding,
    protocol_path: Path,
    manifest_path: Path,
    statistics_path: Path,
) -> None:
    """Two identical runs must record the same resolved-config hash.

    Two things break that otherwise. ``excluded_properties`` is stored as a
    set, so a plain dump orders it by the process hash seed, and ``master_port``
    is drawn at random unless it is exported. A hash that moves on every run
    cannot prove that two runs shared a configuration.
    """
    assert stable_sha256({"a": {"y", "x"}}) == canonical_json_sha256({"a": ["x", "y"]})

    request = _request(tmp_path, protocol_path, manifest_path, statistics_path)
    cell = load_manifest_cell(manifest_path, 25)
    dumps = [
        build_experiment(request, cell, protocol).config.model_dump(mode="python", exclude_computed_fields=True)
        for _ in range(2)
    ]
    for dump in dumps:
        # The flaws this guards against are invisible without these two fields.
        assert isinstance(dump["datasets"]["train"]["excluded_properties"], set)
        assert isinstance(dump["master_port"], int)
    assert dumps[0]["master_port"] != dumps[1]["master_port"]
    assert resolved_config_sha256(dumps[0]) == resolved_config_sha256(dumps[1])
    assert stable_sha256(dumps[0]) != stable_sha256(dumps[1])


def test_a_run_without_checkpoints_cannot_publish_provenance(
    tmp_path: Path,
    protocol: ProtocolBinding,
    protocol_path: Path,
    manifest_path: Path,
    statistics_path: Path,
) -> None:
    """Provenance must never describe a run that produced no weights."""
    request = _request(tmp_path, protocol_path, manifest_path, statistics_path)
    experiment = build_experiment(request, load_manifest_cell(manifest_path, 25), protocol)
    with pytest.raises(RuntimeError, match="no model checkpoints"):
        write_training_provenance(request, experiment.audit, {}, FIXTURE_GIT_STATE)
