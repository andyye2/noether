# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Construction tests for the frozen single-split evaluation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from aero_cfd.multi_fidelity.evaluation import EvaluationRequest, TargetModel, build_evaluation
from aero_cfd.multi_fidelity.experiment import (
    GeometryRendering,
    SourceCheckpoint,
    TrainingRequest,
    build_experiment,
)
from aero_cfd.multi_fidelity.integrity import GitState, sha256_file
from aero_cfd.multi_fidelity.manifest import load_manifest_cell
from aero_cfd.multi_fidelity.protocol import ProtocolBinding
from aero_cfd.multi_fidelity.provenance import write_training_provenance
from aero_cfd.multi_fidelity.statistics import load_statistics_binding

from .conftest import FIXTURE_GIT_STATE

TRAINED_RENDERING = GeometryRendering(supernode_radius=0.1)
CLEAN_EVALUATOR = GitState(commit="e" * 40, dirty=False, status_sha256="f" * 64)


@dataclass(frozen=True)
class TrainedRun:
    """A completed training run with a published sidecar.

    Attributes:
        output_path: Root the run wrote into.
        run_id: Training run identifier.
        checkpoint_sha256: SHA256 of the ``latest`` checkpoint.
    """

    output_path: Path
    run_id: str
    checkpoint_sha256: str


@pytest.fixture
def trained_run(
    tmp_path: Path,
    protocol: ProtocolBinding,
    protocol_path: Path,
    manifest_path: Path,
    statistics_path: Path,
) -> TrainedRun:
    """Produce a real training result through the production code path."""
    request = TrainingRequest(
        dataset_root=Path("/scratch/andyye2/data/drivaerml_subsampled_10x"),
        protocol_path=protocol_path,
        manifest_path=manifest_path,
        statistics_path=statistics_path,
        output_path=tmp_path / "outputs",
        task="common",
        strategy="scratch",
        sample_size=25,
        replicate=0,
        model_seed=7103,
        source=SourceCheckpoint(output_path=Path("/scratch/andyye2/ABUPT/outputs")),
        geometry=TRAINED_RENDERING,
    )
    experiment = build_experiment(request, load_manifest_cell(manifest_path, 25), protocol)
    checkpoints = request.output_path / experiment.audit["run_id"] / "train/checkpoints"
    checkpoints.mkdir(parents=True)
    latest = checkpoints / "ab_upt_cp=latest_model.th"
    latest.write_bytes(b"construction-only-checkpoint")
    write_training_provenance(
        request,
        experiment.audit,
        experiment.config.model_dump(mode="json", exclude_computed_fields=True),
        FIXTURE_GIT_STATE,
    )
    return TrainedRun(
        output_path=request.output_path,
        run_id=str(experiment.audit["run_id"]),
        checkpoint_sha256=sha256_file(latest),
    )


def _request(
    tmp_path: Path,
    protocol_path: Path,
    manifest_path: Path,
    statistics_path: Path,
    run: TrainedRun,
    **overrides: object,
) -> EvaluationRequest:
    """Build an evaluation request that the overrides can specialize."""
    arguments: dict[str, object] = {
        "dataset_root": Path("/scratch/andyye2/data/drivaerml_subsampled_10x"),
        "protocol_path": protocol_path,
        "manifest_path": manifest_path,
        "statistics_path": statistics_path,
        "target": TargetModel(
            output_path=run.output_path,
            run_id=run.run_id,
            expected_sha256=run.checkpoint_sha256,
        ),
        "task": "common",
        "method": "S",
        "replicate": 0,
        "sample_size": 25,
        "budget": "compute_matched",
        "eval_output_path": tmp_path / "eval",
        "output_csv": tmp_path / "metrics.csv",
        "geometry": TRAINED_RENDERING,
    }
    arguments.update(overrides)
    return EvaluationRequest(**arguments)  # type: ignore[arg-type]


def _build(
    request: EvaluationRequest,
    protocol: ProtocolBinding,
    manifest_path: Path,
    *,
    evaluator: GitState = CLEAN_EVALUATOR,
) -> tuple[object, dict]:
    """Run the evaluation builder with the shared bindings."""
    return build_evaluation(
        request,
        protocol=protocol,
        manifest_cell=load_manifest_cell(manifest_path, request.sample_size),
        statistics=load_statistics_binding(request.statistics_path, protocol=protocol),
        evaluator_git_state=evaluator,
    )


def test_evaluation_scores_one_split_and_reuses_the_recorded_rendering(
    tmp_path: Path,
    protocol: ProtocolBinding,
    protocol_path: Path,
    manifest_path: Path,
    statistics_path: Path,
    trained_run: TrainedRun,
) -> None:
    """The evaluated model keeps the geometry it was trained under."""
    request = _request(tmp_path, protocol_path, manifest_path, statistics_path, trained_run)
    config, audit = _build(request, protocol, manifest_path)

    assert set(config.datasets) == {"val"}
    assert config.model.supernode_pooling_config.radius == 0.1
    assert audit["position_scale"] == 1000.0
    assert audit["supernode_radius"] == 0.1
    assert audit["method"] == "S"
    assert audit["evaluation_design_count"] == len(audit["official_evaluation_design_ids"])
    assert config.trainer.callbacks[0].rendering == "ps1000-sr0.1"
    assert config.datasets["val"].pipeline.num_surface_anchor_points == 16384


def test_evaluating_at_another_rendering_is_refused(
    tmp_path: Path,
    protocol: ProtocolBinding,
    protocol_path: Path,
    manifest_path: Path,
    statistics_path: Path,
    trained_run: TrainedRun,
) -> None:
    """A checkpoint may not be scored under a graph it was not trained with."""
    request = _request(
        tmp_path,
        protocol_path,
        manifest_path,
        statistics_path,
        trained_run,
        geometry=GeometryRendering(),
    )
    with pytest.raises(ValueError, match="training provenance mismatch for supernode_radius"):
        _build(request, protocol, manifest_path)


def test_claiming_another_method_is_refused(
    tmp_path: Path,
    protocol: ProtocolBinding,
    protocol_path: Path,
    manifest_path: Path,
    statistics_path: Path,
    trained_run: TrainedRun,
) -> None:
    """Scientific labels come from the sidecar, not from the caller."""
    request = _request(tmp_path, protocol_path, manifest_path, statistics_path, trained_run, method="P-FT")
    with pytest.raises(ValueError, match="training provenance mismatch for method"):
        _build(request, protocol, manifest_path)


def test_a_dirty_evaluator_worktree_is_refused(
    tmp_path: Path,
    protocol: ProtocolBinding,
    protocol_path: Path,
    manifest_path: Path,
    statistics_path: Path,
    trained_run: TrainedRun,
) -> None:
    """Released numbers must come from a committed implementation."""
    request = _request(tmp_path, protocol_path, manifest_path, statistics_path, trained_run)
    dirty = GitState(commit="e" * 40, dirty=True, status_sha256="f" * 64)
    with pytest.raises(ValueError, match="clean implementation Git worktree"):
        _build(request, protocol, manifest_path, evaluator=dirty)


def test_test_split_requires_an_explicit_release(
    tmp_path: Path,
    protocol_path: Path,
    manifest_path: Path,
    statistics_path: Path,
    trained_run: TrainedRun,
) -> None:
    """The held-out split cannot be reached by accident."""
    with pytest.raises(ValueError, match="confirm_test_release"):
        _request(
            tmp_path,
            protocol_path,
            manifest_path,
            statistics_path,
            trained_run,
            evaluation_split="test",
        )


def test_a_tampered_checkpoint_is_refused(
    tmp_path: Path,
    protocol: ProtocolBinding,
    protocol_path: Path,
    manifest_path: Path,
    statistics_path: Path,
    trained_run: TrainedRun,
) -> None:
    """Weights that changed after training cannot be scored."""
    checkpoint = trained_run.output_path / trained_run.run_id / "train/checkpoints/ab_upt_cp=latest_model.th"
    checkpoint.write_bytes(b"tampered-weights")
    request = _request(tmp_path, protocol_path, manifest_path, statistics_path, trained_run)
    with pytest.raises(ValueError, match="checkpoint SHA256 mismatch"):
        _build(request, protocol, manifest_path)


def test_an_undeclared_extra_checkpoint_is_refused(
    tmp_path: Path,
    protocol: ProtocolBinding,
    protocol_path: Path,
    manifest_path: Path,
    statistics_path: Path,
    trained_run: TrainedRun,
) -> None:
    """A checkpoint set that grew after the sidecar was written is suspect."""
    checkpoints = trained_run.output_path / trained_run.run_id / "train/checkpoints"
    (checkpoints / "ab_upt_cp=best_model.loss.val.total_model.th").write_bytes(b"late-arrival")
    request = _request(tmp_path, protocol_path, manifest_path, statistics_path, trained_run)
    with pytest.raises(ValueError, match="checkpoint set mismatch"):
        _build(request, protocol, manifest_path)


def test_expected_checkpoint_hash_must_match_the_selection(
    tmp_path: Path,
    protocol: ProtocolBinding,
    protocol_path: Path,
    manifest_path: Path,
    statistics_path: Path,
    trained_run: TrainedRun,
) -> None:
    """The caller must name the exact weights it believes it is scoring."""
    request = _request(tmp_path, protocol_path, manifest_path, statistics_path, trained_run)
    request = replace(request, target=replace(request.target, expected_sha256="0" * 64))
    with pytest.raises(ValueError, match="target checkpoint SHA256 mismatch"):
        _build(request, protocol, manifest_path)
