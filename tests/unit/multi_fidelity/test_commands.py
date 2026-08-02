# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Tests for the generated Slurm command lists."""

from __future__ import annotations

from pathlib import Path
import shlex

import pytest

from aero_cfd.multi_fidelity.experiment import GeometryRendering, SourceCheckpoint, TrainingRequest, build_experiment
from aero_cfd.multi_fidelity.integrity import sha256_file
from aero_cfd.multi_fidelity.manifest import load_manifest_cell
from aero_cfd.multi_fidelity.protocol import ProtocolBinding
from aero_cfd.multi_fidelity.provenance import write_training_provenance
from research.multi_fidelity.tools import (
    generate_evaluation_commands,
    generate_statistics_commands,
    generate_training_commands,
)
from research.multi_fidelity.tools.paper_arms import ARMS, PaperCell

from .conftest import FIXTURE_GIT_STATE

REMOTE_ROOT = Path("/scratch/andyye2/ABUPT/multi_fidelity")
DATASET_ROOT = Path("/scratch/andyye2/data/drivaerml_subsampled_10x")


def _tokens(command: str) -> list[str]:
    """Split one rendered command line into tokens."""
    return shlex.split(command)


def test_training_commands_cover_the_reported_arms(tmp_path: Path) -> None:
    """Every arm is emitted once with its own explicit geometry rendering."""
    output = tmp_path / "training.txt"
    generate_training_commands.main(
        [
            "--repo-root",
            str(REMOTE_ROOT),
            "--manifest-root",
            str(tmp_path / "manifests"),
            "--stats-root",
            str(tmp_path / "statistics"),
            "--dataset-root",
            str(DATASET_ROOT),
            "--output-path",
            str(tmp_path / "outputs"),
            "--source-output-path",
            "/scratch/andyye2/ABUPT/outputs",
            "--output",
            str(output),
            "--allow-missing-stats",
        ]
    )
    lines = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(ARMS)

    renderings = set()
    for line, arm in zip(lines, ARMS, strict=True):
        tokens = _tokens(line)
        assert tokens[tokens.index("--strategy") + 1] == arm.strategy
        assert tokens[tokens.index("--sample-size") + 1] == "100"
        assert tokens[tokens.index("--replicate") + 1] == "0"
        assert tokens[tokens.index("--model-seed") + 1] == "7103"
        assert tokens[tokens.index("--budget") + 1] == "compute_matched"
        assert tokens[tokens.index("--position-scale") + 1] == "1000"
        renderings.add((arm.strategy, tokens[tokens.index("--supernode-radius") + 1]))
        assert tokens[tokens.index("--project") + 1] == str(REMOTE_ROOT)
    assert renderings == {("scratch", "9"), ("finetune", "9"), ("scratch", "0.1"), ("finetune", "0.1")}


def test_training_generation_requires_the_statistics_artifact(tmp_path: Path) -> None:
    """A run cannot be scheduled before its normalizers exist."""
    with pytest.raises(FileNotFoundError, match="statistics artifact is missing"):
        generate_training_commands.main(
            [
                "--repo-root",
                str(REMOTE_ROOT),
                "--manifest-root",
                str(tmp_path / "manifests"),
                "--stats-root",
                str(tmp_path / "statistics"),
                "--dataset-root",
                str(DATASET_ROOT),
                "--output-path",
                str(tmp_path / "outputs"),
                "--source-output-path",
                "/scratch/andyye2/ABUPT/outputs",
                "--output",
                str(tmp_path / "training.txt"),
            ]
        )


def test_statistics_command_targets_the_shared_cell(tmp_path: Path) -> None:
    """All arms share one statistics artifact, so one command is emitted."""
    output = tmp_path / "statistics.txt"
    generate_statistics_commands.main(
        [
            "--repo-root",
            str(REMOTE_ROOT),
            "--manifest-root",
            str(tmp_path / "manifests"),
            "--stats-root",
            str(tmp_path / "statistics"),
            "--dataset-root",
            str(DATASET_ROOT),
            "--output",
            str(output),
        ]
    )
    lines = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    tokens = _tokens(lines[0])
    cell = PaperCell()
    assert tokens[tokens.index("--n") + 1] == str(cell.sample_size)
    assert tokens[tokens.index("--manifest") + 1].endswith(f"drivaerml_nested_seed{cell.subset_seed}.json")
    assert "research.multi_fidelity.tools.compute_subset_statistics" in tokens


def test_evaluation_commands_carry_the_recorded_geometry(
    tmp_path: Path,
    protocol: ProtocolBinding,
    protocol_path: Path,
    manifest_path: Path,
    statistics_path: Path,
) -> None:
    """Discovered cells evaluate under the rendering their run was trained with.

    Regression: an evaluation command that omits the rendering silently falls
    back to the frozen defaults and is then rejected by the provenance check.
    """
    training_root = tmp_path / "outputs"
    request = TrainingRequest(
        dataset_root=DATASET_ROOT,
        protocol_path=protocol_path,
        manifest_path=manifest_path,
        statistics_path=statistics_path,
        output_path=training_root,
        task="common",
        strategy="scratch",
        sample_size=25,
        replicate=0,
        model_seed=7103,
        source=SourceCheckpoint(output_path=Path("/scratch/andyye2/ABUPT/outputs")),
        geometry=GeometryRendering(supernode_radius=0.1),
    )
    experiment = build_experiment(request, load_manifest_cell(manifest_path, 25), protocol)
    checkpoints = training_root / experiment.audit["run_id"] / "train/checkpoints"
    checkpoints.mkdir(parents=True)
    latest = checkpoints / "ab_upt_cp=latest_model.th"
    latest.write_bytes(b"weights")
    write_training_provenance(request, experiment.audit, {}, FIXTURE_GIT_STATE)

    output = tmp_path / "evaluation.txt"
    generate_evaluation_commands.main(
        [
            "--repo-root",
            str(REMOTE_ROOT),
            "--dataset-root",
            str(DATASET_ROOT),
            "--training-output-root",
            str(training_root),
            "--eval-output-root",
            str(tmp_path / "eval"),
            "--metrics-root",
            str(tmp_path / "metrics"),
            "--split",
            "val",
            "--output",
            str(output),
        ]
    )
    tokens = _tokens(output.read_text(encoding="utf-8").strip())
    assert tokens[tokens.index("--supernode-radius") + 1] == "0.1"
    assert tokens[tokens.index("--position-scale") + 1] == "1000"
    assert tokens[tokens.index("--expected-target-sha256") + 1] == sha256_file(latest)
    assert tokens[tokens.index("--method") + 1] == "S"
    assert tokens[tokens.index("--evaluation-split") + 1] == "val"
    assert "--confirm-test-release" not in tokens


def test_test_split_generation_requires_an_explicit_release(tmp_path: Path) -> None:
    """Held-out data cannot be scheduled without an explicit release."""
    (tmp_path / "outputs").mkdir()
    with pytest.raises(ValueError, match="requires explicit --confirm-test-release"):
        generate_evaluation_commands.main(
            [
                "--repo-root",
                str(REMOTE_ROOT),
                "--dataset-root",
                str(DATASET_ROOT),
                "--training-output-root",
                str(tmp_path / "outputs"),
                "--eval-output-root",
                str(tmp_path / "eval"),
                "--metrics-root",
                str(tmp_path / "metrics"),
                "--split",
                "test",
                "--output",
                str(tmp_path / "evaluation.txt"),
            ]
        )


def test_an_empty_command_file_is_refused(tmp_path: Path) -> None:
    """An empty Slurm array is always an operator error."""
    (tmp_path / "outputs").mkdir()
    with pytest.raises(ValueError, match="no commands were generated"):
        generate_evaluation_commands.main(
            [
                "--repo-root",
                str(REMOTE_ROOT),
                "--dataset-root",
                str(DATASET_ROOT),
                "--training-output-root",
                str(tmp_path / "outputs"),
                "--eval-output-root",
                str(tmp_path / "eval"),
                "--metrics-root",
                str(tmp_path / "metrics"),
                "--split",
                "val",
                "--output",
                str(tmp_path / "evaluation.txt"),
            ]
        )
