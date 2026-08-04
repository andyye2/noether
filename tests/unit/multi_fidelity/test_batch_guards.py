# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Tests for the B2 submission boundary and command selection."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest

from research.multi_fidelity.tools import generate_training_commands

REPO_ROOT = Path(__file__).resolve().parents[3]
SLURM_ROOT = REPO_ROOT / "research/multi_fidelity/slurm"
STATISTICS_BATCH = SLURM_ROOT / "drivaerml_statistics_array.sbatch"
TRAINING_BATCH = SLURM_ROOT / "drivaerml_training_array.sbatch"
VTRANSFER_STATISTICS_BATCH = SLURM_ROOT / "drivaerml_vtransfer_statistics_array.sbatch"
VTRANSFER_TRAINING_BATCH = SLURM_ROOT / "drivaerml_vtransfer_training_array.sbatch"
RETIRED_BATCH = SLURM_ROOT / "drivaerml_paper_array.sbatch"
REMOTE_ROOT = Path("/scratch/andyye2/ABUPT/multi_fidelity_B2")
DATASET_ROOT = Path("/scratch/andyye2/data/drivaerml_subsampled_10x")

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash is unavailable")


@pytest.mark.parametrize(
    "batch",
    [STATISTICS_BATCH, TRAINING_BATCH, VTRANSFER_STATISTICS_BATCH, VTRANSFER_TRAINING_BATCH, RETIRED_BATCH],
)
def test_batch_scripts_are_valid_bash(batch: Path) -> None:
    """A syntax error must be caught before a Slurm submission."""
    assert subprocess.run(["bash", "-n", str(batch)], check=False).returncode == 0


@pytest.mark.parametrize(
    ("batches", "namespace"),
    [
        ((STATISTICS_BATCH, TRAINING_BATCH), "multi_fidelity_B2"),
        ((VTRANSFER_STATISTICS_BATCH, VTRANSFER_TRAINING_BATCH), "multi_fidelity_B2_vtransfer"),
    ],
)
def test_batches_are_bound_to_their_own_namespace(batches: tuple[Path, ...], namespace: str) -> None:
    """Code and logs must never fall back to another experiment's tree."""
    for batch in batches:
        text = batch.read_text(encoding="utf-8")
        assert f"PROJECT_ROOT=/scratch/andyye2/ABUPT/{namespace}\n" in text
        assert f"#SBATCH --output=/scratch/andyye2/ABUPT/slurm/{namespace}/" in text
        assert f"#SBATCH --error=/scratch/andyye2/ABUPT/slurm/{namespace}/" in text


def test_the_vtransfer_batch_cannot_write_into_the_b2_results() -> None:
    """The two namespaces run the same cell and must stay separable."""
    text = VTRANSFER_TRAINING_BATCH.read_text(encoding="utf-8")
    assert "/scratch/andyye2/ABUPT/outputs/multi_fidelity_B2/" in text
    assert "/scratch/andyye2/ABUPT/multi_fidelity_B2_artifacts/" in text


def test_training_batch_freezes_the_two_matched_arms() -> None:
    """The production array accepts only the shared matched geometry cell."""
    text = TRAINING_BATCH.read_text(encoding="utf-8")
    for required in (
        "command_count -ne 2",
        "--strategy scratch",
        "--strategy finetune",
        "--sample-size 100",
        "--budget compute_matched",
        "--position-scale 1000",
        "--supernode-radius 0.1",
        "--replicate 0",
        "--model-seed 7103",
    ):
        assert required in text


def _run_training_guard(
    tmp_path: Path,
    commands: list[str],
    batch: Path = TRAINING_BATCH,
) -> subprocess.CompletedProcess[str]:
    """Execute a training batch's local guards against one command file.

    Every check the guards reach here is local: the script exits long before it
    needs the cluster, the environment, or the dataset.

    Args:
        tmp_path: Directory the command file is written to.
        commands: Command lines, one per array index.
        batch: Batch script to execute.

    Returns:
        The completed process, whose exit code and stderr carry the verdict.
    """
    command_file = tmp_path / "commands.txt"
    command_file.write_text("".join(f"{command}\n" for command in commands), encoding="utf-8")
    digest = hashlib.sha256(command_file.read_bytes()).hexdigest()
    return subprocess.run(
        ["bash", str(batch), "0" * 40, str(command_file), digest],
        env={**os.environ, "SLURM_ARRAY_TASK_ID": "1"},
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_vtransfer_batch_requires_the_feature_on_both_arms(tmp_path: Path) -> None:
    """This namespace holds one pair, and it is the wall-distance pair.

    A plain arm submitted here would be stored beside feature arms and later
    read as if it were their baseline.
    """
    plain = "--strategy scratch --supernode-radius 0.1"
    feature = "--strategy finetune --supernode-radius 0.1 --wall-distance-feature"

    refused = _run_training_guard(tmp_path, [plain, feature], VTRANSFER_TRAINING_BATCH)
    assert refused.returncode == 2
    assert "requires --wall-distance-feature on both arms" in refused.stderr

    paired = _run_training_guard(
        tmp_path,
        [f"{plain} --wall-distance-feature", feature],
        VTRANSFER_TRAINING_BATCH,
    )
    assert "--wall-distance-feature on both arms" not in paired.stderr


def test_training_batch_refuses_to_pair_a_feature_arm_with_a_plain_one(tmp_path: Path) -> None:
    """Both arms must read the same inputs, or the contrast means nothing.

    A mixed submission would differ in initialization *and* in what the model
    is given, which is not the question either arm answers.
    """
    plain = "--strategy scratch --supernode-radius 0.1"
    feature = "--strategy finetune --supernode-radius 0.1 --wall-distance-feature"

    mixed = _run_training_guard(tmp_path, [plain, feature])
    assert mixed.returncode == 2
    assert "--wall-distance-feature" in mixed.stderr

    # Both arms carrying the feature passes this guard and fails later, on the
    # entry-point prefix, which is what a fabricated command line should do.
    paired = _run_training_guard(tmp_path, [f"{plain} --wall-distance-feature", feature])
    assert "both arms must set" not in paired.stderr


def test_training_generator_rejects_duplicate_arms(tmp_path: Path) -> None:
    """Duplicate arm names would make two array tasks overwrite one run ID."""
    with pytest.raises(ValueError, match="must not contain duplicates"):
        generate_training_commands.main(
            [
                "--repo-root",
                str(REMOTE_ROOT),
                "--protocol",
                str(REPO_ROOT / "research/multi_fidelity/experiment_protocol.yaml"),
                "--manifest-root",
                str(tmp_path / "manifests"),
                "--stats-root",
                str(tmp_path / "statistics"),
                "--dataset-root",
                str(DATASET_ROOT),
                "--output-path",
                "/scratch/andyye2/ABUPT/outputs/multi_fidelity_B2/a",
                "--source-output-path",
                "/scratch/andyye2/ABUPT/outputs",
                "--output",
                str(tmp_path / "commands.txt"),
                "--allow-missing-stats",
                "--arms",
                "S-matched",
                "S-matched",
            ]
        )


def test_generator_emits_only_the_selected_matched_pair(tmp_path: Path) -> None:
    """The requested two-arm command file contains one scratch and one transfer run."""
    output = tmp_path / "commands.txt"
    generate_training_commands.main(
        [
            "--repo-root",
            str(REMOTE_ROOT),
            "--protocol",
            str(REPO_ROOT / "research/multi_fidelity/experiment_protocol.yaml"),
            "--manifest-root",
            str(tmp_path / "manifests"),
            "--stats-root",
            str(tmp_path / "statistics"),
            "--dataset-root",
            str(DATASET_ROOT),
            "--output-path",
            "/scratch/andyye2/ABUPT/outputs/multi_fidelity_B2/a",
            "--source-output-path",
            "/scratch/andyye2/ABUPT/outputs",
            "--output",
            str(output),
            "--allow-missing-stats",
            "--arms",
            "S-matched",
            "P-FT-matched",
        ]
    )
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    tokens = [shlex.split(line) for line in lines]
    assert [line[line.index("--strategy") + 1] for line in tokens] == ["scratch", "finetune"]
    assert all(line[line.index("--supernode-radius") + 1] == "0.1" for line in tokens)
    assert all(line[line.index("--position-scale") + 1] == "1000" for line in tokens)
