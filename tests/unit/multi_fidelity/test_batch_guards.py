# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Tests for the B2 submission boundary and command selection."""

from __future__ import annotations

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
RETIRED_BATCH = SLURM_ROOT / "drivaerml_paper_array.sbatch"
REMOTE_ROOT = Path("/scratch/andyye2/ABUPT/multi_fidelity_B2")
DATASET_ROOT = Path("/scratch/andyye2/data/drivaerml_subsampled_10x")

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash is unavailable")


@pytest.mark.parametrize("batch", [STATISTICS_BATCH, TRAINING_BATCH, RETIRED_BATCH])
def test_batch_scripts_are_valid_bash(batch: Path) -> None:
    """A syntax error must be caught before a Slurm submission."""
    assert subprocess.run(["bash", "-n", str(batch)], check=False).returncode == 0


def test_batches_are_bound_to_the_parallel_b2_layout() -> None:
    """Code and logs must never fall back to the prior experiment tree."""
    for batch in (STATISTICS_BATCH, TRAINING_BATCH):
        text = batch.read_text(encoding="utf-8")
        assert "PROJECT_ROOT=/scratch/andyye2/ABUPT/multi_fidelity_B2" in text
        assert "#SBATCH --output=/scratch/andyye2/ABUPT/slurm/multi_fidelity_B2/" in text
        assert "#SBATCH --error=/scratch/andyye2/ABUPT/slurm/multi_fidelity_B2/" in text


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
