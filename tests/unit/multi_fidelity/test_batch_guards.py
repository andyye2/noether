# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Tests for the request guards of the Slurm batch script.

The batch script is the last thing between a generated command and a GPU, and
its guards are the only protection for results that already exist. They are
written to reject a bad request before touching the cluster environment, which
also makes them testable off the cluster.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
BATCH_SCRIPT = REPO_ROOT / "research/multi_fidelity/slurm/drivaerml_paper_array.sbatch"
COMMIT = "a" * 40
OUTPUT_ROOT = f"/scratch/andyye2/ABUPT/outputs/multi_fidelity_paper/{COMMIT}/n100-r0-paper"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash is unavailable")


def _run(*arguments: str, task_id: str = "1") -> subprocess.CompletedProcess[str]:
    """Run the batch script with one array index and capture its report."""
    return subprocess.run(
        ["bash", str(BATCH_SCRIPT), *arguments],
        env={"PATH": "/usr/bin:/bin", "SLURM_ARRAY_TASK_ID": task_id},
        capture_output=True,
        text=True,
        check=False,
    )


def _command_file(tmp_path: Path, *commands: str) -> str:
    """Write a command file and return its path."""
    path = tmp_path / "commands.txt"
    path.write_text("\n".join(commands) + "\n", encoding="utf-8")
    return str(path)


def test_script_is_valid_bash() -> None:
    """A syntax error would only surface hours later inside the queue."""
    assert subprocess.run(["bash", "-n", str(BATCH_SCRIPT)], check=False).returncode == 0


def test_a_partial_commit_is_rejected(tmp_path: Path) -> None:
    """An abbreviated commit cannot identify an implementation state."""
    result = _run("deadbeef", _command_file(tmp_path, f"echo --output-path {OUTPUT_ROOT}"))
    assert result.returncode == 2
    assert "expected a full lowercase commit hash" in result.stderr


def test_an_out_of_range_array_index_is_rejected(tmp_path: Path) -> None:
    """A wider array than the command file would run nothing, silently."""
    result = _run(COMMIT, _command_file(tmp_path, f"echo --output-path {OUTPUT_ROOT}"), task_id="7")
    assert result.returncode == 2
    assert "array index 7 is outside 1..1" in result.stderr


def test_a_command_for_another_commit_is_rejected(tmp_path: Path) -> None:
    """Replaying an old command file must not land in this run's namespace."""
    stale = "/scratch/andyye2/ABUPT/outputs/multi_fidelity_paper/0000/n100-r0-paper"
    result = _run(COMMIT, _command_file(tmp_path, f"echo --output-path {stale}"))
    assert result.returncode == 2
    assert "does not write into the namespace" in result.stderr


@pytest.mark.parametrize(
    "protected",
    [
        "/scratch/andyye2/ABUPT/outputs/multi_fidelity/",
        "/scratch/andyye2/ABUPT/outputs/multi_fidelity_exploratory/",
        "/scratch/andyye2/ABUPT/multi_fidelity_artifacts/cf974078ea633f1fb6ff2178de7fa56ac60c89d5/",
        "/scratch/andyye2/ABUPT/multi_fidelity_stats/",
    ],
)
def test_completed_result_namespaces_are_untouchable(tmp_path: Path, protected: str) -> None:
    """No command may even name a namespace that already holds results."""
    result = _run(COMMIT, _command_file(tmp_path, f"echo {protected}x/{COMMIT}/y"))
    assert result.returncode == 2
    assert "completed-result namespace" in result.stderr


def test_the_test_split_is_never_released_by_this_batch(tmp_path: Path) -> None:
    """Held-out data needs a separate, explicit decision.

    The evaluator and the command generator both refuse it too; this is the
    third gate, on the machine that would actually read the files.
    """
    command = f"echo --output-path {OUTPUT_ROOT} --evaluation-split test"
    result = _run(COMMIT, _command_file(tmp_path, command))
    assert result.returncode == 2
    assert "never releases the held-out test split" in result.stderr


def test_a_valid_request_reaches_the_environment_check(tmp_path: Path) -> None:
    """A well-formed request is only stopped by the cluster being absent."""
    result = _run(COMMIT, _command_file(tmp_path, f"echo --output-path {OUTPUT_ROOT}"))
    assert result.returncode == 2
    assert "required file missing" in result.stderr
