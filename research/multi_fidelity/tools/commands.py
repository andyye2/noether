# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Shared rendering of the shell command lines consumed by the Slurm arrays.

One command per line, shell-quoted, so a batch script can select a line by
array index and a reviewer can read exactly what will run.
"""

from __future__ import annotations

from pathlib import Path
import shlex

REPO_ROOT = Path(__file__).resolve().parents[3]

PROTOCOL_RELATIVE_PATH = Path("research/multi_fidelity/experiment_protocol.yaml")
TRAINING_SCRIPT = Path("recipes/aero_cfd/scripts/run_drivaerml_transfer_strict.py")
EVALUATION_SCRIPT = Path("recipes/aero_cfd/scripts/eval_drivaerml_transfer_frozen.py")
STATISTICS_MODULE = "research.multi_fidelity.tools.compute_subset_statistics"


def python_path(repo_root: Path) -> str:
    """Return the import path a remote command needs.

    Args:
        repo_root: Repository root on the executing host.

    Returns:
        Colon-separated ``PYTHONPATH`` covering the repository, the framework,
        and the aero-CFD recipe package.
    """
    return f"{repo_root}:{repo_root / 'src'}:{repo_root / 'recipes/aero_cfd/src'}"


def uv_command(repo_root: Path, target: list[str], arguments: list[str]) -> str:
    """Render one shell-safe ``uv run`` command line.

    Args:
        repo_root: Repository root on the executing host.
        target: Python invocation, for example ``["-m", "some.module"]``.
        arguments: Arguments passed to the invoked program.

    Returns:
        A single shell-quoted command line.
    """
    return shlex.join(
        [
            "env",
            f"PYTHONPATH={python_path(repo_root)}",
            "uv",
            "run",
            "--project",
            str(repo_root),
            "--no-sync",
            "python",
            *target,
            *arguments,
        ]
    )


def write_command_file(output: Path, commands: list[str]) -> None:
    """Write one command per line and report the count.

    Args:
        output: Destination file.
        commands: Rendered command lines.

    Raises:
        ValueError: If no command was generated, which would otherwise submit
            an empty Slurm array.
    """
    if not commands:
        raise ValueError("no commands were generated")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(f"commands={len(commands)} output={output}")
