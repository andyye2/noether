# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Generate the strict-training command list for the reported arms.

Example:

    .. code-block:: bash

        uv run python -m research.multi_fidelity.tools.generate_training_commands \\
            --repo-root /scratch/andyye2/ABUPT/multi_fidelity \\
            --dataset-root /scratch/andyye2/data/drivaerml_subsampled_10x \\
            --manifest-root  <artifacts>/manifests \\
            --stats-root     <artifacts>/statistics \\
            --output-path    <outputs>/n100-r0-paper \\
            --source-output-path /scratch/andyye2/ABUPT/outputs \\
            --output <artifacts>/commands/training.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aero_cfd.multi_fidelity.experiment import BUDGETS

from .commands import TRAINING_SCRIPT, uv_command, write_command_file
from .paper_arms import ARM_NAMES, ARMS, Arm, PaperCell, arm_by_name


def command_for_arm(
    arm: Arm,
    cell: PaperCell,
    *,
    repo_root: Path,
    protocol_path: Path,
    manifest_root: Path,
    stats_root: Path,
    dataset_root: Path,
    output_path: Path,
    source_output_path: Path,
) -> str:
    """Render the training command line of one arm.

    Args:
        arm: Reported arm.
        cell: The shared paired data cell.
        repo_root: Repository root on the executing host.
        protocol_path: Path of the frozen preregistration.
        manifest_root: Directory holding the materialized manifests.
        stats_root: Directory holding the train-subset statistics.
        dataset_root: Root of the DrivAerML dataset.
        output_path: Root the runs write into.
        source_output_path: Root of the ShapeNet-Car source outputs.

    Returns:
        One shell-quoted command line.
    """
    manifest = manifest_root / f"drivaerml_nested_seed{cell.subset_seed}.json"
    statistics = (
        stats_root / f"seed{cell.subset_seed}" / f"n{cell.sample_size}_{cell.task}_{cell.coordinate_frame}.json"
    )
    return uv_command(
        repo_root,
        [str(repo_root / TRAINING_SCRIPT)],
        [
            "--protocol",
            str(protocol_path),
            "--dataset-root",
            str(dataset_root),
            "--manifest",
            str(manifest),
            "--target-statistics",
            str(statistics),
            "--output-path",
            str(output_path),
            "--source-output-path",
            str(source_output_path),
            "--task",
            cell.task,
            "--strategy",
            arm.strategy,
            "--sample-size",
            str(cell.sample_size),
            "--budget",
            cell.budget,
            "--coordinate-frame",
            cell.coordinate_frame,
            "--position-scale",
            f"{arm.geometry.position_scale:g}",
            "--supernode-radius",
            f"{arm.geometry.supernode_radius:g}",
            "--replicate",
            str(cell.replicate),
            "--model-seed",
            str(cell.model_seed),
            "--eval-point-seed",
            "4242",
        ],
    )


def main(argv: list[str] | None = None) -> None:
    """Write the auditable training command list.

    Args:
        argv: Argument vector; ``None`` reads ``sys.argv``.

    Raises:
        FileNotFoundError: If a required statistics artifact is missing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--stats-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--source-output-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=ARM_NAMES, default=list(ARM_NAMES))
    parser.add_argument("--budget", choices=BUDGETS, default="compute_matched")
    parser.add_argument("--allow-missing-stats", action="store_true")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    protocol_path = args.protocol or repo_root / "research/multi_fidelity/experiment_protocol.yaml"
    cell = PaperCell(budget=args.budget)
    statistics = (
        args.stats_root / f"seed{cell.subset_seed}" / (f"n{cell.sample_size}_{cell.task}_{cell.coordinate_frame}.json")
    )
    if not args.allow_missing_stats and not statistics.is_file():
        raise FileNotFoundError(f"statistics artifact is missing: {statistics}")

    ordered = [arm_by_name(name) for name in args.arms] if args.arms != list(ARM_NAMES) else list(ARMS)
    write_command_file(
        args.output,
        [
            command_for_arm(
                arm,
                cell,
                repo_root=repo_root,
                protocol_path=protocol_path,
                manifest_root=args.manifest_root,
                stats_root=args.stats_root,
                dataset_root=args.dataset_root,
                output_path=args.output_path,
                source_output_path=args.source_output_path,
            )
            for arm in ordered
        ],
    )


if __name__ == "__main__":
    main()
