# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Generate the train-only statistics command for the reported cell.

All reported arms share one data cell, so they share exactly one normalizer
statistics artifact.  Fitting it once and hashing it into every run is what
makes the arms comparable.

Example:

    .. code-block:: bash

        uv run python -m research.multi_fidelity.tools.generate_statistics_commands \\
            --repo-root /scratch/andyye2/ABUPT/multi_fidelity \\
            --dataset-root /scratch/andyye2/data/drivaerml_subsampled_10x \\
            --manifest-root <artifacts>/manifests \\
            --stats-root    <artifacts>/statistics \\
            --output        <artifacts>/commands/statistics.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aero_cfd.multi_fidelity.protocol import load_protocol_binding

from .commands import PROTOCOL_RELATIVE_PATH, REPO_ROOT, STATISTICS_MODULE, uv_command, write_command_file
from .paper_arms import PaperCell

FIELDS: dict[str, list[str]] = {
    "common": ["surface_pressure", "volume_velocity"],
    "full": [
        "surface_pressure",
        "surface_friction",
        "volume_pressure",
        "volume_velocity",
        "volume_vorticity",
    ],
}


def command_for_cell(
    cell: PaperCell,
    *,
    repo_root: Path,
    protocol_path: Path,
    manifest_root: Path,
    stats_root: Path,
    dataset_root: Path,
) -> str:
    """Render the statistics command line of one data cell.

    Args:
        cell: The shared paired data cell.
        repo_root: Repository root on the executing host.
        protocol_path: Path of the frozen preregistration.
        manifest_root: Directory holding the materialized manifests.
        stats_root: Directory the artifacts are written to.
        dataset_root: Root of the DrivAerML dataset.

    Returns:
        One shell-quoted command line.
    """
    seed_root = stats_root / f"seed{cell.subset_seed}"
    stem = f"n{cell.sample_size}_{cell.task}_{cell.coordinate_frame}"
    return uv_command(
        repo_root,
        ["-m", STATISTICS_MODULE],
        [
            "--root",
            str(dataset_root),
            "--manifest",
            str(manifest_root / f"drivaerml_nested_seed{cell.subset_seed}.json"),
            "--protocol",
            str(protocol_path),
            "--n",
            str(cell.sample_size),
            "--fields",
            *FIELDS[cell.task],
            "--coordinate-frame",
            cell.coordinate_frame,
            "--output",
            str(seed_root / f"{stem}.json"),
            "--flat-stats-output",
            str(seed_root / f"{stem}.flat.yaml"),
        ],
    )


def main(argv: list[str] | None = None) -> None:
    """Write the auditable statistics command list.

    Args:
        argv: Argument vector; ``None`` reads ``sys.argv``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--stats-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    write_command_file(
        args.output,
        [
            command_for_cell(
                PaperCell.from_protocol(load_protocol_binding(REPO_ROOT / PROTOCOL_RELATIVE_PATH)),
                repo_root=repo_root,
                protocol_path=args.protocol or repo_root / PROTOCOL_RELATIVE_PATH,
                manifest_root=args.manifest_root,
                stats_root=args.stats_root,
                dataset_root=args.dataset_root,
            )
        ],
    )


if __name__ == "__main__":
    main()
