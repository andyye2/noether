# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Generate deduplicated train-only statistics commands for release phases."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from collections.abc import Sequence

import yaml

if __package__:
    from .generate_training_commands import Cell, Phase, phase_cells
else:
    from generate_training_commands import Cell, Phase, phase_cells

DEFAULT_MANIFEST_RELATIVE_ROOT = Path("research/multi_fidelity/evidence/manifests")
FIELDS = {
    "common": ["surface_pressure", "volume_velocity"],
    "full": [
        "surface_pressure",
        "surface_friction",
        "volume_pressure",
        "volume_velocity",
        "volume_vorticity",
    ],
}


def command_for_statistics_cells(
    cells: Sequence[Cell],
    *,
    repo_root: Path,
    manifest_root: Path,
    protocol_path: Path,
    dataset_root: Path,
    stats_root: Path,
) -> str:
    """Render one single-pass command for nested cells sharing seed/task/frame."""
    if not cells:
        raise ValueError("cells must not be empty")
    ordered = sorted(cells, key=lambda cell: cell.n)
    first = ordered[0]
    identity = (first.subset_seed, first.task, first.frame)
    if any((cell.subset_seed, cell.task, cell.frame) != identity for cell in ordered):
        raise ValueError("grouped statistics cells must share subset seed, task, and frame")
    if len({cell.n for cell in ordered}) != len(ordered):
        raise ValueError("grouped statistics cells must have unique sample sizes")

    manifest = manifest_root / f"drivaerml_nested_seed{first.subset_seed}.json"
    outputs = [stats_root / f"seed{first.subset_seed}" / f"n{cell.n}_{cell.task}_{cell.frame}.json" for cell in ordered]
    flats = [
        stats_root / f"seed{first.subset_seed}" / f"n{cell.n}_{cell.task}_{cell.frame}.flat.yaml" for cell in ordered
    ]
    tool = repo_root / "research/multi_fidelity/tools/compute_subset_statistics.py"
    pythonpath = f"{repo_root}:{repo_root / 'src'}:{repo_root / 'recipes/aero_cfd/src'}"
    arguments = [
        "env",
        f"PYTHONPATH={pythonpath}",
        "uv",
        "run",
        "--project",
        str(repo_root),
        "--no-sync",
        "python",
        str(tool),
        "--root",
        str(dataset_root),
        "--manifest",
        str(manifest),
        "--protocol",
        str(protocol_path),
        "--n",
        *(str(cell.n) for cell in ordered),
        "--fields",
        *FIELDS[first.task],
        "--coordinate-frame",
        first.frame,
        "--output",
        *(str(path) for path in outputs),
        "--flat-stats-output",
        *(str(path) for path in flats),
    ]
    return shlex.join(arguments)


def command_for_statistics_cell(
    cell: Cell,
    *,
    repo_root: Path,
    manifest_root: Path,
    protocol_path: Path,
    dataset_root: Path,
    stats_root: Path,
) -> str:
    """Render a backward-compatible one-cell statistics command."""
    return command_for_statistics_cells(
        [cell],
        repo_root=repo_root,
        manifest_root=manifest_root,
        protocol_path=protocol_path,
        dataset_root=dataset_root,
        stats_root=stats_root,
    )


def generate_statistics_commands(
    protocol: dict,
    phases: Sequence[Phase],
    *,
    repo_root: Path,
    manifest_root: Path,
    protocol_path: Path,
    dataset_root: Path,
    stats_root: Path,
) -> list[str]:
    """Return single-pass commands grouped by subset seed, task, and frame."""
    grouped: dict[tuple[int, str, str], dict[int, Cell]] = {}
    for phase in phases:
        for cell in phase_cells(protocol, phase):
            key = (cell.subset_seed, cell.task, cell.frame)
            grouped.setdefault(key, {}).setdefault(cell.n, cell)
    return [
        command_for_statistics_cells(
            list(grouped[key].values()),
            repo_root=repo_root,
            manifest_root=manifest_root,
            protocol_path=protocol_path,
            dataset_root=dataset_root,
            stats_root=stats_root,
        )
        for key in sorted(grouped)
    ]


def main() -> None:
    """Write one command per unique subset/task/frame statistics artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=("P0", "P1", "P2", "P2a", "P2b", "P2c", "P3", "R1"),
        required=True,
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument(
        "--manifest-root",
        type=Path,
        help="Manifest directory; defaults to <repo-root>/research/multi_fidelity/evidence/manifests",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--stats-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    repo_root = args.repo_root.resolve()
    manifest_root = args.manifest_root or repo_root / DEFAULT_MANIFEST_RELATIVE_ROOT
    commands = generate_statistics_commands(
        protocol,
        args.phases,
        repo_root=repo_root,
        manifest_root=manifest_root,
        protocol_path=args.protocol.resolve(),
        dataset_root=args.dataset_root,
        stats_root=args.stats_root,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(f"phases={','.join(args.phases)} statistics_jobs={len(commands)} output={args.output}")


if __name__ == "__main__":
    main()
