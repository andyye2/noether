# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Generate preregistered strict-training command lists by release phase."""

from __future__ import annotations

import argparse
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from aero_cfd.model.transfer_reset import DEFAULT_RESET_SCOPE

Phase = Literal["P0", "P1", "P2", "P2a", "P2b", "P2c", "P3", "R1"]
DEFAULT_MANIFEST_RELATIVE_ROOT = Path("research/multi_fidelity/evidence/manifests")

#: Runner defaults; a cell that matches one leaves the flag off the command line
#: so that every preregistered phase renders exactly the text it rendered before.
DEFAULT_POSITION_SCALE = 1000.0
DEFAULT_SUPERNODE_RADIUS = 9.0


@dataclass(frozen=True)
class Cell:
    """One train/validation experiment cell."""

    replicate: int
    subset_seed: int
    model_seed: int
    n: int
    task: str
    strategy: str
    budget: str
    frame: str = "shapenet"
    position_scale: float = DEFAULT_POSITION_SCALE
    supernode_radius: float = DEFAULT_SUPERNODE_RADIUS
    reset_scope: str = DEFAULT_RESET_SCOPE


def _protocol_replicates(protocol: dict) -> list[dict[str, int]]:
    """Read and validate the preregistered seed table."""
    entries = protocol["data"]["paired_replicates"]
    if len(entries) != 8:
        raise ValueError(f"protocol must define exactly eight paired replicates, got {len(entries)}")
    for expected, entry in enumerate(entries):
        if entry["replicate"] != expected:
            raise ValueError("replicate IDs must be consecutive and ordered from zero")
    return entries


def phase_cells(protocol: dict, phase: Phase) -> list[Cell]:
    """Expand one staged-release phase into exact cells."""
    reps = _protocol_replicates(protocol)
    sizes = protocol["data"].get("train_sample_sizes")
    expected_sizes = [25, 50, 100, 200, 400]
    if sizes != expected_sizes:
        raise ValueError(f"protocol train_sample_sizes must be {expected_sizes}, got {sizes}")
    cells: list[Cell] = []

    def add(rep_indices, ns, task, strategies, budget, frame="shapenet", **overrides):
        for rep_index in rep_indices:
            rep = reps[rep_index]
            for n in ns:
                for strategy in strategies:
                    cells.append(
                        Cell(
                            replicate=rep_index,
                            subset_seed=rep["subset_seed"],
                            model_seed=rep["model_seed"],
                            n=n,
                            task=task,
                            strategy=strategy,
                            budget=budget,
                            frame=frame,
                            **overrides,
                        )
                    )

    if phase == "P0":
        add([0], [25], "common", ["scratch", "finetune", "linear_probe", "gradual_unfreeze"], "smoke")
        add([0], [25], "full", ["scratch", "finetune"], "smoke")
    elif phase == "P1":
        add(range(3), sizes, "common", ["scratch", "finetune"], "compute_matched")
    elif phase == "P2":
        add(range(3, 8), sizes, "common", ["scratch", "finetune"], "compute_matched")
    elif phase == "P2a":
        add(range(3), sizes, "common", ["scratch", "finetune"], "fixed_epoch")
    elif phase == "P2b":
        add(range(3), [50, 100], "common", ["linear_probe", "gradual_unfreeze"], "compute_matched")
    elif phase == "P2c":
        add(range(3), [50, 100], "common", ["scratch", "finetune"], "compute_matched", "native")
    elif phase == "P3":
        add(range(5), [50, 100, 200, 400], "full", ["scratch", "finetune"], "compute_matched")
    elif phase == "R1":
        # Exploratory, not part of the preregistered ladder. One replicate at the
        # geometry rendering of the reported sr0.1 cells: a scratch anchor, the
        # default-scope transfer arm that anchor is paired with, and the two
        # volume-side scopes under test. Scratch takes no scope, so the four arms
        # differ only in how much of the volume path is inherited.
        add([0], [100], "common", ["scratch"], "compute_matched", supernode_radius=0.1)
        for scope in ("readout", "volume_decoder", "volume_path"):
            add([0], [100], "common", ["finetune"], "compute_matched", supernode_radius=0.1, reset_scope=scope)
    else:
        raise ValueError(f"unsupported phase {phase}")
    return cells


def command_for_cell(
    cell: Cell,
    *,
    repo_root: Path,
    manifest_root: Path,
    protocol_path: Path,
    dataset_root: Path,
    output_path: Path,
    stats_root: Path,
    source_output_path: Path,
) -> str:
    """Render one shell-safe command line."""
    manifest = manifest_root / f"drivaerml_nested_seed{cell.subset_seed}.json"
    statistics = stats_root / f"seed{cell.subset_seed}" / f"n{cell.n}_{cell.task}_{cell.frame}.json"
    runner = repo_root / "recipes/aero_cfd/scripts/run_drivaerml_transfer_strict.py"
    arguments = [
        "env",
        f"PYTHONPATH={repo_root}:{repo_root / 'src'}:{repo_root / 'recipes/aero_cfd/src'}",
        "uv",
        "run",
        "--project",
        str(repo_root),
        "--no-sync",
        "python",
        str(runner),
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
        cell.strategy,
        "--sample-size",
        str(cell.n),
        "--budget",
        cell.budget,
        "--coordinate-frame",
        cell.frame,
        "--replicate",
        str(cell.replicate),
        "--model-seed",
        str(cell.model_seed),
        "--eval-point-seed",
        "4242",
    ]
    if cell.position_scale != DEFAULT_POSITION_SCALE:
        arguments += ["--position-scale", f"{cell.position_scale:g}"]
    if cell.supernode_radius != DEFAULT_SUPERNODE_RADIUS:
        arguments += ["--supernode-radius", f"{cell.supernode_radius:g}"]
    if cell.reset_scope != DEFAULT_RESET_SCOPE:
        arguments += ["--reset-scope", cell.reset_scope]
    return shlex.join(arguments)


def main() -> None:
    """Write one auditable command list for a gated release phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--phase", choices=("P0", "P1", "P2", "P2a", "P2b", "P2c", "P3", "R1"), required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument(
        "--manifest-root",
        type=Path,
        help="Manifest directory; defaults to <repo-root>/research/multi_fidelity/evidence/manifests",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--stats-root", type=Path, required=True)
    parser.add_argument("--source-output-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-missing-stats", action="store_true")
    args = parser.parse_args()

    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    repo_root = args.repo_root.resolve()
    manifest_root = args.manifest_root or repo_root / DEFAULT_MANIFEST_RELATIVE_ROOT
    cells = phase_cells(protocol, args.phase)
    commands = [
        command_for_cell(
            cell,
            repo_root=repo_root,
            manifest_root=manifest_root,
            protocol_path=args.protocol.resolve(),
            dataset_root=args.dataset_root,
            output_path=args.output_path,
            stats_root=args.stats_root,
            source_output_path=args.source_output_path,
        )
        for cell in cells
    ]
    if not args.allow_missing_stats:
        missing = []
        for cell in cells:
            path = args.stats_root / f"seed{cell.subset_seed}" / f"n{cell.n}_{cell.task}_{cell.frame}.json"
            if not path.is_file():
                missing.append(path)
        if missing:
            preview = "\n".join(str(path) for path in missing[:10])
            raise FileNotFoundError(f"{len(missing)} statistics artifacts are missing; first paths:\n{preview}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(f"phase={args.phase} jobs={len(commands)} output={args.output}")


if __name__ == "__main__":
    main()
