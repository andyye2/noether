# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Materialize the preregistered, hashed nested DrivAerML train manifests.

Production manifests are written outside the Git worktree and bind themselves
to the protocol and to the exact implementation commit, so a run can prove
which acquisition ladder and which code produced it.

Example:

    .. code-block:: bash

        uv run python -m research.multi_fidelity.tools.materialize_manifests \\
            --output-dir /scratch/andyye2/ABUPT/multi_fidelity_artifacts/<commit>/manifests
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from aero_cfd.multi_fidelity.integrity import atomic_write_json, read_git_state
from aero_cfd.multi_fidelity.manifest import DEFAULT_SEEDS, DEFAULT_SIZES, build_study_manifest
from aero_cfd.multi_fidelity.protocol import FROZEN_PROTOCOL_STATUS, load_protocol_binding

from .commands import PROTOCOL_RELATIVE_PATH, REPO_ROOT

DEFAULT_PROTOCOL = REPO_ROOT / PROTOCOL_RELATIVE_PATH


def _validate_protocol_grid(protocol_path: Path, seeds: tuple[int, ...], sizes: tuple[int, ...]) -> None:
    """Reject any request that departs from the frozen acquisition grid.

    Args:
        protocol_path: Path of the frozen preregistration.
        seeds: Requested subset seeds.
        sizes: Requested nested prefix sizes.

    Raises:
        ValueError: If the protocol grid or the request differs from the
            preregistered seeds and sizes.
    """
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    protocol_sizes = protocol.get("data", {}).get("train_sample_sizes")
    if protocol_sizes != list(DEFAULT_SIZES):
        raise ValueError(f"protocol train_sample_sizes must be {list(DEFAULT_SIZES)}, got {protocol_sizes}")
    protocol_seeds = tuple(entry["subset_seed"] for entry in protocol["data"]["paired_replicates"])
    if protocol_seeds != DEFAULT_SEEDS:
        raise ValueError(f"protocol subset seeds must be {list(DEFAULT_SEEDS)}, got {list(protocol_seeds)}")
    if seeds != protocol_seeds:
        raise ValueError("--seeds must exactly match the frozen protocol order")
    if sizes != tuple(protocol_sizes):
        raise ValueError("--sizes must exactly match the frozen protocol order")


def main(argv: list[str] | None = None) -> None:
    """Write one manifest per preregistered replicate and verify each file.

    Args:
        argv: Argument vector; ``None`` reads ``sys.argv``.

    Raises:
        ValueError: If the protocol is not frozen, the implementation worktree
            is dirty, or the destination is inside the worktree.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    args = parser.parse_args(argv)

    seeds = tuple(args.seeds)
    sizes = tuple(args.sizes)
    protocol = load_protocol_binding(args.protocol)
    if protocol.status != FROZEN_PROTOCOL_STATUS:
        raise ValueError(f"production manifests require protocol status {FROZEN_PROTOCOL_STATUS!r}")
    _validate_protocol_grid(args.protocol, seeds, sizes)

    git_state = read_git_state(REPO_ROOT)
    if git_state.dirty:
        raise ValueError("production manifests require a clean implementation Git worktree")
    output_dir = args.output_dir.resolve()
    if output_dir == REPO_ROOT or REPO_ROOT in output_dir.parents:
        raise ValueError("production manifests must be written outside the implementation Git worktree")

    for seed in seeds:
        manifest = build_study_manifest(
            seed=seed,
            sizes=sizes,
            study_id=protocol.study_id,
            protocol_sha256=protocol.sha256,
            git_state=git_state,
        )
        output = output_dir / f"drivaerml_nested_seed{seed}.json"
        atomic_write_json(output, manifest)
        print(f"{output} {manifest['manifest_sha256']}")


if __name__ == "__main__":
    main()
