# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Tests for statistics-command manifest path binding."""

from __future__ import annotations

import shlex
from pathlib import Path

from .generate_statistics_commands import generate_statistics_commands


def test_statistics_commands_use_external_manifest_root() -> None:
    """Every deduplicated command reads manifests from the explicit external root."""
    protocol = {
        "data": {
            "train_sample_sizes": [25, 50, 100, 200, 400],
            "paired_replicates": [
                {"replicate": replicate, "subset_seed": 1103 + replicate, "model_seed": 7103 + replicate}
                for replicate in range(8)
            ],
        }
    }
    repo_root = Path("/scratch/andyye2/ABUPT/noether")
    manifest_root = Path("/scratch/andyye2/ABUPT/frozen_manifests")
    commands = generate_statistics_commands(
        protocol,
        ["P0"],
        repo_root=repo_root,
        manifest_root=manifest_root,
        protocol_path=repo_root / "research/multi_fidelity/experiment_protocol.yaml",
        dataset_root=Path("/scratch/andyye2/data/drivaerml_subsampled_10x"),
        stats_root=Path("/scratch/andyye2/ABUPT/stats"),
    )

    assert len(commands) == 2
    for command in commands:
        tokens = shlex.split(command)
        assert tokens[tokens.index("--project") + 1] == str(repo_root)
        assert tokens[tokens.index("--manifest") + 1] == str(manifest_root / "drivaerml_nested_seed1103.json")
        assert tokens[tokens.index("--protocol") + 1] == str(
            repo_root / "research/multi_fidelity/experiment_protocol.yaml"
        )
        assert str(repo_root / "research/multi_fidelity/evidence/manifests") not in command


def test_nested_sample_sizes_are_grouped_into_one_scan_per_seed() -> None:
    """P1 reads each replicate ladder once instead of once per nested N."""
    protocol = {
        "data": {
            "train_sample_sizes": [25, 50, 100, 200, 400],
            "paired_replicates": [
                {"replicate": replicate, "subset_seed": 1103 + replicate, "model_seed": 7103 + replicate}
                for replicate in range(8)
            ],
        }
    }
    commands = generate_statistics_commands(
        protocol,
        ["P1"],
        repo_root=Path("/repo"),
        manifest_root=Path("/manifests"),
        protocol_path=Path("/protocol.yaml"),
        dataset_root=Path("/dataset"),
        stats_root=Path("/stats"),
    )

    assert len(commands) == 3
    for command in commands:
        tokens = shlex.split(command)
        n_values = tokens[tokens.index("--n") + 1 : tokens.index("--fields")]
        outputs = tokens[tokens.index("--output") + 1 : tokens.index("--flat-stats-output")]
        assert n_values == ["25", "50", "100", "200", "400"]
        assert len(outputs) == 5
