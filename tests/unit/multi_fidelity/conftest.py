# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Shared fixtures for the multi-fidelity transfer study.

Manifests are built by the production code path instead of being committed as
fixtures, so a test can never pass against a stale copy of an artifact whose
generator has changed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from aero_cfd.multi_fidelity.integrity import GitState, atomic_write_json, sha256_file
from aero_cfd.multi_fidelity.manifest import DEFAULT_SIZES, build_study_manifest
from aero_cfd.multi_fidelity.protocol import ProtocolBinding, load_protocol_binding

REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = REPO_ROOT / "research/multi_fidelity/experiment_protocol.yaml"

#: Implementation state the fixture manifests claim.  Tests that exercise the
#: implementation binding compare against this value explicitly.
FIXTURE_GIT_STATE = GitState(commit="a" * 40, dirty=False, status_sha256="b" * 64)


@pytest.fixture(scope="session")
def protocol_path() -> Path:
    """Return the frozen preregistration shipped with the repository."""
    return PROTOCOL_PATH


@pytest.fixture(scope="session")
def protocol(protocol_path: Path) -> ProtocolBinding:
    """Return the parsed binding of the frozen preregistration."""
    return load_protocol_binding(protocol_path)


@pytest.fixture
def manifest_path(tmp_path: Path, protocol: ProtocolBinding) -> Path:
    """Materialize the replicate-0 manifest exactly as production does."""
    manifest = build_study_manifest(
        seed=protocol.replicate(0).subset_seed,
        sizes=DEFAULT_SIZES,
        study_id=protocol.study_id,
        protocol_sha256=protocol.sha256,
        git_state=FIXTURE_GIT_STATE,
    )
    path = tmp_path / "manifests" / f"drivaerml_nested_seed{protocol.replicate(0).subset_seed}.json"
    atomic_write_json(path, manifest)
    return path


def write_statistics_artifact(
    path: Path,
    *,
    protocol: ProtocolBinding,
    manifest_raw_sha256: str,
    train_subset_size: int,
    overrides: dict[str, Any] | None = None,
) -> Path:
    """Write a minimal train-only statistics artifact for construction tests.

    Args:
        path: Destination file.
        protocol: Protocol binding the artifact must agree with.
        manifest_raw_sha256: Raw-file SHA256 of the bound manifest.
        train_subset_size: Subset size the artifact was fitted on.
        overrides: Normalizer statistics to replace or add.

    Returns:
        The written path.
    """
    statistics: dict[str, Any] = {
        "raw_pos_min": [-40.0],
        "raw_pos_max": [80.0],
        "surface_pressure_mean": [-200.0],
        "surface_pressure_std": [250.0],
        "volume_velocity_mean": [0.0, 0.0, 16.0],
        "volume_velocity_std": [8.0, 7.0, 16.0],
    }
    statistics.update(overrides or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "study_id": protocol.study_id,
                "protocol_sha256": protocol.sha256,
                "train_subset_size": train_subset_size,
                "coordinate_frame": "shapenet",
                "provenance": {"manifest_sha256": manifest_raw_sha256},
                "leakage_guard": {"target_splits_read": ["train"]},
                "normalizer_stats": statistics,
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def statistics_path(tmp_path: Path, protocol: ProtocolBinding, manifest_path: Path) -> Path:
    """Return a statistics artifact bound to the fixture manifest at N=25."""
    return write_statistics_artifact(
        tmp_path / "statistics.json",
        protocol=protocol,
        manifest_raw_sha256=sha256_file(manifest_path),
        train_subset_size=25,
    )
