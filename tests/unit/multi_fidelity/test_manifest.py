# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Tests for the deterministic nested acquisition manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from aero_cfd.multi_fidelity.integrity import GitState, canonical_json_bytes, sha256_file
from aero_cfd.multi_fidelity.manifest import (
    DEFAULT_SIZES,
    build_study_manifest,
    load_manifest_cell,
    verify_manifest,
)
from aero_cfd.multi_fidelity.protocol import ProtocolBinding

from .conftest import FIXTURE_GIT_STATE


def test_nested_subsets_are_deterministic_prefixes(manifest_path: Path) -> None:
    """Every smaller subset is a prefix of the next larger one."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    subsets = manifest["subsets"]
    for smaller, larger in zip(DEFAULT_SIZES, DEFAULT_SIZES[1:], strict=False):
        assert subsets[str(larger)]["run_ids"][:smaller] == subsets[str(smaller)]["run_ids"]
        assert subsets[str(larger)]["base_dataset_indices"][:smaller] == subsets[str(smaller)]["base_dataset_indices"]


def test_same_seed_reproduces_the_same_manifest(protocol: ProtocolBinding) -> None:
    """The ladder is a pure function of the seed, protocol, and implementation."""
    first = build_study_manifest(
        seed=1103,
        sizes=DEFAULT_SIZES,
        study_id=protocol.study_id,
        protocol_sha256=protocol.sha256,
        git_state=FIXTURE_GIT_STATE,
    )
    second = build_study_manifest(
        seed=1103,
        sizes=DEFAULT_SIZES,
        study_id=protocol.study_id,
        protocol_sha256=protocol.sha256,
        git_state=FIXTURE_GIT_STATE,
    )
    assert first == second


def test_a_tampered_subset_is_rejected(manifest_path: Path) -> None:
    """Editing a subset breaks both the content hash and the permutation."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["subsets"]["100"]["run_ids"][0] += 1
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_manifest(manifest)


def test_a_reseeded_manifest_cannot_keep_its_permutation(manifest_path: Path) -> None:
    """Rewriting the seed and rehashing still fails the recomputed permutation."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["seed"] = 2207
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    with pytest.raises(ValueError, match="do not match the recorded PCG64 seed"):
        verify_manifest(manifest)


def test_cell_binds_protocol_study_and_implementation(manifest_path: Path, protocol: ProtocolBinding) -> None:
    """A loaded cell carries every binding a run must record."""
    cell = load_manifest_cell(
        manifest_path,
        100,
        expected_protocol_sha256=protocol.sha256,
        expected_study_id=protocol.study_id,
    )
    assert len(cell.base_dataset_indices) == 100
    assert len(set(cell.run_ids)) == 100
    assert cell.seed == 1103
    assert cell.raw_file_sha256 == sha256_file(manifest_path)
    assert cell.implementation_commit == FIXTURE_GIT_STATE.commit
    cell.require_implementation(FIXTURE_GIT_STATE)


def test_cell_rejects_a_foreign_protocol(manifest_path: Path, protocol: ProtocolBinding) -> None:
    """A manifest from another preregistration cannot configure a run."""
    with pytest.raises(ValueError, match="manifest/protocol SHA256 mismatch"):
        load_manifest_cell(manifest_path, 100, expected_protocol_sha256="0" * 64)
    with pytest.raises(ValueError, match="manifest/protocol study_id mismatch"):
        load_manifest_cell(manifest_path, 100, expected_study_id="other-study")


def test_cell_rejects_a_different_implementation(manifest_path: Path) -> None:
    """Running code that did not produce the manifest is refused."""
    cell = load_manifest_cell(manifest_path, 100)
    with pytest.raises(ValueError, match="manifest/implementation commit mismatch"):
        cell.require_implementation(GitState(commit="c" * 40, dirty=False, status_sha256="d" * 64))
    with pytest.raises(ValueError, match="manifest/implementation dirty-state mismatch"):
        cell.require_implementation(GitState(commit="a" * 40, dirty=True, status_sha256="d" * 64))


def test_unknown_sample_size_is_rejected(manifest_path: Path) -> None:
    """Only the preregistered ladder sizes can be requested."""
    with pytest.raises(ValueError, match="no subset cell for N=37"):
        load_manifest_cell(manifest_path, 37)
