# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Tests for the binding to the frozen preregistration."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aero_cfd.multi_fidelity.protocol import (
    FROZEN_PROTOCOL_STATUS,
    KNOWN_SOURCE_SHA256,
    ProtocolBinding,
    load_protocol_binding,
    require_frozen_protocol_for_execution,
)


def test_repository_protocol_is_frozen_and_pins_the_audited_source(protocol: ProtocolBinding) -> None:
    """The shipped protocol is executable and names the audited checkpoint."""
    assert protocol.status == FROZEN_PROTOCOL_STATUS
    assert protocol.source_primary_checkpoint_tag == "latest"
    assert protocol.source_primary_sha256 == KNOWN_SOURCE_SHA256[("latest", None)]
    require_frozen_protocol_for_execution(protocol)


def test_replicate_table_is_complete_and_ordered(protocol: ProtocolBinding) -> None:
    """Replicate labels index directly into the preregistered seed table."""
    assert len(protocol.replicates) == 8
    assert protocol.replicate(0).subset_seed == 1103
    assert protocol.replicate(0).model_seed == 7103
    assert [entry.replicate for entry in protocol.replicates] == list(range(8))


def test_a_draft_protocol_blocks_production_execution(protocol_path: Path, tmp_path: Path) -> None:
    """Target data may not be touched before the explicit freeze."""
    payload = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    payload["status"] = "preregistration_draft"
    draft = tmp_path / "draft_protocol.yaml"
    draft.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="production execution is blocked"):
        require_frozen_protocol_for_execution(load_protocol_binding(draft))


def test_a_substituted_source_checkpoint_is_rejected(protocol_path: Path, tmp_path: Path) -> None:
    """The protocol cannot name a source the audited constants do not know."""
    payload = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    payload["source"]["primary_sha256"] = "0" * 64
    mismatched = tmp_path / "mismatched_protocol.yaml"
    mismatched.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="disagrees with the audited constant"):
        load_protocol_binding(mismatched)


def test_a_truncated_replicate_table_is_rejected(protocol_path: Path, tmp_path: Path) -> None:
    """Dropping replicates would silently change the paired design."""
    payload = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    payload["data"]["paired_replicates"] = payload["data"]["paired_replicates"][:3]
    truncated = tmp_path / "truncated_protocol.yaml"
    truncated.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly 8 paired replicates"):
        load_protocol_binding(truncated)
