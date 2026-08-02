# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Binding to one train-subset-only normalizer-statistics artifact."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .protocol import ProtocolBinding


@dataclass(frozen=True)
class StatisticsBinding:
    """Identity of the statistics artifact that normalizes one run.

    Attributes:
        path: Resolved artifact path.
        sha256: Raw-file SHA256 recorded in the training sidecar.
        protocol_sha256: Protocol SHA256 the artifact repeats.
        study_id: Study identifier the artifact repeats.
    """

    path: Path
    sha256: str
    protocol_sha256: str
    study_id: str


def load_statistics_binding(statistics_path: Path, *, protocol: ProtocolBinding) -> StatisticsBinding:
    """Validate the protocol/study binding of a statistics artifact and hash it.

    The artifact's leakage guard and field contents are validated later by the
    preset that turns it into concrete normalizers; this function only fixes
    its identity.

    Args:
        statistics_path: Path of the statistics artifact.
        protocol: Protocol binding the artifact must agree with.

    Returns:
        The validated :class:`StatisticsBinding`.

    Raises:
        TypeError: If the artifact root is not a mapping.
        ValueError: If the artifact belongs to another protocol or study.
    """
    raw = statistics_path.read_bytes()
    artifact: Any = json.loads(raw.decode("utf-8"))
    if not isinstance(artifact, dict):
        raise TypeError("statistics artifact root must be a mapping")
    protocol_sha256 = artifact.get("protocol_sha256")
    study_id = artifact.get("study_id")
    if protocol_sha256 != protocol.sha256:
        raise ValueError(
            f"statistics/protocol SHA256 mismatch: statistics={protocol_sha256}, protocol={protocol.sha256}"
        )
    if study_id != protocol.study_id:
        raise ValueError(f"statistics/protocol study_id mismatch: statistics={study_id}, protocol={protocol.study_id}")
    return StatisticsBinding(
        path=statistics_path.resolve(),
        sha256=hashlib.sha256(raw).hexdigest(),
        protocol_sha256=protocol_sha256,
        study_id=study_id,
    )
