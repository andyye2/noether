# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Binding to the frozen preregistration.

The YAML file is the integrity anchor of the study: its raw-file SHA256 is
embedded in every manifest, statistics artifact, training sidecar, and
evaluation audit.  Only the fields parsed here are runtime contracts; the rest
of the document is a scientific record.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import yaml

from .integrity import is_sha256

FROZEN_PROTOCOL_STATUS = "frozen_before_first_target_job"
DRAFT_PROTOCOL_STATUS = "preregistration_draft"

#: SHA256 of every audited ShapeNet-Car source checkpoint, keyed by
#: ``(checkpoint_tag, model_info)``.  The confirmatory source is the raw
#: epoch-500 ``latest`` checkpoint; the other two are sensitivity-only and are
#: deliberately not reachable from the strict runner.
KNOWN_SOURCE_SHA256 = {
    ("latest", None): "261a46b7464d50c26301db60758feaabc27ce3d4f30b511e826201d32154fe38",
    ("best_model.loss.test.total", None): "221ce5a1a6803876b5d0fc6157a8f06aa5299abe104c4ee9fcbef3f0aaffc5f7",
    ("latest", "ema=0.9999"): "f1200056bee2a2be2cac0f390bf9aaa049d1b1f29b680bab0721969ad8ba4aeb",
}

#: Number of paired replicates the preregistration fixes.
REPLICATE_COUNT = 8


@dataclass(frozen=True)
class Replicate:
    """One preregistered paired replicate.

    Attributes:
        replicate: Zero-based replicate label.
        subset_seed: PCG64 seed that generates the acquisition ladder.
        model_seed: Seed shared by the paired scratch and transfer runs.
    """

    replicate: int
    subset_seed: int
    model_seed: int


@dataclass(frozen=True)
class ProtocolBinding:
    """Immutable identity of the preregistration used by one run.

    Attributes:
        path: Resolved path of the YAML file that was read.
        sha256: Raw-file SHA256 that downstream artifacts must repeat.
        study_id: Study identifier shared by all artifacts.
        status: Freeze status; production work requires the frozen value.
        replicates: The eight preregistered paired replicates, in order.
        source_primary_checkpoint_tag: Confirmatory source checkpoint tag.
        source_primary_sha256: Confirmatory source checkpoint SHA256.
    """

    path: Path
    sha256: str
    study_id: str
    status: str
    replicates: tuple[Replicate, ...]
    source_primary_checkpoint_tag: str
    source_primary_sha256: str

    def replicate(self, index: int) -> Replicate:
        """Return one preregistered replicate.

        Args:
            index: Zero-based replicate label.

        Returns:
            The matching :class:`Replicate`.

        Raises:
            IndexError: If the label is outside the preregistered range.
        """
        return self.replicates[index]


def _parse_replicates(protocol: dict[str, Any]) -> tuple[Replicate, ...]:
    """Validate and return the preregistered seed table."""
    entries = protocol.get("data", {}).get("paired_replicates")
    if not isinstance(entries, list) or len(entries) != REPLICATE_COUNT:
        raise ValueError(f"protocol must define exactly {REPLICATE_COUNT} paired replicates")
    replicates: list[Replicate] = []
    for expected, entry in enumerate(entries):
        if not isinstance(entry, dict) or entry.get("replicate") != expected:
            raise ValueError("protocol replicate IDs must be consecutive and ordered from zero")
        subset_seed = entry.get("subset_seed")
        model_seed = entry.get("model_seed")
        if not isinstance(subset_seed, int) or not isinstance(model_seed, int):
            raise ValueError(f"protocol replicate {expected} has invalid subset/model seeds")
        replicates.append(Replicate(replicate=expected, subset_seed=subset_seed, model_seed=model_seed))
    return tuple(replicates)


def load_protocol_binding(protocol_path: Path) -> ProtocolBinding:
    """Load the preregistration and return its immutable identity and seed table.

    Args:
        protocol_path: Path of ``experiment_protocol.yaml``.

    Returns:
        The parsed :class:`ProtocolBinding`.

    Raises:
        TypeError: If the document root is not a mapping.
        ValueError: If a runtime-contract field is missing, malformed, or
            disagrees with :data:`KNOWN_SOURCE_SHA256`.
    """
    raw = protocol_path.read_bytes()
    protocol = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(protocol, dict):
        raise TypeError("protocol root must be a mapping")

    study_id = protocol.get("study_id")
    if not isinstance(study_id, str) or not study_id:
        raise ValueError("protocol must define a non-empty study_id")
    status = protocol.get("status")
    if status not in {DRAFT_PROTOCOL_STATUS, FROZEN_PROTOCOL_STATUS}:
        raise ValueError(
            f"protocol status must be {DRAFT_PROTOCOL_STATUS!r} or {FROZEN_PROTOCOL_STATUS!r}, got {status!r}"
        )

    source = protocol.get("source")
    if not isinstance(source, dict):
        raise ValueError("protocol must define a source mapping")
    primary_checkpoint_tag = source.get("primary_checkpoint_tag")
    primary_sha256 = source.get("primary_sha256")
    if not isinstance(primary_checkpoint_tag, str) or not primary_checkpoint_tag:
        raise ValueError("protocol source.primary_checkpoint_tag must be a non-empty string")
    if not is_sha256(primary_sha256):
        raise ValueError("protocol source.primary_sha256 must be a lowercase SHA256")
    known_primary_sha256 = KNOWN_SOURCE_SHA256.get((primary_checkpoint_tag, None))
    if known_primary_sha256 is None or primary_sha256 != known_primary_sha256:
        raise ValueError(
            "protocol primary source checkpoint disagrees with the audited constant: "
            f"tag={primary_checkpoint_tag!r}, protocol={primary_sha256}, known={known_primary_sha256}"
        )

    return ProtocolBinding(
        path=protocol_path.resolve(),
        sha256=hashlib.sha256(raw).hexdigest(),
        study_id=study_id,
        status=status,
        replicates=_parse_replicates(protocol),
        source_primary_checkpoint_tag=primary_checkpoint_tag,
        source_primary_sha256=primary_sha256,
    )


def require_frozen_protocol_for_execution(binding: ProtocolBinding) -> None:
    """Reject production target-data work until the preregistration is frozen.

    Args:
        binding: Protocol binding under which a run was requested.

    Raises:
        ValueError: If the protocol is still a draft.
    """
    if binding.status != FROZEN_PROTOCOL_STATUS:
        raise ValueError(
            f"production execution is blocked while protocol status is {binding.status!r}; "
            f"set it to {FROZEN_PROTOCOL_STATUS!r} only after explicit approval"
        )
