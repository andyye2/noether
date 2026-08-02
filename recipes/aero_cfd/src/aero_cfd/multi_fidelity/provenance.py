# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""The training provenance sidecar that links training to evaluation.

A run publishes exactly one sidecar after it succeeds.  The evaluator refuses
to read a checkpoint that is not described by a matching sidecar, so the
sidecar is the only supported way to carry a training identity forward.

Schema 2 makes the geometry rendering mandatory.  Schema 1 allowed it to be
absent and silently assumed the frozen defaults, which meant a sidecar written
before the rendering options existed was indistinguishable from one written by
a run that had genuinely used them.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .experiment import METHOD_BY_STRATEGY, TrainingRequest
from .integrity import GitState, atomic_write_json, is_sha256, sha256_file, stable_sha256

PROVENANCE_FILENAME = "training_provenance.json"
PROVENANCE_SCHEMA_VERSION = 2
PROVENANCE_KIND = "drivaerml_transfer_training_provenance"

#: Top-level configuration fields the runtime assigns rather than the
#: experiment. ``master_port`` is drawn at random unless ``MASTER_PORT`` is
#: exported, so including it would make the identity hash differ between two
#: runs of the very same cell.
NON_IDENTITY_CONFIG_FIELDS = ("master_port",)


def resolved_config_sha256(resolved_config: dict[str, Any]) -> str:
    """Hash a resolved configuration by its experiment identity alone.

    Args:
        resolved_config: Python-mode dump of the resolved configuration.

    Returns:
        Lowercase SHA256 over the dump without runtime-assigned fields and
        without any dependence on set iteration order.
    """
    identity = {key: value for key, value in resolved_config.items() if key not in NON_IDENTITY_CONFIG_FIELDS}
    return stable_sha256(identity)


def write_training_provenance(
    request: TrainingRequest,
    audit: dict[str, Any],
    resolved_config: dict[str, Any],
    git_state: GitState,
) -> Path:
    """Hash every model checkpoint and atomically publish training provenance.

    Args:
        request: The executed experiment cell.
        audit: Audit record returned by the config builder.
        resolved_config: Python-mode dump of the resolved configuration. It is
            hashed order-independently, so two identical runs agree even though
            the schema stores some fields as sets.
        git_state: Implementation state of the training process.

    Returns:
        Path of the published sidecar.

    Raises:
        RuntimeError: If a successful run produced no model checkpoints.
    """
    run_stage_path = request.output_path / str(audit["run_id"]) / str(audit["stage_name"])
    checkpoint_paths = sorted((run_stage_path / "checkpoints").glob("*_model.th"))
    if not checkpoint_paths:
        raise RuntimeError(f"successful training produced no model checkpoints in {run_stage_path / 'checkpoints'}")

    payload = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "kind": PROVENANCE_KIND,
        "study_id": audit["study_id"],
        "protocol_path": audit["protocol_path"],
        "protocol_sha256": audit["protocol_sha256"],
        "implementation_git_commit": git_state.commit,
        "implementation_git_dirty": git_state.dirty,
        "implementation_git_status_sha256": git_state.status_sha256,
        "manifest_path": audit["manifest"]["path"],
        "manifest_raw_sha256": audit["manifest"]["raw_file_sha256"],
        "manifest_payload_sha256": audit["manifest"]["payload_sha256"],
        "target_statistics_path": audit["target_statistics"]["path"],
        "target_statistics_sha256": audit["target_statistics"]["sha256"],
        "source_checkpoint": audit["source_checkpoint"],
        "source_checkpoint_sha256": audit["source_checkpoint_sha256"],
        "task": audit["task"],
        "strategy": audit["strategy"],
        "method": audit["method"],
        "replicate": audit["replicate"],
        "train_sample_size": audit["train_sample_size"],
        "coordinate_frame": audit["coordinate_frame"],
        "position_scale": audit["position_scale"],
        "supernode_radius": audit["supernode_radius"],
        "supernode_radius_position_fraction": audit["supernode_radius_position_fraction"],
        "supernode_radius_raw_units": audit["supernode_radius_raw_units"],
        "budget": request.budget,
        "expected_updates": audit["budget"]["expected_updates"],
        "subset_seed": audit["manifest"]["seed"],
        "model_seed": audit["model_seed"],
        "training_pipeline_seed": audit["training_pipeline_seed"],
        "data_loader_seed": audit["data_loader_seed"],
        "validation_seed": audit["validation_seed"],
        "evaluation_seed": audit["evaluation_seed"],
        "resolved_config_sha256": resolved_config_sha256(resolved_config),
        "resolved_config_sha256_excludes": list(NON_IDENTITY_CONFIG_FIELDS),
        "run_id": audit["run_id"],
        "stage_name": audit["stage_name"],
        "model_checkpoints_sha256": {path.name: sha256_file(path) for path in checkpoint_paths},
    }
    sidecar = run_stage_path / PROVENANCE_FILENAME
    atomic_write_json(sidecar, payload)
    return sidecar


@dataclass(frozen=True)
class TrainingProvenance:
    """A validated training sidecar.

    Attributes:
        path: Sidecar path.
        sha256: Raw-file SHA256 of the sidecar.
        payload: Parsed sidecar mapping.
    """

    path: Path
    sha256: str
    payload: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        """Return one recorded field.

        Args:
            key: Sidecar field name.

        Returns:
            The recorded value.
        """
        return self.payload[key]


def load_training_provenance(sidecar_path: Path) -> TrainingProvenance:
    """Read one sidecar and validate the fields every consumer relies on.

    Args:
        sidecar_path: Path of ``training_provenance.json``.

    Returns:
        The validated :class:`TrainingProvenance`.

    Raises:
        TypeError: If the sidecar root is not a mapping.
        ValueError: If the schema, strategy/method pair, source binding,
            geometry rendering, or checkpoint hash table is invalid.
    """
    raw = sidecar_path.read_bytes()
    payload: Any = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("training provenance sidecar root must be a mapping")
    if payload.get("schema_version") != PROVENANCE_SCHEMA_VERSION or payload.get("kind") != PROVENANCE_KIND:
        raise ValueError(
            "unsupported training provenance sidecar schema: "
            f"schema_version={payload.get('schema_version')!r}, kind={payload.get('kind')!r}; "
            f"expected {PROVENANCE_SCHEMA_VERSION} and {PROVENANCE_KIND!r}"
        )

    strategy = str(payload.get("strategy"))
    if METHOD_BY_STRATEGY.get(strategy) != payload.get("method"):
        raise ValueError(
            f"training provenance strategy/method mismatch: strategy={strategy!r}, method={payload.get('method')!r}"
        )
    source_checkpoint = payload.get("source_checkpoint")
    source_checkpoint_sha256 = payload.get("source_checkpoint_sha256")
    if strategy == "scratch":
        if source_checkpoint is not None or source_checkpoint_sha256 is not None:
            raise ValueError("scratch training provenance must not record a source checkpoint")
    elif not isinstance(source_checkpoint, str) or not Path(source_checkpoint).is_absolute():
        raise ValueError(f"transfer training provenance must record an absolute source path: {source_checkpoint!r}")
    elif not is_sha256(source_checkpoint_sha256):
        raise ValueError("transfer training provenance must record the source checkpoint SHA256")

    for key in ("position_scale", "supernode_radius"):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"training provenance must record numeric {key}, got {value!r}")
    if not is_sha256(payload.get("resolved_config_sha256")):
        raise ValueError("training provenance has no valid resolved config SHA256")
    if not isinstance(payload.get("implementation_git_dirty"), bool):
        raise ValueError("training provenance has no implementation Git dirty state")

    checkpoint_hashes = payload.get("model_checkpoints_sha256")
    if not isinstance(checkpoint_hashes, dict) or not checkpoint_hashes:
        raise ValueError("training provenance has no model checkpoint hashes")
    for basename, digest in checkpoint_hashes.items():
        if not isinstance(basename, str) or not is_sha256(digest):
            raise ValueError("training provenance checkpoint hash mapping is malformed")

    return TrainingProvenance(
        path=sidecar_path.resolve(),
        sha256=hashlib.sha256(raw).hexdigest(),
        payload=payload,
    )
