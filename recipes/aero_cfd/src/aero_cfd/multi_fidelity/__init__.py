# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Library for the ShapeNet-Car to DrivAerML multi-fidelity transfer study.

The modules below hold every rule that a run must satisfy, so the entry points
under ``recipes/aero_cfd/scripts`` and the research tools under
``research/multi_fidelity/tools`` stay thin command-line wrappers and cannot
drift from each other.
"""

from .integrity import GitState, atomic_write_json, canonical_json_sha256, read_git_state, sha256_file
from .manifest import ManifestCell, build_study_manifest, load_manifest_cell, verify_manifest
from .protocol import ProtocolBinding, load_protocol_binding, require_frozen_protocol_for_execution

__all__ = [
    "GitState",
    "ManifestCell",
    "ProtocolBinding",
    "atomic_write_json",
    "build_study_manifest",
    "canonical_json_sha256",
    "load_manifest_cell",
    "load_protocol_binding",
    "read_git_state",
    "require_frozen_protocol_for_execution",
    "sha256_file",
    "verify_manifest",
]
