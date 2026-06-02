#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class CheckpointMetadata:
    uri: str
    local_path: Path
    sha256: str | None
    size: int | None


class Provider(Protocol):
    """
    Simple provider interface for checkpoint fetching (URI -> local path).
    """

    scheme: str

    def can_handle(self, uri: str) -> bool: ...

    def estimate_size(self, uri: str) -> int | None: ...

    def fetch(self, uri: str, destination_dir: Path, *, compute_hash: bool = True) -> tuple[Path, str | None]: ...
