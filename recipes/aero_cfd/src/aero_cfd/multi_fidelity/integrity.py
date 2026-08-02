# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Content hashing, atomic publication, and Git-state capture.

Every artifact of the study is bound to the others by SHA256, so the hashing
helpers must produce byte-identical results regardless of which entry point
called them.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, TypeGuard

SHA256_ALPHABET = "0123456789abcdef"


def is_sha256(value: object) -> TypeGuard[str]:
    """Return whether a value is a lowercase 64-character SHA256 digest.

    Args:
        value: Candidate digest.

    Returns:
        ``True`` for a lowercase hexadecimal digest of the correct length.
    """
    return isinstance(value, str) and len(value) == 64 and all(character in SHA256_ALPHABET for character in value)


def is_git_commit(value: object) -> TypeGuard[str]:
    """Return whether a value is a lowercase 40-character Git commit.

    Args:
        value: Candidate commit.

    Returns:
        ``True`` for a full lowercase hexadecimal commit hash.
    """
    return isinstance(value, str) and len(value) == 40 and all(character in SHA256_ALPHABET for character in value)


def canonical_json_bytes(payload: Any) -> bytes:
    """Return the stable UTF-8 JSON representation used for content hashes.

    Args:
        payload: JSON-compatible object.

    Returns:
        Sorted, separator-normalized UTF-8 bytes.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_json_sha256(payload: Any) -> str:
    """Hash a JSON-compatible object through its canonical representation.

    Args:
        payload: JSON-compatible object.

    Returns:
        Lowercase SHA256 digest.
    """
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _order_independent(value: Any) -> Any:
    """Return a representation whose ordering does not depend on iteration.

    Sets carry no order, so two identical configurations would otherwise hash
    differently depending on the process hash seed.
    """
    if isinstance(value, set | frozenset):
        return sorted(_order_independent(item) for item in value)
    if isinstance(value, dict):
        return {str(key): _order_independent(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_order_independent(item) for item in value]
    if value is None or isinstance(value, str | bool | int | float):
        return value
    return str(value)


def stable_sha256(value: Any) -> str:
    """Hash an arbitrary Python object reproducibly across processes.

    Unlike :func:`canonical_json_sha256` this accepts values a Pydantic model
    dumps in Python mode, including sets and paths, and is therefore the right
    hash for a resolved configuration.

    Args:
        value: Object to hash.

    Returns:
        Lowercase SHA256 digest.
    """
    return canonical_json_sha256(_order_independent(value))


def sha256_file(path: Path, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    """Hash a file without deserializing it.

    Args:
        path: File to read.
        chunk_bytes: Read size; only affects memory use.

    Returns:
        Lowercase SHA256 digest of the file bytes.
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Durably replace one JSON artifact after writing it completely.

    A reader therefore never observes a truncated artifact, which matters
    because the sidecars are consumed by later stages as provenance.

    Args:
        path: Destination file.
        payload: JSON-compatible mapping.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as file:
            file.write((json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class GitState:
    """Implementation state of one repository at the moment of execution.

    Attributes:
        commit: Lowercase 40-character ``HEAD`` commit.
        dirty: Whether tracked or untracked files differ from ``HEAD``.
        status_sha256: SHA256 of the porcelain status, which pins *which*
            files differed without embedding the diff itself.
    """

    commit: str
    dirty: bool
    status_sha256: str

    def __post_init__(self) -> None:
        """Reject a commit that cannot identify an implementation state."""
        if not is_git_commit(self.commit):
            raise ValueError(f"invalid implementation Git commit {self.commit!r}")


def read_git_state(repo_root: Path) -> GitState:
    """Capture the implementation commit and dirty status without mutating Git.

    Args:
        repo_root: Repository that holds the implementation.

    Returns:
        The captured :class:`GitState`.

    Raises:
        subprocess.CalledProcessError: If ``repo_root`` is not a repository.
    """
    commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
    ).stdout
    return GitState(commit=commit, dirty=bool(status.strip()), status_sha256=hashlib.sha256(status).hexdigest())
