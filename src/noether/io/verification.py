#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
from pathlib import Path  # noqa: TCH003
from typing import Any, Literal, TypeVar

from loguru import logger

HashType = Literal["sha256"]
T = TypeVar("T")
R = TypeVar("R")


class FailAction(str, Enum):
    WARN = "warn"
    DELETE = "delete"
    REDOWNLOAD = "redownload"
    ABORT = "abort"

    def describe(self) -> str:
        return {
            self.WARN: "Show a warning but continue",
            self.DELETE: "Delete corrupted files",
            self.REDOWNLOAD: "Force re-download corrupted files",
            self.ABORT: "Stop execution on first error",
        }[self]


@dataclass(frozen=True)
class FileRecord:
    path: str  # relative POSIX path w.r.t root, string for easier JSON serialization as Path will be more hustle
    size: int
    hash: str | None  # optional if unavailable
    etag: str | None  # optional for services that provide an etag, like S3
    source: dict[str, Any] | None  # optional, e.g. {provider: "huggingface", repo: "...", rev: "...", filename: "..."}


@dataclass
class VerificationResult:
    ok: list[str]
    missing: list[str]  # listed in manifest but not on disk
    extra: list[str]  # on disk but not in manifest
    size_mismatch: list[str]
    hash_mismatch: list[str]


class VerificationType(str, Enum):
    SIZE = "size"
    HASH = "hash"


# =====================================================================================================================
#                                               PARALLEL HELPERS
# ---------------------------------------------------------------------------------------------------------------------
class ParallelErrors(Exception):
    def __init__(self, errors: list[tuple[Any, BaseException]]) -> None:
        super().__init__(f"{len(errors)} errors encountered")
        self.errors = errors  # [(input, exception), ...]


Manifest = dict[str, FileRecord]  # key = relative path


def parallel_map_collect_errors(
    fn: Callable[[T], R],
    items: Iterable[T],
    max_workers: int = 8,
) -> list[R]:
    results: list[R] = []
    failures: list[tuple[T, BaseException]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(fn, item): item for item in items}
        for _future in as_completed(future_to_item):
            item = future_to_item[_future]
            try:
                results.append(_future.result())
            except BaseException as exc:
                failures.append((item, exc))
    if failures:
        raise ParallelErrors(failures)
    return results


# =====================================================================================================================


# =====================================================================================================================
#                                                       HASHING
# ---------------------------------------------------------------------------------------------------------------------
def hash_file(path: Path, chunk_size_mb: int = 1) -> str:
    """Reads file bytes as chunks and hashes them.

    Args:
        path: Input file path.
        chunk_size_mb: Chunk size in megabytes.

    Returns:
        - The hash of the file.
    """
    chunk_size = chunk_size_mb * 1024 * 1024

    hash_fn = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash_fn.update(chunk)
    return hash_fn.hexdigest()


# =====================================================================================================================


# =====================================================================================================================
#                                               MANIFEST BUILD / IO
# ---------------------------------------------------------------------------------------------------------------------
def build_manifest(root: Path, jobs: int = 4, include_hash: bool = True) -> Manifest:
    """Creates a manifest from the given root directory.

    Args:
        root: Input root directory.
        jobs: A number of jobs to run in parallel.
        include_hash: If True, include the hash of the file in the manifest. Defaults to True.

    Returns:
        - A dictionary containing the manifest.
    """
    root = root.resolve()
    files = [x for x in root.rglob("*") if x.is_file()]
    logger.info(f"Building manifest for {len(files)} files under {root}")

    def _to_record(path: Path) -> tuple[str, FileRecord]:
        _relative_path = str(path.relative_to(root).as_posix())
        _hash_value = hash_file(path) if include_hash else None
        return (
            _relative_path,
            FileRecord(
                path=_relative_path,
                size=path.stat().st_size,
                hash=_hash_value,
                etag=None,
                source=None,
            ),
        )

    manifest = Manifest()
    # Fail-collect: if any file hashing/stat fails, raise aggregated error with inputs
    for relative_path, record in parallel_map_collect_errors(_to_record, files, max_workers=jobs):
        manifest[relative_path] = record

    return manifest


def save_manifest(manifest: Manifest, path: Path) -> None:
    """Saves the given manifest to the given path.

    Args:
        manifest: Input manifest to save.
        path: Output file path.

    Returns:
        - None
    """
    data = {key: asdict(value) for key, value in manifest.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
    logger.info(f"Manifest saved to {path}")


def load_manifest(path: Path) -> Manifest:
    """Loads the given manifest from the given path.
    Args:
        path: Input file path.

    Returns:
        - A dictionary containing the manifest.
    """
    data = json.loads(path.read_text())
    result: Manifest = {}
    for relative_path, record in data.items():
        result[relative_path] = FileRecord(**record)
    return result


# =====================================================================================================================


# =====================================================================================================================
#                                                   VERIFICATION
# ---------------------------------------------------------------------------------------------------------------------
def verify_tree(
    root: Path,
    manifest: Manifest,
    jobs: int = 4,
    require_hash: bool = True,
) -> VerificationResult:
    """Verifies the given manifest against the given root directory.

    Args:
        root: Input root directory.
        manifest: Input manifest to verify.
        jobs: Number of jobs to run in parallel.
        require_hash: If True, require hash of the file in the manifest. Defaults to True.

    Returns:
        - An instance of VerificationResult.
    """
    root = root.resolve()
    on_disk = {str(path.relative_to(root).as_posix()) for path in root.rglob("*") if path.is_file()}
    in_manifest = set(manifest.keys())

    missing = sorted(in_manifest - on_disk)
    extra = sorted(on_disk - in_manifest)
    candidates = sorted(in_manifest & on_disk)

    size_mismatch: list[str] = []
    hash_mismatch: list[str] = []
    files_okay: list[str] = []

    def _check(relative_path: str) -> tuple[str, VerificationType | None]:
        _path = root / relative_path
        _record = manifest[relative_path]
        # Check filesize:
        if _path.stat().st_size != _record.size:
            return relative_path, VerificationType.SIZE
        # Check hash, if available:
        if _record.hash is not None:
            digest = hash_file(_path)
            if digest != _record.hash:
                return relative_path, VerificationType.HASH
        elif require_hash:
            # Manifest lacks hash, but we require it -> treat as mismatch:
            return relative_path, VerificationType.HASH
        return relative_path, None

    # Collect all failures; if any task raises, surface them together
    for rel, err in parallel_map_collect_errors(_check, candidates, max_workers=jobs):
        if err is None:
            files_okay.append(rel)
        elif err is VerificationType.SIZE:
            size_mismatch.append(rel)
        else:
            hash_mismatch.append(rel)

    return VerificationResult(
        ok=files_okay,
        missing=missing,
        extra=extra,
        size_mismatch=sorted(size_mismatch),
        hash_mismatch=sorted(hash_mismatch),
    )


# =====================================================================================================================
