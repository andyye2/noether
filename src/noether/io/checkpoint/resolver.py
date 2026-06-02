#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from loguru import logger

from noether.io.checkpoint.providers.file_provider import FileProvider
from noether.io.checkpoint.providers.huggingface_provider import HFProvider
from noether.io.checkpoint.types import CheckpointMetadata, Provider

PROVIDERS: list[Provider] = [FileProvider(), HFProvider()]
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(uri: str) -> str:
    # very simple slug; adjust if you want subdirs by org/name
    return uri.replace("://", "__").replace("/", "_").replace("?", "_").replace("#", "_").replace("=", "-")


def _sanitize(name: str) -> str:
    return _SAFE.sub("_", name).strip("._-") or "checkpoint"


def _strip_all_suffixes(name: str) -> str:
    path = Path(name)
    base = path.name
    for suffix in path.suffixes:
        base = base.removesuffix(suffix)
    return base


def _last_ext(name: str) -> str:
    return Path(name).suffix or ""


def _filename_hint(uri: str) -> str:
    """Derive a target filename from a checkpoint URI or local path.
    - hf://... ?filename=NAME  -> NAME (required)
    - s3://bucket/key          -> basename of key
    - file://... or plain path -> basename of the path
    """
    p = urlparse(uri)
    scheme = (p.scheme or "").lower()
    if scheme == "hf":
        q = parse_qs(p.query)
        fn = q.get("filename", [None])[0]
        if not fn:
            raise ValueError("hf:// URI requires ?filename=<name>")
        return fn
    if scheme == "s3":
        name = Path(p.path).name
        if not name:
            raise ValueError("s3:// URI must include an object key")
        return name
    if scheme == "file":
        return Path(p.path).name
    # Plain local path (no scheme) or other schemes default to basename:
    return Path(uri).name


def _plan_dest(cache_dir: Path, filename_hint: str) -> tuple[Path, Path]:
    """Decide the final directory and file path based on a filename hint.
    Dir uses the filename :strong:without any extensions; file = dir/<name><last_ext>.

    Args:
        cache_dir: A cache directory where the checkpoint is.
        filename_hint: A filename hint to be used as filename.

    Returns:
        - tuple: A checkpoint dir and a complete checkpoint path.
    """
    base = _sanitize(_strip_all_suffixes(filename_hint))
    ext = _last_ext(filename_hint)
    checkpoint_dir = cache_dir / base
    checkpoint_path = checkpoint_dir / f"{base}{ext}"
    return checkpoint_dir, checkpoint_path


def _ensure_space(destination_dir: Path, estimate: int | None, min_free_bytes: int | None) -> None:
    """Inplace check to check existence of free space on disk.
    If no `estimate` and `min_free_bytes` are specified, the check does nothing.

    Args:
        destination_dir: Destination directory.
        estimate: How many bytes to be written.
        min_free_bytes: Minimum free space to check.

    Raises:
        RuntimeError: if there is not enough space available.

    """
    if estimate is None and min_free_bytes is None:
        return
    usage = shutil.disk_usage(destination_dir)
    required_bytes = (estimate or 0) + (min_free_bytes or 0)
    if usage.free < required_bytes:
        raise RuntimeError(f"Insufficient space: need ≈{required_bytes} bytes, free={usage.free} bytes")


def resolve_checkpoint(
    uri_or_path: str | Path,
    *,
    cache_dir: Path = Path.home() / ".cache" / "emmi" / "checkpoints",
    expected_sha256: str | None = None,
    verify_load: str | None = None,  # "pt" | "ts" | "onnx"
    min_free_bytes: int | None = None,
    compute_hash: bool = True,
) -> CheckpointMetadata:
    """Resolve a checkpoint reference (URI or local path) to a verified local file.

    - Downloads to cache if remote (atomic write).
    - Optional disk space check (estimate + min_free_bytes).
    - Optional sha256 validation.
    - Writes meta.json in the cache folder.
    - [Optional] Load smoke-test (pt|ts|onnx).

    Args:
        uri_or_path: URI or local path.
        cache_dir: Cache directory.
        expected_sha256: Expected checksum.
        verify_load: Whether to verify checkpoint loading of a specific type "pt" | "ts" | "onnx". Defaults to None.
        min_free_bytes: Minimum free space. Defaults to None.
        compute_hash: Whether to compute hash. Defaults to True.

    Returns:
        An instance of `CheckpointMetadata`.
    """
    uri = str(uri_or_path)
    provider: Provider | None = next((_provider for _provider in PROVIDERS if _provider.can_handle(uri)), None)
    if provider is None:
        raise ValueError(f"No provider can handle {uri!r}")

    filename_hint = _filename_hint(uri)
    checkpoint_dir, checkpoint_path = _plan_dest(cache_dir, filename_hint)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    est = provider.estimate_size(uri)
    _ensure_space(checkpoint_dir, est, min_free_bytes)

    local_path, sha_from_fetch = provider.fetch(uri, checkpoint_dir, compute_hash=compute_hash)

    # Ensure the file resides at the planned path (dir/name without extra suffixes):
    if local_path != checkpoint_path and local_path.exists():
        try:
            if local_path.parent == checkpoint_dir:
                # Same directory: just rename into canonical filename:
                local_path.rename(checkpoint_path)
            else:
                # Different location (e.g., file:// provider): copy into cache dir:
                shutil.copy2(local_path, checkpoint_path)
            local_path = checkpoint_path
        except Exception as e:
            logger.warning(f"Could not place checkpoint at {checkpoint_path}: {e}")

    sha = sha_from_fetch
    if expected_sha256 and not sha:
        # If user asked for verification but provider didn't compute hash, compute now.
        from noether.io.verification import hash_file

        sha = hash_file(local_path)

    if expected_sha256 and sha != expected_sha256:
        raise RuntimeError("SHA-256 mismatch for checkpoint")

    meta = CheckpointMetadata(
        uri=uri,
        local_path=local_path,
        sha256=sha,
        size=local_path.stat().st_size if local_path.exists() else None,
    )
    data = asdict(meta)
    data["local_path"] = str(meta.local_path)  # ensure JSON serializable
    data["created_at"] = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    (checkpoint_dir / "meta.json").write_text(json.dumps(data, indent=2))

    if verify_load:
        _smoke_load(local_path, verify_load)

    logger.info(f"Checkpoint ready → {local_path}")
    return meta


def _smoke_load(path: Path, kind: str) -> None:
    """Attempts to load a checkpoint from disk.

    Args:
        path: Path to checkpoint.
        kind: Checkpoint kind ("pt", "ts", "onnx").

    Raises:
        ValueError: If `kind` is not supported.
    """
    kind = kind.lower()
    if kind == "pt":
        import torch

        _ = torch.load(str(path), map_location="cpu")
        return
    if kind in ("ts", "torchscript", "jit"):
        import torch

        _ = torch.jit.load(str(path), map_location="cpu")
        return
    if kind == "onnx":
        import onnx  # type: ignore

        _ = onnx.load(str(path))
        return
    raise ValueError(f"Unknown verify_load={kind!r}")
