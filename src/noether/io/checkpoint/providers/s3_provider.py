#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from pathlib import Path

from botocore.exceptions import ClientError  # type: ignore[import-untyped]

from noether.io.checkpoint.iohash import atomic_write_and_hash
from noether.io.checkpoint.types import Provider
from noether.io.interfaces.s3 import get_s3_client
from noether.io.logging_progress import LogProgress


def _split_s3(uri: str) -> tuple[str, str]:
    """Parse s3://bucket/key... into (bucket, key)."""
    if not uri.startswith("s3://"):
        raise ValueError("S3 URI must start with 's3://'")
    rest = uri[5:]  # strip "s3://"
    if "/" not in rest:
        raise ValueError("S3 URI must be 's3://<bucket>/<key>'")
    bucket, key = rest.split("/", 1)
    if not bucket or not key.strip():
        raise ValueError("S3 URI must include non-empty bucket and key")
    return bucket, key


class S3Provider(Provider):
    scheme = "s3"

    def can_handle(self, uri: str) -> bool:
        return uri.startswith("s3://")

    def estimate_size(self, uri: str) -> int | None:
        s3 = get_s3_client()
        bucket, key = _split_s3(uri)
        try:
            head = s3.head_object(Bucket=bucket, Key=key)
            return int(head.get("ContentLength", 0)) or None
        except ClientError:
            return None

    def fetch(self, uri: str, destination_dir: Path, *, compute_hash: bool = True) -> tuple[Path, str | None]:
        s3 = get_s3_client()
        bucket, key = _split_s3(uri)
        destination_dir.mkdir(parents=True, exist_ok=True)

        object_name = Path(key).name
        out_path = destination_dir / object_name

        # HEAD for size (progress):
        total = None
        try:
            head = s3.head_object(Bucket=bucket, Key=key)
            total = head.get("ContentLength")
        except ClientError:
            # public/unsigned or no HEAD perms; progress will still work without total
            pass

        # GET streaming body:
        obj = s3.get_object(Bucket=bucket, Key=key)
        body = obj["Body"]

        def _chunks():
            for part in body.iter_chunks(1024 * 1024):
                if part:
                    yield part

        progress = LogProgress(label=object_name, total_bytes=total)
        try:
            sha = atomic_write_and_hash(
                out_path,
                _chunks(),
                compute_hash=compute_hash,
                progress=progress,
            )
        finally:
            try:
                body.close()
            except Exception:
                pass

        return out_path, sha
