#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from pathlib import Path

from noether.io.checkpoint.types import Provider
from noether.io.verification import hash_file


class FileProvider(Provider):
    """Local filesystem (supports file:// and plain paths)."""

    scheme = "file"

    def can_handle(self, uri: str) -> bool:
        """Returns True if the URI points to a local file.

        Args:
            uri: Input URI where the file is located.

        Returns:
            - bool: Whether the URI is a local file or not.
        """
        # file://... or any existing local path
        if uri.startswith("file://"):
            return True
        return Path(uri).exists()

    def estimate_size(self, uri: str) -> int | None:
        """Estimate the size of a local file.

        Args:
            uri: Input URI where the file is located.

        Returns:
            - int: Estimated size of the local file.
            - None: If no size is available.
        """
        p = Path(uri.replace("file://", ""))
        return p.stat().st_size if p.exists() else None

    def fetch(self, uri: str, destination_dir: Path, *, compute_hash: bool = True) -> tuple[Path, str | None]:
        """Returns a tuple with a local file path for a given URI and a corresponding hash (if possible to create).

        Args:
            uri: Input URI where the file is located.
            destination_dir: Destination directory where the file need to be located.
            compute_hash: Compute the hash of the file on the local file system.

        Returns:
            - tuple[Path, str | None]: Local file path and an option hash value.
        """
        # Not copying; just point to the local path. Hash if requested.
        p = Path(uri.replace("file://", ""))
        sha = hash_file(p) if compute_hash and p.is_file() else None
        return p, sha
        # ---

        # src = Path(uri.replace("file://", "")).expanduser().resolve()
        # destination_dir.mkdir(parents=True, exist_ok=True)
        #
        # # If already inside dst_dir, just return it:
        # try:
        #     if destination_dir.resolve() in src.parents:
        #         return src
        # except Exception:
        #     pass
        #
        # target = (destination_dir / src.name).resolve()
        # if target.exists():
        #     return target  # assume cached
        #
        # # Try hardlink -> fallback to symlink → fallback to copy:
        # try:
        #     os.link(src, target)
        # except OSError:
        #     try:
        #         target.symlink_to(src)
        #     except OSError:
        #         shutil.copy2(src, target)
        #
        # return target
