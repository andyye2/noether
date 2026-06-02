#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from pathlib import Path
from typing import Any, Literal


def validate_path(
    path: str | Path,
    exists: Literal["must", "must_not", "any"] = "must",
    suffix: str | None = None,
    mkdir: bool = False,
) -> Path:
    """Converts a string to a Path, validates it, and optionally creates it.

    Args:
        path: The path string to validate.
        exists:
            - "must": Raises FileNotFoundError if the path doesn't exist.
            - "must_not": Raises FileExistsError if the path already exists.
            - "any": Performs no existence check.
        suffix: If provided, checks if the path ends with this suffix.
        mkdir: If True, creates the directory path (like mkdir -p).

    Returns:
        Path: The validated path.
    """
    path = Path(path).expanduser()

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)

    if exists == "must" and not path.exists():
        raise FileNotFoundError(f"Path '{path.as_posix()}' does not exist")
    if exists == "must_not" and path.exists():
        raise FileExistsError(f"Path '{path.as_posix()}' already exists")

    if suffix is not None and not path.as_posix().endswith(suffix):
        raise ValueError(f"'{path.as_posix()}' doesn't end with '{suffix}'")

    return path


def select_with_path(obj: dict[str, Any] | list[Any] | object, path: str | None) -> object:
    """Access values of an object, a list or a dictionary using a string path.

    Args:
        obj: The object to access.
        path: The path to the value, e.g. "a.b.c" or "a[0].b.c".
    """
    if path is not None and len(path) > 0:
        for p in path.split("."):
            if isinstance(obj, dict):
                obj = obj[p]
            elif isinstance(obj, list):
                obj = obj[int(p)]
            else:
                obj = getattr(obj, p)

    return obj
