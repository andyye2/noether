# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Inspect a Noether model checkpoint without constructing the training stack.

The command emits a compact JSON document containing checkpoint metadata, tensor
counts, parameter counts, and tensor shapes.  It intentionally uses
``weights_only=True`` and never mutates the checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def checkpoint_summary(path: Path) -> dict[str, Any]:
    """Return a JSON-serializable summary of a Noether model checkpoint.

    Args:
        path: Path to a ``*_model.th`` checkpoint.

    Returns:
        Checkpoint metadata and state-dict tensor shapes.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        TypeError: If the checkpoint or its state dict has an unexpected type.
        KeyError: If a wrapped checkpoint does not contain ``state_dict``.
    """
    if not path.is_file():
        raise FileNotFoundError(path)

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint).__name__}")

    is_wrapped = "state_dict" in checkpoint
    if is_wrapped:
        state_dict = checkpoint["state_dict"]
    else:
        if any(not isinstance(value, torch.Tensor) for value in checkpoint.values()):
            raise KeyError("Checkpoint has metadata but no 'state_dict' key")
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected state_dict mapping, got {type(state_dict).__name__}")

    tensor_shapes: dict[str, list[int]] = {}
    parameter_count = 0
    dtypes: dict[str, int] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise TypeError("State dict must map string keys to tensors")
        tensor_shapes[key] = list(value.shape)
        parameter_count += value.numel()
        dtype = str(value.dtype).removeprefix("torch.")
        dtypes[dtype] = dtypes.get(dtype, 0) + 1

    metadata = {key: _json_safe(value) for key, value in checkpoint.items() if key != "state_dict"}
    return {
        "path": str(path.resolve()),
        "file_size_bytes": path.stat().st_size,
        "wrapped_state_dict": is_wrapped,
        "metadata": metadata,
        "tensor_count": len(tensor_shapes),
        "parameter_count": parameter_count,
        "dtypes": dtypes,
        "tensor_shapes": tensor_shapes,
    }


def _json_safe(value: Any) -> Any:
    """Convert small checkpoint metadata values into JSON-safe objects."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    return repr(value)


def main() -> None:
    """Parse command-line arguments and print the checkpoint summary as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to a Noether *_model.th checkpoint")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()

    rendered = json.dumps(checkpoint_summary(args.checkpoint), indent=2, sort_keys=True)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
