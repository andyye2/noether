#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from collections.abc import Iterator
from typing import Protocol, runtime_checkable


@runtime_checkable
class SizedIterable(Protocol):
    """Sampler needs to implement __len__ and be iterable such that the type checking doesn't complain."""

    def __iter__(self) -> Iterator[int]: ...

    def __len__(self) -> int: ...
