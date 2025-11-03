from __future__ import annotations

import math
from typing import Sequence, Tuple

SizeLike = int | Sequence[int] | None


def normalize_shape(size: SizeLike) -> Tuple[int, ...]:
    """Convert a size argument to a canonical ``tuple`` form."""
    if size is None:
        return ()
    if isinstance(size, int):
        if size < 0:
            raise ValueError("size must be non-negative.")
        return (size,)
    shape = tuple(int(dim) for dim in size)
    if any(dim < 0 for dim in shape):
        raise ValueError("size entries must be non-negative.")
    return shape


def total_size(shape: Tuple[int, ...]) -> int:
    if not shape:
        return 1
    return math.prod(shape)


__all__ = ["SizeLike", "normalize_shape", "total_size"]
