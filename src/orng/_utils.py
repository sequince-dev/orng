from __future__ import annotations

import math
import operator
from numbers import Real
from typing import Any, Sequence, Tuple

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


def integer_bounds(
    low: int, high: int | None, minimum: int, maximum: int
) -> tuple[int, int]:
    """Validate scalar integer bounds before a backend can clip or wrap them."""
    lower = operator.index(low)
    upper = operator.index(high) if high is not None else lower
    if high is None:
        lower = 0
    if lower >= upper:
        raise ValueError("low must be less than high.")
    if lower < minimum or upper > maximum:
        raise ValueError("Integer bounds must fit in the requested dtype.")
    return lower, upper


def normalize_axis(axis: int, ndim: int) -> int:
    """Validate an axis without accepting out-of-range values modulo ndim."""
    axis = operator.index(axis)
    if not -ndim <= axis < ndim:
        raise ValueError("axis is out of bounds for the input array.")
    return axis % ndim


def validate_nonnegative_scalar(value: Any, name: str) -> None:
    """Reject invalid Python scalar parameters without converting JAX tracers."""
    if isinstance(value, Real) and (value < 0 or not math.isfinite(value)):
        raise ValueError(f"{name} must be finite and non-negative.")


def poisson_rate_limit(lam: Any, maximum: int) -> float:
    """Reserve ten standard deviations below the output integer limit."""
    validate_nonnegative_scalar(lam, "lam")
    limit = maximum - 10 * math.sqrt(maximum)
    if isinstance(lam, Real) and float(lam) > limit:
        raise ValueError("lam is too large for the requested dtype.")
    return limit


__all__ = ["SizeLike", "normalize_shape", "total_size"]
