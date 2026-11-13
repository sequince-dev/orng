from __future__ import annotations

import copy
from collections.abc import Mapping
from numbers import Integral
from typing import Any

from .._utils import (
    SizeLike,
    integer_bounds,
    normalize_axis,
    normalize_shape,
    poisson_rate_limit,
    validate_nonnegative_scalar,
)


class NumPyBackend:
    def __init__(self, *, seed: int | None, generator: Any | None) -> None:
        self._impl = NumPyFunctionalBackend(pure=False)
        self._state = self._impl.init_state(seed=seed, generator=generator)

    def integers(
        self, *, low: int, high: int | None, size: SizeLike, dtype: Any | None
    ) -> Any:
        result, self._state = self._impl.integers(
            self._state, low=low, high=high, size=size, dtype=dtype
        )
        return result

    def permutation(self, x: int | Any, *, axis: int) -> Any:
        result, self._state = self._impl.permutation(self._state, x, axis=axis)
        return result

    def exponential(
        self, *, scale: Any, size: SizeLike, dtype: Any | None
    ) -> Any:
        result, self._state = self._impl.exponential(
            self._state, scale=scale, size=size, dtype=dtype
        )
        return result

    def poisson(self, *, lam: Any, size: SizeLike, dtype: Any | None) -> Any:
        result, self._state = self._impl.poisson(
            self._state, lam=lam, size=size, dtype=dtype
        )
        return result

    def random(self, *, size: SizeLike, dtype: Any | None) -> Any:
        result, self._state = self._impl.random(
            self._state,
            size=size,
            dtype=dtype,
        )
        return result

    def uniform(
        self,
        *,
        low: Any,
        high: Any,
        size: SizeLike,
        dtype: Any | None,
    ) -> Any:
        result, self._state = self._impl.uniform(
            self._state,
            low=low,
            high=high,
            size=size,
            dtype=dtype,
        )
        return result

    def normal(
        self,
        *,
        loc: Any,
        scale: Any,
        size: SizeLike,
        dtype: Any | None,
    ) -> Any:
        result, self._state = self._impl.normal(
            self._state,
            loc=loc,
            scale=scale,
            size=size,
            dtype=dtype,
        )
        return result

    def gamma(
        self,
        *,
        shape: Any,
        scale: Any,
        size: SizeLike,
        dtype: Any | None,
    ) -> Any:
        result, self._state = self._impl.gamma(
            self._state,
            shape=shape,
            scale=scale,
            size=size,
            dtype=dtype,
        )
        return result

    def choice(
        self,
        population: int | Any,
        *,
        size: SizeLike,
        replace: bool,
        probabilities: Any | None,
    ) -> Any:
        result, self._state = self._impl.choice(
            self._state,
            population,
            size=size,
            replace=replace,
            probabilities=probabilities,
        )
        return result

    def state(self) -> Any:
        """Return the current backend-native generator state."""
        return self._state

    def set_state(self, state: Any) -> None:
        """Replace the current backend-native generator state."""
        self._state = self._impl.init_state(seed=None, generator=state)

    def state_dict(self) -> dict[str, Any]:
        generator = self._state
        return {
            "bit_generator": type(generator.bit_generator).__name__,
            "bit_generator_state": copy.deepcopy(
                generator.bit_generator.state
            ),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        try:
            bit_generator_name = state["bit_generator"]
            bit_generator_state = state["bit_generator_state"]
        except KeyError as exc:
            raise ValueError(
                "NumPy RNG state requires 'bit_generator' and "
                "'bit_generator_state'."
            ) from exc
        bit_generator_cls = getattr(
            self._impl._np.random, bit_generator_name, None
        )
        if bit_generator_cls is None:
            raise ValueError(
                f"Unknown NumPy bit generator '{bit_generator_name}'."
            )
        bit_generator = bit_generator_cls()
        bit_generator.state = copy.deepcopy(bit_generator_state)
        self._state = self._impl._np.random.Generator(bit_generator)


class NumPyFunctionalBackend:
    def __init__(self, *, pure: bool = True) -> None:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "NumPy backend requires the 'numpy' package to be installed. "
                "Install it with `pip install orng[numpy]`."
            ) from exc
        self._np = np
        self._pure = pure

    def init_state(self, *, seed: int | None, generator: Any | None) -> Any:
        np = self._np
        if generator is None:
            gen = np.random.default_rng(seed)
        elif isinstance(generator, np.random.Generator):
            gen = generator
        else:
            raise TypeError(
                "generator must be a numpy.random.Generator when using the "
                "NumPy backend."
            )
        if self._pure:
            return copy.deepcopy(gen.bit_generator.state)
        return gen

    def _generator_from_state(self, state: Any) -> Any:
        if not self._pure:
            if not isinstance(state, self._np.random.Generator):
                raise TypeError(
                    "state must be a numpy.random.Generator when pure=False."
                )
            return state
        gen = self._np.random.default_rng()
        gen.bit_generator.state = copy.deepcopy(state)
        return gen

    def _result_and_next_state(self, gen: Any, result: Any) -> tuple[Any, Any]:
        if self._pure:
            return result, copy.deepcopy(gen.bit_generator.state)
        return result, gen

    def integers(
        self,
        state: Any,
        *,
        low: int,
        high: int | None = None,
        size: SizeLike = None,
        dtype: Any | None = None,
    ) -> tuple[Any, Any]:
        xp = self._np
        sample_dtype = xp.dtype(dtype if dtype is not None else xp.int64)
        if sample_dtype not in (xp.dtype(xp.int32), xp.dtype(xp.int64)):
            raise TypeError("dtype must be int32 or int64.")
        limits = xp.iinfo(sample_dtype)
        low, high = integer_bounds(low, high, int(limits.min), int(limits.max))
        gen = self._generator_from_state(state)
        result = gen.integers(
            low, high, size=normalize_shape(size), dtype=sample_dtype
        )
        return self._result_and_next_state(gen, result)

    def exponential(
        self,
        state: Any,
        *,
        scale: Any = 1.0,
        size: SizeLike = None,
        dtype: Any | None = None,
    ) -> tuple[Any, Any]:
        xp = self._np
        sample_dtype = xp.dtype(dtype if dtype is not None else xp.float64)
        if sample_dtype not in (xp.dtype(xp.float32), xp.dtype(xp.float64)):
            raise TypeError("dtype must be float32 or float64.")
        validate_nonnegative_scalar(scale, "scale")
        parameter = xp.asarray(scale)
        if bool(xp.any((parameter < 0) | ~xp.isfinite(parameter))):
            raise ValueError("scale must be finite and non-negative.")
        shape = None if size is None else normalize_shape(size)
        gen = self._generator_from_state(state)
        result = gen.exponential(scale=parameter, size=shape)
        result = xp.asarray(result, dtype=sample_dtype)
        return self._result_and_next_state(gen, result)

    def poisson(
        self,
        state: Any,
        *,
        lam: Any = 1.0,
        size: SizeLike = None,
        dtype: Any | None = None,
    ) -> tuple[Any, Any]:
        xp = self._np
        sample_dtype = xp.dtype(dtype if dtype is not None else xp.int64)
        if sample_dtype not in (xp.dtype(xp.int32), xp.dtype(xp.int64)):
            raise TypeError("dtype must be int32 or int64.")
        limit = poisson_rate_limit(lam, int(xp.iinfo(sample_dtype).max))
        parameter = xp.asarray(lam)
        if bool(xp.any((parameter < 0) | ~xp.isfinite(parameter))):
            raise ValueError("lam must be finite and non-negative.")
        if bool(xp.any(parameter > limit)):
            raise ValueError("lam is too large for the requested dtype.")
        shape = None if size is None else normalize_shape(size)
        gen = self._generator_from_state(state)
        result = gen.poisson(lam=parameter, size=shape)
        result = xp.asarray(result, dtype=sample_dtype)
        return self._result_and_next_state(gen, result)

    def permutation(
        self, state: Any, x: int | Any, *, axis: int = 0
    ) -> tuple[Any, Any]:
        np = self._np
        if isinstance(x, Integral):
            if x < 0:
                raise ValueError("x must be non-negative.")
            domain = np.arange(x)
        else:
            domain = np.asarray(x)
        axis = normalize_axis(axis, domain.ndim)
        gen = self._generator_from_state(state)
        result = gen.permutation(domain, axis=axis)
        return self._result_and_next_state(gen, result)

    def random(
        self,
        state: Any,
        *,
        size: SizeLike,
        dtype: Any | None = None,
    ) -> tuple[Any, Any]:
        gen = self._generator_from_state(state)
        result = gen.random(size=size, dtype=dtype)
        return self._result_and_next_state(gen, result)

    def uniform(
        self,
        state: Any,
        *,
        low: Any,
        high: Any,
        size: SizeLike,
        dtype: Any | None = None,
    ) -> tuple[Any, Any]:
        gen = self._generator_from_state(state)
        result = gen.uniform(low=low, high=high, size=size)
        if dtype is not None:
            result = self._np.asarray(result, dtype=dtype)
        return self._result_and_next_state(gen, result)

    def normal(
        self,
        state: Any,
        *,
        loc: Any,
        scale: Any,
        size: SizeLike,
        dtype: Any | None = None,
    ) -> tuple[Any, Any]:
        gen = self._generator_from_state(state)
        result = gen.normal(loc=loc, scale=scale, size=size)
        if dtype is not None:
            result = self._np.asarray(result, dtype=dtype)
        return self._result_and_next_state(gen, result)

    def gamma(
        self,
        state: Any,
        *,
        shape: Any,
        scale: Any,
        size: SizeLike,
        dtype: Any | None = None,
    ) -> tuple[Any, Any]:
        gen = self._generator_from_state(state)
        result = gen.gamma(shape=shape, scale=scale, size=size)
        if dtype is not None:
            result = self._np.asarray(result, dtype=dtype)
        return self._result_and_next_state(gen, result)

    def choice(
        self,
        state: Any,
        population: int | Any,
        *,
        size: SizeLike,
        replace: bool,
        probabilities: Any | None,
    ) -> tuple[Any, Any]:
        gen = self._generator_from_state(state)
        result = gen.choice(
            population,
            size=size,
            replace=replace,
            p=probabilities,
        )
        return self._result_and_next_state(gen, result)


__all__ = ["NumPyBackend", "NumPyFunctionalBackend"]
