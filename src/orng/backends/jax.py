from __future__ import annotations

import secrets
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


class JAXBackend:
    def __init__(self, *, seed: int | None, key: Any | None) -> None:
        self._impl = JAXFunctionalBackend()
        self._state = self._impl.init_state(seed=seed, generator=key)

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
        """Return the current JAX random key."""
        return self._state

    def set_state(self, state: Any) -> None:
        """Replace the current JAX random key."""
        self._state = self._impl.init_state(seed=None, generator=state)

    def state_dict(self) -> dict[str, Any]:
        return {
            "key_data": self._impl._jnp.array(
                self._impl._jax.random.key_data(self._state)
            ),
            "implementation": str(
                self._impl._jax.random.key_impl(self._state)
            ),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        try:
            key_data = state["key_data"]
            implementation = state["implementation"]
        except KeyError as exc:
            raise ValueError(
                "JAX RNG state requires 'key_data' and 'implementation'."
            ) from exc
        self._state = self._impl._jax.random.wrap_key_data(
            self._impl._jnp.array(key_data),
            impl=implementation,
        )


class JAXFunctionalBackend:
    def __init__(self) -> None:
        try:
            import jax
            import jax.numpy as jnp
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "JAX backend requires the 'jax' package to be installed. "
                "Install it with `pip install orng[jax]`."
            ) from exc

        self._jax = jax
        self._jnp = jnp

    def init_state(self, *, seed: int | None, generator: Any | None) -> Any:
        if generator is not None:
            return generator
        if seed is None:
            seed = secrets.randbits(32)
        return self._jax.random.key(seed)

    def integers(
        self,
        state: Any,
        *,
        low: int,
        high: int | None = None,
        size: SizeLike = None,
        dtype: Any | None = None,
    ) -> tuple[Any, Any]:
        jnp = self._jnp
        sample_dtype = jnp.dtype(dtype if dtype is not None else int)
        if sample_dtype not in (jnp.dtype(jnp.int32), jnp.dtype(jnp.int64)):
            raise TypeError("dtype must be int32 or int64.")
        canonical_dtype = self._jax.dtypes.canonicalize_dtype(sample_dtype)
        if dtype is not None and canonical_dtype != sample_dtype:
            raise ValueError("Enable JAX X64 mode to request a 64-bit dtype.")
        sample_dtype = canonical_dtype
        limits = jnp.iinfo(sample_dtype)
        low, high = integer_bounds(low, high, int(limits.min), int(limits.max))
        key, next_state = self._jax.random.split(state)
        result = self._jax.random.randint(
            key, normalize_shape(size), low, high, dtype=sample_dtype
        )
        return result, next_state

    def permutation(
        self, state: Any, x: int | Any, *, axis: int = 0
    ) -> tuple[Any, Any]:
        if isinstance(x, Integral):
            if x < 0:
                raise ValueError("x must be non-negative.")
            domain = self._jnp.arange(x)
        else:
            domain = self._jnp.asarray(x)
        axis = normalize_axis(axis, domain.ndim)
        key, next_state = self._jax.random.split(state)
        result = self._jax.random.permutation(key, domain, axis=axis)
        return result, next_state

    def exponential(
        self,
        state: Any,
        *,
        scale: Any = 1.0,
        size: SizeLike = None,
        dtype: Any | None = None,
    ) -> tuple[Any, Any]:
        jnp = self._jnp
        sample_dtype = jnp.dtype(dtype if dtype is not None else jnp.float32)
        if sample_dtype not in (
            jnp.dtype(jnp.float32),
            jnp.dtype(jnp.float64),
        ):
            raise TypeError("dtype must be float32 or float64.")
        validate_nonnegative_scalar(scale, "scale")
        canonical_dtype = self._jax.dtypes.canonicalize_dtype(sample_dtype)
        if dtype is not None and canonical_dtype != sample_dtype:
            raise ValueError("Enable JAX X64 mode to request a 64-bit dtype.")
        sample_dtype = canonical_dtype
        scales = jnp.asarray(scale, dtype=sample_dtype)
        shape = scales.shape if size is None else normalize_shape(size)
        scales = jnp.broadcast_to(scales, shape)
        key, next_state = self._jax.random.split(state)
        result = (
            self._jax.random.exponential(key, shape=shape, dtype=sample_dtype)
            * scales
        )
        return result, next_state

    def poisson(
        self,
        state: Any,
        *,
        lam: Any = 1.0,
        size: SizeLike = None,
        dtype: Any | None = None,
    ) -> tuple[Any, Any]:
        jnp = self._jnp
        sample_dtype = jnp.dtype(dtype if dtype is not None else int)
        if sample_dtype not in (jnp.dtype(jnp.int32), jnp.dtype(jnp.int64)):
            raise TypeError("dtype must be int32 or int64.")
        canonical_dtype = self._jax.dtypes.canonicalize_dtype(sample_dtype)
        if dtype is not None and canonical_dtype != sample_dtype:
            raise ValueError("Enable JAX X64 mode to request a 64-bit dtype.")
        sample_dtype = canonical_dtype
        poisson_rate_limit(lam, int(jnp.iinfo(sample_dtype).max))
        rates = jnp.asarray(lam)
        shape = rates.shape if size is None else normalize_shape(size)
        rates = jnp.broadcast_to(rates, shape)
        key, next_state = self._jax.random.split(state)
        result = self._jax.random.poisson(
            key, rates, shape=shape, dtype=sample_dtype
        )
        return result, next_state

    def random(
        self,
        state: Any,
        *,
        size: SizeLike,
        dtype: Any | None = None,
    ) -> tuple[Any, Any]:
        key, next_state = self._jax.random.split(state)
        shape = normalize_shape(size)
        sample_dtype = dtype if dtype is not None else self._jnp.float32
        low = self._jnp.array(0.0, dtype=sample_dtype)
        high = self._jnp.array(1.0, dtype=sample_dtype)
        if shape:
            result = self._jax.random.uniform(
                key,
                shape=shape,
                minval=low,
                maxval=high,
                dtype=sample_dtype,
            )
        else:
            result = self._jax.random.uniform(
                key,
                shape=(1,),
                minval=low,
                maxval=high,
                dtype=sample_dtype,
            )[0]
        return result, next_state

    def uniform(
        self,
        state: Any,
        *,
        low: Any,
        high: Any,
        size: SizeLike,
        dtype: Any | None = None,
    ) -> tuple[Any, Any]:
        key, next_state = self._jax.random.split(state)
        shape = normalize_shape(size)
        sample_dtype = dtype if dtype is not None else self._jnp.float32
        low_arr = self._jnp.asarray(low, dtype=sample_dtype)
        high_arr = self._jnp.asarray(high, dtype=sample_dtype)
        if shape:
            result = self._jax.random.uniform(
                key,
                shape=shape,
                minval=low_arr,
                maxval=high_arr,
                dtype=sample_dtype,
            )
        else:
            result = self._jax.random.uniform(
                key,
                shape=(1,),
                minval=low_arr,
                maxval=high_arr,
                dtype=sample_dtype,
            )[0]
        return result, next_state

    def normal(
        self,
        state: Any,
        *,
        loc: Any,
        scale: Any,
        size: SizeLike,
        dtype: Any | None = None,
    ) -> tuple[Any, Any]:
        key, next_state = self._jax.random.split(state)
        shape = normalize_shape(size)
        sample_dtype = dtype if dtype is not None else self._jnp.float32
        if shape:
            standard = self._jax.random.normal(
                key,
                shape=shape,
                dtype=sample_dtype,
            )
        else:
            standard = self._jax.random.normal(
                key,
                shape=(1,),
                dtype=sample_dtype,
            )[0]
        return standard * scale + loc, next_state

    def gamma(
        self,
        state: Any,
        *,
        shape: Any,
        scale: Any,
        size: SizeLike,
        dtype: Any | None = None,
    ) -> tuple[Any, Any]:
        key, next_state = self._jax.random.split(state)
        sample_shape = normalize_shape(size)
        sample_dtype = dtype if dtype is not None else self._jnp.float32
        concentration = self._jnp.asarray(shape, dtype=sample_dtype)
        scale_arr = self._jnp.asarray(scale, dtype=sample_dtype)
        if sample_shape:
            draw_shape = sample_shape + self._jnp.shape(concentration)
        else:
            draw_shape = self._jnp.shape(concentration)
        gamma_samples = self._jax.random.gamma(
            key,
            concentration,
            shape=draw_shape,
            dtype=sample_dtype,
        )
        scaled = gamma_samples * scale_arr
        return scaled, next_state

    def choice(
        self,
        state: Any,
        population: int | Any,
        *,
        size: SizeLike,
        replace: bool,
        probabilities: Any | None,
    ) -> tuple[Any, Any]:
        key, next_state = self._jax.random.split(state)
        shape = normalize_shape(size)
        jax_shape = shape if shape else None
        if isinstance(population, int):
            domain = population
        else:
            domain = self._jnp.asarray(population)
        probs = (
            None if probabilities is None else self._jnp.asarray(probabilities)
        )
        result = self._jax.random.choice(
            key,
            domain,
            shape=jax_shape,
            replace=replace,
            p=probs,
        )
        return result, next_state


__all__ = ["JAXBackend", "JAXFunctionalBackend"]
