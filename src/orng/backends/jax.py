from __future__ import annotations

import secrets
from typing import Any

from .._utils import SizeLike, normalize_shape


class JAXBackend:
    def __init__(self, *, seed: int | None, key: Any | None) -> None:
        self._impl = JAXFunctionalBackend()
        self._state = self._impl.init_state(seed=seed, generator=key)

    def random(self, *, size: SizeLike, dtype: Any | None) -> Any:
        self._state, result = self._impl.random(
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
        self._state, result = self._impl.uniform(
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
        self._state, result = self._impl.normal(
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
        self._state, result = self._impl.gamma(
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
        self._state, result = self._impl.choice(
            self._state,
            population,
            size=size,
            replace=replace,
            probabilities=probabilities,
        )
        return result


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

    def random(
        self,
        state: Any,
        *,
        size: SizeLike,
        dtype: Any | None,
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
        return next_state, result

    def uniform(
        self,
        state: Any,
        *,
        low: Any,
        high: Any,
        size: SizeLike,
        dtype: Any | None,
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
        return next_state, result

    def normal(
        self,
        state: Any,
        *,
        loc: Any,
        scale: Any,
        size: SizeLike,
        dtype: Any | None,
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
        return next_state, standard * scale + loc

    def gamma(
        self,
        state: Any,
        *,
        shape: Any,
        scale: Any,
        size: SizeLike,
        dtype: Any | None,
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
        return next_state, scaled

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
        return next_state, result


__all__ = ["JAXBackend", "JAXFunctionalBackend"]
