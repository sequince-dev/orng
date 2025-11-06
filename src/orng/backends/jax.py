from __future__ import annotations

from typing import Any

from .._utils import SizeLike, normalize_shape


class JAXBackend:
    def __init__(self, *, seed: int | None, key: Any | None) -> None:
        try:
            import jax
            from jax import config as jax_config

            if not getattr(jax_config, "x64_enabled", False):
                jax_config.update("jax_enable_x64", True)

            import jax.numpy as jnp
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "JAX backend requires the 'jax' package to be installed. "
                "Install it with `pip install orng[jax]`."
            ) from exc

        self._jax = jax
        self._jnp = jnp
        if key is None:
            if seed is None:
                seed = 0
            self._key = jax.random.key(seed)
        else:
            self._key = key

    def _next_key(self) -> Any:
        key, self._key = self._jax.random.split(self._key)
        return key

    def random(self, *, size: SizeLike, dtype: Any | None) -> Any:
        key = self._next_key()
        shape = normalize_shape(size)
        dtype = dtype if dtype is not None else self._jnp.float32
        low = self._jnp.array(0.0, dtype=dtype)
        high = self._jnp.array(1.0, dtype=dtype)
        if shape:
            return self._jax.random.uniform(
                key,
                shape=shape,
                minval=low,
                maxval=high,
                dtype=dtype,
            )
        return self._jax.random.uniform(
            key,
            shape=(1,),
            minval=low,
            maxval=high,
            dtype=dtype,
        )[0]

    def uniform(
        self,
        *,
        low: Any,
        high: Any,
        size: SizeLike,
        dtype: Any | None,
    ) -> Any:
        key = self._next_key()
        shape = normalize_shape(size)
        dtype = dtype if dtype is not None else self._jnp.float32
        low_arr = self._jnp.asarray(low, dtype=dtype)
        high_arr = self._jnp.asarray(high, dtype=dtype)
        if shape:
            return self._jax.random.uniform(
                key,
                shape=shape,
                minval=low_arr,
                maxval=high_arr,
                dtype=dtype,
            )
        return self._jax.random.uniform(
            key,
            shape=(1,),
            minval=low_arr,
            maxval=high_arr,
            dtype=dtype,
        )[0]

    def normal(
        self,
        *,
        loc: Any,
        scale: Any,
        size: SizeLike,
        dtype: Any | None,
    ) -> Any:
        key = self._next_key()
        shape = normalize_shape(size)
        dtype = dtype if dtype is not None else self._jnp.float32
        if shape:
            standard = self._jax.random.normal(
                key,
                shape=shape,
                dtype=dtype,
            )
        else:
            standard = self._jax.random.normal(
                key,
                shape=(1,),
                dtype=dtype,
            )[0]
        return standard * scale + loc

    def choice(
        self,
        population: int | Any,
        *,
        size: SizeLike,
        replace: bool,
        probabilities: Any | None,
    ) -> Any:
        key = self._next_key()
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
        if not shape:
            return result
        return result


__all__ = ["JAXBackend"]
