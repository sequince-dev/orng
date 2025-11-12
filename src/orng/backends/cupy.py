from __future__ import annotations

from typing import Any

from .._utils import SizeLike


class CuPyBackend:
    def __init__(self, *, seed: int | None, generator: Any | None) -> None:
        try:
            import cupy as cp
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "CuPy backend requires the 'cupy' package to be installed. "
                "Install it with `pip install orng[cupy]`."
            ) from exc

        self._cupy = cp
        if generator is None:
            self._generator = cp.random.default_rng(seed)
        elif isinstance(generator, cp.random.Generator):
            self._generator = generator
        else:
            raise TypeError(
                "generator must be a cupy.random.Generator when using the "
                "CuPy backend."
            )

    def random(self, *, size: SizeLike, dtype: Any | None) -> Any:
        return self._generator.random(size=size, dtype=dtype)

    def uniform(
        self,
        *,
        low: Any,
        high: Any,
        size: SizeLike,
        dtype: Any | None,
    ) -> Any:
        return self._generator.uniform(
            low=low,
            high=high,
            size=size,
            dtype=dtype,
        )

    def normal(
        self,
        *,
        loc: Any,
        scale: Any,
        size: SizeLike,
        dtype: Any | None,
    ) -> Any:
        standard = self._generator.standard_normal(
            size=size,
            dtype=dtype,
        )
        result = loc + standard * scale
        return result

    def gamma(
        self,
        *,
        shape: Any,
        scale: Any,
        size: SizeLike,
        dtype: Any | None,
    ) -> Any:
        result = self._generator.gamma(
            shape=shape,
            scale=scale,
            size=size,
        )
        if dtype is not None:
            return self._cupy.asarray(result, dtype=dtype)
        return result

    def choice(
        self,
        population: int | Any,
        *,
        size: SizeLike,
        replace: bool,
        probabilities: Any | None,
    ) -> Any:
        return self._generator.choice(
            population,
            size=size,
            replace=replace,
            p=probabilities,
        )


__all__ = ["CuPyBackend"]
