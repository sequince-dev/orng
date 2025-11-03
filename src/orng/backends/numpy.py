from __future__ import annotations

from typing import Any

from .._utils import SizeLike


class NumPyBackend:
    def __init__(self, *, seed: int | None, generator: Any | None) -> None:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "NumPy backend requires the 'numpy' package to be installed. "
                "Install it with `pip install orng[numpy]`."
            ) from exc

        if generator is None:
            self._generator = np.random.default_rng(seed)
        elif isinstance(generator, np.random.Generator):
            self._generator = generator
        else:
            raise TypeError(
                "generator must be a numpy.random.Generator when using the "
                "NumPy backend."
            )

    def random(self, *, size: SizeLike, dtype: Any | None) -> Any:
        return self._generator.random(size=size, dtype=dtype)

    def normal(
        self,
        *,
        loc: Any,
        scale: Any,
        size: SizeLike,
        dtype: Any | None,
    ) -> Any:
        result = self._generator.normal(
            loc=loc,
            scale=scale,
            size=size,
        )
        if dtype is not None:
            import numpy as np

            return np.asarray(result, dtype=dtype)
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


__all__ = ["NumPyBackend"]
