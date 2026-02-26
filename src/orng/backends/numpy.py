from __future__ import annotations

import copy
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

    def uniform(
        self,
        *,
        low: Any,
        high: Any,
        size: SizeLike,
        dtype: Any | None,
    ) -> Any:
        result = self._generator.uniform(
            low=low,
            high=high,
            size=size,
        )
        if dtype is not None:
            import numpy as np

            return np.asarray(result, dtype=dtype)
        return result

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


class NumPyFunctionalBackend:
    def __init__(self) -> None:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "NumPy backend requires the 'numpy' package to be installed. "
                "Install it with `pip install orng[numpy]`."
            ) from exc
        self._np = np

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
        return copy.deepcopy(gen.bit_generator.state)

    def _generator_from_state(self, state: Any) -> Any:
        gen = self._np.random.default_rng()
        gen.bit_generator.state = copy.deepcopy(state)
        return gen

    def _next_state_and_result(self, gen: Any, result: Any) -> tuple[Any, Any]:
        return copy.deepcopy(gen.bit_generator.state), result

    def random(
        self,
        state: Any,
        *,
        size: SizeLike,
        dtype: Any | None,
    ) -> tuple[Any, Any]:
        gen = self._generator_from_state(state)
        result = gen.random(size=size, dtype=dtype)
        return self._next_state_and_result(gen, result)

    def uniform(
        self,
        state: Any,
        *,
        low: Any,
        high: Any,
        size: SizeLike,
        dtype: Any | None,
    ) -> tuple[Any, Any]:
        gen = self._generator_from_state(state)
        result = gen.uniform(low=low, high=high, size=size)
        if dtype is not None:
            result = self._np.asarray(result, dtype=dtype)
        return self._next_state_and_result(gen, result)

    def normal(
        self,
        state: Any,
        *,
        loc: Any,
        scale: Any,
        size: SizeLike,
        dtype: Any | None,
    ) -> tuple[Any, Any]:
        gen = self._generator_from_state(state)
        result = gen.normal(loc=loc, scale=scale, size=size)
        if dtype is not None:
            result = self._np.asarray(result, dtype=dtype)
        return self._next_state_and_result(gen, result)

    def gamma(
        self,
        state: Any,
        *,
        shape: Any,
        scale: Any,
        size: SizeLike,
        dtype: Any | None,
    ) -> tuple[Any, Any]:
        gen = self._generator_from_state(state)
        result = gen.gamma(shape=shape, scale=scale, size=size)
        if dtype is not None:
            result = self._np.asarray(result, dtype=dtype)
        return self._next_state_and_result(gen, result)

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
        return self._next_state_and_result(gen, result)


__all__ = ["NumPyBackend", "NumPyFunctionalBackend"]
