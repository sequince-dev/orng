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
        # CuPy's Generator lacks choice, so we implement the core semantics.
        cp = self._cupy
        gen = self._generator

        if isinstance(population, int):
            n = population
            values = None
        else:
            values = cp.asarray(population)
            n = values.shape[0]

        target_size = size if size is not None else ()

        probs = None
        if probabilities is not None:
            probs = cp.asarray(probabilities, dtype=cp.float64)
            probs = probs / cp.sum(probs)

        if replace:
            if probs is None:
                indices = gen.integers(0, n, size=target_size)
            else:
                # Inverse-CDF sampling: draw U(0,1) then locate within the CDF.
                cdf = cp.cumsum(probs)
                draws = gen.random(size=target_size)
                indices = cp.searchsorted(cdf, draws)
        else:
            flat_k = (
                1
                if target_size == ()
                else int(cp.prod(cp.asarray(target_size)))
            )
            if probs is None:
                indices = gen.permutation(n)[:flat_k]
            else:
                # Gumbel-top-k trick for weighted sampling without replacement.
                # Ref: https://arxiv.org/abs/1611.00712 (Gumbel-Softmax).
                # Inverse CDF for Gumbel: -log(-log(U)), U~Uniform(0,1)
                gumbels = -cp.log(-cp.log(gen.random(n)))
                keys = cp.log(probs) + gumbels
                # Select the top-k keys
                indices = cp.argpartition(keys, -flat_k)[-flat_k:]
            if target_size != ():
                indices = cp.reshape(indices, target_size)
            else:
                indices = indices[0]

        if values is None:
            return indices
        return values[indices]


__all__ = ["CuPyBackend"]
