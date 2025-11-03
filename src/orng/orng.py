"""Backend-aware random number generation helpers.

This module introduces :class:`ArrayRNG`, a small facade that mimics the subset
of ``numpy.random.Generator`` APIs. The class presents a uniform interface
across NumPy, PyTorch, CuPy, and JAX.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ._utils import SizeLike
from .backends import create_backend


class RNGBackend(Protocol):
    """Protocol representing the shim each backend must implement."""

    def random(self, *, size: SizeLike, dtype: Any | None) -> Any: ...

    def normal(
        self,
        *,
        loc: Any,
        scale: Any,
        size: SizeLike,
        dtype: Any | None,
    ) -> Any: ...

    def choice(
        self,
        population: int | Any,
        *,
        size: SizeLike,
        replace: bool,
        probabilities: Any | None,
    ) -> Any: ...


@dataclass
class ArrayRNG:
    """Facade exposing ``numpy.random.Generator``-style helpers across backends.

    Parameters
    ----------
    backend:
        String identifier for the target library.  Accepted values are
        ``"numpy"``, ``"torch"``/``"pytorch"``, ``"cupy"``, and ``"jax"``.
    seed:
        Optional seed used when constructing a fresh generator.
    generator:
        Optional backend-specific state to wrap.  The expected value depends on
        the backend:

        * NumPy – ``numpy.random.Generator``.
        * PyTorch – ``torch.Generator``.
        * CuPy – ``cupy.random.Generator``.
        * JAX – a ``jax.random.KeyArray`` generated via ``jax.random.key``.

        When supplied, the instance is wrapped instead of creating a new
        generator from ``seed``.
    device:
        Optional device specification forwarded to backends that understand the
        concept (currently PyTorch).
    """

    backend: str = "numpy"
    seed: int | None = None
    generator: Any | None = None
    device: Any | None = None

    def __post_init__(self) -> None:
        self._impl: RNGBackend = create_backend(
            self.backend,
            seed=self.seed,
            generator=self.generator,
            device=self.device,
        )

    # Public API -----------------------------------------------------------------

    def random(
        self,
        size: SizeLike = None,
        *,
        dtype: Any | None = None,
    ) -> Any:
        """Draw samples from ``Uniform[0, 1)``."""
        return self._impl.random(size=size, dtype=dtype)

    def normal(
        self,
        loc: Any = 0.0,
        scale: Any = 1.0,
        size: SizeLike = None,
        *,
        dtype: Any | None = None,
    ) -> Any:
        """Draw samples from a normal distribution."""
        return self._impl.normal(
            loc=loc,
            scale=scale,
            size=size,
            dtype=dtype,
        )

    def choice(
        self,
        a: int | Any,
        size: SizeLike = None,
        replace: bool = True,
        p: Any | None = None,
    ) -> Any:
        """Sample from a discrete domain."""
        return self._impl.choice(
            a,
            size=size,
            replace=replace,
            probabilities=p,
        )


__all__ = ["ArrayRNG"]
