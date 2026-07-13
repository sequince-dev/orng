"""Backend-aware random number generation helpers.

This module introduces :class:`RandomGenerator`, a small facade that mimics the subset
of ``numpy.random.Generator`` APIs. The class presents a uniform interface
across NumPy, PyTorch, CuPy, and JAX.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from ._utils import SizeLike
from .backends import create_backend, infer_backend_name_from_xp


class RNGBackend(Protocol):
    """Protocol representing the shim each backend must implement."""

    _state: Any

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...

    def random(self, *, size: SizeLike, dtype: Any | None) -> Any: ...

    def uniform(
        self,
        *,
        low: Any,
        high: Any,
        size: SizeLike,
        dtype: Any | None,
    ) -> Any: ...

    def normal(
        self,
        *,
        loc: Any,
        scale: Any,
        size: SizeLike,
        dtype: Any | None,
    ) -> Any: ...

    def gamma(
        self,
        *,
        shape: Any,
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
class RandomGenerator:
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

    @classmethod
    def from_xp(
        cls,
        xp: Any,
        *,
        seed: int | None = None,
        generator: Any | None = None,
        device: Any | None = None,
    ) -> "RandomGenerator":
        return cls(
            backend=infer_backend_name_from_xp(xp),
            seed=seed,
            generator=generator,
            device=device,
        )

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

    def uniform(
        self,
        low: Any = 0.0,
        high: Any = 1.0,
        size: SizeLike = None,
        *,
        dtype: Any | None = None,
    ) -> Any:
        """Draw samples from ``Uniform[low, high)``."""
        return self._impl.uniform(
            low=low,
            high=high,
            size=size,
            dtype=dtype,
        )

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

    def gamma(
        self,
        shape: Any,
        scale: Any = 1.0,
        size: SizeLike = None,
        *,
        dtype: Any | None = None,
    ) -> Any:
        """Draw samples from a gamma distribution."""
        return self._impl.gamma(
            shape=shape,
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

    def to_functional(
        self,
        *,
        pure: bool | None = None,
    ) -> tuple[Any, Any]:
        """Return the matching functional backend and current backend state."""
        from .functional import create_functional_backend

        if pure is None:
            pure = self.backend == "jax"
        backend = create_functional_backend(
            self.backend,
            pure=pure,
        )
        return backend, self._impl._state

    def state_dict(self) -> dict[str, Any]:
        """Return a detached, versioned snapshot of the generator state."""
        backend = self.backend.lower()
        if backend == "pytorch":
            backend = "torch"
        return {
            "version": 1,
            "backend": backend,
            "state": self._impl.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore this generator from :meth:`state_dict` output."""
        _, backend, backend_state = _validate_state_dict(state)
        current_backend = self.backend.lower()
        if current_backend == "pytorch":
            current_backend = "torch"
        if backend != current_backend:
            raise ValueError(
                f"Cannot load '{backend}' RNG state into a "
                f"'{current_backend}' generator."
            )
        self._impl.load_state_dict(backend_state)
        if backend == "torch":
            self.device = backend_state.get("device")

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "RandomGenerator":
        """Construct a generator from :meth:`state_dict` output."""
        _, backend, backend_state = _validate_state_dict(state)
        device = backend_state.get("device") if backend == "torch" else None
        rng = cls(backend=backend, device=device)
        rng._impl.load_state_dict(backend_state)
        return rng


class ArrayRNG(RandomGenerator):
    """Deprecated alias for :class:`RandomGenerator`."""

    def __post_init__(self) -> None:
        import warnings

        warnings.warn(
            "ArrayRNG is deprecated and will be removed in a future release. "
            "Please use orng.RandomGenerator instead.",
            FutureWarning,
        )
        super().__post_init__()


def _validate_state_dict(
    state: Mapping[str, Any],
) -> tuple[int, str, Mapping[str, Any]]:
    if not isinstance(state, Mapping):
        raise TypeError("ORNG state must be a mapping.")
    missing = {
        key for key in ("version", "backend", "state") if key not in state
    }
    if missing:
        raise ValueError(
            "ORNG state is missing required keys: "
            + ", ".join(sorted(missing))
        )
    version = state["version"]
    if version != 1:
        raise ValueError(f"Unsupported ORNG state version {version!r}.")
    backend = state["backend"]
    if not isinstance(backend, str):
        raise TypeError("ORNG state 'backend' must be a string.")
    backend = backend.lower()
    if backend == "pytorch":
        backend = "torch"
    backend_state = state["state"]
    if not isinstance(backend_state, Mapping):
        raise TypeError("ORNG backend state must be a mapping.")
    return version, backend, backend_state


__all__ = ["ArrayRNG", "RandomGenerator"]
