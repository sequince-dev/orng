from __future__ import annotations

from typing import Any, Callable, Dict, Protocol

from ._utils import SizeLike
from .backends.cupy import CuPyFunctionalBackend
from .backends.jax import JAXFunctionalBackend
from .backends.numpy import NumPyFunctionalBackend
from .backends.torch import TorchFunctionalBackend


class FunctionalBackend(Protocol):
    def init_state(
        self,
        *,
        seed: int | None,
        generator: Any | None,
    ) -> Any: ...

    def random(
        self,
        state: Any,
        *,
        size: SizeLike,
        dtype: Any | None,
    ) -> tuple[Any, Any]: ...

    def uniform(
        self,
        state: Any,
        *,
        low: Any,
        high: Any,
        size: SizeLike,
        dtype: Any | None,
    ) -> tuple[Any, Any]: ...

    def normal(
        self,
        state: Any,
        *,
        loc: Any,
        scale: Any,
        size: SizeLike,
        dtype: Any | None,
    ) -> tuple[Any, Any]: ...

    def gamma(
        self,
        state: Any,
        *,
        shape: Any,
        scale: Any,
        size: SizeLike,
        dtype: Any | None,
    ) -> tuple[Any, Any]: ...

    def choice(
        self,
        state: Any,
        population: int | Any,
        *,
        size: SizeLike,
        replace: bool,
        probabilities: Any | None,
    ) -> tuple[Any, Any]: ...


FunctionalFactory = Callable[..., FunctionalBackend]


_FUNCTIONAL_FACTORIES: Dict[str, FunctionalFactory] = {
    "numpy": lambda *, device=None, pure=True: NumPyFunctionalBackend(
        pure=pure
    ),
    "cupy": lambda *, device=None, pure=True: CuPyFunctionalBackend(pure=pure),
    "torch": lambda *, device=None, pure=True: TorchFunctionalBackend(
        device=device,
        pure=pure,
    ),
    "pytorch": lambda *, device=None, pure=True: TorchFunctionalBackend(
        device=device,
        pure=pure,
    ),
    "jax": lambda *, device=None, pure=True: JAXFunctionalBackend(),
}


def create_functional_backend(
    name: str,
    *,
    device: Any | None = None,
    pure: bool = True,
) -> FunctionalBackend:
    try:
        factory = _FUNCTIONAL_FACTORIES[name.lower()]
    except KeyError as exc:  # pragma: no cover - defensive
        supported = "', '".join(
            sorted({k for k in _FUNCTIONAL_FACTORIES if k != "pytorch"})
        )
        raise ValueError(
            f"Unsupported backend '{name}'. Expected one of '{supported}'."
        ) from exc
    return factory(device=device, pure=pure)


__all__ = [
    "FunctionalBackend",
    "create_functional_backend",
]
