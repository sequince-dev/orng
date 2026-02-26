from __future__ import annotations

from typing import Any, Callable, Dict

from .cupy import CuPyBackend
from .jax import JAXBackend
from .numpy import NumPyBackend
from .torch import TorchBackend

BackendFactory = Callable[..., Any]

_FACTORIES: Dict[str, BackendFactory] = {
    "numpy": lambda *, seed=None, generator=None, device=None: NumPyBackend(
        seed=seed,
        generator=generator,
    ),
    "cupy": lambda *, seed=None, generator=None, device=None: CuPyBackend(
        seed=seed,
        generator=generator,
    ),
    "torch": lambda *, seed=None, generator=None, device=None: TorchBackend(
        seed=seed,
        generator=generator,
        device=device,
    ),
    "pytorch": lambda *, seed=None, generator=None, device=None: TorchBackend(
        seed=seed,
        generator=generator,
        device=device,
    ),
    "jax": lambda *, seed=None, generator=None, device=None: JAXBackend(
        seed=seed,
        key=generator,
    ),
}


def create_backend(
    name: str,
    *,
    seed: int | None,
    generator: Any | None,
    device: Any | None,
):
    try:
        factory = _FACTORIES[name.lower()]
    except KeyError as exc:  # pragma: no cover - defensive
        supported = "', '".join(
            sorted({k for k in _FACTORIES if k != "pytorch"})
        )
        raise ValueError(
            f"Unsupported backend '{name}'. Expected one of '{supported}'."
        ) from exc
    return factory(seed=seed, generator=generator, device=device)


__all__ = ["create_backend"]
