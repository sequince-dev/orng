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


def infer_backend_name_from_xp(xp: Any) -> str:
    try:
        from array_api_compat import (
            is_cupy_namespace,
            is_jax_namespace,
            is_numpy_namespace,
            is_torch_namespace,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Inferring ORNG backends from an array namespace requires "
            "'array_api_compat'."
        ) from exc

    if is_numpy_namespace(xp):
        return "numpy"
    if is_jax_namespace(xp):
        return "jax"
    if is_torch_namespace(xp):
        return "torch"
    if is_cupy_namespace(xp):
        return "cupy"
    raise ValueError("Unsupported array namespace for ORNG backend inference.")


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


def create_backend_from_xp(
    xp: Any,
    *,
    seed: int | None,
    generator: Any | None,
    device: Any | None,
):
    return create_backend(
        infer_backend_name_from_xp(xp),
        seed=seed,
        generator=generator,
        device=device,
    )


__all__ = [
    "create_backend",
    "create_backend_from_xp",
    "infer_backend_name_from_xp",
]
