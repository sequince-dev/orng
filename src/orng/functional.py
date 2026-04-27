from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Protocol, TypeAlias

from ._utils import SizeLike
from .backends import infer_backend_name_from_xp
from .backends.cupy import CuPyFunctionalBackend
from .backends.jax import JAXFunctionalBackend
from .backends.numpy import NumPyFunctionalBackend
from .backends.torch import TorchFunctionalBackend

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np
    import torch
    from jax import Array as JAXArray

    from .backends.torch import TorchFunctionalState

    NumPyGenerator: TypeAlias = np.random.Generator
    TorchGenerator: TypeAlias = torch.Generator
    CuPyGenerator: TypeAlias = cp.random.Generator
    JAXKey: TypeAlias = JAXArray
    NumPyPureState: TypeAlias = dict[str, Any]
    CuPyPureState: TypeAlias = dict[str, Any]
else:
    NumPyGenerator = Any
    TorchGenerator = Any
    CuPyGenerator = Any
    JAXKey = Any
    TorchFunctionalState = Any
    NumPyPureState = Any
    CuPyPureState = Any


FunctionalSeed: TypeAlias = int | None
FunctionalGenerator: TypeAlias = (
    NumPyGenerator | TorchGenerator | CuPyGenerator | JAXKey | None
)
FunctionalState: TypeAlias = (
    NumPyPureState
    | NumPyGenerator
    | TorchFunctionalState
    | TorchGenerator
    | CuPyPureState
    | CuPyGenerator
    | JAXKey
)


class FunctionalBackend(Protocol):
    """Protocol for ORNG's pure functional random backends.

    The ``generator`` argument to :meth:`init_state` must match the concrete
    backend:

    - NumPy: ``numpy.random.Generator``
    - PyTorch: ``torch.Generator``
    - CuPy: ``cupy.random.Generator``
    - JAX: a PRNG key array, typically from ``jax.random.key(...)``

    When ``generator`` is ``None``, a new backend-native state is created from
    ``seed``. If ``seed`` is also ``None``, a random seed is used.

    The ``state`` argument and returned next state depend on the backend and
    ``pure`` mode:

    - NumPy:
      - ``pure=True``: NumPy bit-generator state ``dict``
      - ``pure=False``: ``numpy.random.Generator``
    - PyTorch:
      - ``pure=True``: :class:`orng.backends.torch.TorchFunctionalState`
      - ``pure=False``: ``torch.Generator``
    - CuPy:
      - ``pure=True``: CuPy bit-generator state ``dict``
      - ``pure=False``: ``cupy.random.Generator``
    - JAX:
      - always: a JAX PRNG key array
    """

    def init_state(
        self,
        *,
        seed: FunctionalSeed,
        generator: FunctionalGenerator,
    ) -> FunctionalState: ...

    def random(
        self,
        state: FunctionalState,
        *,
        size: SizeLike,
        dtype: Any | None,
    ) -> tuple[FunctionalState, Any]: ...

    def uniform(
        self,
        state: FunctionalState,
        *,
        low: Any,
        high: Any,
        size: SizeLike,
        dtype: Any | None,
    ) -> tuple[FunctionalState, Any]: ...

    def normal(
        self,
        state: FunctionalState,
        *,
        loc: Any,
        scale: Any,
        size: SizeLike,
        dtype: Any | None,
    ) -> tuple[FunctionalState, Any]: ...

    def gamma(
        self,
        state: FunctionalState,
        *,
        shape: Any,
        scale: Any,
        size: SizeLike,
        dtype: Any | None,
    ) -> tuple[FunctionalState, Any]: ...

    def choice(
        self,
        state: FunctionalState,
        population: int | Any,
        *,
        size: SizeLike,
        replace: bool,
        probabilities: Any | None,
    ) -> tuple[FunctionalState, Any]: ...


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
    """Create a functional random backend by name.

    Parameters
    ----------
    name:
        Backend identifier. Supported values are ``"numpy"``, ``"torch"``,
        ``"cupy"``, and ``"jax"``.
    device:
        Optional device forwarded to backends that support it, currently
        PyTorch.
    pure:
        Whether stateful backends should expose a pure state value instead of
        mutating a backend-native generator object in place. JAX is always pure
        and rejects ``pure=False``.
    """
    normalized_name = name.lower()
    if normalized_name == "jax" and not pure:
        raise ValueError(
            "JAX functional backend is always pure; `pure=False` is not "
            "supported."
        )
    try:
        factory = _FUNCTIONAL_FACTORIES[normalized_name]
    except KeyError as exc:  # pragma: no cover - defensive
        supported = "', '".join(
            sorted({k for k in _FUNCTIONAL_FACTORIES if k != "pytorch"})
        )
        raise ValueError(
            f"Unsupported backend '{name}'. Expected one of '{supported}'."
        ) from exc
    return factory(device=device, pure=pure)


def create_functional_backend_from_xp(
    xp: Any,
    *,
    device: Any | None = None,
    pure: bool | None = None,
) -> FunctionalBackend:
    """Create a functional backend inferred from an array namespace.

    The namespace is resolved via ``array_api_compat.is_<name>_namespace``.
    When ``pure`` is omitted, ORNG defaults to pure functional state for all
    backends.
    """
    name = infer_backend_name_from_xp(xp)
    if pure is None:
        pure = True
    return create_functional_backend(name, device=device, pure=pure)


__all__ = [
    "FunctionalBackend",
    "FunctionalGenerator",
    "FunctionalSeed",
    "FunctionalState",
    "create_functional_backend_from_xp",
    "create_functional_backend",
    "infer_backend_name_from_xp",
]
