import importlib.metadata

try:
    __version__ = importlib.metadata.version("orng")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

from .backends import create_backend_from_xp, infer_backend_name_from_xp
from .functional import (
    create_functional_backend,
    create_functional_backend_from_xp,
)
from .orng import ArrayRNG

__all__ = [
    "ArrayRNG",
    "create_backend_from_xp",
    "create_functional_backend",
    "create_functional_backend_from_xp",
    "infer_backend_name_from_xp",
    "__version__",
]
