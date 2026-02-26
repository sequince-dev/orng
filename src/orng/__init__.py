import importlib.metadata

try:
    __version__ = importlib.metadata.version("orng")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

from .functional import create_functional_backend
from .orng import ArrayRNG

__all__ = ["ArrayRNG", "create_functional_backend", "__version__"]
