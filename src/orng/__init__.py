import importlib.metadata

try:
    __version__ = importlib.metadata.version("orng")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

from .orng import ArrayRNG

__all__ = ["ArrayRNG", "__version__"]
