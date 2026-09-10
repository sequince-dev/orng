import sys
from types import SimpleNamespace

import numpy as np
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--fake-cupy",
        action="store_true",
        help="Run CuPy backend cases with NumPy and the installed CuPy API",
    )


@pytest.fixture
def cupy_generator_api():
    """Inspect the installed class without creating any CUDA objects."""
    cp = pytest.importorskip("cupy")
    methods = frozenset(
        name
        for name in dir(cp.random.Generator)
        if not name.startswith("_")
        and callable(getattr(cp.random.Generator, name))
    )
    return cp.__version__, methods


@pytest.fixture
def numpy_cupy(monkeypatch, cupy_generator_api):
    """NumPy execution restricted to the installed CuPy Generator API.

    This checks method availability, not CUDA behaviour or every signature.
    Bit-generator state is NumPy state, solely for exercising ORNG's plumbing.
    """
    version, methods = cupy_generator_api

    class Generator:
        def __init__(self, seed=None):
            self._rng = np.random.default_rng(seed)

        @property
        def bit_generator(self):
            return self._rng.bit_generator

        def __getattr__(self, name):
            if name not in methods:
                raise AttributeError(
                    f"CuPy {version} Generator has no method {name!r}"
                )
            method = getattr(self._rng, name)
            if name in {"uniform", "beta"}:
                # CuPy accepts dtype here; NumPy does not.
                def with_dtype(*args, dtype=np.float64, **kwargs):
                    return np.asarray(method(*args, **kwargs), dtype=dtype)

                return with_dtype
            if name in {
                "random",
                "standard_normal",
                "standard_gamma",
                "standard_exponential",
            }:

                def with_default_dtype(*args, **kwargs):
                    if kwargs.get("dtype", np.float64) is None:
                        kwargs["dtype"] = np.float64
                    return method(*args, **kwargs)

                return with_default_dtype
            return method

    # Only the generator API is restricted; array operations use NumPy.
    namespace = SimpleNamespace(
        **{name: getattr(np, name) for name in dir(np) if name != "random"},
        random=SimpleNamespace(
            Generator=Generator,
            default_rng=Generator,
            PCG64=np.random.PCG64,
        ),
        asnumpy=np.asarray,
    )
    monkeypatch.setitem(sys.modules, "cupy", namespace)
    return namespace
