from __future__ import annotations

import pytest

from orng import ArrayRNG
from orng.backends.cupy import CuPyBackend
from orng.backends.jax import JAXBackend
from orng.backends.numpy import NumPyBackend
from orng.backends.torch import TorchBackend


@pytest.fixture(params=["numpy", "torch", "cupy", "jax"])
def backend_name(request):
    if request.param == "cupy":
        cupy = pytest.importorskip("cupy")
        try:
            cupy.cuda.runtime.getDeviceCount()
        except (ImportError, cupy.cuda.runtime.CUDARuntimeError) as exc:
            pytest.skip(f"CuPy runtime unavailable: {exc}")
    elif request.param == "jax":
        pytest.importorskip("jax")
        pytest.importorskip("jax.numpy")
    elif request.param == "torch":
        pytest.importorskip("torch")
    elif request.param == "numpy":
        pytest.importorskip("numpy")
    else:
        raise ValueError(
            f"Unexpected backend fixture parameter: {request.param}"
        )
    return request.param


@pytest.fixture()
def seed():
    return 42


@pytest.fixture()
def backend_case(backend_name):
    name = backend_name

    if name == "numpy":
        np = pytest.importorskip("numpy")

        def new_backend(seed):
            return NumPyBackend(seed=seed, generator=None)

        def assert_array(arr, shape, dtype):
            assert isinstance(arr, np.ndarray)
            assert tuple(arr.shape) == shape
            if dtype is not None:
                assert arr.dtype == dtype

        def assert_close(a, b):
            np.testing.assert_allclose(a, b)

        def to_list(arr):
            values = arr.tolist()
            return values if isinstance(values, list) else [values]

        return {
            "name": name,
            "seed": 123,
            "new_backend": new_backend,
            "assert_array": assert_array,
            "assert_close": assert_close,
            "to_list": to_list,
            "dtype64": np.float64,
            "dtype32": np.float32,
        }

    if name == "torch":
        torch = pytest.importorskip("torch")

        def new_backend(seed):
            return TorchBackend(seed=seed, generator=None, device=None)

        def assert_array(arr, shape, dtype):
            assert isinstance(arr, torch.Tensor)
            assert tuple(arr.shape) == shape
            if dtype is not None:
                assert arr.dtype == dtype

        def assert_close(a, b):
            assert torch.allclose(a, b)

        def to_list(arr):
            if isinstance(arr, torch.Tensor):
                return arr.cpu().tolist()
            return [arr]

        return {
            "name": name,
            "seed": 123,
            "new_backend": new_backend,
            "assert_array": assert_array,
            "assert_close": assert_close,
            "to_list": to_list,
            "dtype64": torch.float64,
            "dtype32": torch.float32,
        }

    if name == "cupy":
        cp = pytest.importorskip("cupy")
        try:
            cp.cuda.runtime.getDeviceCount()
        except cp.cuda.runtime.CUDARuntimeError as exc:
            pytest.skip(f"CuPy runtime unavailable: {exc}")

        def new_backend(seed):
            return CuPyBackend(seed=seed, generator=None)

        def assert_array(arr, shape, dtype):
            assert isinstance(arr, cp.ndarray)
            assert tuple(arr.shape) == shape
            if dtype is not None:
                assert arr.dtype == dtype

        def assert_close(a, b):
            cp.testing.assert_allclose(a, b)

        def to_list(arr):
            return arr.get().tolist()

        return {
            "name": name,
            "seed": 123,
            "new_backend": new_backend,
            "assert_array": assert_array,
            "assert_close": assert_close,
            "to_list": to_list,
            "dtype64": cp.float64,
            "dtype32": cp.float32,
        }

    if name == "jax":
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")
        array_type = getattr(jax, "Array", type(jnp.asarray(0)))

        def new_backend(seed):
            return JAXBackend(seed=seed, key=None)

        def assert_array(arr, shape, dtype):
            assert isinstance(arr, array_type)
            assert tuple(arr.shape) == shape
            if dtype is not None:
                assert arr.dtype == dtype

        def assert_close(a, b):
            assert bool(jnp.allclose(a, b))

        def to_list(arr):
            values = arr.tolist()
            return values if isinstance(values, list) else [values]

        return {
            "name": name,
            "seed": 123,
            "new_backend": new_backend,
            "assert_array": assert_array,
            "assert_close": assert_close,
            "to_list": to_list,
            "dtype64": jnp.float64,
            "dtype32": jnp.float32,
        }

    raise AssertionError(f"Unexpected backend fixture parameter: {name}")


@pytest.fixture(params=["random", "uniform", "normal", "gamma", "choice"])
def method_case(request):
    if request.param == "random":
        return {
            "method": "random",
            "args": {},
            "kwargs": {"size": (2, 3), "dtype": None},
        }
    elif request.param == "uniform":
        return {
            "method": "uniform",
            "args": {},
            "kwargs": {"low": 0.0, "high": 1.0, "size": (2, 3), "dtype": None},
        }
    elif request.param == "normal":
        return {
            "method": "normal",
            "args": {},
            "kwargs": {
                "loc": 0.0,
                "scale": 1.0,
                "size": (2, 3),
                "dtype": None,
            },
        }
    elif request.param == "gamma":
        return {
            "method": "gamma",
            "args": {},
            "kwargs": {
                "shape": 2.0,
                "scale": 2.0,
                "size": (2, 3),
                "dtype": None,
            },
        }
    elif request.param == "choice":
        return {
            "method": "choice",
            "args": {"a": [0, 1, 2, 3]},
            "kwargs": {"size": (2, 3), "replace": True, "probabilities": None},
        }
    else:
        raise ValueError(
            f"Unexpected method fixture parameter: {request.param}"
        )


def test_backend_seeding(backend_case, method_case):
    seed = backend_case["seed"]
    backend = backend_case["new_backend"](seed)
    clone = backend_case["new_backend"](seed)

    method = getattr(backend, method_case["method"])
    clone_method = getattr(clone, method_case["method"])

    result = method(*method_case["args"].values(), **method_case["kwargs"])
    backend_case["assert_array"](
        result,
        shape=method_case["kwargs"]["size"],
        dtype=method_case["kwargs"].get("dtype", None),
    )

    clone_result = clone_method(
        *method_case["args"].values(), **method_case["kwargs"]
    )
    backend_case["assert_close"](result, clone_result)


@pytest.fixture()
def rng(backend_name, seed):
    return ArrayRNG(backend_name, seed=seed)


def test_array_rng_random(rng):
    x = rng.random(size=(10, 2))
    assert x.shape == (10, 2)


def test_array_rng_uniform(rng):
    x = rng.uniform(low=-1.0, high=1.0, size=(5, 3))
    assert x.shape == (5, 3)


def test_array_rng_normal(rng):
    x = rng.normal(loc=0.0, scale=2.0, size=(4, 4))
    assert x.shape == (4, 4)


def test_array_rng_gamma(rng):
    x = rng.gamma(shape=2.0, scale=2.0, size=(6, 2))
    assert x.shape == (6, 2)


def test_array_rng_choice(rng):
    x = rng.choice([10, 20, 30], size=(3, 3), replace=True)
    assert x.shape == (3, 3)
    assert all(v in [10, 20, 30] for v in x.flatten())
