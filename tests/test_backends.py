from __future__ import annotations

import pytest

from orng import ArrayRNG
from orng.backends.cupy import CuPyBackend
from orng.backends.jax import JAXBackend
from orng.backends.numpy import NumPyBackend
from orng.backends.torch import TorchBackend


@pytest.fixture(params=["numpy", "torch", "cupy", "jax"])
def backend_case(request):
    name = request.param

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


def test_backend_random_normal_choice(backend_case):
    seed = backend_case["seed"]
    backend = backend_case["new_backend"](seed)
    clone = backend_case["new_backend"](seed)

    rand = backend.random(size=(2, 2), dtype=backend_case["dtype64"])
    backend_case["assert_array"](rand, (2, 2), backend_case["dtype64"])

    clone_rand = clone.random(size=(2, 2), dtype=backend_case["dtype64"])
    backend_case["assert_close"](rand, clone_rand)

    uniform = backend.uniform(
        low=-1.0,
        high=1.0,
        size=(2, 2),
        dtype=backend_case["dtype32"],
    )
    backend_case["assert_array"](uniform, (2, 2), backend_case["dtype32"])
    clone_uniform = clone.uniform(
        low=-1.0,
        high=1.0,
        size=(2, 2),
        dtype=backend_case["dtype32"],
    )
    backend_case["assert_close"](uniform, clone_uniform)

    normal = backend.normal(
        loc=0.0,
        scale=1.0,
        size=3,
        dtype=backend_case["dtype32"],
    )
    backend_case["assert_array"](normal, (3,), backend_case["dtype32"])

    choice = backend.choice(
        [0, 1, 2, 3],
        size=2,
        replace=True,
        probabilities=None,
    )
    values = backend_case["to_list"](choice)
    assert len(values) == 2
    assert all(int(v) in {0, 1, 2, 3} for v in values)


def test_array_rng_integration_per_backend(backend_case):
    seed = backend_case["seed"]
    rng_a = ArrayRNG(backend=backend_case["name"], seed=seed)
    rng_b = ArrayRNG(backend=backend_case["name"], seed=seed)

    rand_a = rng_a.random(size=(2, 2), dtype=backend_case["dtype64"])
    rand_b = rng_b.random(size=(2, 2), dtype=backend_case["dtype64"])
    backend_case["assert_array"](rand_a, (2, 2), backend_case["dtype64"])
    backend_case["assert_close"](rand_a, rand_b)

    uniform = rng_a.uniform(
        low=0.0,
        high=5.0,
        size=3,
        dtype=backend_case["dtype32"],
    )
    backend_case["assert_array"](uniform, (3,), backend_case["dtype32"])

    normal = rng_a.normal(
        loc=0.0,
        scale=1.0,
        size=3,
        dtype=backend_case["dtype32"],
    )
    backend_case["assert_array"](normal, (3,), backend_case["dtype32"])

    choice = rng_a.choice(
        [0, 1, 2, 3],
        size=2,
        replace=True,
        p=None,
    )
    values = backend_case["to_list"](choice)
    assert len(values) == 2
    assert all(int(v) in {0, 1, 2, 3} for v in values)
