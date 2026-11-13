from __future__ import annotations

from importlib import import_module
from typing import Any

import pytest

from orng import RandomGenerator
from orng.backends.cupy import CuPyBackend
from orng.backends.jax import JAXBackend
from orng.backends.numpy import NumPyBackend
from orng.backends.torch import TorchBackend


@pytest.fixture(params=["numpy", "torch", "cupy", "jax"])
def backend_name(request):
    if request.param == "cupy":
        if request.config.getoption("--fake-cupy"):
            request.getfixturevalue("numpy_cupy")
            return "cupy"
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
            return cp.asnumpy(arr).tolist()

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
    return RandomGenerator(backend_name, seed=seed)


def test_array_rng_random(rng):
    x = rng.random(size=(10, 2))
    assert x.shape == (10, 2)


def test_array_rng_uniform(rng):
    x = rng.uniform(low=-1.0, high=1.0, size=(5, 3))
    assert x.shape == (5, 3)


def test_array_rng_normal(rng):
    x = rng.normal(loc=0.0, scale=2.0, size=(4, 4))
    assert x.shape == (4, 4)


def test_random_generator_state_round_trip(rng, backend_case):
    rng.normal(size=(3, 2))
    state = rng.state_dict()

    expected = rng.normal(size=(4, 3))
    restored = RandomGenerator.from_state_dict(state)
    actual = restored.normal(size=(4, 3))

    backend_case["assert_close"](actual, expected)


def test_random_generator_load_state_dict(rng, backend_case):
    rng.uniform(size=(5,))
    state = rng.state_dict()
    expected = rng.uniform(size=(5,))

    rng.uniform(size=(5,))
    rng.load_state_dict(state)
    actual = rng.uniform(size=(5,))

    backend_case["assert_close"](actual, expected)


def test_random_generator_set_state(rng, backend_case):
    backend_name = backend_case["name"]
    source = RandomGenerator(backend_name, seed=987)
    expected_source = RandomGenerator(backend_name, seed=987)
    state = source.state()

    rng.set_state(state)
    assert rng.state() is state
    actual = rng.normal(size=(4, 3))
    expected = expected_source.normal(size=(4, 3))

    backend_case["assert_close"](actual, expected)


def test_array_rng_gamma(rng):
    x = rng.gamma(shape=2.0, scale=2.0, size=(6, 2))
    assert x.shape == (6, 2)

    x = rng.gamma(shape=[1.0, 2.0, 3.0], scale=1.0)
    assert x.shape == (3,)

    x = rng.gamma(shape=2.0, scale=[1.0, 2.0, 3.0])
    assert x.shape == (3,)


def test_array_rng_choice(rng):
    x = rng.choice([10, 20, 30], size=(3, 3), replace=True)
    assert x.shape == (3, 3)
    assert all(v in [10, 20, 30] for v in x.flatten())


def test_choice_with_probabilities_and_no_replacement(backend_case):
    backend = backend_case["new_backend"](123)
    values = [1, 2, 3, 4]
    probs = [0.7, 0.1, 0.1, 0.1]

    draws = backend.choice(
        values,
        size=(2, 3),
        replace=True,
        probabilities=probs,
    )
    backend_case["assert_array"](draws, shape=(2, 3), dtype=None)
    assert all(
        value in values for value in backend_case["to_list"](draws.reshape(-1))
    )

    without_replacement = backend.choice(
        values,
        size=(3,),
        replace=False,
        probabilities=None,
    )
    backend_case["assert_array"](without_replacement, shape=(3,), dtype=None)
    assert len(set(backend_case["to_list"](without_replacement))) == 3


@pytest.fixture
def sampling_namespace(backend_name: str) -> Any:
    return import_module(
        "jax.numpy" if backend_name == "jax" else backend_name
    )


@pytest.fixture(
    params=[
        ("integers", (), {"low": -3, "high": 8, "size": (4, 5)}),
        ("permutation", (40,), {}),
        ("exponential", (), {"scale": 2.0, "size": (4, 5)}),
        ("poisson", (), {"lam": 3.0, "size": (4, 5)}),
    ],
    ids=["integers", "permutation", "exponential", "poisson"],
)
def new_sampling_method(request: Any) -> tuple[str, tuple, dict]:
    return request.param


def _host_array(value: Any, name: str) -> Any:
    np = import_module("numpy")
    if name == "torch":
        return value.cpu().numpy()
    if name == "cupy":
        return import_module("cupy").asnumpy(value)
    return np.asarray(value)


def test_new_methods_checkpoint_and_state_advancement(
    backend_name: str, new_sampling_method: tuple[str, tuple, dict]
) -> None:
    np = import_module("numpy")
    method, args, kwargs = new_sampling_method
    rng = RandomGenerator(backend_name, seed=723)
    snapshot = rng.state_dict()
    first = getattr(rng, method)(*args, **kwargs)
    second = getattr(rng, method)(*args, **kwargs)
    assert not np.array_equal(
        _host_array(first, backend_name), _host_array(second, backend_name)
    )
    restored = RandomGenerator.from_state_dict(snapshot)
    for expected in (first, second):
        actual = getattr(restored, method)(*args, **kwargs)
        np.testing.assert_array_equal(
            _host_array(actual, backend_name),
            _host_array(expected, backend_name),
        )
    np.testing.assert_array_equal(
        _host_array(rng.random(size=10), backend_name),
        _host_array(restored.random(size=10), backend_name),
    )


@pytest.mark.parametrize("pure", [True, False])
def test_new_functional_methods_thread_the_managed_state(
    backend_name: str,
    new_sampling_method: tuple[str, tuple, dict],
    pure: bool,
) -> None:
    from orng.functional import create_functional_backend

    if backend_name == "jax" and not pure:
        pytest.skip("JAX has no mutable functional mode.")
    np = import_module("numpy")
    method, args, kwargs = new_sampling_method
    backend = create_functional_backend(backend_name, pure=pure)
    state = backend.init_state(seed=723, generator=None)
    call = getattr(backend, method)
    first, next_state = call(state, *args, **kwargs)
    if pure:
        replay, _ = call(state, *args, **kwargs)
        np.testing.assert_array_equal(
            _host_array(first, backend_name), _host_array(replay, backend_name)
        )
    else:
        assert next_state is state
    second, next_state = call(next_state, *args, **kwargs)
    assert not np.array_equal(
        _host_array(first, backend_name), _host_array(second, backend_name)
    )
    # The mutable object API and the functional API must use the same stream.
    rng = RandomGenerator(backend_name, seed=723)
    for expected in (first, second):
        actual = getattr(rng, method)(*args, **kwargs)
        np.testing.assert_array_equal(
            _host_array(actual, backend_name),
            _host_array(expected, backend_name),
        )
    following, _ = backend.random(next_state, size=10)
    np.testing.assert_array_equal(
        _host_array(following, backend_name),
        _host_array(rng.random(size=10), backend_name),
    )


@pytest.mark.parametrize("size", [None, (), 0, (2, 0), (3, 4)])
def test_integers_shape_bounds_and_dtype(
    rng: RandomGenerator, sampling_namespace: Any, size: Any
) -> None:
    xp = sampling_namespace
    result = rng.integers(-2, 3, size=size, dtype=xp.int32)
    shape = (
        () if size is None else ((size,) if isinstance(size, int) else size)
    )
    assert result.shape == shape
    assert result.dtype == xp.int32
    assert bool(xp.all((result >= -2) & (result < 3)))
    assert int(rng.integers(1)) == 0


@pytest.mark.parametrize("low,high", [(0, None), (3, 3), (4, 2)])
def test_integers_rejects_empty_intervals(
    rng: RandomGenerator, low: int, high: int | None
) -> None:
    with pytest.raises(ValueError, match="low must be less"):
        rng.integers(low, high)


def test_integers_rejects_noninteger_and_overflowing_bounds(
    rng: RandomGenerator, sampling_namespace: Any
) -> None:
    with pytest.raises(TypeError):
        invalid_low: Any = 1.5
        rng.integers(invalid_low, 4)
    with pytest.raises(ValueError, match="bounds must fit"):
        rng.integers(0, 2**32, dtype=sampling_namespace.int32)


@pytest.mark.parametrize(
    "method,param", [("exponential", "scale"), ("poisson", "lam")]
)
def test_new_distributions_broadcast_to_output_shape(
    rng: RandomGenerator, sampling_namespace: Any, method: str, param: str
) -> None:
    xp = sampling_namespace
    dtype = xp.float32 if method == "exponential" else xp.int32
    call = getattr(rng, method)
    values = (
        xp.asarray([0.0, 2.0])
        if rng.backend != "torch"
        else xp.tensor([0.0, 2.0])
    )
    result = call(**{param: values}, size=(5, 2), dtype=dtype)
    assert result.shape == (5, 2)
    assert result.dtype == dtype
    assert bool(xp.all(result >= 0))
    assert bool(xp.all(result[:, 0] == 0))
    assert call(**{param: values}).shape == (2,)
    assert call(**{param: 2.0}).shape == ()
    assert call(**{param: 2.0}, size=()).shape == ()
    assert call(**{param: values}, size=(0, 2)).shape == (0, 2)
    with pytest.raises((ValueError, RuntimeError)):
        call(**{param: values}, size=(3,))
    with pytest.raises((ValueError, RuntimeError)):
        call(**{param: values}, size=())


@pytest.mark.parametrize(
    "method,param", [("exponential", "scale"), ("poisson", "lam")]
)
@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_new_distributions_reject_invalid_scalar_parameters(
    rng: RandomGenerator, method: str, param: str, value: float
) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        getattr(rng, method)(**{param: value})


@pytest.mark.parametrize("method", ["integers", "poisson", "exponential"])
def test_new_distributions_reject_invalid_dtypes(
    rng: RandomGenerator, sampling_namespace: Any, method: str
) -> None:
    kwargs = {"low": 3} if method == "integers" else {}
    with pytest.raises(TypeError, match="dtype must be"):
        getattr(rng, method)(**kwargs, dtype=bool)


def test_permutation_preserves_slices_and_input(
    rng: RandomGenerator, sampling_namespace: Any
) -> None:
    xp = sampling_namespace
    np = import_module("numpy")
    original = xp.arange(24).reshape(2, 3, 4)
    shuffled = rng.permutation(original, axis=-2)
    assert shuffled.shape == original.shape
    assert shuffled.dtype == original.dtype
    np.testing.assert_array_equal(
        _host_array(original, rng.backend), np.arange(24).reshape(2, 3, 4)
    )
    # Each column determines the same row permutation, across both batches.
    host = _host_array(shuffled, rng.backend)
    order = host[0, :, 0] // 4
    np.testing.assert_array_equal(
        host, np.arange(24).reshape(2, 3, 4)[:, order, :]
    )
    assert sorted(_host_array(rng.permutation(30), rng.backend)) == list(
        range(30)
    )
    assert rng.permutation(0).shape == (0,)
    assert rng.permutation(original, axis=0).shape == original.shape
    with pytest.raises(ValueError, match="non-negative"):
        rng.permutation(-1)
    with pytest.raises(ValueError, match="axis"):
        rng.permutation(original, axis=3)
    with pytest.raises(ValueError, match="axis"):
        rng.permutation(
            xp.asarray(1) if rng.backend != "torch" else xp.tensor(1)
        )


def test_new_distribution_moments(rng: RandomGenerator) -> None:
    np = import_module("numpy")
    n = 30000
    integers = _host_array(rng.integers(-2, 4, size=n), rng.backend)
    assert abs(integers.mean() - 0.5) < 0.08
    counts = np.bincount(integers + 2, minlength=6)
    assert np.all(np.abs(counts - n / 6) < 400)
    exponential = _host_array(rng.exponential(scale=2.0, size=n), rng.backend)
    # About six standard errors for the mean and a conservative variance bound.
    assert abs(exponential.mean() - 2.0) < 0.08
    assert abs(exponential.var() - 4.0) < 0.4
    poisson = _host_array(rng.poisson(lam=4.0, size=n), rng.backend)
    assert abs(poisson.mean() - 4.0) < 0.08
    assert abs(poisson.var() - 4.0) < 0.3


def test_jax_new_methods_compile_with_dynamic_parameters() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from orng.functional import create_functional_backend

    backend = create_functional_backend("jax")
    state = backend.init_state(seed=987, generator=None)

    def step(key: Any, scale: Any, lam: Any, values: Any) -> tuple:
        integers, key = backend.integers(key, low=-2, high=7, size=(3, 2))
        exponential, key = backend.exponential(key, scale=scale, size=(3, 2))
        poisson, key = backend.poisson(key, lam=lam, size=(3, 2))
        shuffled, key = backend.permutation(key, values, axis=1)
        return integers, exponential, poisson, shuffled, key

    parameters = (
        jnp.array([0.0, 2.0]),
        jnp.array([0.0, 4.0]),
        jnp.arange(6).reshape(2, 3),
    )
    expected = step(state, *parameters)
    actual = jax.jit(step)(state, *parameters)
    for eager, compiled in zip(expected[:-1], actual[:-1]):
        assert bool(jnp.allclose(eager, compiled, rtol=1e-6, atol=1e-6))
    assert bool(
        jnp.array_equal(
            jax.random.key_data(expected[-1]), jax.random.key_data(actual[-1])
        )
    )
    assert bool(jnp.all(actual[1][:, 0] == 0))
    assert bool(jnp.all(actual[2][:, 0] == 0))


@pytest.mark.parametrize("enabled", [False, True])
def test_jax_new_methods_respect_x64_configuration(enabled: bool) -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", enabled)
    try:
        rng = RandomGenerator("jax", seed=5)
        if not enabled:
            for method, kwargs in [
                ("integers", {"low": 5}),
                ("poisson", {}),
                ("exponential", {}),
            ]:
                dtype = jnp.float64 if method == "exponential" else jnp.int64
                with pytest.raises(ValueError, match="X64"):
                    getattr(rng, method)(**kwargs, dtype=dtype)
        else:
            assert rng.integers(5, size=3, dtype=jnp.int64).dtype == jnp.int64
            assert rng.poisson(size=3, dtype=jnp.int64).dtype == jnp.int64
            assert (
                rng.exponential(size=3, dtype=jnp.float64).dtype == jnp.float64
            )
            assert rng.integers(5).dtype == jnp.int64
            assert rng.poisson().dtype == jnp.int64
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("dtype_name", ["int64", "float64"])
def test_new_methods_support_64_bit_dtypes(
    rng: RandomGenerator, sampling_namespace: Any, dtype_name: str
) -> None:
    if rng.backend == "jax":
        pytest.skip("JAX X64 modes are covered separately.")
    xp = sampling_namespace
    if dtype_name == "int64":
        assert rng.integers(5, size=3, dtype=xp.int64).dtype == xp.int64
        assert rng.poisson(size=3, dtype=xp.int64).dtype == xp.int64
    else:
        assert rng.exponential(size=3, dtype=xp.float64).dtype == xp.float64


def test_cupy_permutation_retries_tied_keys(monkeypatch: Any) -> None:
    import sys

    from numpy.random import Generator

    np = import_module("numpy")
    from orng.backends.cupy import CuPyFunctionalBackend

    class TiedGenerator(Generator):
        calls = 0

        def random(self, size: Any = None, dtype: Any = None) -> Any:
            self.calls += 1
            return np.array(
                [0.1, 0.1, 0.2] if self.calls == 1 else [0.7, 0.2, 0.4]
            )

    monkeypatch.setitem(sys.modules, "cupy", np)
    generator = TiedGenerator(np.random.PCG64(1))
    backend = CuPyFunctionalBackend(pure=False)
    state = backend.init_state(seed=None, generator=generator)
    result, next_state = backend.permutation(state, 3)
    np.testing.assert_array_equal(result, [1, 2, 0])
    assert generator.calls == 2
    assert next_state is state


def test_torch_new_methods_keep_samples_on_cuda(
    new_sampling_method: tuple[str, tuple, dict],
) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("PyTorch CUDA is unavailable.")
    method, args, kwargs = new_sampling_method
    rng = RandomGenerator("torch", seed=723, device="cuda")
    result = getattr(rng, method)(*args, **kwargs)
    assert result.device.type == "cuda"


def test_poisson_rejects_rates_that_would_overflow(
    rng: RandomGenerator, sampling_namespace: Any
) -> None:
    with pytest.raises(ValueError, match="too large"):
        rng.poisson(lam=2**32, dtype=sampling_namespace.int32)
    if rng.backend != "jax":
        with pytest.raises(ValueError, match="too large"):
            rng.poisson(lam=[1.0, 2**32], dtype=sampling_namespace.int32)
