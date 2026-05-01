from __future__ import annotations

import pytest

from orng.functional import (
    create_functional_backend,
    create_functional_backend_from_xp,
    infer_backend_name_from_xp,
)


def _check_backend_available(name: str) -> None:
    if name == "numpy":
        pytest.importorskip("numpy")
    elif name == "torch":
        pytest.importorskip("torch")
    elif name == "jax":
        pytest.importorskip("jax")
        pytest.importorskip("jax.numpy")
    elif name == "cupy":
        cp = pytest.importorskip("cupy")
        try:
            cp.cuda.runtime.getDeviceCount()
        except (ImportError, cp.cuda.runtime.CUDARuntimeError) as exc:
            pytest.skip(f"CuPy runtime unavailable: {exc}")
    else:  # pragma: no cover - defensive
        raise AssertionError(f"Unexpected backend fixture parameter: {name}")


@pytest.fixture(params=["numpy", "torch", "jax", "cupy"])
def functional_backend_case(request):
    name = request.param
    _check_backend_available(name)
    backend = create_functional_backend(name)

    if name == "numpy":
        np = pytest.importorskip("numpy")
        return {
            "backend": backend,
            "dtype": np.float32,
            "assert_close": np.testing.assert_allclose,
        }
    if name == "torch":
        torch = pytest.importorskip("torch")

        def assert_close(a, b):
            assert torch.allclose(a, b)

        return {
            "backend": backend,
            "dtype": torch.float32,
            "assert_close": assert_close,
        }
    if name == "jax":
        jnp = pytest.importorskip("jax.numpy")

        def assert_close(a, b):
            assert bool(jnp.allclose(a, b))

        return {
            "backend": backend,
            "dtype": jnp.float32,
            "assert_close": assert_close,
        }

    cp = pytest.importorskip("cupy")

    def assert_close(a, b):
        cp.testing.assert_allclose(a, b)

    return {
        "backend": backend,
        "dtype": cp.float32,
        "assert_close": assert_close,
    }


def test_functional_backend_reproducibility(functional_backend_case):
    backend = functional_backend_case["backend"]
    dtype = functional_backend_case["dtype"]
    assert_close = functional_backend_case["assert_close"]

    state_a = backend.init_state(seed=123, generator=None)
    state_b = backend.init_state(seed=123, generator=None)

    state_a, a1 = backend.random(state_a, size=(2, 3), dtype=dtype)
    state_b, b1 = backend.random(state_b, size=(2, 3), dtype=dtype)
    assert_close(a1, b1)

    state_a, a2 = backend.uniform(
        state_a,
        low=-1.0,
        high=2.0,
        size=(4,),
        dtype=dtype,
    )
    state_b, b2 = backend.uniform(
        state_b,
        low=-1.0,
        high=2.0,
        size=(4,),
        dtype=dtype,
    )
    assert_close(a2, b2)

    state_a, a3 = backend.normal(
        state_a,
        loc=0.5,
        scale=1.5,
        size=(3,),
        dtype=dtype,
    )
    state_b, b3 = backend.normal(
        state_b,
        loc=0.5,
        scale=1.5,
        size=(3,),
        dtype=dtype,
    )
    assert_close(a3, b3)

    state_a, a4 = backend.gamma(
        state_a,
        shape=2.0,
        scale=2.0,
        size=(2, 2),
        dtype=dtype,
    )
    state_b, b4 = backend.gamma(
        state_b,
        shape=2.0,
        scale=2.0,
        size=(2, 2),
        dtype=dtype,
    )
    assert_close(a4, b4)

    state_a, a5 = backend.choice(
        state_a,
        [0, 1, 2, 3],
        size=(3,),
        replace=True,
        probabilities=None,
    )
    state_b, b5 = backend.choice(
        state_b,
        [0, 1, 2, 3],
        size=(3,),
        replace=True,
        probabilities=None,
    )
    assert_close(a5, b5)


def test_jax_functional_backend_works_under_jit():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    backend = create_functional_backend("jax")
    state = backend.init_state(seed=7, generator=None)

    @jax.jit
    def step(key):
        next_key, sample = backend.normal(
            key,
            loc=0.0,
            scale=1.0,
            size=(8,),
            dtype=jnp.float32,
        )
        return next_key, sample

    state, sample_1 = step(state)
    state, sample_2 = step(state)
    assert not bool(jnp.allclose(sample_1, sample_2))


def test_jax_functional_backend_explicit_compile():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    backend = create_functional_backend("jax")
    state = backend.init_state(seed=11, generator=None)

    def program(key, loc):
        next_key, sample = backend.normal(
            key,
            loc=loc,
            scale=1.0,
            size=(4,),
            dtype=jnp.float32,
        )
        return next_key, sample

    lowered = jax.jit(program).lower(
        state,
        jnp.asarray(0.25, dtype=jnp.float32),
    )
    compiled = lowered.compile()
    next_state, sample = compiled(state, jnp.asarray(0.25, dtype=jnp.float32))

    assert sample.shape == (4,)
    assert sample.dtype == jnp.float32
    assert not bool(jnp.array_equal(next_state, state))


def test_jax_functional_backend_gamma_scalar():
    jnp = pytest.importorskip("jax.numpy")

    backend = create_functional_backend("jax")
    state = backend.init_state(seed=0, generator=None)

    next_state, sample = backend.gamma(
        state,
        shape=2.0,
        scale=3.0,
        size=None,
        dtype=jnp.float32,
    )

    assert sample.shape == ()
    assert sample.dtype == jnp.float32
    assert not bool(jnp.array_equal(next_state, state))


def test_jax_functional_backend_seed_none_uses_entropy(monkeypatch):
    jax = pytest.importorskip("jax")

    backend = create_functional_backend("jax")
    monkeypatch.setattr(
        "orng.backends.jax.secrets.randbits", lambda bits: 1234
    )

    state = backend.init_state(seed=None, generator=None)

    assert bool(jax.numpy.array_equal(state, jax.random.key(1234)))


def test_jax_functional_backend_rejects_impure_mode():
    pytest.importorskip("jax")

    with pytest.raises(
        ValueError,
        match="always pure",
    ):
        create_functional_backend("jax", pure=False)


def test_numpy_functional_backend_fast_state_skips_copying():
    np = pytest.importorskip("numpy")

    backend = create_functional_backend("numpy", pure=False)
    state = backend.init_state(seed=123, generator=None)
    assert isinstance(state, np.random.Generator)

    next_state, sample = backend.random(state, size=(3,), dtype=np.float32)
    assert next_state is state
    assert sample.shape == (3,)


def test_infer_backend_name_from_xp_numpy():
    np = pytest.importorskip("numpy")
    assert infer_backend_name_from_xp(np) == "numpy"


def test_infer_backend_name_from_xp_jax():
    jnp = pytest.importorskip("jax.numpy")
    assert infer_backend_name_from_xp(jnp) == "jax"


def test_create_functional_backend_from_xp_jax_is_pure():
    jnp = pytest.importorskip("jax.numpy")

    backend = create_functional_backend_from_xp(jnp)
    state = backend.init_state(seed=0, generator=None)
    next_state, sample = backend.normal(
        state,
        loc=0.0,
        scale=1.0,
        size=(4,),
        dtype=jnp.float32,
    )

    assert sample.shape == (4,)
    assert sample.dtype == jnp.float32
    assert not bool(jnp.array_equal(next_state, state))
