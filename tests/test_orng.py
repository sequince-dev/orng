from __future__ import annotations

import builtins
import sys

import pytest

from orng import ArrayRNG
from orng._utils import normalize_shape, total_size
from orng.backends import _FACTORIES
from orng.backends import numpy as numpy_backend


def test_normalize_shape_accepts_valid_inputs():
    assert normalize_shape(None) == ()
    assert normalize_shape(5) == (5,)
    assert normalize_shape([2, 3]) == (2, 3)


def test_normalize_shape_rejects_negative_dimensions():
    with pytest.raises(ValueError):
        normalize_shape(-1)
    with pytest.raises(ValueError):
        normalize_shape([2, -1])


def test_total_size_handles_scalar_and_shapes():
    assert total_size(()) == 1
    assert total_size((3,)) == 3
    assert total_size((2, 4)) == 8


def test_array_rng_delegates_to_backends(monkeypatch):
    class StubBackend:
        def __init__(self, *, seed, generator, device):
            self.seed = seed
            self.generator = generator
            self.device = device
            self.calls: list[tuple[str, tuple]] = []

        def random(self, *, size, dtype):
            self.calls.append(("random", size, dtype))
            return ("random", size, dtype, self.seed)

        def uniform(self, *, low, high, size, dtype):
            self.calls.append(("uniform", low, high, size, dtype))
            return ("uniform", low, high, size, dtype, self.seed)

        def normal(self, *, loc, scale, size, dtype):
            self.calls.append(("normal", loc, scale, size, dtype))
            return ("normal", loc, scale, size, dtype, self.seed)

        def gamma(self, *, shape, scale, size, dtype):
            self.calls.append(("gamma", shape, scale, size, dtype))
            return ("gamma", shape, scale, size, dtype, self.seed)

        def choice(self, population, *, size, replace, probabilities):
            self.calls.append(
                ("choice", population, size, replace, probabilities)
            )
            return (
                "choice",
                population,
                size,
                replace,
                probabilities,
                self.seed,
            )

    instances: list[StubBackend] = []

    def fake_factory(**kwargs):
        backend = StubBackend(**kwargs)
        instances.append(backend)
        return backend

    monkeypatch.setitem(_FACTORIES, "numpy", fake_factory)

    rng = ArrayRNG(backend="numpy", seed=7, generator="sentinel", device="cpu")
    assert len(instances) == 1
    backend = instances[0]
    assert backend.seed == 7
    assert backend.generator == "sentinel"
    assert backend.device == "cpu"

    random_result = rng.random(size=5, dtype="float32")
    assert random_result == ("random", 5, "float32", 7)
    assert backend.calls[0] == ("random", 5, "float32")

    uniform_result = rng.uniform(
        low=-1.0,
        high=2.0,
        size=(1, 2),
        dtype="float16",
    )
    assert uniform_result == ("uniform", -1.0, 2.0, (1, 2), "float16", 7)
    assert backend.calls[1] == ("uniform", -1.0, 2.0, (1, 2), "float16")

    normal_result = rng.normal(
        loc=1.5, scale=2.0, size=(2, 2), dtype="float64"
    )
    assert normal_result == ("normal", 1.5, 2.0, (2, 2), "float64", 7)
    assert backend.calls[2] == ("normal", 1.5, 2.0, (2, 2), "float64")

    gamma_result = rng.gamma(
        shape=2.0,
        scale=3.0,
        size=(4,),
        dtype="float32",
    )
    assert gamma_result == ("gamma", 2.0, 3.0, (4,), "float32", 7)
    assert backend.calls[3] == ("gamma", 2.0, 3.0, (4,), "float32")

    choice_result = rng.choice(
        ["a", "b"],
        size=1,
        replace=False,
        p=[0.4, 0.6],
    )
    assert choice_result == ("choice", ["a", "b"], 1, False, [0.4, 0.6], 7)
    assert backend.calls[4] == (
        "choice",
        ["a", "b"],
        1,
        False,
        [0.4, 0.6],
    )


def test_array_rng_rejects_unknown_backend():
    with pytest.raises(ValueError):
        ArrayRNG(backend="unknown")


def test_array_rng_forwards_generator_to_jax(monkeypatch):
    captured_kwargs = {}

    class DummyBackend:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        def random(self, *, size, dtype):
            return None

        def normal(self, *, loc, scale, size, dtype):
            return None

        def gamma(self, *, shape, scale, size, dtype):
            return None

        def choice(self, population, *, size, replace, probabilities):
            return None

    monkeypatch.setitem(
        _FACTORIES, "jax", lambda **kwargs: DummyBackend(**kwargs)
    )

    key = ("jax-key",)
    ArrayRNG(backend="jax", generator=key)

    assert captured_kwargs["generator"] is key


def test_numpy_backend_import_error_mentions_extra(monkeypatch):
    # Ensure numPy import attempts go through builtins.__import__
    monkeypatch.delitem(sys.modules, "numpy", raising=False)
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "numpy":
            raise ImportError("missing numpy")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError) as err:
        numpy_backend.NumPyBackend(seed=None, generator=None)

    assert "pip install orng[numpy]" in str(err.value)
