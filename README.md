# Omni RNG (`orng`)

`orng` provides a thin wrapper over several Array API–compatible random number
generators. It mirrors a subset of the `numpy.random.Generator` API:

- `random`
- `uniform`
- `normal`
- `choice`
- `gamma`
- `integers`
- `permutation`
- `exponential`
- `poisson`

letting you pick the underlying backend at runtime. The following backends are
currently supported:

- `numpy`
- `torch`
- `cupy`
- `jax`

## Installation

`orng` can be install from PyPI using `pip`:

```bash
pip install orng
```

Backends are optional extras that you can install as needed:

```bash
pip install "orng[numpy]"   # NumPy RNG support
pip install "orng[torch]"   # PyTorch RNG support
pip install "orng[cupy]"    # CuPy RNG support
pip install "orng[jax]"     # JAX RNG support
```

You can also combine extras, e.g. `pip install "orng[numpy,torch]"`.

## Quick Start

```python
from orng import RandomGenerator

rng = RandomGenerator(backend="numpy", seed=42)
samples = rng.normal(loc=0.0, scale=1.0, size=5)
uniform = rng.uniform(low=-1.0, high=1.0, size=(2, 2))
indices = rng.integers(0, 10, size=5)
shuffled = rng.permutation(10)
waiting_times = rng.exponential(scale=2.0, size=5)
counts = rng.poisson(lam=3.0, size=5)
```

The backend module is imported lazily. If the requested library is missing,
`RandomGenerator` will raise an informative `ImportError` that points to the matching
extra.

## Saving and restoring state

`RandomGenerator` exposes a versioned state mapping for checkpointing:

```python
state = rng.state_dict()

restored = RandomGenerator.from_state_dict(state)
assert restored.backend == rng.backend
```

An existing generator can also be reset in place:

```python
rng.load_state_dict(state)
```

The returned mapping is detached from the live generator. It contains native backend arrays where applicable.

## Functional Backend API

For JAX and other functional workflows, `orng` also provides a pure API in
`orng.functional`:

```python
from orng.functional import create_functional_backend

backend = create_functional_backend("numpy")
state = backend.init_state(seed=42, generator=None)

x, state = backend.normal(state, loc=0.0, scale=1.0, size=(4,), dtype=None)
y, state = backend.uniform(state, low=-1.0, high=1.0, size=(2, 2), dtype=None)
```

Every sampling call takes an explicit `state` and returns
`(sample, next_state)`. This avoids mutable RNG objects inside compiled code.

By default this API is pure (`pure=True`). On stateful backends (`numpy`,
`torch`, and `cupy`) this snapshots RNG state each call. For lower overhead on
those backends, you can opt into a trusted mutable fast path with
`pure=False`:

```python
backend = create_functional_backend("numpy", pure=False)
state = backend.init_state(seed=42, generator=None)  # numpy.random.Generator
x, state = backend.normal(state, loc=0.0, scale=1.0, size=(4,))
```

The JAX functional backend is always pure and does not support `pure=False`.

Supported functional methods:

- `random`
- `uniform`
- `normal`
- `choice`
- `gamma`
- `integers`
- `permutation`
- `exponential`
- `poisson`

`integers(low, high=None, size=None, dtype=None)` uses scalar integer bounds
and a half-open interval: `low` is inclusive and `high` is exclusive. With
only one bound, it samples from zero up to that bound. Both bounds must fit
in the requested signed `int32` or `int64` dtype. Array-valued bounds and the
NumPy `endpoint` option are not supported.

`permutation(x, axis=0)` returns a shuffled copy of an array, or a permuted
`arange(x)` for a non-negative integer. It applies the same permutation to all
slices along the selected axis and preserves the input array's dtype.

For `exponential(scale=1.0, size=None, dtype=None)` and
`poisson(lam=1.0, size=None, dtype=None)`, `size` specifies the full output
shape, and parameter arrays must broadcast to that shape. When `size=None`,
the parameter shape determines the output. Scale and rate parameters must
be finite and non-negative; zero produces zero samples. Poisson rates must
remain below the output integer limit with a ten-standard-deviation margin.
Exponential uses the scale (mean), rather than its reciprocal rate, and supports
`float32` and `float64` output. Poisson supports `int32` and `int64` output.

The new integer-valued methods default to `int64` on NumPy, CuPy, and PyTorch;
JAX follows its X64 configuration. Exponential defaults to `float64` on NumPy
and CuPy and `float32` on PyTorch and JAX. Explicit 64-bit dtypes on JAX require
X64 mode. Under JAX `jit`, integer bounds, output shapes, and permutation axes
must be static; array-valued scale and rate parameters can be dynamic and must
satisfy their constraints before the compiled call.

The functional versions take `state` first and return `(sample, next_state)`:

```python
indices, state = backend.integers(state, low=0, high=10, size=(5,))
shuffled, state = backend.permutation(state, 10)
waiting_times, state = backend.exponential(state, scale=2.0, size=(5,))
counts, state = backend.poisson(state, lam=3.0, size=(5,))
```

### JAX Compilation Example

```python
import jax
import jax.numpy as jnp
from orng.functional import create_functional_backend

backend = create_functional_backend("jax")
state = backend.init_state(seed=0, generator=None)

@jax.jit
def step(key):
    sample, next_key = backend.normal(
        key, loc=0.0, scale=1.0, size=(8,), dtype=jnp.float32
    )
    return sample, next_key

sample, state = step(state)
```

### Functional State Reference

The functional API follows the native conventions of each backend rather than
introducing a wrapper state type.

`init_state(seed=..., generator=...)` accepts backend-specific generator
inputs:

| Backend | `generator` argument |
|---------|----------------------|
| `numpy` | `numpy.random.Generator` |
| `torch` | `torch.Generator` |
| `cupy`  | `cupy.random.Generator` |
| `jax`   | JAX PRNG key array, typically from `jax.random.key(...)` |

If `generator=None`, ORNG creates a new backend-native state from `seed`. If
`seed=None`, the backend chooses a fresh random seed using its usual behavior.

The `state` value passed into every sampling method also matches the backend:

| Backend | `pure=True` state | `pure=False` state |
|---------|-------------------|--------------------|
| `numpy` | NumPy bit-generator state `dict` | `numpy.random.Generator` |
| `torch` | `TorchFunctionalState` | `torch.Generator` |
| `cupy`  | CuPy bit-generator state `dict` | `cupy.random.Generator` |
| `jax`   | JAX PRNG key array | not supported |

For example, NumPy in pure mode snapshots and returns a bit-generator state
dictionary each call, while `pure=False` threads a `numpy.random.Generator`
through the same functional interface. JAX always uses and returns a PRNG key.

### Backend State Reference

When you pass the optional `generator` argument to `RandomGenerator`, the expected
object depends on the backend:

| Backend | Generator argument |
|---------|--------------------|
| `numpy` | `numpy.random.Generator` |
| `torch` | `torch.Generator` |
| `cupy`  | `cupy.random.Generator` |
| `jax`   | `jax.random.KeyArray` (from `jax.random.key`) |

This lets you wrap an existing RNG/key instead of seeding a new one.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, checks, and release guidelines,
and [AI_POLICY.md](AI_POLICY.md) for guidance on AI-assisted contributions.
Release notes are in [CHANGELOG.md](CHANGELOG.md).

## Citing

If you find `orng` useful in your work, please cite the corresponding [DOI](https://doi.org/10.5281/zenodo.17544786).
