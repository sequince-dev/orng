# Omni RNG (`orng`)

`orng` provides a thin facade over several Array API–compatible random number
generators. It mirrors the subset of the `numpy.random.Generator` API:

- `random`
- `uniform`
- `normal`
- `choice`
- `gamma`

letting you pick the underlying backend at runtime. The following backends are
currently supported:

- `numpy`
- `torch`
- `cupy`
- `jax`

## Installation

The core package only depends on the standard Python library:

```bash
pip install orng
```

Backends are optional extras that you can install on demand:

```bash
pip install "orng[numpy]"   # NumPy RNG support
pip install "orng[torch]"   # PyTorch RNG support
pip install "orng[cupy]"    # CuPy RNG support
pip install "orng[jax]"     # JAX RNG support
```

You can also combine extras, e.g. `pip install "orng[numpy,torch]"`.

## Quick Start

```python
from orng import ArrayRNG

rng = ArrayRNG(backend="numpy", seed=42)
samples = rng.normal(loc=0.0, scale=1.0, size=5)
uniform = rng.uniform(low=-1.0, high=1.0, size=(2, 2))
```

The backend module is imported lazily. If the requested library is missing,
`ArrayRNG` will raise an informative `ImportError` that points to the matching
extra.

## Functional Backend API

For JAX and other functional workflows, `orng` also provides a pure API in
`orng.functional`:

```python
from orng.functional import create_functional_backend

backend = create_functional_backend("numpy")
state = backend.init_state(seed=42, generator=None)

state, x = backend.normal(state, loc=0.0, scale=1.0, size=(4,), dtype=None)
state, y = backend.uniform(state, low=-1.0, high=1.0, size=(2, 2), dtype=None)
```

Every sampling call takes an explicit `state` and returns
`(next_state, sample)`. This avoids mutable RNG objects inside compiled code.

By default this API is pure (`pure=True`). On stateful backends (`numpy`,
`torch`, and `cupy`) this snapshots RNG state each call. For lower overhead on
those backends, you can opt into a trusted mutable fast path with
`pure=False`:

```python
backend = create_functional_backend("numpy", pure=False)
state = backend.init_state(seed=42, generator=None)  # numpy.random.Generator
state, x = backend.normal(state, loc=0.0, scale=1.0, size=(4,), dtype=None)
```

The JAX functional backend is always pure and does not support `pure=False`.

Supported functional methods:

- `random`
- `uniform`
- `normal`
- `choice`
- `gamma`

### JAX Compilation Example

```python
import jax
import jax.numpy as jnp
from orng.functional import create_functional_backend

backend = create_functional_backend("jax")
state = backend.init_state(seed=0, generator=None)

@jax.jit
def step(key):
    next_key, sample = backend.normal(
        key, loc=0.0, scale=1.0, size=(8,), dtype=jnp.float32
    )
    return next_key, sample

state, sample = step(state)
```

### Functional State Reference

`init_state(seed=..., generator=...)` expects backend-specific generator
inputs:

| Backend | Generator argument |
|---------|--------------------|
| `numpy` | `numpy.random.Generator` |
| `torch` | `torch.Generator` |
| `cupy`  | `cupy.random.Generator` |
| `jax`   | `jax.random.KeyArray` (from `jax.random.key`) |

If `generator=None`, a new backend-specific state is created from `seed`.
If `seed=None`, a fresh backend-specific random seed is used.

### Backend State Reference

When you pass the optional `generator` argument to `ArrayRNG`, the expected
object depends on the backend:

| Backend | Generator argument |
|---------|--------------------|
| `numpy` | `numpy.random.Generator` |
| `torch` | `torch.Generator` |
| `cupy`  | `cupy.random.Generator` |
| `jax`   | `jax.random.KeyArray` (from `jax.random.key`) |

This lets you wrap an existing RNG/key instead of seeding a new one.

## Project Layout

```
orng/
├── src/orng/
│   ├── __init__.py      # package exports
│   ├── _utils.py        # shared helpers (internal)
│   ├── orng.py          # ArrayRNG facade
│   └── backends/        # backend-specific implementations
└── README.md
```

Each backend class lives in its own module under `orng/backends/`, keeping the
core facade compact and making optional dependencies easy to manage.
