# Omni RNG (`orng`)

`orng` provides a thin facade over several Array API–compatible random number
generators. It mirrors the subset of the `numpy.random.Generator` API currently
needed by the Sequince project, while letting you pick the underlying backend at
runtime (`numpy`, `torch`, `cupy`, or `jax`).

## Installation

The core package only depends on `array-api-compat`:

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
```

The backend module is imported lazily. If the requested library is missing,
`ArrayRNG` will raise an informative `ImportError` that points to the matching
extra.

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
