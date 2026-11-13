# Changelog

Notable changes to ORNG are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Historical entries below summarise stable releases; changes from prereleases
are included in the corresponding stable release.

## [Unreleased]

### Added

- `integers`, `permutation`, `exponential`, and `poisson` in the object and
  functional APIs for all four backends, including shape, dtype, RNG-state,
  checkpoint replay, and JAX compilation support.
- Pixi development environments for NumPy, JAX, CPU PyTorch, and CUDA CuPy,
  with test, lint, formatting, and type-check tasks used locally and in CI.
  ([#17](https://github.com/sequince-dev/orng/pull/17))
- Contribution guidelines, an AI contribution policy, agent instructions,
  and a changelog.

## [0.2.1] - 2026-09-10

### Fixed

- Use supported CuPy Generator methods for `RandomGenerator.choice`.
  ([#16](https://github.com/sequince-dev/orng/pull/16))

## [0.2.0] - 2026-08-06

### Added

- Detached, versioned RNG checkpoints through `state_dict`, `load_state_dict`,
  and `from_state_dict`.
  ([#14](https://github.com/sequince-dev/orng/pull/14))
- Access to backend-native RNG state through `state` and `set_state`.
  ([#15](https://github.com/sequince-dev/orng/pull/15))

## [0.1.0] - 2026-06-04

First stable release, including the following functionality developed during
the prereleases.

### Added

- A unified random-number generation interface for NumPy, PyTorch, CuPy, and JAX,
  supporting `random`, `uniform`, `normal`, `choice`, and `gamma`.
- A functional API returning `(sample, next_state)`, with pure state handling
  and an optional mutable fast path for stateful backends.
  ([#8](https://github.com/sequince-dev/orng/pull/8),
  [#10](https://github.com/sequince-dev/orng/pull/10))
- Optional `dtype` arguments in the functional API.
  ([#11](https://github.com/sequince-dev/orng/pull/11))
- Citation guidelines in the README.
  ([#13](https://github.com/sequince-dev/orng/pull/13))

### Deprecated

- `ArrayRNG` in favour of `RandomGenerator`.
  ([#9](https://github.com/sequince-dev/orng/pull/9))

### Fixed

- Normal sampling with the CuPy backend.
  ([#5](https://github.com/sequince-dev/orng/pull/5))
- The deprecation warning for `ArrayRNG`.
  ([#12](https://github.com/sequince-dev/orng/pull/12))

[Unreleased]: https://github.com/sequince-dev/orng/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/sequince-dev/orng/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/sequince-dev/orng/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/sequince-dev/orng/releases/tag/v0.1.0
