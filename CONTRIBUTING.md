# Contributing to ORNG

Contributions to ORNG are welcome, including bug reports, tests, performance
improvements, and new sampling functionality. This guide describes the development
workflow and conventions we use to keep changes easy to review and maintain.

## Community and responsibility

Treat other contributors with respect and follow the
[Python Community Code of Conduct](https://www.python.org/psf/codeofconduct/).
Contributions must comply with the project's [license](LICENSE). Preserve
required attribution when incorporating third-party material.

AI-assisted contributions are welcome under our [AI policy](AI_POLICY.md).
Contributors are responsible for understanding, validating, and explaining their
changes, including their numerical correctness and performance.

## Scope of contributions

Keep each pull request focused on one problem or a closely related set of changes.
Discuss substantial API changes, new dependencies, or new algorithms in an issue
before investing significant implementation effort. Small fixes can go directly
to a pull request.

ORNG provides a thin random-number generation interface over NumPy, PyTorch,
CuPy, and JAX. Keep the object and functional APIs consistent, preserve lazy
imports of optional backends, and prefer existing interfaces and helpers.
Introduce abstractions only for a concrete shared need. Avoid unrelated cleanup
in bug fixes and feature changes.

## Development setup

Fork [sequince-dev/orng](https://github.com/sequince-dev/orng) on GitHub,
then clone your fork and configure local `main` to track `upstream/main`:

```bash
git clone git@github.com:YOUR_USERNAME/orng.git
cd orng
git remote add upstream https://github.com/sequince-dev/orng.git
git fetch upstream
git switch main
git branch --set-upstream-to=upstream/main main
```

Keep local `main` for upstream changes and do development on separate branches.
Before starting each contribution, update `main` and create a branch:

```bash
git switch main
git pull --ff-only
git switch -c fix-descriptive-name
```

Here, `upstream` is the ORNG repository and `origin` is your fork. The tracking
setting makes `git pull --ff-only` on `main` fetch from upstream. Push contribution
branches explicitly to `origin`, as shown in the pull request instructions below.

Use Pixi for development. Follow the
[Pixi installation instructions](https://pixi.prefix.dev/latest/installation/)
to install it first. The environments, dependencies, and tasks are defined
in `pyproject.toml`. The repository does not track a lock file; Pixi resolves
dependencies and generates `pixi.lock` locally. With Pixi installed, set up the
default environment and Git hooks:

```bash
pixi install
pixi run pre-commit install
```

This installs the code-check hooks configured in `.pre-commit-config.yaml`.

The default environment installs ORNG in editable mode and includes the core
test and lint tools. Source changes take effect without reinstalling the package.
When intentionally changing dependencies, update `pyproject.toml` and run
`pixi install` to update your environment. Do not include generated lock files
in the pull request.

## Code style

- Follow the existing module's conventions and use Ruff for formatting and import
  ordering. Python code uses four-space indentation and a 79-character line limit.
- Use descriptive names. Conventional mathematical names are appropriate when
  their meaning is clear locally.
- Document public functions and classes with NumPy-style docstrings, including
  array shapes, return values, and important assumptions where relevant.
- Use comments to explain numerical reasoning, constraints, and non-obvious
  choices. Avoid comments that simply restate the code.
- Preserve supported array backends, devices, and dtypes. Use the existing array
  helpers and RNG interfaces rather than introducing implicit NumPy conversions
  or global random state.
- Keep eager and functional behavior consistent where both are supported. Keep
  host scalar conversions outside compiled operations and validate results at
  appropriate boundaries.
- Avoid unnecessary array copies, host-device transfers, and retention of unused
  arrays. Expensive diagnostics and logging should not add work when disabled.
  Support performance claims with reproducible measurements.

## Typing

Add parameter and return annotations to new functions and methods, including
private helpers and tests where practical. When changing an existing interface,
update its annotations and affected callers together. Annotate stored attributes
when their types are not clear from initialization.

- Use modern Python syntax supported by Python 3.10, such as `list[str]`,
  `dict[str, float]`, and `int | None`. Prefer collection interfaces
  such as `Mapping` or `Sequence` when callers need not supply a concrete container.
- Reuse `SizeLike` from `orng._utils` and the generator and state aliases from
  `orng.functional`. Use backend-specific types only where that backend is
  required. Do not introduce runtime imports of optional backends for annotations.
- Describe array shapes, dtype restrictions, and scalar-versus-batch behavior in
  docstrings. An annotation alone does not express these constraints;
  validate them at appropriate runtime boundaries.
- Give callbacks explicit signatures and reuse existing protocols and type
  aliases. Prefer a precise union or protocol to `Any` when the accepted values
  have a known contract. Keep `Any` limited to genuinely dynamic interfaces.
- Use `TYPE_CHECKING` for annotation-only imports when needed to avoid import
  cycles or loading optional dependencies. Ensure annotations do not introduce
  new runtime dependency requirements.
- Treat `cast` and `# type: ignore[error-code]` as exceptions. Keep them narrowly
  scoped and explain non-obvious reasons; neither replaces runtime validation.
  Fix inaccurate annotations instead of broadly suppressing errors.

The pre-commit hooks run mypy using the configuration in `pyproject.toml`.
For a standalone type check during development, use `pixi run typecheck`.
Passing type checks complements numerical and runtime tests; it does not replace
them.

## Documentation

User documentation lives in `README.md` and public API docstrings. Update both
when changing an interface. Use NumPy-style docstrings with parameter types,
shapes, return values, and constraints, keeping them consistent with annotations.
ORNG does not currently have a separate documentation build.

## Checks and tests

Run the following checks before requesting review:

```bash
pixi run pre-commit run --all-files
pixi run test
```

Pre-commit covers Ruff linting and formatting, mypy type checking, spelling,
whitespace, YAML validation, and large-file checks. It does not run the test
suite, so run pytest separately with `pixi run test`.

Pre-commit may modify files to fix formatting or other issues. Review those
changes, stage the relevant files, and rerun the checks before committing.
Configuration in `.pre-commit-config.yaml` and `pyproject.toml` is authoritative.

Tests use pytest and live in `tests/`. Follow the existing test layout, fixtures,
and parameterization. During development, run the relevant file or individual test:

```bash
pixi run python -m pytest tests/test_backends.py
pixi run python -m pytest tests/test_backends.py -k choice
```

Add regression tests for bug fixes and tests for new behavior. Test observable
outcomes and numerical invariants rather than implementation details. Use fixed
RNG seeds and justified numerical tolerances for stochastic tests. Cover relevant
edge cases, such as scalar and empty output shapes, broadcasting, invalid
parameters, probabilities, and dtype precision. Test generator advancement,
reproducibility within each backend, detached checkpoint round trips, and
`pure=True` versus `pure=False` state handling where relevant. Do not require
identical random samples across different backends. Exercise JAX compilation
when changing functional operations used under `jit`.

The default environment includes NumPy. Use the configured environments for
changes affecting optional backends:

```bash
pixi run -e all-backends test
pixi run -e jax test
pixi run -e torch test
pixi run -e py310 test
```

`all-backends` includes NumPy, JAX, and CPU PyTorch. CuPy requires a compatible
NVIDIA GPU and driver on Linux; select the matching CUDA environment:

```bash
pixi run -e cupy-cuda12 check-cupy
pixi run -e cupy-cuda12 test
# For CUDA 13, use cupy-cuda13 instead.
```

CI also checks the installed CuPy Generator API using a CPU stub build and the
NumPy-backed `--fake-cupy` fixture. That fixture checks method availability and
ORNG's wrapper logic; it does not validate CUDA execution, native RNG state, or
every CuPy signature. Report the environments tested and any relevant skips or
unavailable backends in the pull request. A skipped backend test does not verify
that backend.

The CI test matrix covers NumPy, JAX, and CPU PyTorch together on Python
3.10–3.13 using `ci-py310`, `ci-py311`, `ci-py312`, and `all-backends`.
Use the matching environment when reproducing a CI failure locally:

```bash
pixi run -e ci-py310 test
```

## Commit messages

Use uppercase SciPy-style prefixes with optional parenthesized scopes for commit
subjects and pull request titles:

```text
PREFIX(scope): short imperative summary
PREFIX: short imperative summary
PREFIX/PREFIX(scope): short imperative summary
```

The scope identifies the affected module or area, such as `numpy`, `torch`,
or `checkpoint`. Start it with a lowercase letter and use only lowercase letters,
digits, dots, underscores, or hyphens. Keep it short, and omit it when the change
spans the project or a scope would add little information.

Choose the prefix that best describes the change:

| Prefix | Use |
| --- | --- |
| `BUG` | Fix incorrect behavior |
| `ENH` | Add or extend functionality |
| `PERF` | Improve runtime or memory use without changing intended behavior |
| `API` | Change a public interface or introduce an incompatibility |
| `DEP` | Deprecate functionality |
| `MAINT` | Refactor or maintain code without changing intended behavior |
| `TST` | Add or improve tests |
| `DOC` | Update documentation |
| `BLD` | Change packaging, dependencies, or build configuration |
| `CI` | Change continuous integration workflows |
| `REL` | Prepare or maintain a release |
| `STY` | Make formatting-only changes |

Use `BUG` consistently for bug fixes rather than alternating with `FIX`. A bug fix
that includes a regression test still uses `BUG`; choose the prefix for the main
purpose of the change.

Use one prefix by default, or two separated by `/` when both describe the change,
for example `MAINT/PERF(torch): avoid duplicate conversions`. The optional
scope follows both prefixes. Use at most two prefixes.

These are contribution conventions; ORNG does not currently enforce commit
messages or pull request titles with a hook or CI validator.

Keep the subject concise, preferably within 72 characters, and omit its trailing
period. Keep the prefix uppercase and start the summary after `: ` with a
lowercase letter. Use imperative wording such as "guard", "add", or "avoid".
When useful, add a blank line and a body explaining the problem, the resulting
behavior, and any important tradeoffs. Reference related issues; use
`Closes #123` only when the change fully resolves that issue.

Examples:

```text
BUG(cupy): use supported generator methods for choice

Keep sampling on the managed generator so checkpointing and reproducibility
continue to work with the CuPy backend.
```

```text
PERF(torch): avoid unnecessary array copies
DOC: clarify contribution guidelines
REL: prepare release 0.3.0
```

Make commits in coherent units. Pull request titles should also follow this
format and describe the final change. When squash-merging, use the same
conventions for the resulting commit.

## Pull requests and review

For each pull request with user-visible changes, update
[CHANGELOG.md][changelog] under `Unreleased` in the same pull request. Follow
the entry categories and guidance in the releases section below.

Push your branch and open a pull request targeting `sequince-dev/orng:main`:

```bash
git push --set-upstream origin fix-descriptive-name
```

Use a draft pull request for work in progress or early feedback. In the description:

- Explain the problem and resulting behavior, and link related issues.
- Describe relevant API changes, numerical assumptions, and compatibility effects.
- Summarize validation, including test commands and results. Include reproducible
  benchmark conditions for performance changes.

Respond to reviewer feedback and keep the description aligned with the final
implementation. Reviewers should distinguish required changes from suggestions,
explain their reasoning, and raise unrelated findings in separate issues.

Maintainers should wait for applicable CI checks and review discussions to be
resolved before merging. Check the squash commit title and message, link or close
resolved issues, and delete merged branches in the main repository when no longer
needed.

## Releases

Maintain [CHANGELOG.md][changelog] using
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Add user-visible changes
under `Unreleased`, grouped as `Added`, `Changed`, `Deprecated`, `Removed`,
`Fixed`, or `Security`; omit empty groups. Describe the effect on users,
especially changes to scientific results, APIs, RNG behaviour, or checkpoint
compatibility. Routine internal refactoring and CI maintenance do not normally
need entries.

Releases are coordinated by maintainers. Versions are derived from Git tags by
`setuptools-scm`; do not maintain a separate hard-coded version number. Use tags
such as `v0.1.0` and release notes describing user-visible changes and any
incompatibilities.

When preparing a release, move the unreleased entries into a section headed
`## [VERSION] - YYYY-MM-DD`, using the actual release date, and leave a fresh
`Unreleased` section at the top. Link each version heading to the comparison with
the preceding release tag, and update the `Unreleased` link to compare the new tag
with `HEAD`. For an initial release section, link to its GitHub release. Use PEP
440 versions for prereleases, such as `0.3.0a1` with the Git tag `v0.3.0a1`.
Use the version's changelog section as the GitHub release notes.

Before publishing, verify the intended commit, applicable CI results, and the
built distribution in a clean environment. Publishing a GitHub release triggers
the PyPI workflow in `.github/workflows/publish.yml`. Check that the workflow
completes successfully. Published distributions cannot be overwritten, so verify
the tag and artifacts before publishing.

[changelog]: CHANGELOG.md
