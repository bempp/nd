# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Project overview

`nd` (Numerical Discretisation) is a Rust 2024 Cargo workspace for finite-element and boundary-element discretisation. Crates, in dependency order:

- `ndelement`: finite-element definitions and tabulation (Lagrange, Raviart–Thomas, Nédélec first kind), reference cells, polynomials, quadrature, push-forward/pull-back maps, DOF transformations, and VTK reorderings.
- `ndmesh`: single-element-type and mixed meshes (builders, geometry, topology), Gmsh/RON import/export, VTK export, built-in shapes (cube, screen, sphere), and optional MPI-distributed meshes with Coupe/SCOTCH partitioning. Depends on `ndelement` and re-exports it.
- `ndfunctionspace`: DOF maps and function spaces (serial, plus parallel with `mpi`). Depends on both of the above.
- `ndelement-python`: PyO3 + `numpy` extension (`cdylib` named `ndelement`, Python module `ndelement._ndelementrs`) with a Maturin-built Python package. Depends on `ndelement`.

All crates are BSD-3-Clause, version `0.4.0-dev`, hosted at https://codeberg.org/nd-project/nd. Linear algebra comes from `rlst` (`DynArray`, `RlstScalar`, etc.).

## Workspace layout

- Root `Cargo.toml`: workspace members, and `[patch.crates-io]` pointing the crates at their local paths (plus a git fork of `scotch`). Run commands from the repo root unless a crate-specific command is intended.
- `default-members` excludes `ndelement-python`, so a bare `cargo test` / `cargo clippy` at the root does **not** build the Python extension. CI clippy uses `--workspace`, which does.
- `nd*/src/`: implementation; the public API is defined by each `src/lib.rs` and its module tree. Traits live in `traits.rs` / `traits/`, enums and small types in `types.rs`.
- `nd*/examples/`: runnable examples. Several need features (`serde`, `mpi`, `scotch`, `coupe`) declared via `required-features`; MPI examples carry `[package.metadata.example.*.templated-examples]` for `cargo templated-examples`.
- `ndmesh/tests/`: integration tests and committed Gmsh fixtures (`*.msh`, `*.geo`). `ndmesh/meshes/` holds further mesh data.
- `ndelement/src/bindings.rs`: legacy C bindings, currently disabled (`// pub mod bindings;` in `lib.rs`). Not part of the build.
- `ndelement-python/python/`: Python package (`ndelement/`), `.pyi` stubs for the Rust module (`ndelement/_ndelementrs/`), tests (`test/`), and Sphinx docs (`docs/`).
- `.forgejo/workflows/`: authoritative CI (`run-tests.yml`: Rust feature matrix, release-mode examples, MPI examples, Python 3.10–3.14 tests; `style-checks.yml`: fmt, clippy, ruff, mypy).
- `.github/workflows/`: docs build/deploy to GitHub Pages (`docs.yml`) and PyPI release (`ndelement-release.yml`).
- `joss/`: JOSS paper sources. 
- `katex-header.html`: KaTeX header for rustdoc math.
- `RELEASE.md`: release procedure. Do not publish, tag, or change versions unless explicitly asked.

## Key conventions

- `FiniteElement::tabulate(points, nderivs, data)`: `points` has shape `[tdim, npoints]` (one point per column); `data` has shape `[npoints, value_size, nderivatives, ndofs]` (DOF index slowest in memory, so a whole table can be passed to `apply_dof_*`), obtainable from `tabulate_array_shape(nderivs, npoints)`. The derivative index ordering per cell type is documented on the trait in `ndelement/src/traits.rs`.
- Mesh cell point ordering follows DefElement's Lagrange element ordering (see the `ndmesh` crate docs in `ndmesh/src/lib.rs`).
- Preserve established point, DOF, and sub-entity ordering conventions. Correctness-sensitive changes (tabulation, orientations, DOF transformations, maps, topology) need targeted regression tests.

## Rust conventions

- Edition 2024; formatting is plain `cargo fmt` (no rustfmt config).
- Clippy lints in each crate's `Cargo.toml`: `wildcard_imports = "forbid"` (use explicit imports), and `todo`, `unimplemented`, `unreachable`, `dbg_macro`, `print_stdout`, `print_stderr` warn. Examples and doctests may print; library code should not.
- Crate roots have `#![warn(missing_docs)]`: public items need `///` docs; module/crate docs use `//!`. Rustdoc math uses KaTeX (`$...$`).
- `strict` feature: `#![cfg_attr(feature = "strict", deny(warnings), deny(unused_crate_dependencies))]`. New code must be clean under `strict`. Dependencies used only in some configurations are silenced with `use crate_name as _;` (see the `#[cfg(test)] mod test` blocks in each `lib.rs`).
- BLAS/LAPACK: the three Rust libraries take `blas-src`/`lapack-src` (Accelerate on macOS, system OpenBLAS on Linux) as **dev-dependencies**, linked via `extern crate` in the `#[cfg(test)]` module of `lib.rs`. Examples and doctests must include the same hidden `extern crate blas_src; extern crate lapack_src;` (plus `openblas_src` on Linux) lines — copy the pattern from existing doctests/examples. `ndelement-python` has them as normal dependencies since the extension must link them.
- Unit tests go in `#[cfg(test)]` modules next to the code; cross-crate or fixture-based tests go in `tests/`. Use `approx` for floating-point comparisons; `paste` is available for generating parametrised tests.
- Add or update a focused example when it materially clarifies a public API.

## Features

- `ndelement`: `mpi`, `serde` (none default).
- `ndmesh`: `serde` (default; also enables `ndelement/serde` and `ron`), `mpi` (propagates to `rlst/mpi`, `ndelement/mpi`), `coupe`, `scotch`.
- `ndfunctionspace`: `mpi` (propagates to `ndelement/mpi`, `ndmesh/mpi`).
- `ndelement-python`: `serde` (default), `mpi`; Maturin builds it with `--no-default-features --features serde`.
- All crates: `strict`.
- When adding a cross-crate capability, propagate features consistently through the dependent crates.

## Commands

Run the narrowest relevant check first, then workspace checks for shared changes.

```bash
# Format and default tests
cargo fmt -- --check
cargo test
cargo test -p ndelement some_test_name     # single crate / single test

# Feature matrix used by CI (RUST_MIN_STACK avoids stack overflows in some tests)
RUST_MIN_STACK=8388608 cargo test --no-default-features --features "strict"
RUST_MIN_STACK=8388608 cargo test --features "strict"
RUST_MIN_STACK=8388608 cargo test --all-features

# Clippy as in CI (repeat with each feature configuration above)
cargo clippy --workspace --features "strict" -- -D warnings
cargo clippy --workspace --tests --features "strict" -- -D warnings
cargo clippy --workspace --examples --features "strict" -- -D warnings

# Release-mode examples, and MPI examples (needs cargo-templated-examples, cargo-mpirun, MPI)
RUST_MIN_STACK=8388608 cargo test --examples --release --features "strict"
cargo templated-examples NPROCESSES 4

# Docs as in CI (nightly)
RUSTDOCFLAGS="--html-in-header katex-header.html" cargo +nightly doc --no-deps -Zunstable-options -Zrustdoc-scrape-examples --all-features
```

`--all-features`, `mpi`, `scotch` and `coupe` need native MPI, OpenBLAS/LAPACK, SCOTCH, CMake and libclang. Don't assume they are installed locally; say so if a check could not be run.

## Python bindings (`ndelement-python`)

Python ≥ 3.10, built with Maturin. The Rust side (`src/lib.rs`, `src/element_wrapper.rs`) defines the `_ndelementrs` PyO3 module; the pure-Python wrappers in `python/ndelement/*.py` call into it, and `python/ndelement/_ndelementrs/*.pyi` stubs must be kept in sync with the Rust module. Style: Ruff with 100-char lines and Google docstrings; mypy with `ignore_missing_imports`.

```bash
cd ndelement-python
cargo test --features "strict"
python3 -m ruff check python
python3 -m ruff format --check python
python3 -m mypy python

# Build and test, as in CI
uv venv .venv && source .venv/bin/activate
maturin develop --release
python -m pytest python/test
```

## Change discipline

- Keep changes scoped. Never edit build output under `target/` or generated files such as `_*.msh`, `_*.ron`, `_*.vtu` (test outputs, git-ignored).
- Don't modify unrelated untracked files or release artifacts.
- Before finishing, report which checks were run and which were blocked by missing native MPI/BLAS/SCOTCH dependencies.
