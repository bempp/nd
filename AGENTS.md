# nd contributor guide

## Project overview

`nd` is a Rust 2024 workspace for numerical discretisation and finite-element software. It contains three Rust libraries, in dependency order, and a Python bindings subcrate:

- `ndelement`: finite-element definitions, tabulation, reference cells, and mappings.
- `ndmesh`: finite-element meshes, geometry/topology, import/export, and optional distributed/MPI support; depends on `ndelement`.
- `ndfunctionspace`: finite-element DOF maps and function spaces; depends on both other crates.
- `ndelement-python`: CFFI Python bindings for `ndelement` and its Python package; depends on `ndelement`.

All crates are BSD-3-Clause licensed and currently use version `0.4.0-dev`. The workspace root `Cargo.toml` patches the local crates; run workspace commands from the repository root unless a crate-specific command is intended.

## Layout

- `nd*/src/`: library implementation. Public APIs are exposed through each crate's `src/lib.rs` and module hierarchy.
- `nd*/examples/`: runnable usage examples; some require optional features or MPI.
- `ndmesh/tests/`: integration tests and committed Gmsh fixture files.
- `ndelement-python/python/`: Python package built from the Rust extension with Maturin.
- `.forgejo/workflows/`: Rust CI and style checks; treat these as the authoritative CI command matrix.
- `.github/workflows/`: Python test/release/docs automation.
- `RELEASE.md`: release-only procedure. Do not publish, tag, or change release versions unless explicitly asked.

## Rust conventions

- Use Rust edition 2024 and let `cargo fmt` determine formatting; there is no repository-local rustfmt configuration.
- The crates forbid Clippy wildcard imports. Prefer explicit imports or grouped imports.
- Crate roots warn on missing documentation. Public APIs should have `///` documentation and crate-level docs should use `//!`.
- The `strict` feature upgrades warnings and unused dependencies to errors. New code must be clean with this feature.
- Keep numerical and topological conventions explicit. Preserve established point/order conventions and add targeted regression tests for correctness-sensitive changes.
- Place unit tests near implementation in `#[cfg(test)]` modules. Put cross-crate or fixture-based coverage in `tests/`. Use `approx` for floating-point comparisons where exact equality is not mathematically appropriate.
- Add or update a focused example when it materially clarifies a public API.

## Features and platform dependencies

- `ndmesh` enables `serde` by default. `mpi`, `coupe`, and `scotch` are optional; MPI propagates through the workspace crates.
- BLAS/LAPACK provider crates are dev-dependencies in the three Rust libraries. Their unit tests, examples, doctests, and integration tests explicitly link Accelerate on macOS and system OpenBLAS on Linux.
- `ndelement-python` keeps those providers as normal target dependencies because its release-built extension needs them. All-feature and MPI checks require MPI, OpenBLAS/LAPACK, SCOTCH, CMake, and libclang; do not assume these are installed locally.
- MPI examples use `mpirun`; the CI example suite uses `cargo-templated-examples` and `cargo-mpirun`.

## Validation

Run the narrowest relevant checks first, then workspace checks for shared changes.

```bash
# Format and default workspace tests
cargo fmt -- --check
cargo test

# Strict feature matrix (requires the relevant native dependencies)
RUST_MIN_STACK=8388608 cargo test --no-default-features --features "strict"
RUST_MIN_STACK=8388608 cargo test --features "strict"
RUST_MIN_STACK=8388608 cargo test --all-features

# CI lint checks for a chosen feature configuration
cargo clippy --features "strict" -- -D warnings
cargo clippy --tests --features "strict" -- -D warnings
cargo clippy --examples --features "strict" -- -D warnings

# CI release-mode example compilation
RUST_MIN_STACK=8388608 cargo test --examples --release --features "strict"
```

For documentation changes that affect public Rust APIs, also run the appropriate `cargo doc` check. The complete CI docs build needs nightly because it uses `-Zunstable-options -Zrustdoc-scrape-examples`.

## Python bindings (`ndelement-python`)

The Python package requires Python 3.10+ and uses Maturin. Its style configuration lives in `ndelement-python/pyproject.toml`: Ruff uses a 100-character line length, Google docstring convention, and mypy ignores missing imports.

```bash
cd ndelement-python
python3 -m ruff check python
python3 -m ruff format --check python
python3 -m mypy python
# After installing/developing the extension, as in CI:
python -m pytest python/test
```

CI creates a virtual environment with `uv venv .venv`, runs `maturin develop`, then runs `python -m pytest python/test`.

## Change discipline

- Keep changes scoped; do not edit generated build output under `target/`.
- Do not modify unrelated untracked files (including the repository's current `package.json` and `package-lock.json`) or release artifacts.
- Update Cargo feature propagation consistently when adding a cross-crate capability.
- Before finishing, report which checks were run and any checks blocked by unavailable native MPI/BLAS/SCOTCH dependencies.
