# nd: Numerical Discretisation in Rust

This repo contains a number of crates that can be used for solving problems using numerical
discretisation in Rust, for example by using the finite element method or boundary element
method.

## The crates

### [ndelement](ndelement/)
[![DefElement verification](https://defelement.org/badges/ndelement.svg)](https://defelement.org/verification/ndelement.html)
[![crates.io](https://img.shields.io/crates/v/ndelement?color=blue)](https://crates.io/crates/ndelement)
[![docs.rs](https://img.shields.io/docsrs/ndelement?label=docs.rs)](https://docs.rs/ndelement/latest/ndelement/)
[![PyPI](https://img.shields.io/pypi/v/ndelement?color=blue&label=PyPI&logo=pypi&logoColor=white)](https://pypi.org/project/ndelement/)

ndelement is an open-source library written in Rust that can be used to create finite elements on 1D, 2D, or 3D reference cells.

#### Using ndelement
##### Rust
You can use the latest release of ndelement by adding the following to `[dependencies]` section of your Cargo.toml file:

```toml
ndelement = "0.3.0"
```

##### Python
You can install the latest release of ndelement by running:

```bash
pip3 install ndelement
```

The Python functionality of the library can be tested by running:
```bash
python -m pytest ndelement/python/test
```

### [ndgrid](ndgrid/)
[![crates.io](https://img.shields.io/crates/v/ndgrid?color=blue)](https://crates.io/crates/ndgrid)

ndgrid is an open-source library written in Rust for handling finite element grids/meshes.

### Using ndgrid
You can use the latest release of ndgrid by adding the following to `[dependencies]` section of your Cargo.toml file:

```toml
ndgrid = "0.1.5"
```

## Documentation
The latest documentation of the crates in this repo is available at
[bempp.github.io/nd/](https://bempp.github.io/nd/).

## Testing
The Rust functionality of the library can be tested by running:
```bash
cargo test
```

## Examples
Examples of use can be found in the examples folder of each crate, for example
[the ndelement examples are in the folder ndelement/examples](ndelement/examples/) and
[the ndgrid examples are in the folder ndgrid/examples](ndgrid/examples/).

## Getting help
Errors in the library should be added to the [GitHub issue tracker](https://github.com/bempp/nd/issues).

Questions about the crates and their use can be asked on the [Bempp Discourse](https://bempp.discourse.group).

## Licence
All the crates included here are licensed under a BSD 3-Clause licence. Full text of the licence can be found [here](LICENSE).
