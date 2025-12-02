# ndelement

[![DefElement verification](https://defelement.org/badges/ndelement.svg)](https://defelement.org/verification/ndelement.html)
[![crates.io](https://img.shields.io/crates/v/ndelement?color=blue)](https://crates.io/crates/ndelement)
[![docs.rs](https://img.shields.io/docsrs/ndelement?label=docs.rs)](https://docs.rs/ndelement/latest/ndelement/)
[![PyPI](https://img.shields.io/pypi/v/ndelement?color=blue&label=PyPI&logo=pypi&logoColor=white)](https://pypi.org/project/ndelement/)

ndelement is an open-source library written in Rust that can be used to create finite elements on 1D, 2D, or 3D reference cells.

## Using ndelement
### Rust
You can use the latest release of ndelement by adding the following to `[dependencies]` section of your Cargo.toml file:

```toml
ndelement = "0.3.0"
```

### Python
You can install the latest release of ndelement by running:

```bash
pip3 install ndelement
```

The Python functionality of the library can be tested by running:
```bash
python -m pytest ndelement/python/test
```

## Documentation
The latest documentation of nelement is available at
[bempp.github.io/nd/rust/ndelement](https://bempp.github.io/nd/rust/ndelement/) (Rust)
and [bempp.github.io/nd/python/ndelement](https://bempp.github.io/nd/python/ndelement/) (Python).

## Testing
The Rust functionality of the library can be tested by running:
```bash
cargo test
```

## Examples
Examples of use can be found in the [examples](examples/) folder.

## Getting help
Errors in should be added to the [nd GitHub issue tracker](https://github.com/bempp/nd/issues).

Questions about use can be asked on the [Bempp Discourse](https://bempp.discourse.group).

## Licence
ndelement is licensed under a BSD 3-Clause licence.
