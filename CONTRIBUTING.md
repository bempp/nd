# How to contribute

## Making suggestions
If you find a bug in nd, please report it on the [issue tracker](https://codeberg.org/nd-project/nd/issues)
and label it "bug".

If you want to suggest a new feature or an improvement of a current feature, you can submit this
on the [issue tracker](https://codeberg.org/nd-project/nd/issues) and label it "enhancement".

## Contributing directly
If you want to directly submit code to nd, you can do this by forking the nd, making changes on the fork,
then submitting a pull request. If you want to contribute, but are unsure where to start, have a look at the
[issues labelled "good first issue"](https://codeberg.org/nd-project/nd/issues?labels=1727108&state=open).

### Crates
The code in nd is split into various crates, with each on in its own directory:

- [ndelement/](ndelement/) contains the ndelement create, which defines finite element basis functions
- [ndelement-python/](ndelement-python/) contains Python wrappers to the ndelement crate
- [ndmesh/](ndmesh/) contains the ndmesh create, which contains functionality for creating and working
  with finite element meshes
- [ndfunctionspace/](ndfunctionspace/) contains the ndfunctionspace create, which contains functionality
  for creating and working with finite element function spaces

### Unit tests and code quality
In order for a pull request to be merged, various status checks must pass:

- Rust unit tests must all pass. These can be run locally with `cargo test --features "strict"`.
  The strict feature is used in all our crates to enable additional checks as well as turning all
  warnings into errors.
- All examples must successfully fun. These can be run locally using
  [cargo-templated-examples](https://github.com/mscroggs/cargo-templated-examples)
  with `cargo templated-examples NPROCESSES 4`.
- Rust format checks. Automatic reformatting can be done locally by running `cargo fmt`.
- Rust clippy checks. These can be run locally with `cargo clippy`.
- Python unit tests in the ndelement wrappers must all pass. The Python wrappers in `ndelement-python`
  can be build by running `maturin develop` in that subdiretory. The tests can then be run locally with
  `python -m pytest python/test`.
- Python ruff style checks. These can be run locally with `python -m ruff check` and `python -m ruff format --check`.

These tests will be run on every pull request by out CI. If the checks fail, the reason for the failure
can be seen by clicking on the "Details" link next to the failed run.

## Code of conduct
We expect all our contributors to follow [the Contributor Covenant](CODE_OF_CONDUCT.md). Any unacceptable
behaviour can be reported to Matthew (nd@mscroggs.co.uk) or Timo (tbetcke@ucl.ac.uk).

----

Thank-you for considering contributing to nd!
