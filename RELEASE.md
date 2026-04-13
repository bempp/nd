# Making a release

## Version numbering
The nd crates use semantic version numbering: version numbers have the format `[x].[y].[z]`.
If you are releasing a major version, you should increment `[x]` and set `[y]` and `[z]` to 0.
If you are releasing a minor version, you should increment `[y]` and set `[z]`
to zero. If you are releasing a bugfix, you should increment `[z]`.

## Releasing all crates

To make a new release of all the crates that make up nd:

0) If you are yet to make a release on your current computer, run `cargo login` and copy an API
   key from https://crates.io/me

1) Checkout the `main` branch and `git pull`, then checkout a new branch called `release-v[x].[y].[z]`:
   ```bash
   git checkout main
   git pull
   git checkout -b release-v[x].[y].[z]
   ```

2) Update the version numbers in `nd*/Cargo.toml`, `nd*/pyproject.toml` and `nd*/README.md`.

3) Update the version number in every section of `README.md`.

4) Run `cargo publish --dry-run` in each crate's subfolder and fix any errors.

5) Commit your changes and push, open a pull request to merge changes back into main, and merge the
   pull request.

6) [Create a release on Codeberg](https://codeberg.org/nd-project/nd/releases/new) from the `main` branch.
   The release tag and title should be `v[x].[y].[z]` (where `[x]`, `[y]` and `[z]` are as in step 2).
   In the "Describe this release" box, you should bullet point the main changes since the last
   release.

7) Run `cargo publish` in each crate's subfolder. This will push the new version to crates.io.
   Note: this cannot be undone, but you can use `cargo yank` to mark a version as unsuitable for use.

8) Open a pull request to `main` to update the version numbers in `nd*/Cargo.toml` and `nd*/pyproject.toml`
   to `[x].[y].[z]-dev`

9) Add the release to the next issue of [Scientific Computing in Rust Monthly](https://github.com/rust-scicomp/scientific-computing-in-rust-monthly)


## Releasing individual crates

To make a new release of the subcrate nd[crate], follow the following steps:

0) Follow step 0 in "Releasing all crates".

1) Checkout the `main` branch and `git pull`, then checkout a new branch called `nd[crate]-v[x].[y].[z]`:
   ```bash
   git checkout main
   git pull
   git checkout -b nd[crate]-v[x].[y].[z]
   ```

2) Update the version numbers in `nd[crate]/Cargo.toml` and `nd[crate]/README.md`, and `nd[crate]/pyproject.toml` if it exists.

3) Update the version number in the "Using nd[crate]" section of `README.md`.

4) Run `cargo publish --dry-run` in `nd[crate]/` and fix any errors.

5) Commit your changes and push, open a pull request to merge changes back into main, and merge the
   pull request.

6) [Create a release on Codeberg](https://codeberg.org/nd-project/nd/releases/new) from the `main` branch.
   The release tag and title should be `nd[crate] v[x].[y].[z]` (where `[x]`, `[y]` and `[z]` are as in step 2).
   In the "Describe this release" box, you should bullet point the main changes since the last
   release.

7) Run `cargo publish` in `nd[crate]/`. This will push the new version to crates.io.
   Note: this cannot be undone, but you can use `cargo yank` to mark a version as unsuitable for use.

8) Open a pull request to `main` to update the version numbers in `nd[crate]/Cargo.toml` and `nd[crate]/pyproject.toml` (if it exists)
   to `[x].[y].[z]-dev`

9) Add the release to the next issue of [Scientific Computing in Rust Monthly](https://github.com/rust-scicomp/scientific-computing-in-rust-monthly)
