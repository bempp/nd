//! A library for creating finite element DOF maps and function spaces.

#![cfg_attr(feature = "strict", deny(warnings), deny(unused_crate_dependencies))]
#![warn(missing_docs)]

mod function_space;
pub mod traits;
pub use function_space::SerialFunctionSpace;
#[cfg(feature = "mpi")]
pub use function_space::ParallelFunctionSpace;

use ndelement as _;
use ndgrid as _;
