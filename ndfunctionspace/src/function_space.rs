//! Function spaces
mod serial;
#[cfg(feature = "mpi")]
mod parallel;
pub use serial::SerialFunctionSpace;
#[cfg(feature = "mpi")]
pub use parallel::ParallelFunctionSpace;
