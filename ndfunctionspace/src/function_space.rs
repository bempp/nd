//! Function spaces
#[cfg(feature = "mpi")]
mod parallel;
mod serial;
#[cfg(feature = "mpi")]
pub use parallel::ParallelFunctionSpaceImpl;
pub use serial::FunctionSpaceImpl;
