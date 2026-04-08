//! Meshes
pub mod local_mesh;
#[cfg(feature = "mpi")]
mod parallel_builder;
#[cfg(feature = "mpi")]
mod parallel_mesh;

pub use local_mesh::{MixedMesh, MixedMeshBuilder, SingleElementMesh, SingleElementMeshBuilder};
#[cfg(feature = "mpi")]
pub use parallel_mesh::ParallelMeshImpl;
