//! Traits

mod builder;
mod entity;
mod geometry;
mod geometry_map;
mod io;
mod mesh;
mod topology;

pub use builder::Builder;
#[cfg(feature = "mpi")]
pub use builder::ParallelBuilder;
pub(crate) use builder::{GeometryBuilder, MeshBuilder, TopologyBuilder};
pub use entity::Entity;
pub use geometry::{Geometry, Point};
pub use geometry_map::GeometryMap;
#[cfg(feature = "serde")]
pub(crate) use io::ConvertToSerializable;
pub use io::{GmshExport, GmshImport};
#[cfg(feature = "serde")]
pub use io::{RONExport, RONImport};
#[cfg(feature = "mpi")]
#[cfg(feature = "serde")]
pub use io::{RONExportParallel, RONImportParallel};
pub use mesh::Mesh;
#[cfg(feature = "mpi")]
pub use mesh::{DistributableMesh, ParallelMesh};
pub use topology::Topology;
