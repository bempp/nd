//! Mesh builder
use crate::{traits::Mesh, types::Scalar};
#[cfg(feature = "mpi")]
use crate::{traits::ParallelMesh, types::GraphPartitioner};
#[cfg(feature = "mpi")]
use mpi::traits::Communicator;
use std::fmt::Debug;
use std::hash::Hash;

/// A builder is a factory that creates meshes.
///
/// After instantiation points and cells can be added.
/// To build the actual mesh call [Builder::create_mesh].
pub trait Builder {
    /// Type used as identifier of different entity types
    type EntityDescriptor: Debug + PartialEq + Eq + Clone + Copy + Hash;

    /// The type of the mesh that the builder creates
    type Mesh: Mesh<EntityDescriptor = Self::EntityDescriptor>;
    /// The floating point type used for coordinates
    type T: Scalar;
    /// The type of the data that is input to add a cell
    type CellData<'a>;

    /// Add a point to the mesh
    fn add_point(&mut self, id: usize, data: &[Self::T]);

    /// Add parametric coordinates for a point (optional)
    fn add_point_parametric_coords(&mut self, _id: usize, _entity_dim: usize, _coords: &[Self::T]);

    /// Add a cell to the mesh
    fn add_cell(&mut self, id: usize, cell_data: Self::CellData<'_>);

    /// Add a cell to the mesh
    fn add_cell_from_nodes_and_type(
        &mut self,
        id: usize,
        nodes: &[usize],
        cell_type: Self::EntityDescriptor,
        cell_degree: usize,
    );

    /// Create the mesh
    fn create_mesh(&self) -> Self::Mesh;

    /// Number of points
    fn point_count(&self) -> usize;

    /// Number of cells
    fn cell_count(&self) -> usize;

    /// Get the insertion ids of each point
    fn point_indices_to_ids(&self) -> &[usize];

    /// Get the insertion ids of each cell
    fn cell_indices_to_ids(&self) -> &[usize];

    /// Get the indices of the points of a cell
    fn cell_points(&self, index: usize) -> &[usize];

    /// Get the indices of the points of a cell
    fn cell_vertices(&self, index: usize) -> &[usize];

    /// Get the coordinates of a point
    fn point(&self, index: usize) -> &[Self::T];

    /// Get all points
    fn points(&self) -> &[Self::T];

    /// Get parametric coordinates for a point, if available
    fn point_parametric_coords(&self, _index: usize) -> Option<(usize, &[Self::T])>;

    /// Get the type of a cell
    fn cell_type(&self, index: usize) -> Self::EntityDescriptor;

    /// Get the degree of a cell's geometry
    fn cell_degree(&self, index: usize) -> usize;

    /// Geometric dimension
    fn gdim(&self) -> usize;

    /// Topoligical dimension
    fn tdim(&self) -> usize;

    /// Number of points in a cell with the given type and degree
    fn npts(&self, cell_type: Self::EntityDescriptor, degree: usize) -> usize;
}

/// Trait for building a geometry
///
/// This trait is usually not called by the user. It provides
/// an interface to building the geometry information of the mesh.
pub(crate) trait GeometryBuilder: Builder {
    /// Mesh geometry type
    type MeshGeometry;

    /// Create geometry
    fn create_geometry(
        &self,
        point_ids: &[usize],
        coordinates: &[Self::T],
        cell_points: &[usize],
        cell_types: &[Self::EntityDescriptor],
        cell_degrees: &[usize],
    ) -> Self::MeshGeometry;
}

/// Trait for building a topology
///
/// This trait is usually not called by the user. It provides
/// an interface to building the topology information of the mesh.
pub(crate) trait TopologyBuilder: Builder {
    /// Mesh topology type
    type MeshTopology;

    /// Create topology
    fn create_topology(
        &self,
        vertex_ids: Vec<usize>,
        cell_ids: Vec<usize>,
        cells: &[usize],
        cell_types: &[Self::EntityDescriptor],
    ) -> Self::MeshTopology;

    /// Extract the cell vertices from the cell points
    fn extract_vertices(
        &self,
        cell_points: &[usize],
        cell_types: &[Self::EntityDescriptor],
        cell_degrees: &[usize],
    ) -> Vec<usize>;
}

/// Trait for building a mesh from topology and geometry
///
/// This trait is usually not called by the user. It provides
/// an interface to building the mesh from a given topology and Geometry.
pub(crate) trait MeshBuilder: Builder + GeometryBuilder + TopologyBuilder {
    /// Create topology
    fn create_mesh_from_topology_geometry(
        &self,
        topology: <Self as TopologyBuilder>::MeshTopology,
        geometry: <Self as GeometryBuilder>::MeshGeometry,
    ) -> <Self as Builder>::Mesh;
}

/// MPI parallelized mesh builder
#[cfg(feature = "mpi")]
pub trait ParallelBuilder: Builder {
    /// Parallel mesh type
    type ParallelMesh<'a, C: Communicator + 'a>: ParallelMesh<C = C>
    where
        Self: 'a;

    /// Create a parallel mesh (call from root)
    fn create_parallel_mesh_root<'a, C: Communicator>(
        &self,
        comm: &'a C,
        partitioner: GraphPartitioner,
    ) -> Self::ParallelMesh<'a, C>;

    /// Create a parallel mesh (call from other processes)
    fn create_parallel_mesh<'a, C: Communicator>(
        &self,
        comm: &'a C,
        root_rank: i32,
    ) -> Self::ParallelMesh<'a, C>;
}
