//! Function space traits
#[cfg(feature = "mpi")]
use mpi::traits::Communicator;
use ndelement::traits::FiniteElement;
use ndmesh::{traits::Mesh, types::Ownership};
use rlst::RlstScalar;
use std::fmt::Debug;
use std::hash::Hash;

/// Function space.
///
/// There will be three ways that the degrees of freedom (DOFs) are numbered:
///
/// 1. The local DOF numbering is the numbering of the DOFs on a single cell.
/// 2. The process DOF numbering is the numbering of the DOFs local to the current process.
///    This includes owned and ghost DOFs.
/// 3. The global DOF numbering is the numberinf of the DOFs across all processes. Note that
///    if the mpi feature is not enabaled, then this is the same as the process DOF numbering.
///
/// DOFs included in function spaces are either owned or ghosts. Owned DOFs are owned by the
/// current process. Ghost DOFs are owned by another process but information about them
/// is known by the current process because (eg) they neighbour a cell on this process.
/// If the mpi feature is disabled, all DOFs will be owned DOFs.
pub trait FunctionSpace {
    /// Scalar type
    type T: RlstScalar;
    /// Scalar type for geometry
    type TMesh: RlstScalar;
    /// Type used as identifier of different entity types
    type EntityDescriptor: Debug + PartialEq + Eq + Clone + Copy + Hash;
    /// The type for the mesh this function space is defined on
    type Mesh: Mesh<EntityDescriptor = Self::EntityDescriptor, T = Self::TMesh>;
    /// The type for the finite element this mesh is defined by
    type FiniteElement: FiniteElement<CellType = Self::EntityDescriptor, T = Self::T>;

    /// The mesh that this function space is defined on
    fn mesh(&self) -> &Self::Mesh;

    /// The finite elements used in this function space
    fn elements(&self) -> &[Self::FiniteElement];

    /// A list of entity indices that use the element with the given index
    fn entities_by_element(&self, element_index: usize) -> Option<&[usize]>;

    /// Get the process DOF numbers associated with the given entity
    fn entity_dofs(
        &self,
        entity_type: Self::EntityDescriptor,
        entity_number: usize,
    ) -> Option<&[usize]>;

    /// Get the process DOF numbers associated with the closure of the given entity
    ///
    /// The closure of an entity includes the lower dimensional entities that are on the
    /// boundary of the entity. For example, the closure of a triangle includes its
    /// edges and vertices.
    fn entity_closure_dofs(
        &self,
        entity_type: Self::EntityDescriptor,
        entity_number: usize,
    ) -> Option<&[usize]>;

    /// Get the number of process DOFs
    ///
    /// This count includes owned and ghost DOFs
    fn process_size(&self) -> usize;

    /// Get the number of owned process DOFs
    fn process_owned_size(&self) -> usize;

    /// Get the number of DOFs on all processes
    fn global_size(&self) -> usize;

    /// Get the global DOF index associated with a process DOF index
    fn global_dof_index(&self, process_dof_index: usize) -> usize;

    /// Get ownership of a process DOF
    fn ownership(&self, process_dof_index: usize) -> Ownership;
}

/// MPI parallel function space.
#[cfg(feature = "mpi")]
pub trait ParallelFunctionSpace {
    /// Process space type   
    type ProcessSpace: FunctionSpace;

    /// Communicator
    type C: Communicator;

    /// MPI communicator
    fn comm(&self) -> &Self::C;

    /// Space on the current process
    fn process_space(&self) -> &Self::ProcessSpace;

    /// Get the number of DOFs on all processes
    fn global_size(&self) -> usize;
}
