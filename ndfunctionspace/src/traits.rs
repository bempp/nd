//! Function space traits
#[cfg(feature = "mpi")]
use mpi::traits::Communicator;
use ndelement::traits::FiniteElement;
use ndgrid::{traits::Grid, types::Ownership};
use std::fmt::Debug;
use std::hash::Hash;

/// Function space.
pub trait FunctionSpace {
    /// Type used as identifier of different entity types
    type EntityDescriptor: Debug + PartialEq + Eq + Clone + Copy + Hash;
    /// The type for the grid this function space is defined on
    type Grid: Grid<EntityDescriptor = Self::EntityDescriptor>;
    /// The type for the finite element this grid is defined by
    type FiniteElement: FiniteElement<CellType = Self::EntityDescriptor>;

    /// The grid that this function space is defined on
    fn grid(&self) -> &Self::Grid;

    /// The finite elements used in this function space
    fn elements(&self) -> &[Self::FiniteElement];

    /// A list of entity indices that use the element with the given index
    fn entities_by_element(&self, element_index: usize) -> Option<&[usize]>;

    /// Get the local DOFs numbers associated with the given entity
    fn entity_dofs(
        &self,
        entity_type: Self::EntityDescriptor,
        entity_number: usize,
    ) -> Option<&[usize]>;

    /// Get the local DOFs numbers associated with the closure of the given entity
    fn entity_closure_dofs(
        &self,
        entity_type: Self::EntityDescriptor,
        entity_number: usize,
    ) -> Option<&[usize]>;

    /// Get the number of DOFs associated with the local process
    fn local_size(&self) -> usize;

    /// Get the number of DOFs on all processes
    fn global_size(&self) -> usize;

    /// Get the global DOF index associated with a local DOF index
    fn global_dof_index(&self, local_dof_index: usize) -> usize;

    /// Get ownership of a local DOF
    fn ownership(&self, local_dof_index: usize) -> Ownership;
}

/// MPI parallel function space.
#[cfg(feature = "mpi")]
pub trait ParallelFunctionSpace {
    /// Local space type   
    type LocalSpace: FunctionSpace;

    /// Communicator
    type C: Communicator;

    /// MPI communicator
    fn comm(&self) -> &Self::C;

    /// Local space on the current process
    fn local_space(&self) -> &Self::LocalSpace;

    /// Get the number of DOFs on all processes
    fn global_size(&self) -> usize;
}
