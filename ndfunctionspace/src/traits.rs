//! Function space traits
use ndelement::traits::FiniteElement;
use ndgrid::{traits::Grid, types::Ownership};

/// Function space.
pub trait FunctionSpace {
    /// The type for the grid this function space is defined on
    type Grid: Grid;
    /// The type for the finite element this grid is defined by
    type FiniteElement: FiniteElement;

    /// The grid that this function space is defined on
    fn grid(&self) -> &Self::Grid;

    /// The finite elements used in this function space
    fn elements(&self) -> &[Self::FiniteElement];

    /// A list of cell indices that use the element with the given index
    fn cells_by_element(&self, element_index: usize) -> Option<&[usize]>;

    /// Get the local DOFs numbers associated with the given entity
    fn entity_dofs(
        &self,
        entity_type: <Self::Grid as Grid>::EntityDescriptor,
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
