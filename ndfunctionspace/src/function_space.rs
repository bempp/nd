//! Function space traits
use crate::traits::FunctionSpace as FunctionSpaceTrait;
use ndelement::traits::FiniteElement;
use ndgrid::{traits::Grid, types::Ownership};
use std::collections::HashMap;

/// Function space.
pub struct FunctionSpace<'a, G: Grid, F: FiniteElement> {
    grid: &'a G,
    elements: Vec<F>,
    entity_dofs: HashMap<G::EntityDescriptor, Vec<Vec<usize>>>,
}

impl<'a, G: Grid, F: FiniteElement> FunctionSpaceTrait for FunctionSpace<'a, G, F> {
    type Grid = G;
    type FiniteElement = F;

    fn grid(&self) -> &G {
        self.grid
    }

    fn elements(&self) -> &[F] {
        &self.elements
    }

    fn entities_by_element(&self, element_index: usize) -> Option<&[usize]> {
        todo!();
    }

    fn entity_dofs(
        &self,
        entity_type: <Self::Grid as Grid>::EntityDescriptor,
        entity_number: usize,
    ) -> Option<&[usize]> {
        if let Some(i) = self.entity_dofs.get(&entity_type) {
            if let Some(j) = i.get(entity_number) {
                Some(j)
            } else {
                None
            }
        } else { None }
    }

    fn local_size(&self) -> usize {
        todo!();
    }

    fn global_size(&self) -> usize {
        todo!();
    }

    fn global_dof_index(&self, local_dof_index: usize) -> usize {
        todo!();
    }

    fn ownership(&self, local_dof_index: usize) -> Ownership {
        todo!();
    }
}
