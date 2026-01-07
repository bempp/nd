//! Parallel function space
use crate::{function_space::SerialFunctionSpace, traits::FunctionSpace};
use itertools::izip;
use ndelement::{
    reference_cell,
    traits::{ElementFamily, MappedFiniteElement},
    types::ReferenceCellType,
};
use ndgrid::{
    traits::{Entity, Grid, Topology, ParallelGrid},
    types::Ownership,
};
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

/// Function space.
pub struct ParallelFunctionSpace<
    'a,
    E: Debug + PartialEq + Eq + Clone + Copy + Hash,
    G: Grid<EntityDescriptor = E>,
    PG: ParallelGrid<LocalGrid=G>,
    F: MappedFiniteElement<CellType = E>,
> {
    grid: &'a PG,
    local_space: SerialFunctionSpace<'a, E, G, F>,
}

impl<
    'a,
    G: Grid<EntityDescriptor = ReferenceCellType>,
    PG: ParallelGrid<LocalGrid=G>,
    F: MappedFiniteElement<CellType = ReferenceCellType>,
> ParallelFunctionSpace<'a, ReferenceCellType, G, PG, F>
{
    /// Create a new parallel function space
    pub fn new<EF: ElementFamily<FiniteElement = F, CellType = ReferenceCellType>>(
        grid: &'a PG,
        family: &EF,
    ) -> Self {
        let local_space = SerialFunctionSpace::new(grid.local_grid(), family);
        Self {
            grid,
            local_space,
        }
    }
}

impl<
    'a,
    E: Debug + PartialEq + Eq + Clone + Copy + Hash,
    G: Grid<EntityDescriptor = E>,
    PG: ParallelGrid<LocalGrid=G>,
    F: MappedFiniteElement<CellType = E>,
> ParallelFunctionSpace<'a, E, G, PG, F>
{
    pub fn global_size(&self) -> usize { 0 }

    pub fn local_space(&self) -> &SerialFunctionSpace<'a, E, G, F> {
        &self.local_space
    }
}
