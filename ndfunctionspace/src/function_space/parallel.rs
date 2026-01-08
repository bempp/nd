//! Parallel function space
use crate::{
    function_space::FunctionSpaceImpl,
    traits::{FunctionSpace, ParallelFunctionSpace},
};
use mpi::{
    collective::SystemOperation,
    traits::{Communicator, CommunicatorCollectives, Equivalence},
};
use ndelement::{
    traits::{ElementFamily, MappedFiniteElement},
    types::ReferenceCellType,
};
use ndgrid::{
    traits::{Entity, Grid, ParallelGrid},
    types::Ownership,
};
use rlst::distributed_tools::array_tools::all_to_allv;
use std::fmt::Debug;
use std::hash::Hash;

/// Local function space on a process
pub struct LocalFunctionSpace<S: FunctionSpace> {
    space: S,
    global_size: usize,
    global_dof_indices: Vec<usize>,
    ownership: Vec<Ownership>,
}

impl<S: FunctionSpace> LocalFunctionSpace<S> {
    fn new(
        space: S,
        global_size: usize,
        global_dof_indices: Vec<usize>,
        ownership: Vec<Ownership>,
    ) -> Self {
        Self {
            space,
            global_size,
            global_dof_indices,
            ownership,
        }
    }
}

impl<S: FunctionSpace> FunctionSpace for LocalFunctionSpace<S> {
    type EntityDescriptor = S::EntityDescriptor;
    type Grid = S::Grid;
    type FiniteElement = S::FiniteElement;

    fn grid(&self) -> &S::Grid {
        self.space.grid()
    }

    fn elements(&self) -> &[S::FiniteElement] {
        self.space.elements()
    }

    fn entities_by_element(&self, element_index: usize) -> Option<&[usize]> {
        self.space.entities_by_element(element_index)
    }

    fn entity_dofs(
        &self,
        entity_type: S::EntityDescriptor,
        entity_number: usize,
    ) -> Option<&[usize]> {
        self.space.entity_dofs(entity_type, entity_number)
    }

    fn entity_closure_dofs(
        &self,
        entity_type: S::EntityDescriptor,
        entity_number: usize,
    ) -> Option<&[usize]> {
        self.space.entity_closure_dofs(entity_type, entity_number)
    }

    fn local_size(&self) -> usize {
        self.space.local_size()
    }

    fn global_size(&self) -> usize {
        self.global_size
    }

    fn global_dof_index(&self, local_dof_index: usize) -> usize {
        self.global_dof_indices[local_dof_index]
    }

    fn ownership(&self, local_dof_index: usize) -> Ownership {
        self.ownership[local_dof_index]
    }
}

/// Parallel function space.
pub struct ParallelFunctionSpaceImpl<
    'a,
    E: Debug + PartialEq + Eq + Clone + Copy + Hash,
    G: Grid<EntityDescriptor = E>,
    PG: ParallelGrid<LocalGrid = G>,
    F: MappedFiniteElement<CellType = E>,
    S: FunctionSpace<Grid = G, EntityDescriptor = E, FiniteElement = F>,
> {
    grid: &'a PG,
    local_space: LocalFunctionSpace<S>,
    global_size: usize,
}

/// A temporary DOF index value used when constructing a parallel DOF map
#[derive(Debug, Clone, PartialEq)]
enum DofIndex {
    /// No value. This should only be used as a placeholder value
    None,
    /// A local DOF
    Local,
    /// A DOF owned by another process, storing the process id, index of ghost in the list of ghosts, and DOF index on that entity
    Ghost(usize, usize, usize),
}

#[derive(Equivalence, Debug, Clone, PartialEq)]
struct GhostEntity {
    entity_type: ReferenceCellType,
    entity_index: usize,
}

impl GhostEntity {
    fn new(entity_type: ReferenceCellType, entity_index: usize) -> Self {
        Self {
            entity_type,
            entity_index,
        }
    }
}

impl<
    'a,
    G: Grid<EntityDescriptor = ReferenceCellType>,
    PG: ParallelGrid<LocalGrid = G>,
    F: MappedFiniteElement<CellType = ReferenceCellType>,
>
    ParallelFunctionSpaceImpl<
        'a,
        ReferenceCellType,
        G,
        PG,
        F,
        FunctionSpaceImpl<'a, ReferenceCellType, G, F>,
    >
{
    /// Create a new parallel function space
    pub fn new<EF: ElementFamily<FiniteElement = F, CellType = ReferenceCellType>>(
        grid: &'a PG,
        family: &EF,
    ) -> Self {
        let comm = grid.comm();
        let local_grid = grid.local_grid();
        let serial_space = FunctionSpaceImpl::new(local_grid, family);

        let mut dof_indices = vec![DofIndex::None; serial_space.local_size()];

        // Entities to ask other processes about
        let mut ask_for = vec![vec![]; comm.size() as usize];

        // Identify which DOFs are associated with owned entities and which are associated with ghosts
        for dim in 0..=local_grid.topology_dim() {
            for entity_type in local_grid.entity_types(dim) {
                for entity in local_grid.entity_iter(*entity_type) {
                    let dofs = serial_space
                        .entity_dofs(*entity_type, entity.local_index())
                        .unwrap();
                    if !dofs.is_empty() {
                        match entity.ownership() {
                            Ownership::Owned => {
                                for j in dofs {
                                    dof_indices[*j] = DofIndex::Local
                                }
                            }
                            Ownership::Ghost(process, entity_index) => {
                                for (i, j) in dofs.iter().enumerate() {
                                    dof_indices[*j] =
                                        DofIndex::Ghost(process, ask_for[process].len(), i);
                                }
                                ask_for[process].push((*entity_type, entity_index));
                            }
                            _ => {
                                panic!("Invalid ownership");
                            }
                        }
                    }
                }
            }
        }

        // Number the DOFs owned by this process, starting at 0
        let mut ndofs = 0;
        let mut local_dof_indices = vec![None; serial_space.local_size()];
        for (i, j) in dof_indices.iter().enumerate() {
            if let DofIndex::Local = j {
                local_dof_indices[i] = Some(ndofs);
                ndofs += 1;
            }
        }

        // Communicate number of DOFs to all processes
        let mut global_size = 0;
        let mut offset = 0;
        comm.exclusive_scan_into(&ndofs, &mut offset, SystemOperation::sum());
        comm.all_reduce_into(&ndofs, &mut global_size, SystemOperation::sum());

        // Ask other processes about their ghosts
        let counts = ask_for.iter().map(|i| i.len()).collect::<Vec<_>>();
        let ask_for_flat = ask_for
            .iter()
            .flatten()
            .map(|(t, i)| GhostEntity::new(*t, *i))
            .collect::<Vec<_>>();

        let (recv_counts, recv_data) = all_to_allv(comm, &counts, &ask_for_flat);

        // Get indices to send to other processes
        let asked_for_local = recv_data
            .iter()
            .map(|entity| {
                serial_space
                    .entity_dofs(entity.entity_type, entity.entity_index)
                    .unwrap()[0]
            })
            .collect::<Vec<_>>();
        let asked_for_global = asked_for_local
            .iter()
            .map(|i| {
                let dof = local_dof_indices[*i].unwrap();
                dof + offset
            })
            .collect::<Vec<_>>();

        // Get indices of ghosts from other processes
        let (_, local_ghost_data) = all_to_allv(comm, &recv_counts, &asked_for_local);
        let (_, global_ghost_data) = all_to_allv(comm, &recv_counts, &asked_for_global);

        let mut start = 0;
        let local_ghost_indices = counts
            .iter()
            .map(|c| {
                let data = &local_ghost_data[start..start + c];
                start += c;
                data.to_vec()
            })
            .collect::<Vec<_>>();
        let mut start = 0;
        let global_ghost_indices = counts
            .iter()
            .map(|c| {
                let data = &global_ghost_data[start..start + c];
                start += c;
                data.to_vec()
            })
            .collect::<Vec<_>>();

        // Assign global numbers to DOFs on this process
        // Here it is assumed that the DOFs associated with each entity are contiguously numbered
        let offset = 1;
        let global_dof_indices = dof_indices
            .iter()
            .enumerate()
            .map(|(i, index)| match index {
                DofIndex::Local => offset + local_dof_indices[i].unwrap(),
                DofIndex::Ghost(process, index, number) => {
                    global_ghost_indices[*process][*index] + *number
                }
                DofIndex::None => {
                    panic!("Unset DOF index");
                }
            })
            .collect::<Vec<_>>();
        let ownership = dof_indices
            .iter()
            .map(|i| match i {
                DofIndex::Local => Ownership::Owned,
                DofIndex::Ghost(process, index, number) => {
                    Ownership::Ghost(*process, local_ghost_indices[*process][*index] + *number)
                }
                DofIndex::None => {
                    panic!("Unset DOF index");
                }
            })
            .collect::<Vec<_>>();

        let local_space =
            LocalFunctionSpace::new(serial_space, global_size, global_dof_indices, ownership);
        Self {
            grid,
            local_space,
            global_size,
        }
    }
}

impl<
    'a,
    E: Debug + PartialEq + Eq + Clone + Copy + Hash,
    G: Grid<EntityDescriptor = E>,
    PG: ParallelGrid<LocalGrid = G>,
    F: MappedFiniteElement<CellType = E>,
    S: FunctionSpace<Grid = G, EntityDescriptor = E, FiniteElement = F>,
> ParallelFunctionSpace for ParallelFunctionSpaceImpl<'a, E, G, PG, F, S>
{
    type LocalSpace = LocalFunctionSpace<S>;
    type C = PG::C;

    fn comm(&self) -> &PG::C {
        self.grid.comm()
    }

    fn local_space(&self) -> &LocalFunctionSpace<S> {
        &self.local_space
    }

    fn global_size(&self) -> usize {
        self.global_size
    }
}
