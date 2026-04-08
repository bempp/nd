//! Parallel mesh
#[cfg(feature = "serde")]
use crate::traits::{ConvertToSerializable, RONImportParallel};
use crate::{
    mesh::local_mesh::LocalMesh,
    traits::{Mesh, ParallelMesh},
    types::{Ownership, Scalar},
};
use mpi::traits::Communicator;
use rlst::distributed_tools::IndexLayout;
#[cfg(feature = "serde")]
use std::hash::Hash;
use std::{collections::HashMap, fmt::Debug};

/// Parallel mesh
#[derive(Debug)]
pub struct ParallelMeshImpl<'a, C: Communicator, G: Mesh + Sync> {
    comm: &'a C,
    local_mesh: LocalMesh<G>,
    cell_layout: std::rc::Rc<IndexLayout<'a, C>>,
}

impl<'a, C: Communicator, G: Mesh + Sync> ParallelMeshImpl<'a, C, G> {
    /// Create new
    pub fn new(
        comm: &'a C,
        serial_mesh: G,
        ownership: HashMap<G::EntityDescriptor, Vec<Ownership>>,
        global_indices: HashMap<G::EntityDescriptor, Vec<usize>>,
    ) -> Self {
        let local_mesh = LocalMesh::new(serial_mesh, ownership, global_indices);
        let owned_cell_count = local_mesh.owned_cell_count();
        Self {
            comm,
            local_mesh,
            cell_layout: std::rc::Rc::new(IndexLayout::from_local_counts(owned_cell_count, comm)),
        }
    }
}

#[cfg(feature = "serde")]
impl<
    'a,
    EntityDescriptor: Debug + PartialEq + Eq + Clone + Copy + Hash + serde::Serialize,
    C: Communicator + 'a,
    G: Mesh<EntityDescriptor = EntityDescriptor> + Sync + ConvertToSerializable,
> RONImportParallel<'a, C> for ParallelMeshImpl<'a, C, G>
where
    for<'de2> <G as ConvertToSerializable>::SerializableType: serde::Deserialize<'de2>,
    for<'de2> EntityDescriptor: serde::Deserialize<'de2>,
    Self: 'a,
{
    fn create_from_ron_info(comm: &'a C, local_mesh: LocalMesh<G>) -> Self {
        let owned_cell_count = local_mesh.owned_cell_count();
        Self {
            comm,
            local_mesh,
            cell_layout: std::rc::Rc::new(IndexLayout::from_local_counts(owned_cell_count, comm)),
        }
    }
}

impl<T: Scalar, C: Communicator, G: Mesh<T = T> + Sync> ParallelMesh
    for ParallelMeshImpl<'_, C, G>
{
    type LocalMesh = LocalMesh<G>;

    type C = C;

    type T = T;
    fn comm(&self) -> &C {
        self.comm
    }
    fn local_mesh(&self) -> &Self::LocalMesh {
        &self.local_mesh
    }

    fn cell_layout(&self) -> std::rc::Rc<IndexLayout<'_, C>> {
        self.cell_layout.clone()
    }
}
