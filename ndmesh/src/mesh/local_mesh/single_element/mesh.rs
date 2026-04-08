//! Single element mesh
#[cfg(feature = "mpi")]
use crate::ParallelMeshImpl;
#[cfg(feature = "mpi")]
use crate::{
    SingleElementMeshBuilder,
    traits::{Builder, DistributableMesh, ParallelBuilder},
    types::GraphPartitioner,
};
#[cfg(feature = "serde")]
use crate::{
    geometry::single_element::SerializableGeometry, topology::single_type::SerializableTopology,
    traits::ConvertToSerializable,
};
use crate::{
    geometry::{GeometryMap, SingleElementEntityGeometry, SingleElementGeometry},
    topology::single_type::{SingleTypeEntityTopology, SingleTypeTopology},
    traits::{Entity, Mesh},
    types::{Ownership, Scalar},
};
#[cfg(feature = "mpi")]
use mpi::traits::{Communicator, Equivalence};
use ndelement::{
    ciarlet::{CiarletElement, LagrangeElementFamily},
    map::IdentityMap,
    reference_cell,
    traits::{ElementFamily, FiniteElement, MappedFiniteElement},
    types::{Continuity, ReferenceCellType},
};
use rlst::{
    Array, ValueArrayImpl,
    dense::{base_array::BaseArray, data_container::VectorContainer},
    rlst_dynamic_array,
};

/// Single element mesh entity
#[derive(Debug)]
pub struct SingleElementMeshEntity<
    'a,
    T: Scalar,
    E: MappedFiniteElement<CellType = ReferenceCellType, T = T>,
> {
    mesh: &'a SingleElementMesh<T, E>,
    cell_index: usize,
    entity_dim: usize,
    entity_index: usize,
}

impl<'e, T: Scalar, E: MappedFiniteElement<CellType = ReferenceCellType, T = T>>
    SingleElementMeshEntity<'e, T, E>
{
    /// Create new
    pub fn new(
        mesh: &'e SingleElementMesh<T, E>,
        cell_index: usize,
        entity_dim: usize,
        entity_index: usize,
    ) -> Self {
        Self {
            mesh,
            cell_index,
            entity_dim,
            entity_index,
        }
    }
}
impl<T: Scalar, E: MappedFiniteElement<CellType = ReferenceCellType, T = T>> Entity
    for SingleElementMeshEntity<'_, T, E>
{
    type T = T;
    type EntityDescriptor = ReferenceCellType;
    type Topology<'a>
        = SingleTypeEntityTopology<'a>
    where
        Self: 'a;
    type Geometry<'a>
        = SingleElementEntityGeometry<'a, T, E>
    where
        Self: 'a;
    fn entity_type(&self) -> ReferenceCellType {
        self.mesh.topology.entity_types()[self.entity_dim]
    }
    fn local_index(&self) -> usize {
        self.mesh
            .topology
            .cell_entity_index(self.cell_index, self.entity_dim, self.entity_index)
    }
    fn global_index(&self) -> usize {
        self.local_index()
    }
    fn geometry(&self) -> Self::Geometry<'_> {
        SingleElementEntityGeometry::new(
            &self.mesh.geometry,
            self.cell_index,
            self.entity_dim,
            self.entity_index,
        )
    }
    fn topology(&self) -> Self::Topology<'_> {
        SingleTypeEntityTopology::new(&self.mesh.topology, self.entity_type(), self.local_index())
    }
    fn ownership(&self) -> Ownership {
        Ownership::Owned
    }
    fn id(&self) -> Option<usize> {
        self.mesh
            .topology
            .entity_id(self.entity_dim, self.local_index())
    }
}

/// Single element mesh entity iterator
#[derive(Debug)]
pub struct SingleElementMeshEntityIter<
    'a,
    T: Scalar,
    E: MappedFiniteElement<CellType = ReferenceCellType, T = T>,
> {
    mesh: &'a SingleElementMesh<T, E>,
    entity_type: ReferenceCellType,
    index: usize,
}

impl<'a, T: Scalar, E: MappedFiniteElement<CellType = ReferenceCellType, T = T>>
    SingleElementMeshEntityIter<'a, T, E>
{
    /// Create new
    pub fn new(mesh: &'a SingleElementMesh<T, E>, entity_type: ReferenceCellType) -> Self {
        Self {
            mesh,
            entity_type,
            index: 0,
        }
    }
}
impl<'a, T: Scalar, E: MappedFiniteElement<CellType = ReferenceCellType, T = T>> Iterator
    for SingleElementMeshEntityIter<'a, T, E>
{
    type Item = SingleElementMeshEntity<'a, T, E>;

    fn next(&mut self) -> Option<SingleElementMeshEntity<'a, T, E>> {
        self.index += 1;
        self.mesh.entity(self.entity_type, self.index - 1)
    }
}

/// Serial single element mesh
#[derive(Debug)]
pub struct SingleElementMesh<T: Scalar, E: MappedFiniteElement<CellType = ReferenceCellType, T = T>>
{
    topology: SingleTypeTopology,
    geometry: SingleElementGeometry<T, E>,
}

#[cfg(feature = "serde")]
#[derive(serde::Serialize, Debug, serde::Deserialize)]
#[serde(bound = "for<'de2> T: serde::Deserialize<'de2>")]
pub struct SerializableMesh<T: Scalar + serde::Serialize>
where
    for<'de2> T: serde::Deserialize<'de2>,
{
    topology: SerializableTopology,
    geometry: SerializableGeometry<T>,
}

#[cfg(feature = "serde")]
impl<T: Scalar + serde::Serialize> ConvertToSerializable
    for SingleElementMesh<T, CiarletElement<T, IdentityMap, T>>
{
    type SerializableType = SerializableMesh<T>;
    fn to_serializable(&self) -> SerializableMesh<T> {
        SerializableMesh {
            topology: self.topology.to_serializable(),
            geometry: self.geometry.to_serializable(),
        }
    }
    fn from_serializable(s: SerializableMesh<T>) -> Self {
        Self {
            topology: SingleTypeTopology::from_serializable(s.topology),
            geometry: SingleElementGeometry::from_serializable(s.geometry),
        }
    }
}

impl<T: Scalar, E: MappedFiniteElement<CellType = ReferenceCellType, T = T>>
    SingleElementMesh<T, E>
{
    /// Create new
    pub fn new(topology: SingleTypeTopology, geometry: SingleElementGeometry<T, E>) -> Self {
        Self { topology, geometry }
    }
}

impl<T: Scalar> SingleElementMesh<T, CiarletElement<T, IdentityMap, T>> {
    /// Create new from raw data
    pub fn new_from_raw_data(
        coordinates: &[T],
        gdim: usize,
        cells: &[usize],
        cell_type: ReferenceCellType,
        geometry_degree: usize,
    ) -> Self {
        let npts = coordinates.len() / gdim;
        let mut points = rlst_dynamic_array!(T, [gdim, npts]);
        points.data_mut().unwrap().copy_from_slice(coordinates);

        let family = LagrangeElementFamily::<T, T>::new(geometry_degree, Continuity::Standard);

        let geometry = SingleElementGeometry::<T, CiarletElement<T, IdentityMap, T>>::new(
            cell_type, points, cells, &family,
        );

        let points_per_cell = family.element(cell_type).dim();
        let tpoints_per_cell = reference_cell::entity_counts(cell_type)[0];
        let ncells = cells.len() / points_per_cell;

        let mut tcells = vec![0; tpoints_per_cell * ncells];
        for c in 0..ncells {
            tcells[c * tpoints_per_cell..(c + 1) * tpoints_per_cell].copy_from_slice(
                &cells[c * points_per_cell..c * points_per_cell + tpoints_per_cell],
            );
        }

        let topology = SingleTypeTopology::new(&tcells, cell_type, None, None);

        Self { topology, geometry }
    }
}

impl<T: Scalar, E: MappedFiniteElement<CellType = ReferenceCellType, T = T>> Mesh
    for SingleElementMesh<T, E>
{
    type T = T;
    type Entity<'a>
        = SingleElementMeshEntity<'a, T, E>
    where
        Self: 'a;
    type GeometryMap<'a>
        = GeometryMap<'a, T, BaseArray<VectorContainer<T>, 2>, BaseArray<VectorContainer<usize>, 2>>
    where
        Self: 'a;
    type EntityDescriptor = ReferenceCellType;
    type EntityIter<'a>
        = SingleElementMeshEntityIter<'a, T, E>
    where
        Self: 'a;

    fn geometry_dim(&self) -> usize {
        self.geometry.dim()
    }
    fn topology_dim(&self) -> usize {
        self.topology.dim()
    }

    fn entity(
        &self,
        entity_type: ReferenceCellType,
        local_index: usize,
    ) -> Option<Self::Entity<'_>> {
        let dim = reference_cell::dim(entity_type);
        if local_index < self.topology.entity_count(entity_type) {
            if dim == self.topology_dim() {
                Some(SingleElementMeshEntity::new(self, local_index, dim, 0))
            } else {
                let cell = self.topology.upward_connectivity[dim][self.topology_dim() - dim - 1]
                    [local_index][0];
                let dc = &self.topology.downward_connectivity[self.topology_dim()][dim];
                let index = (0..dc.shape()[0])
                    .position(|i| dc[[i, cell]] == local_index)
                    .unwrap();
                Some(SingleElementMeshEntity::new(self, cell, dim, index))
            }
        } else {
            None
        }
    }

    fn entity_types(&self, dim: usize) -> &[ReferenceCellType] {
        &self.topology.entity_types()[dim..dim + 1]
    }

    fn entity_count(&self, entity_type: ReferenceCellType) -> usize {
        self.topology.entity_count(entity_type)
    }

    fn entity_iter(&self, entity_type: ReferenceCellType) -> Self::EntityIter<'_> {
        SingleElementMeshEntityIter::new(self, entity_type)
    }

    fn entity_from_id(
        &self,
        entity_type: ReferenceCellType,
        id: usize,
    ) -> Option<Self::Entity<'_>> {
        let entity_dim = reference_cell::dim(entity_type);
        self.topology.ids_to_indices[entity_dim]
            .get(&id)
            .map(|i| self.entity(entity_type, *i))?
    }

    fn geometry_map<Array2Impl: ValueArrayImpl<T, 2>>(
        &self,
        entity_type: ReferenceCellType,
        geometry_degree: usize,
        points: &Array<Array2Impl, 2>,
    ) -> GeometryMap<'_, T, BaseArray<VectorContainer<T>, 2>, BaseArray<VectorContainer<usize>, 2>>
    {
        debug_assert!(points.shape()[0] == reference_cell::dim(entity_type));
        debug_assert!(geometry_degree == self.geometry.element().lagrange_superdegree());
        if entity_type == self.topology.entity_types()[self.topology_dim()] {
            GeometryMap::new(
                self.geometry.element(),
                points,
                self.geometry.points(),
                self.geometry.cells(),
            )
        } else {
            unimplemented!();
        }
    }
}

#[cfg(feature = "mpi")]
impl<T: Scalar + Equivalence, E: MappedFiniteElement<CellType = ReferenceCellType, T = T>>
    DistributableMesh for SingleElementMesh<T, E>
{
    type ParallelMesh<'a, C: Communicator + 'a> =
        ParallelMeshImpl<'a, C, SingleElementMesh<T, CiarletElement<T, IdentityMap, T>>>;

    fn distribute<'a, C: Communicator>(
        &self,
        comm: &'a C,
        partitioner: GraphPartitioner,
    ) -> Self::ParallelMesh<'a, C> {
        let e = self.geometry.element();
        let pts = self.geometry.points();
        let cells = self.geometry.cells();
        let mut b = SingleElementMeshBuilder::<T>::new_with_capacity(
            self.geometry.dim(),
            pts.shape()[1],
            cells.shape()[1],
            (e.cell_type(), e.lagrange_superdegree()),
        );
        for p in 0..pts.shape()[1] {
            b.add_point(
                p,
                &pts.data().unwrap()[p * pts.shape()[0]..(p + 1) * pts.shape()[0]],
            );
        }
        for c in 0..cells.shape()[1] {
            b.add_cell(
                c,
                &cells.data().unwrap()[c * cells.shape()[0]..(c + 1) * cells.shape()[0]],
            );
        }
        b.create_parallel_mesh_root(comm, partitioner)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::traits::Topology;
    use itertools::izip;
    use ndelement::{
        ciarlet::{CiarletElement, LagrangeElementFamily},
        reference_cell,
        types::Continuity,
    };
    use rlst::rlst_dynamic_array;

    fn example_mesh_triangle() -> SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>> {
        let mut points = rlst_dynamic_array!(f64, [3, 4]);
        *points.get_mut([0, 0]).unwrap() = 0.0;
        *points.get_mut([1, 0]).unwrap() = 0.0;
        *points.get_mut([2, 0]).unwrap() = 1.0;
        *points.get_mut([0, 1]).unwrap() = 1.0;
        *points.get_mut([1, 1]).unwrap() = 0.0;
        *points.get_mut([2, 1]).unwrap() = 0.0;
        *points.get_mut([0, 2]).unwrap() = 0.0;
        *points.get_mut([1, 2]).unwrap() = 1.0;
        *points.get_mut([2, 2]).unwrap() = 0.0;
        *points.get_mut([0, 3]).unwrap() = 2.0;
        *points.get_mut([1, 3]).unwrap() = 1.0;
        *points.get_mut([2, 3]).unwrap() = 0.0;
        let family = LagrangeElementFamily::<f64>::new(1, Continuity::Standard);
        SingleElementMesh::new(
            SingleTypeTopology::new(&[0, 1, 2, 2, 1, 3], ReferenceCellType::Triangle, None, None),
            SingleElementGeometry::<f64, CiarletElement<f64, IdentityMap, f64>>::new(
                ReferenceCellType::Triangle,
                points,
                &[0, 1, 2, 2, 1, 3],
                &family,
            ),
        )
    }

    #[test]
    fn test_edges_triangle() {
        let mesh = example_mesh_triangle();
        let conn = reference_cell::connectivity(ReferenceCellType::Triangle);
        for edge in mesh.entity_iter(ReferenceCellType::Interval) {
            let cell = mesh
                .entity(ReferenceCellType::Triangle, edge.cell_index)
                .unwrap();
            for (i, v) in izip!(
                &conn[1][edge.entity_index][0],
                edge.topology().sub_entity_iter(ReferenceCellType::Point)
            ) {
                assert_eq!(v, cell.topology().sub_entity(ReferenceCellType::Point, *i));
            }
        }
    }
}
