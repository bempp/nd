//! Traits for a mesh entity
use super::{Entity, GeometryMap};
use crate::traits::{Geometry, Point};
#[cfg(feature = "mpi")]
use crate::types::GraphPartitioner;
use crate::types::{Ownership, Scalar};
use itertools::izip;
#[cfg(feature = "mpi")]
use mpi::traits::Communicator;
use ndelement::reorderings::vtk_ordering;
use ndelement::types::ReferenceCellType;
#[cfg(feature = "mpi")]
use rlst::distributed_tools::IndexLayout;
use rlst::{Array, DynArray, EvaluateArray, ValueArrayImpl, rlst_dynamic_array};
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;
use std::iter::Iterator;
#[cfg(feature = "mpi")]
use std::rc::Rc;
use vtkio::model::{
    Attributes, ByteOrder, CellType, Cells, DataSet, UnstructuredGridPiece, Version, VertexNumbers,
};
use vtkio::{IOBuffer, Vtk};

/// A mesh provides access to entities, their geometrical and their topological properties.
pub trait Mesh {
    /// Scalar type
    type T: Scalar;

    /// Type used as identifier of different entity types
    type Entity<'a>: Entity<EntityDescriptor = Self::EntityDescriptor, T = Self::T>
    where
        Self: 'a;

    /// Geometry map type
    type GeometryMap<'a>: GeometryMap<T = Self::T>
    where
        Self: 'a;

    /// Type used as identifier of different entity types
    type EntityDescriptor: Debug + PartialEq + Eq + Clone + Copy + Hash;

    /// Iterator over sub-entities
    type EntityIter<'a>: Iterator<Item = Self::Entity<'a>>
    where
        Self: 'a;

    /// Dimension of the geometry of this mesh
    fn geometry_dim(&self) -> usize;

    /// Dimension of the topology of this mesh
    fn topology_dim(&self) -> usize;

    /// An entity in this mesh
    fn entity(
        &self,
        entity_type: Self::EntityDescriptor,
        local_index: usize,
    ) -> Option<Self::Entity<'_>>;

    /// The entity types of topological dimension `dim` contained in this mesh
    fn entity_types(&self, tdim: usize) -> &[Self::EntityDescriptor];

    /// Number of entities of type `entity_type`
    fn entity_count(&self, entity_type: Self::EntityDescriptor) -> usize;

    /// Number of cells in the mesh
    fn cell_count(&self) -> usize {
        self.entity_types(self.topology_dim())
            .iter()
            .map(|&t| self.entity_count(t))
            .sum()
    }

    /// Number of points in the local mesh
    ///
    /// This is different from `self.entity_count(ReferenceCellType::Point)`.
    /// While the entity count only considers topology information, that is vertices
    /// on the edges of elements, this method considers geometry information, that is also
    /// the points in the interior of edges, faces, and volumes.
    ///
    /// # Remark
    /// - The point count returns the number of points associated with owned cells only
    /// - The point indices generally cannot be assumed to be contiguous for owned cells. Hence,
    ///   unless a specific mesh has a fast implementation, a generic HashSet needs to be used to
    ///   count unique points.
    fn point_count(&self) -> usize {
        // Need to add one since the iterator gives back the highest index. But we start counting
        // at zero.
        let mut point_indices = HashSet::<usize>::new();
        self.cell_types().iter().for_each(|t| {
            self.entity_iter(*t)
                .filter(|e| matches!(e.ownership(), Ownership::Owned))
                .for_each(|e| point_indices.extend(e.geometry().point_indices().iter()))
        });

        point_indices.len()
    }

    /// Return an array containing all geometric points of all cells and a vector of the local indices of the points.
    ///
    /// This functions return a tuple `(points, indices)`. The `points` array has dimension `(gdim, npoints)` with `gdim` the geometric
    /// dimension and `npoints` the number of geometry points associated with owned cells. `indices` is a vector whose ith position contains
    /// the local index of the point in column i.
    fn owned_points(&self) -> (DynArray<Self::T, 2>, Vec<usize>) {
        let npoints = self.point_count();
        let mut points = rlst_dynamic_array!(Self::T, [self.geometry_dim(), npoints]);
        let mut indices = Vec::<usize>::with_capacity(npoints);
        let mut already_processed = HashSet::<usize>::new();

        let mut count = 0;
        for t in self.cell_types().iter() {
            for cell in self
                .entity_iter(*t)
                .filter(|e| matches!(e.ownership(), Ownership::Owned))
            {
                let geometry = cell.geometry();
                for point in geometry.points() {
                    let point_index = point.index();
                    if !already_processed.contains(&point_index) {
                        point.coords(points.r_mut().col(count).data_mut().unwrap());
                        indices.push(point_index);
                        already_processed.insert(point_index);
                        count += 1;
                    }
                }
            }
        }
        // An important sanity test is that the inserted points is identical to npoints.
        assert_eq!(count, npoints);

        (points, indices)
    }

    /// Return a vector of connectivities for each owned cell.
    ///
    /// The returned `connectivity` array is a Vec<Vec<usize>>. The point indices associated with cell 0
    /// are obtained as `connectivity[0]`. The ordering of the point indices is the natural ND ordering.
    fn owned_connectivity(&self) -> Vec<Vec<usize>> {
        let mut connectivity = Vec::<Vec<usize>>::new();

        for t in self.cell_types().iter() {
            for cell in self
                .entity_iter(*t)
                .filter(|e| matches!(e.ownership(), Ownership::Owned))
            {
                let geometry = cell.geometry();
                connectivity.push(geometry.point_indices());
            }
        }

        connectivity
    }

    /// Return the reference cell types for each owned cell in the mesh.
    ///
    /// # Remark
    /// This method returns an array that contains for each owned cell the corresponding
    /// cell type. In contrast, the method [Mesh::cell_types] returns a collection of all available
    /// cell types in the mesh.
    fn owned_cell_types(&self) -> Vec<Self::EntityDescriptor> {
        let mut cell_types = Vec::<Self::EntityDescriptor>::new();

        for t in self.cell_types().iter() {
            for cell in self
                .entity_iter(*t)
                .filter(|e| matches!(e.ownership(), Ownership::Owned))
            {
                cell_types.push(cell.entity_type());
            }
        }

        cell_types
    }

    /// Return the cell types in the mesh
    fn cell_types(&self) -> &[Self::EntityDescriptor] {
        let tdim = self.topology_dim();
        self.entity_types(tdim)
    }

    /// Owned cell count
    ///
    /// Note. The default implementation iterates through all mesh to count the number of owned elements.
    /// Override this method if a more efficient implementation is available.
    fn owned_cell_count(&self) -> usize {
        self.cell_types()
            .iter()
            .map(|t| {
                self.entity_iter(*t)
                    .filter(|e| matches!(e.ownership(), Ownership::Owned))
                    .count()
            })
            .sum()
    }

    /// Iterator over entities
    fn entity_iter(&self, entity_type: Self::EntityDescriptor) -> Self::EntityIter<'_>;

    /// An entity in this mesh from an insertion id
    fn entity_from_id(
        &self,
        entity_type: Self::EntityDescriptor,
        id: usize,
    ) -> Option<Self::Entity<'_>>;

    /// Geometry map from reference entity to physical entities at the given points
    ///
    /// `points` should have shape [entity_topology_dim, npts] and use column-major ordering
    fn geometry_map<Array2Impl: ValueArrayImpl<Self::T, 2>>(
        &self,
        entity_type: Self::EntityDescriptor,
        geometry_degree: usize,
        points: &Array<Array2Impl, 2>,
    ) -> Self::GeometryMap<'_>;

    /// Export the local mesh as a vtk structure
    fn as_vtk(&self) -> Vtk
    where
        Self: Mesh<EntityDescriptor = ReferenceCellType>,
    {
        let (points, local_indices) = self.owned_points();
        let cells = self.owned_connectivity();
        let cell_types = self.owned_cell_types();

        // We need to create a hash map that maps from local indices to column in the points array.
        let point_index_map: HashMap<usize, usize> =
            HashMap::from_iter(izip!(local_indices.iter().copied(), 0..local_indices.len()));

        // We set up the connectivity and offsets arrays
        let mut vtk_connectivity = Vec::<u64>::with_capacity(cells.len());
        let mut offsets = Vec::<u64>::with_capacity(1 + cells.len());

        let mut reorderings = HashMap::<(ReferenceCellType, usize), Vec<usize>>::new();
        let mut vtk_types = Vec::<CellType>::with_capacity(cells.len());

        let mut offset_count = 0;
        for (cell_type, cell) in izip!(cell_types, cells) {
            let npts = cell.len() as u64;

            let reordering = reorderings
                .entry((cell_type, npts as usize))
                .or_insert(vtk_ordering(cell_type, npts as usize));

            let mut reordered_cell = Vec::<u64>::with_capacity(npts as usize);

            for &permuted_index in reordering.iter() {
                // We need to remember that the exported points array does not necessary have
                // the same indexing as the points array in ND. Hence, we need to map the point indices
                // with point_index_map.
                reordered_cell.push(*point_index_map.get(&cell[permuted_index]).unwrap() as u64);
            }

            vtk_connectivity.extend_from_slice(&reordered_cell);
            offset_count += npts;

            offsets.push(offset_count);
            vtk_types.push(reference_cell_type_to_vtk(cell_type));
        }

        // Convert points to f64 vector
        let points = points.cast::<f64>().eval().data().unwrap().to_vec();

        vtkio::Vtk {
            version: Version::XML { major: 2, minor: 2 },
            title: String::new(),
            byte_order: ByteOrder::LittleEndian,
            file_path: None,
            data: DataSet::inline(UnstructuredGridPiece {
                points: IOBuffer::F64(points),
                cells: Cells {
                    cell_verts: VertexNumbers::XML {
                        connectivity: vtk_connectivity,
                        offsets,
                    },
                    types: vtk_types,
                },
                data: Attributes {
                    ..Default::default()
                },
            }),
        }
    }
}

/// Return the VTK type associated with the ND cell type
fn reference_cell_type_to_vtk(cell_type: ReferenceCellType) -> CellType {
    match cell_type {
        ReferenceCellType::Point => CellType::Vertex,
        ReferenceCellType::Interval => CellType::LagrangeCurve,
        ReferenceCellType::Triangle => CellType::LagrangeTriangle,
        ReferenceCellType::Quadrilateral => CellType::LagrangeQuadrilateral,
        ReferenceCellType::Tetrahedron => CellType::LagrangeTetrahedron,
        ReferenceCellType::Hexahedron => CellType::LagrangeHexahedron,
        _ => unimplemented!(),
    }
}

/// Definition of an MPI parallel mesh
#[cfg(feature = "mpi")]
pub trait ParallelMesh {
    /// Type of the Mesh
    type T: Scalar;

    /// Local mesh type
    type LocalMesh: Mesh<T = Self::T>;

    /// Communicator
    type C: Communicator;

    /// MPI communicator
    fn comm(&self) -> &Self::C;

    /// Local mesh on the current process
    fn local_mesh(&self) -> &Self::LocalMesh;

    /// Return the cell index layout that describes where each global cell lives
    fn cell_layout(&self) -> Rc<IndexLayout<'_, Self::C>>;

    /// Return the global number of cells
    fn global_cell_count(&self) -> usize {
        self.cell_layout().number_of_global_indices()
    }
}

/// A mesh that can be be distributed across processes
#[cfg(feature = "mpi")]
pub trait DistributableMesh {
    /// Parallel mesh type when distributed
    type ParallelMesh<'a, C: Communicator + 'a>: ParallelMesh<C = C>;

    /// Distribute this mesh in parallel
    fn distribute<'a, C: Communicator>(
        &self,
        comm: &'a C,
        partitioner: GraphPartitioner,
    ) -> Self::ParallelMesh<'a, C>;
}
