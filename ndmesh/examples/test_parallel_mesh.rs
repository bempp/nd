use mpi::{
    collective::SystemOperation, environment::Universe, topology::Communicator,
    traits::CommunicatorCollectives,
};
use ndelement::{ciarlet::CiarletElement, map::IdentityMap, types::ReferenceCellType};
use ndmesh::{
    SingleElementMesh, SingleElementMeshBuilder,
    mesh::ParallelMeshImpl,
    traits::{
        Builder, Entity, Mesh, ParallelBuilder, ParallelMesh, RONExportParallel, RONImportParallel,
    },
    types::{GraphPartitioner, Ownership},
};

/// Test that a graph partitioner works
fn run_partitioner_test<C: Communicator>(comm: &C, partitioner: GraphPartitioner) {
    let n = 10;

    let mut b = SingleElementMeshBuilder::<f64>::new(2, (ReferenceCellType::Quadrilateral, 1));

    let rank = comm.rank();
    let mesh = if rank == 0 {
        let mut i = 0;
        for y in 0..n {
            for x in 0..n {
                b.add_point(i, &[x as f64 / (n - 1) as f64, y as f64 / (n - 1) as f64]);
                i += 1;
            }
        }

        let mut i = 0;
        for y in 0..n - 1 {
            for x in 0..n - 1 {
                let sw = n * y + x;
                b.add_cell(i, &[sw, sw + 1, sw + n, sw + n + 1]);
                i += 1;
            }
        }

        b.create_parallel_mesh_root(comm, partitioner)
    } else {
        b.create_parallel_mesh(comm, 0)
    };

    // Check that owned cells are sorted ahead of ghost cells

    let cell_count_owned = mesh
        .local_mesh()
        .entity_iter(ReferenceCellType::Quadrilateral)
        .filter(|entity| entity.is_owned())
        .count();

    // Now check that the first `cell_count_owned` entities are actually owned.
    for cell in mesh
        .local_mesh()
        .entity_iter(ReferenceCellType::Quadrilateral)
        .take(cell_count_owned)
    {
        assert!(cell.is_owned())
    }

    // Now make sure that the indices of the global cells are in consecutive order

    let mut cell_global_count = mesh.cell_layout().local_range().0;

    for cell in mesh
        .local_mesh()
        .entity_iter(ReferenceCellType::Quadrilateral)
        .take(cell_count_owned)
    {
        assert_eq!(cell.global_index(), cell_global_count);
        cell_global_count += 1;
    }

    // Get the global indices.

    let global_vertices = mesh
        .local_mesh()
        .entity_iter(ReferenceCellType::Point)
        .filter(|e| matches!(e.ownership(), Ownership::Owned))
        .map(|e| e.global_index())
        .collect::<Vec<_>>();

    let nvertices = global_vertices.len();

    let global_cells = mesh
        .local_mesh()
        .entity_iter(ReferenceCellType::Quadrilateral)
        .filter(|e| matches!(e.ownership(), Ownership::Owned))
        .map(|e| e.global_index())
        .collect::<Vec<_>>();

    let ncells = global_cells.len();

    let mut total_cells: usize = 0;
    let mut total_vertices: usize = 0;

    comm.all_reduce_into(&ncells, &mut total_cells, SystemOperation::sum());
    comm.all_reduce_into(&nvertices, &mut total_vertices, SystemOperation::sum());

    assert_eq!(total_cells, (n - 1) * (n - 1));
    assert_eq!(total_vertices, n * n);
}

fn create_single_element_mesh_data(b: &mut SingleElementMeshBuilder<f64>, n: usize) {
    for y in 0..n {
        for x in 0..n {
            b.add_point(
                y * n + x,
                &[x as f64 / (n - 1) as f64, y as f64 / (n - 1) as f64, 0.0],
            );
        }
    }

    for i in 0..n - 1 {
        for j in 0..n - 1 {
            b.add_cell(
                i * (n - 1) + j,
                &[j * n + i, j * n + i + 1, j * n + i + n, j * n + i + n + 1],
            );
        }
    }
}

fn example_single_element_mesh<C: Communicator>(
    comm: &C,
    n: usize,
) -> ParallelMeshImpl<'_, C, SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>>> {
    let rank = comm.rank();

    let mut b = SingleElementMeshBuilder::<f64>::new(3, (ReferenceCellType::Quadrilateral, 1));

    if rank == 0 {
        create_single_element_mesh_data(&mut b, n);
        b.create_parallel_mesh_root(comm, GraphPartitioner::None)
    } else {
        b.create_parallel_mesh(comm, 0)
    }
}

/// Test that meshes can be exported as RON in parallel
fn test_parallel_export<C: Communicator>(comm: &C) {
    let size = comm.size();

    let n = 10;
    let mesh = example_single_element_mesh(comm, n);
    let filename = format!("_examples_parallel_io_{size}ranks.ron");
    mesh.export_as_ron(&filename);
}

/// Test that meshes can be imported from RON in parallel
fn test_parallel_import<C: Communicator>(comm: &C) {
    use ndmesh::traits::ParallelMesh;

    let size = comm.size();

    let filename = format!("_examples_parallel_io_{size}ranks.ron");
    let mesh = ParallelMeshImpl::<
        '_,
        C,
        SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>>,
    >::import_from_ron(comm, &filename);

    let n = 10;
    let mesh2 = example_single_element_mesh(comm, n);

    assert_eq!(
        mesh.local_mesh().entity_count(ReferenceCellType::Point),
        mesh2.local_mesh().entity_count(ReferenceCellType::Point)
    );
}

/// Test that using non-contiguous numbering does not cause panic
fn test_noncontiguous_numbering<C: Communicator>(comm: &C) {
    let rank = comm.rank();
    let mut b = SingleElementMeshBuilder::<f64>::new(3, (ReferenceCellType::Quadrilateral, 1));

    let g = if rank == 0 {
        let n = 5;
        for y in 0..n {
            for x in 0..n {
                b.add_point(
                    2 * (y * n + x) + 5,
                    &[x as f64 / (n - 1) as f64, y as f64 / (n - 1) as f64, 0.0],
                );
            }
        }

        for i in 0..n - 1 {
            for j in 0..n - 1 {
                b.add_cell(
                    3 * (i * (n - 1) + j),
                    &[
                        2 * (j * n + i) + 5,
                        2 * (j * n + i + 1) + 5,
                        2 * (j * n + i + n) + 5,
                        2 * (j * n + i + n + 1) + 5,
                    ],
                );
            }
        }

        b.create_parallel_mesh_root(comm, GraphPartitioner::None)
    } else {
        b.create_parallel_mesh(comm, 0)
    };

    assert!(g.local_mesh().entity_count(ReferenceCellType::Point) > 0);
}

/// Run tests
fn main() {
    let universe: Universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();

    if rank == 0 {
        println!("Testing non-contiguous numbering");
    }
    test_noncontiguous_numbering(&world);

    if rank == 0 {
        println!("Testing parallel mesh export");
    }
    test_parallel_export(&world);

    world.barrier();

    if rank == 0 {
        println!("Testing parallel mesh import");
    }
    test_parallel_import(&world);

    if rank == 0 {
        println!("Testing partitioning using GraphPartitioner::None");
    }
    run_partitioner_test(&world, GraphPartitioner::None);

    let mut p = vec![];
    for i in 0..81 {
        p.push(i % world.size() as usize);
    }
    if rank == 0 {
        println!("Testing partitioning using GraphPartitioner::Manual");
    }
    run_partitioner_test(&world, GraphPartitioner::Manual(p));

    if rank == 0 {
        println!("Testing partitioning using GraphPartitioner::Coupe");
    }
    run_partitioner_test(&world, GraphPartitioner::Coupe);

    if rank == 0 {
        println!("Testing partitioning using GraphPartitioner::Scotch");
    }
    run_partitioner_test(&world, GraphPartitioner::Scotch);
}
