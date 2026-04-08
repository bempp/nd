use itertools::izip;
use mpi::{collective::CommunicatorCollectives, environment::Universe, traits::Communicator};
use ndelement::{ciarlet::CiarletElement, map::IdentityMap, types::ReferenceCellType};
use ndmesh::traits::{
    DistributableMesh, Entity, Mesh, ParallelBuilder, ParallelMesh, RONExportParallel,
    RONImportParallel, Topology,
};
use ndmesh::{
    ParallelMeshImpl, SingleElementMesh, SingleElementMeshBuilder, shapes, types::GraphPartitioner,
};

/// Mesh I/O
///
/// Demonstration of importing and exporting a mesh in parallel
///
/// Serial I/O is demonstrated in the example `io.rs`
fn main() {
    let universe: Universe = mpi::initialize().unwrap();
    let comm = universe.world();
    let rank = comm.rank();

    let g = if rank == 0 {
        // Create a mesh using the shapes module: unit_cube_boundary will mesh the surface of a cube
        let serial_g = shapes::unit_cube_boundary::<f64>(4, 5, 4, ReferenceCellType::Triangle);

        // Distribute this mesh across processes
        serial_g.distribute(&comm, GraphPartitioner::None)
    } else {
        let b = SingleElementMeshBuilder::<f64>::new(3, (ReferenceCellType::Triangle, 1));
        b.create_parallel_mesh(&comm, 0)
    };

    // If the serde option is used, the raw mesh data can be exported in RON format
    g.export_as_ron("_unit_cube_boundary_parallel.ron");

    // Wait for export to finish
    comm.barrier();

    // A mesh can be re-imported from raw RON data. Note that it must be imported on the same number of processes as it was exported using
    let g2 = ParallelMeshImpl::<'_, _, SingleElementMesh::<f64, CiarletElement<f64, IdentityMap, f64>>>::import_from_ron(&comm, "_unit_cube_boundary_parallel.ron");

    // Print the first 5 cells of each mesh on process 0
    if rank == 0 {
        println!("The first 5 cells of the mesh");
        for (cell, cell2) in izip!(
            g.local_mesh().entity_iter(ReferenceCellType::Triangle),
            g2.local_mesh().entity_iter(ReferenceCellType::Triangle)
        )
        .take(5)
        {
            println!(
                "{:?} {:?}",
                cell.topology()
                    .sub_entity_iter(ReferenceCellType::Point)
                    .collect::<Vec<_>>(),
                cell2
                    .topology()
                    .sub_entity_iter(ReferenceCellType::Point)
                    .collect::<Vec<_>>(),
            );
        }
    }
}
