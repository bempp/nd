use mpi::{environment::Universe, traits::Communicator};
use ndelement::{types::{ReferenceCellType, Continuity}, ciarlet::LagrangeElementFamily};
use ndgrid::{shapes::{unit_cube, unit_cube_distributed}, types::GraphPartitioner};
use ndfunctionspace::{SerialFunctionSpace, ParallelFunctionSpace, traits::FunctionSpace};

/// Test parallel function space
fn test_parallel_function_space<C: Communicator>(comm: &C) {
    let grid = unit_cube_distributed::<f64, _>(comm, GraphPartitioner::None, 4, 4, 4, ReferenceCellType::Tetrahedron);

    let family = LagrangeElementFamily::<f64>::new(2, Continuity::Standard);
    let space = ParallelFunctionSpace::new(&grid, &family);
    let serial_grid = unit_cube::<f64>(4, 4, 4, ReferenceCellType::Tetrahedron);
    let serial_space = SerialFunctionSpace::new(&serial_grid, &family);

    println!("{} {} {}", comm.rank(), space.local_space().global_size(), serial_space.global_size());

    assert_eq!(space.global_size(), serial_space.global_size());
}

/// Run tests
fn main() {
    let universe: Universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();

    if rank == 0 {
        println!("Testing parallel function space");
    }
    test_parallel_function_space(&world);
}
