use mpi::{environment::Universe, traits::Communicator};
use ndelement::{
    ciarlet::{LagrangeElementFamily, LagrangeVariant},
    types::{Continuity, ReferenceCellType},
};
use ndfunctionspace::{
    FunctionSpaceImpl, ParallelFunctionSpaceImpl,
    traits::{FunctionSpace, ParallelFunctionSpace},
};
use ndmesh::{
    shapes::{unit_cube, unit_cube_distributed},
    types::GraphPartitioner,
};

/// Test parallel function space
fn test_parallel_function_space<C: Communicator>(comm: &C) {
    let mesh = unit_cube_distributed::<f64, _>(
        comm,
        GraphPartitioner::None,
        4,
        4,
        4,
        ReferenceCellType::Tetrahedron,
    );

    let family =
        LagrangeElementFamily::<f64>::new(2, Continuity::Standard, LagrangeVariant::Equispaced);
    let space = ParallelFunctionSpaceImpl::new(&mesh, &family);
    let serial_mesh = unit_cube::<f64>(4, 4, 4, ReferenceCellType::Tetrahedron);
    let serial_space = FunctionSpaceImpl::new(&serial_mesh, &family);

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
