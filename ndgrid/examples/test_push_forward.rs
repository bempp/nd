use ndelement::{ciarlet::lagrange, types::{ReferenceCellType, Continuity}, traits::{FiniteElement, MappedFiniteElement}};
use ndgrid::{
    grid::local_grid::SingleElementGridBuilder,
    traits::{Builder, Entity, Grid, GeometryMap},
    types::{GraphPartitioner, Ownership},
};
use rlst::{DynArray, rlst_dynamic_array};
use itertools::izip;
use approx::assert_relative_eq;

/// Test Lagrnage push forward
fn test_lagrange_push_forward() {
    let mut b = SingleElementGridBuilder::<f64>::new(3, (ReferenceCellType::Triangle, 1));
    b.add_point(0, &[0.0, 0.0, 0.0]);
    b.add_point(1, &[1.0, 0.0, 0.0]);
    b.add_point(2, &[2.0, 0.0, 1.0]);
    b.add_point(3, &[0.0, 1.0, 0.0]);
    b.add_cell(0, &[0, 1, 3]);
    b.add_cell(1, &[1, 2, 3]);
    let grid = b.create_grid();

    let e = lagrange::create::<f64, f64>(ReferenceCellType::Triangle, 4, Continuity::Standard);

    let npts = 5;

    let mut cell0_points = rlst_dynamic_array!(f64, [2, npts]);
    for i in 0..npts {
        cell0_points[[1, i]] = i as f64 / (npts - 1) as f64;
        cell0_points[[0, i]] = 1.0 - cell0_points[[1, i]];
    }
    let mut cell0_table = DynArray::<f64, 4>::from_shape(e.tabulate_array_shape(0, npts));
    e.tabulate(&cell0_points, 0, &mut cell0_table);

    let mut points = rlst_dynamic_array!(f64, [2, npts]);
    for i in 0..npts {
        points[[0, i]] = 0.0;
        points[[1, i]] = i as f64 / (npts - 1) as f64;
    }

    let mut table = DynArray::<f64, 4>::from_shape(e.tabulate_array_shape(0, npts));
    e.tabulate(&points, 0, &mut table);

    let gmap = grid.geometry_map(ReferenceCellType::Triangle, 1, points.data().unwrap());

    let mut jacobians = rlst_dynamic_array!(f64, [npts, grid.geometry_dim(), grid.topology_dim()]);
    let mut jinv = rlst_dynamic_array!(f64, [npts, grid.topology_dim(), grid.geometry_dim()]);
    let mut jdets = vec![0.0; npts];

    gmap.jacobians_inverses_dets(
        1,
        jacobians.data_mut().unwrap(),
        jinv.data_mut().unwrap(),
        &mut jdets,
    );

    let mut cell1_table = DynArray::<f64, 4>::from_shape(table.shape());
    e.push_forward(&table, 0, &jacobians, &jdets, &jinv, &mut cell1_table);

    for (cell0_dof, cell1_dof) in izip!(e.entity_closure_dofs(1, 1).unwrap(), e.entity_closure_dofs(1, 0).unwrap()) {
        for i in 0..npts {
            assert_relative_eq!(
                cell0_table[[0, i, *cell0_dof, 0]],
                cell1_table[[0, i, *cell1_dof, 0]],
                epsilon = 1e-10
            );
        }
    }
}

/// Run tests
fn main() {
    println!("Testing Lagrange push forward (identity)");
    test_lagrange_push_forward();
}
