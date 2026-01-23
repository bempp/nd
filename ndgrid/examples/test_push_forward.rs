use approx::assert_relative_eq;
use itertools::izip;
use ndelement::{
    ciarlet::{lagrange, nedelec, raviart_thomas},
    traits::{FiniteElement, MappedFiniteElement},
    types::{Continuity, ReferenceCellType},
};
use ndgrid::{
    grid::local_grid::SingleElementGridBuilder,
    traits::{Builder, GeometryMap, Grid},
};
use rlst::{DynArray, rlst_dynamic_array};

/// Test Lagrange push forward
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

    let gmap = grid.geometry_map(ReferenceCellType::Triangle, 1, &points);

    let mut jacobians = rlst_dynamic_array!(f64, [grid.geometry_dim(), grid.topology_dim(), npts]);
    let mut jinv = rlst_dynamic_array!(f64, [grid.topology_dim(), grid.geometry_dim(), npts]);
    let mut jdets = vec![0.0; npts];

    gmap.jacobians_inverses_dets(1, &mut jacobians, &mut jinv, &mut jdets);

    let mut cell1_table = DynArray::<f64, 4>::from_shape(table.shape());
    e.push_forward(&table, 0, &jacobians, &jdets, &jinv, &mut cell1_table);

    // Check that basis functions are continuous between cells
    for (cell0_dof, cell1_dof) in izip!(
        e.entity_closure_dofs(1, 1).unwrap(),
        e.entity_closure_dofs(1, 0).unwrap()
    ) {
        for i in 0..npts {
            assert_relative_eq!(
                cell0_table[[0, i, *cell0_dof, 0]],
                cell1_table[[0, i, *cell1_dof, 0]],
                epsilon = 1e-10
            );
        }
    }
}

/// Test Ravaiart-Thomas push forward
fn test_rt_push_forward() {
    let mut b = SingleElementGridBuilder::<f64>::new(3, (ReferenceCellType::Triangle, 1));
    b.add_point(0, &[0.0, 0.0, 0.0]);
    b.add_point(1, &[1.0, 0.0, 0.0]);
    b.add_point(2, &[2.0, 0.0, 1.0]);
    b.add_point(3, &[0.0, 1.0, 0.0]);
    b.add_cell(0, &[0, 1, 3]);
    b.add_cell(1, &[1, 2, 3]);
    let grid = b.create_grid();

    let e =
        raviart_thomas::create::<f64, f64>(ReferenceCellType::Triangle, 4, Continuity::Standard);

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

    let gmap = grid.geometry_map(ReferenceCellType::Triangle, 1, &points);

    let mut jacobians = rlst_dynamic_array!(f64, [grid.geometry_dim(), grid.topology_dim(), npts]);
    let mut jinv = rlst_dynamic_array!(f64, [grid.topology_dim(), grid.geometry_dim(), npts]);
    let mut jdets = vec![0.0; npts];

    gmap.jacobians_inverses_dets(1, &mut jacobians, &mut jinv, &mut jdets);

    let mut cell1_table =
        DynArray::<f64, 4>::from_shape([table.shape()[0], table.shape()[1], table.shape()[2], 3]);
    e.push_forward(&table, 0, &jacobians, &jdets, &jinv, &mut cell1_table);

    // Check that basis functions dotted with normal to edge are continuous between cells
    for (cell0_dof, cell1_dof) in izip!(
        e.entity_closure_dofs(1, 0).unwrap(),
        e.entity_closure_dofs(1, 1).unwrap()
    ) {
        for i in 0..npts {
            assert_relative_eq!(
                (cell0_table[[0, i, *cell0_dof, 0]] + cell0_table[[0, i, *cell0_dof, 1]])
                    / f64::sqrt(2.0),
                (cell1_table[[0, i, *cell1_dof, 0]]
                    + cell1_table[[0, i, *cell1_dof, 1]]
                    + 2.0 * cell1_table[[0, i, *cell1_dof, 2]])
                    / f64::sqrt(6.0),
                epsilon = 1e-10
            );
        }
    }
}

/// Test Nedelec push forward
fn test_nc_push_forward() {
    let mut b = SingleElementGridBuilder::<f64>::new(3, (ReferenceCellType::Triangle, 1));
    b.add_point(0, &[0.0, 0.0, 0.0]);
    b.add_point(1, &[1.0, 0.0, 0.0]);
    b.add_point(2, &[2.0, 0.0, 1.0]);
    b.add_point(3, &[0.0, 1.0, 0.0]);
    b.add_cell(0, &[0, 1, 3]);
    b.add_cell(1, &[1, 2, 3]);
    let grid = b.create_grid();

    let e = nedelec::create::<f64, f64>(ReferenceCellType::Triangle, 4, Continuity::Standard);

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

    let gmap = grid.geometry_map(ReferenceCellType::Triangle, 1, &points);

    let mut jacobians = rlst_dynamic_array!(f64, [grid.geometry_dim(), grid.topology_dim(), npts]);
    let mut jinv = rlst_dynamic_array!(f64, [grid.topology_dim(), grid.geometry_dim(), npts]);
    let mut jdets = vec![0.0; npts];

    gmap.jacobians_inverses_dets(1, &mut jacobians, &mut jinv, &mut jdets);

    let mut cell1_table =
        DynArray::<f64, 4>::from_shape([table.shape()[0], table.shape()[1], table.shape()[2], 3]);
    e.push_forward(&table, 0, &jacobians, &jdets, &jinv, &mut cell1_table);

    // Check that basis functions dotted with tangent to edge are continuous between cells
    for (cell0_dof, cell1_dof) in izip!(
        e.entity_closure_dofs(1, 0).unwrap(),
        e.entity_closure_dofs(1, 1).unwrap()
    ) {
        for i in 0..npts {
            assert_relative_eq!(
                (cell0_table[[0, i, *cell0_dof, 0]] - cell0_table[[0, i, *cell0_dof, 1]])
                    / f64::sqrt(2.0),
                (cell1_table[[0, i, *cell1_dof, 0]] - cell1_table[[0, i, *cell1_dof, 1]])
                    / f64::sqrt(2.0),
                epsilon = 1e-10
            );
        }
    }
}

/// Run tests
fn main() {
    println!("Testing Lagrange push forward (identity)");
    test_lagrange_push_forward();

    println!("Testing Raviart-Thomas push forward (contravariant Piola)");
    test_rt_push_forward();

    println!("Testing Nedelec push forward (covariant Piola)");
    test_nc_push_forward();
}
