use approx::assert_relative_eq;
use ndelement::{
    ciarlet::{LagrangeElementFamily, RaviartThomasElementFamily},
    traits::{FiniteElement, MappedFiniteElement},
    types::{Continuity, ReferenceCellType},
};
use ndfunctionspace::{FunctionSpaceImpl, traits::FunctionSpace};
use ndgrid::{
    shapes::regular_sphere,
    traits::{Entity, GeometryMap, Grid},
};
use quadraturerules::{Domain, QuadratureRule, single_integral_quadrature};
use rlst::{DynArray, rlst_dynamic_array};

/// Test values in Lagrange mass matrix
fn test_lagrange_mass_matrix() {
    let grid = regular_sphere(0);

    let family = LagrangeElementFamily::<f64>::new(1, Continuity::Standard);
    let space = FunctionSpaceImpl::new(&grid, &family);

    let mut mass_matrix = rlst_dynamic_array!(f64, [space.local_size(), space.local_size()]);

    let element = &space.elements()[0];

    let (p, w) = single_integral_quadrature(
        QuadratureRule::XiaoGimbutas,
        Domain::Triangle,
        2 * element.lagrange_superdegree(),
    )
    .unwrap();
    let npts = w.len();
    let mut pts = rlst_dynamic_array!(f64, [2, npts]);
    for i in 0..w.len() {
        for j in 0..2 {
            *pts.get_mut([j, i]).unwrap() = p[3 * i + j];
        }
    }
    let wts = w.iter().map(|i| *i / 2.0).collect::<Vec<_>>();

    let mut table = DynArray::<f64, 4>::from_shape(element.tabulate_array_shape(0, npts));
    element.tabulate(&pts, 0, &mut table);

    let gmap = grid.geometry_map(ReferenceCellType::Triangle, 1, pts.data().unwrap());
    let mut jacobians = vec![0.0; grid.geometry_dim() * grid.topology_dim() * npts];
    let mut jinv = vec![0.0; grid.geometry_dim() * grid.topology_dim() * npts];
    let mut jdets = vec![0.0; npts];

    for cell in grid.entity_iter(ReferenceCellType::Triangle) {
        let dofs = space
            .entity_closure_dofs(ReferenceCellType::Triangle, cell.local_index())
            .unwrap();
        gmap.jacobians_inverses_dets(cell.local_index(), &mut jacobians, &mut jinv, &mut jdets);
        for (test_i, test_dof) in dofs.iter().enumerate() {
            for (trial_i, trial_dof) in dofs.iter().enumerate() {
                *mass_matrix.get_mut([*test_dof, *trial_dof]).unwrap() += wts
                    .iter()
                    .enumerate()
                    .map(|(i, w)| {
                        jdets[i]
                            * *w
                            * *table.get([0, i, test_i, 0]).unwrap()
                            * *table.get([0, i, trial_i, 0]).unwrap()
                    })
                    .sum::<f64>();
            }
        }
    }

    for i in 0..6 {
        assert_relative_eq!(mass_matrix[[i, i]], 0.5773502691896255, epsilon = 1e-10);
    }
    for i in 0..6 {
        for j in 0..6 {
            if i != j && mass_matrix[[i, j]].abs() > 0.001 {
                assert_relative_eq!(mass_matrix[[i, j]], 0.1443375672974061, epsilon = 1e-10);
            }
        }
    }
}

/// Test values in Raviart-Thomas mass matrix
fn test_rt_mass_matrix() {
    let grid = regular_sphere(0);

    let family = RaviartThomasElementFamily::<f64>::new(1, Continuity::Standard);
    let space = FunctionSpaceImpl::new(&grid, &family);

    let mut mass_matrix = rlst_dynamic_array!(f64, [space.local_size(), space.local_size()]);

    let element = &space.elements()[0];

    let (p, w) = single_integral_quadrature(
        QuadratureRule::XiaoGimbutas,
        Domain::Triangle,
        2 * element.lagrange_superdegree(),
    )
    .unwrap();
    let npts = w.len();
    let mut pts = rlst_dynamic_array!(f64, [2, npts]);
    for i in 0..w.len() {
        for j in 0..2 {
            *pts.get_mut([j, i]).unwrap() = p[3 * i + j];
        }
    }
    let wts = w.iter().map(|i| *i / 2.0).collect::<Vec<_>>();

    let mut table = DynArray::<f64, 4>::from_shape(element.tabulate_array_shape(0, npts));
    element.tabulate(&pts, 0, &mut table);
    let mut pushed_table = rlst_dynamic_array!(
        f64,
        [table.shape()[0], table.shape()[1], table.shape()[2], 3]
    );

    let gmap = grid.geometry_map(ReferenceCellType::Triangle, 1, pts.data().unwrap());
    let mut jacobians = rlst_dynamic_array!(f64, [npts, grid.geometry_dim(), grid.topology_dim()]);
    let mut jinv = rlst_dynamic_array!(f64, [npts, grid.topology_dim(), grid.geometry_dim()]);
    let mut jdets = vec![0.0; npts];

    for cell in grid.entity_iter(ReferenceCellType::Triangle) {
        let dofs = space
            .entity_closure_dofs(ReferenceCellType::Triangle, cell.local_index())
            .unwrap();
        gmap.jacobians_inverses_dets(
            cell.local_index(),
            jacobians.data_mut().unwrap(),
            jinv.data_mut().unwrap(),
            &mut jdets,
        );
        element.push_forward(&table, 0, &jacobians, &jdets, &jinv, &mut pushed_table);

        dbg!(table.data().unwrap());
        dbg!(pushed_table.data().unwrap());

        for (test_i, test_dof) in dofs.iter().enumerate() {
            for (trial_i, trial_dof) in dofs.iter().enumerate() {
                *mass_matrix.get_mut([*test_dof, *trial_dof]).unwrap() += wts
                    .iter()
                    .enumerate()
                    .map(|(i, w)| {
                        jdets[i]
                            * *w
                            * (0..3)
                                .map(|j| {
                                    *pushed_table.get([0, i, test_i, j]).unwrap()
                                        * *pushed_table.get([0, i, trial_i, j]).unwrap()
                                })
                                .sum::<f64>()
                    })
                    .sum::<f64>();
            }
        }
    }

    for i in 0..12 {
        for j in 0..12 {
            println!("{}", mass_matrix[[i, j]]);
        }
        println!();
    }

    for i in 0..12 {
        assert_relative_eq!(mass_matrix[[i, i]], 0.9622504486493761, epsilon = 1e-10);
    }
    for i in 0..12 {
        for j in 0..12 {
            if i != j && mass_matrix[[i, j]].abs() > 0.001 {
                assert_relative_eq!(
                    mass_matrix[[i, j]].abs(),
                    0.9622504486493758,
                    epsilon = 1e-10
                );
            }
        }
    }
}

/// Run tests
fn main() {
    println!("Testing Lagrange mass matrix");
    test_lagrange_mass_matrix();

    println!("Testing Raviart-Thomas mass matrix");
    test_rt_mass_matrix();
}
