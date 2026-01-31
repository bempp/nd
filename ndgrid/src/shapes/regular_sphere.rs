//! Regular sphere grid

#[cfg(feature = "mpi")]
use crate::{ParallelGridImpl, traits::ParallelBuilder, types::GraphPartitioner};
use crate::{
    grid::local_grid::{SingleElementGrid, SingleElementGridBuilder},
    traits::Builder,
    types::Scalar,
};
#[cfg(feature = "mpi")]
use mpi::traits::{Communicator, Equivalence};
use ndelement::{ciarlet::CiarletElement, map::IdentityMap, types::ReferenceCellType};
use std::collections::{HashMap, hash_map::Entry::Vacant};

/// Add points and cells for regular sphere to builder
fn regular_sphere_triangle_add_points_and_cells<T: Scalar>(
    b: &mut SingleElementGridBuilder<T>,
    refinement_level: u32,
) {
    b.add_point(0, &[T::zero(), T::zero(), T::one()]);
    b.add_point(1, &[T::one(), T::zero(), T::zero()]);
    b.add_point(2, &[T::zero(), T::one(), T::zero()]);
    b.add_point(3, &[-T::one(), T::zero(), T::zero()]);
    b.add_point(4, &[T::zero(), -T::one(), T::zero()]);
    b.add_point(5, &[T::zero(), T::zero(), -T::one()]);
    let mut point_n = 6;

    let mut cells = vec![
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 1],
        [5, 2, 1],
        [5, 3, 2],
        [5, 4, 3],
        [5, 1, 4],
    ];
    let mut v = [
        [T::zero(), T::zero(), T::zero()],
        [T::zero(), T::zero(), T::zero()],
        [T::zero(), T::zero(), T::zero()],
    ];

    for _level in 0..refinement_level {
        let mut edge_points = HashMap::new();
        let mut new_cells = Vec::with_capacity(4 * cells.len());
        for c in &cells {
            for (i, v_i) in v.iter_mut().enumerate() {
                for (j, v_ij) in v_i.iter_mut().enumerate() {
                    *v_ij = b.points[3 * c[i] + j];
                }
            }
            let edges = [[1, 2], [0, 2], [0, 1]]
                .iter()
                .map(|[i, j]| {
                    let mut pt_i = c[*i];
                    let mut pt_j = c[*j];
                    if pt_i > pt_j {
                        std::mem::swap(&mut pt_i, &mut pt_j);
                    }
                    if let Vacant(e) = edge_points.entry((pt_i, pt_j)) {
                        let v_i = v[*i];
                        let v_j = v[*j];
                        let mut new_pt = [
                            T::from(0.5).unwrap() * (v_i[0] + v_j[0]),
                            T::from(0.5).unwrap() * (v_i[1] + v_j[1]),
                            T::from(0.5).unwrap() * (v_i[2] + v_j[2]),
                        ];
                        let size = (new_pt.iter().map(|x| x.powi(2)).sum::<T>()).sqrt();
                        for i in new_pt.iter_mut() {
                            *i /= size;
                        }
                        b.add_point(point_n, &new_pt);
                        e.insert(point_n);
                        point_n += 1;
                    }
                    edge_points[&(pt_i, pt_j)]
                })
                .collect::<Vec<_>>();
            new_cells.push([c[0], edges[2], edges[1]]);
            new_cells.push([c[1], edges[0], edges[2]]);
            new_cells.push([c[2], edges[1], edges[0]]);
            new_cells.push([edges[0], edges[1], edges[2]]);
        }
        cells = new_cells;
    }
    for (i, v) in cells.iter().enumerate() {
        b.add_cell(i, v);
    }
}

/// Add points and cells for regular sphere to builder
fn regular_sphere_quadrilateral_add_points_and_cells<T: Scalar>(
    b: &mut SingleElementGridBuilder<T>,
    refinement_level: u32,
) {
    let k = T::from(1.0 / 3.0).unwrap().sqrt();
    b.add_point(0, &[-k, -k, -k]);
    b.add_point(1, &[-k, -k, k]);
    b.add_point(2, &[-k, k, -k]);
    b.add_point(3, &[-k, k, k]);
    b.add_point(4, &[k, -k, -k]);
    b.add_point(5, &[k, -k, k]);
    b.add_point(6, &[k, k, -k]);
    b.add_point(7, &[k, k, k]);

    let mut point_n = 8;

    let mut cells = vec![
        [0, 1, 2, 3],
        [4, 6, 5, 7],
        [0, 4, 1, 5],
        [2, 3, 6, 7],
        [0, 2, 4, 6],
        [1, 5, 3, 7],
    ];
    let mut v = [
        [T::zero(), T::zero(), T::zero()],
        [T::zero(), T::zero(), T::zero()],
        [T::zero(), T::zero(), T::zero()],
        [T::zero(), T::zero(), T::zero()],
    ];

    for _level in 0..refinement_level {
        let mut edge_points = HashMap::new();
        let mut new_cells = Vec::with_capacity(4 * cells.len());
        for c in &cells {
            for (i, v_i) in v.iter_mut().enumerate() {
                for (j, v_ij) in v_i.iter_mut().enumerate() {
                    *v_ij = b.points[3 * c[i] + j];
                }
            }
            let edges = [[0, 1], [0, 2], [1, 3], [2, 3]]
                .iter()
                .map(|[i, j]| {
                    let mut pt_i = c[*i];
                    let mut pt_j = c[*j];
                    if pt_i > pt_j {
                        std::mem::swap(&mut pt_i, &mut pt_j);
                    }
                    if let Vacant(e) = edge_points.entry((pt_i, pt_j)) {
                        let v_i = v[*i];
                        let v_j = v[*j];
                        let mut new_pt = [
                            T::from(0.5).unwrap() * (v_i[0] + v_j[0]),
                            T::from(0.5).unwrap() * (v_i[1] + v_j[1]),
                            T::from(0.5).unwrap() * (v_i[2] + v_j[2]),
                        ];
                        let size = (new_pt.iter().map(|x| x.powi(2)).sum::<T>()).sqrt();
                        for i in new_pt.iter_mut() {
                            *i /= size;
                        }
                        b.add_point(point_n, &new_pt);
                        e.insert(point_n);
                        point_n += 1;
                    }
                    edge_points[&(pt_i, pt_j)]
                })
                .collect::<Vec<_>>();

            new_cells.push([c[0], edges[0], edges[1], point_n]);
            new_cells.push([edges[0], c[1], point_n, edges[2]]);
            new_cells.push([edges[1], point_n, c[2], edges[3]]);
            new_cells.push([point_n, edges[2], edges[3], c[3]]);
            let mut new_pt = [
                T::from(0.25).unwrap() * v.iter().map(|i| i[0]).sum(),
                T::from(0.25).unwrap() * v.iter().map(|i| i[1]).sum(),
                T::from(0.25).unwrap() * v.iter().map(|i| i[2]).sum(),
            ];
            let size = (new_pt.iter().map(|x| x.powi(2)).sum::<T>()).sqrt();
            for i in new_pt.iter_mut() {
                *i /= size;
            }
            b.add_point(point_n, &new_pt);
            point_n += 1;
        }
        cells = new_cells;
    }
    for (i, v) in cells.iter().enumerate() {
        b.add_cell(i, v);
    }
}
/// Create a surface grid of a regular sphere
///
/// A regular sphere is created by starting with a regular octahedron. The shape is then refined `refinement_level` times.
/// Each time the grid is refined, each triangle is split into four triangles (by adding lines connecting the midpoints of
/// each edge). The new points are then scaled so that they are a distance of 1 from the origin.
pub fn regular_sphere<T: Scalar>(
    refinement_level: u32,
    cell_type: ReferenceCellType,
) -> SingleElementGrid<T, CiarletElement<T, IdentityMap, T>> {
    let mut b = SingleElementGridBuilder::new_with_capacity(
        3,
        match cell_type {
            ReferenceCellType::Triangle => 2 + usize::pow(4, refinement_level + 1),
            ReferenceCellType::Quadrilateral => 2 + 6 * usize::pow(4, refinement_level),
            _ => {
                panic!("Unsupported cell type: {cell_type:?}");
            }
        },
        match cell_type {
            ReferenceCellType::Triangle => 2 * usize::pow(4, refinement_level + 1),
            ReferenceCellType::Quadrilateral => 6 * usize::pow(4, refinement_level),
            _ => {
                panic!("Unsupported cell type: {cell_type:?}");
            }
        },
        (cell_type, 1),
    );
    match cell_type {
        ReferenceCellType::Triangle => {
            regular_sphere_triangle_add_points_and_cells(&mut b, refinement_level)
        }
        ReferenceCellType::Quadrilateral => {
            regular_sphere_quadrilateral_add_points_and_cells(&mut b, refinement_level)
        }
        _ => {
            panic!("Unsupported cell type: {cell_type:?}");
        }
    }
    b.create_grid()
}

/// Create a grid of a regular sphere distributed in parallel
#[cfg(feature = "mpi")]
pub fn regular_sphere_distributed<T: Scalar + Equivalence, C: Communicator>(
    comm: &C,
    partitioner: GraphPartitioner,
    refinement_level: u32,
    cell_type: ReferenceCellType,
) -> ParallelGridImpl<'_, C, SingleElementGrid<T, CiarletElement<T, IdentityMap, T>>> {
    let mut b = SingleElementGridBuilder::new(3, (cell_type, 1));
    if comm.rank() == 0 {
        match cell_type {
            ReferenceCellType::Triangle => {
                regular_sphere_triangle_add_points_and_cells(&mut b, refinement_level)
            }
            ReferenceCellType::Quadrilateral => {
                regular_sphere_quadrilateral_add_points_and_cells(&mut b, refinement_level)
            }
            _ => {
                panic!("Unsupported cell type: {cell_type:?}");
            }
        }
        b.create_parallel_grid_root(comm, partitioner)
    } else {
        b.create_parallel_grid(comm, 0)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::traits::{GeometryMap, Grid};
    use approx::assert_relative_eq;
    use paste::paste;
    use rlst::rlst_dynamic_array;

    macro_rules! test_regular_sphere {
        ($cell:ident, $order:expr) => {
            paste! {
                #[test]
                fn [<test_regular_sphere_ $cell:lower _ $order>]() {
                    let _g = regular_sphere::<f64>([<$order>], ReferenceCellType::[<$cell>]);
                }
            }
        };
    }

    test_regular_sphere!(Triangle, 0);
    test_regular_sphere!(Triangle, 1);
    test_regular_sphere!(Triangle, 2);
    test_regular_sphere!(Triangle, 3);
    test_regular_sphere!(Quadrilateral, 0);
    test_regular_sphere!(Quadrilateral, 1);
    test_regular_sphere!(Quadrilateral, 2);
    test_regular_sphere!(Quadrilateral, 3);

    macro_rules! test_normals {
        ($cell:ident, $order:expr) => {
            paste! {
                #[test]
                fn [<test_normal_is_outward_ $cell:lower _ $order>]() {
                    let g = regular_sphere::<f64>([<$order>], ReferenceCellType::[<$cell>]);
                    let mut points = rlst_dynamic_array!(f64, [2, 1]);
                    points[[0, 0]] = 1.0 / 3.0;
                    points[[1, 0]] = 1.0 / 3.0;
                    let map = g.geometry_map(ReferenceCellType::[<$cell>], 1, &points);
                    let mut mapped_pt = rlst_dynamic_array!(f64, [3, 1]);
                    let mut j = rlst_dynamic_array!(f64, [3, 2, 1]);
                    let mut jinv = rlst_dynamic_array!(f64, [2, 3, 1]);
                    let mut jdet = vec![0.0];
                    let mut normal = rlst_dynamic_array!(f64, [3, 1]);
                    for i in 0..g.entity_count(ReferenceCellType::[<$cell>]) {
                        map.physical_points(i, &mut mapped_pt);
                        map.jacobians_inverses_dets_normals(i, &mut j, &mut jinv, &mut jdet, &mut normal);
                        let dot = mapped_pt
                            .iter_value()
                            .zip(normal.iter_value())
                            .map(|(i, j)| i * j)
                            .sum::<f64>();
                        println!("{i}");
                        assert!(dot > 0.0);
                    }
                }

                #[test]
                fn [<test_normal_is_unit_ $cell:lower _ $order>]() {
                    let g = regular_sphere::<f64>([<$order>], ReferenceCellType::[<$cell>]);
                    let mut points = rlst_dynamic_array!(f64, [2, 1]);
                    points[[0, 0]] = 1.0 / 3.0;
                    points[[1, 0]] = 1.0 / 3.0;
                    let map = g.geometry_map(ReferenceCellType::[<$cell>], 1, &points);
                    let mut mapped_pt = rlst_dynamic_array!(f64, [3, 1]);
                    let mut j = rlst_dynamic_array!(f64, [3, 2, 1]);
                    let mut jinv = rlst_dynamic_array!(f64, [2, 3, 1]);
                    let mut jdet = vec![0.0];
                    let mut normal = rlst_dynamic_array!(f64, [3, 1]);
                    for i in 0..g.entity_count(ReferenceCellType::[<$cell>]) {
                        map.physical_points(i, &mut mapped_pt);
                        map.jacobians_inverses_dets_normals(i, &mut j, &mut jinv, &mut jdet, &mut normal);
                        let dot = normal
                            .iter_value()
                            .zip(normal.iter_value())
                            .map(|(i, j)| i * j)
                            .sum::<f64>();
                        assert_relative_eq!(dot, 1.0, epsilon = 1e-10);
                    }
                }
            }
        };
    }

    test_normals!(Triangle, 0);
    test_normals!(Triangle, 1);
    test_normals!(Triangle, 2);
    test_normals!(Quadrilateral, 0);
    test_normals!(Quadrilateral, 1);
    test_normals!(Quadrilateral, 2);
}
