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

/// Add points and cells for a square screen to builder
fn screen_add_points_and_cells<T: Scalar>(
    b: &mut SingleElementGridBuilder<T>,
    ncells: usize,
    cell_type: ReferenceCellType,
) {
    let zero = T::from(0.0).unwrap();
    let n = T::from(ncells).unwrap();
    for y in 0..ncells + 1 {
        for x in 0..ncells + 1 {
            b.add_point(
                y * (ncells + 1) + x,
                &[T::from(x).unwrap() / n, T::from(y).unwrap() / n, zero],
            );
        }
    }
    match cell_type {
        ReferenceCellType::Triangle => {
            for y in 0..ncells {
                for x in 0..ncells {
                    b.add_cell(
                        2 * y * ncells + 2 * x,
                        &[
                            y * (ncells + 1) + x,
                            y * (ncells + 1) + x + 1,
                            y * (ncells + 1) + x + ncells + 2,
                        ],
                    );
                    b.add_cell(
                        2 * y * ncells + 2 * x + 1,
                        &[
                            y * (ncells + 1) + x,
                            y * (ncells + 1) + x + ncells + 2,
                            y * (ncells + 1) + x + ncells + 1,
                        ],
                    );
                }
            }
        }
        ReferenceCellType::Quadrilateral => {
            for y in 0..ncells {
                for x in 0..ncells {
                    b.add_cell(
                        y * ncells + x,
                        &[
                            y * (ncells + 1) + x,
                            y * (ncells + 1) + x + 1,
                            y * (ncells + 1) + x + ncells + 1,
                            y * (ncells + 1) + x + ncells + 2,
                        ],
                    );
                }
            }
        }
        _ => {
            panic!("Unsupported cell type: {cell_type:?}");
        }
    }
}

/// Create a grid of a square screen
///
/// Create a grid of the square \[0,1\]^2. The input ncells is the number of cells
/// along each side of the square.
pub fn screen<T: Scalar>(
    ncells: usize,
    cell_type: ReferenceCellType,
) -> SingleElementGrid<T, CiarletElement<T, IdentityMap, T>> {
    let mut b = SingleElementGridBuilder::new_with_capacity(
        3,
        (ncells + 1) * (ncells + 1),
        match cell_type {
            ReferenceCellType::Quadrilateral => ncells * ncells,
            ReferenceCellType::Triangle => 2 * ncells * ncells,
            _ => {
                panic!("Unsupported cell type: {cell_type:?}");
            }
        },
        (cell_type, 1),
    );
    screen_add_points_and_cells(&mut b, ncells, cell_type);
    b.create_grid()
}

/// Create a grid of a square screen distributed in parallel
#[cfg(feature = "mpi")]
pub fn screen_distributed<T: Scalar + Equivalence, C: Communicator>(
    comm: &C,
    partitioner: GraphPartitioner,
    ncells: usize,
    cell_type: ReferenceCellType,
) -> ParallelGridImpl<'_, C, SingleElementGrid<T, CiarletElement<T, IdentityMap, T>>> {
    let mut b = SingleElementGridBuilder::new(3, (cell_type, 1));
    if comm.rank() == 0 {
        screen_add_points_and_cells(&mut b, ncells, cell_type);
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

    #[test]
    fn test_screen_triangles() {
        let _g1 = screen::<f64>(1, ReferenceCellType::Triangle);
        let _g2 = screen::<f64>(2, ReferenceCellType::Triangle);
        let _g3 = screen::<f64>(3, ReferenceCellType::Triangle);
    }
    #[test]
    fn test_screen_triangles_normals() {
        for i in 1..5 {
            let g = screen::<f64>(i, ReferenceCellType::Triangle);
            let points = vec![1.0 / 3.0, 1.0 / 3.0];
            let map = g.geometry_map(ReferenceCellType::Triangle, 1, &points);
            let mut mapped_pt = vec![0.0; 3];
            let mut j = vec![0.0; 6];
            let mut jinv = vec![0.0; 6];
            let mut jdet = vec![0.0];
            let mut normal = vec![0.0; 3];
            for i in 0..g.entity_count(ReferenceCellType::Triangle) {
                map.physical_points(i, &mut mapped_pt);
                map.jacobians_dets_normals(i, &mut j, &mut jinv, &mut jdet, &mut normal);
                assert!(normal[2] > 0.0);
                assert_relative_eq!(normal[2], 1.0);
            }
        }
    }

    #[test]
    fn test_screen_quadrilaterals() {
        let _g1 = screen::<f64>(1, ReferenceCellType::Quadrilateral);
        let _g2 = screen::<f64>(2, ReferenceCellType::Quadrilateral);
        let _g3 = screen::<f64>(3, ReferenceCellType::Quadrilateral);
    }

    #[test]
    fn test_screen_quadrilaterals_normals() {
        for i in 1..5 {
            let g = screen::<f64>(i, ReferenceCellType::Quadrilateral);
            let points = vec![1.0 / 3.0, 1.0 / 3.0];
            let map = g.geometry_map(ReferenceCellType::Quadrilateral, 1, &points);
            let mut mapped_pt = vec![0.0; 3];
            let mut j = vec![0.0; 6];
            let mut jinv = vec![0.0; 6];
            let mut jdet = vec![0.0];
            let mut normal = vec![0.0; 3];
            for i in 0..g.entity_count(ReferenceCellType::Quadrilateral) {
                map.physical_points(i, &mut mapped_pt);
                map.jacobians_dets_normals(i, &mut j, &mut jinv, &mut jdet, &mut normal);
                assert!(normal[2] > 0.0);
                assert_relative_eq!(normal[2], 1.0);
            }
        }
    }
}
