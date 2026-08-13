//! Regular sphere mesh

use super::resample_cells;
#[cfg(feature = "mpi")]
use crate::{ParallelMeshImpl, traits::ParallelBuilder, types::GraphPartitioner};
use crate::{
    mesh::local_mesh::{SingleElementMesh, SingleElementMeshBuilder},
    traits::Builder,
    types::Scalar,
};
#[cfg(feature = "mpi")]
use mpi::traits::{Communicator, Equivalence};
use ndelement::{ciarlet::CiarletElement, map::IdentityMap, types::ReferenceCellType};

/// Add points and cells for a square screen to builder
fn screen_add_points_and_cells<T: Scalar>(
    b: &mut SingleElementMeshBuilder<T>,
    ncells: usize,
    cell_type: ReferenceCellType,
    degree: usize,
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
            let mut cells = vec![];
            for y in 0..ncells {
                for x in 0..ncells {
                    cells.push([
                        y * (ncells + 1) + x,
                        y * (ncells + 1) + x + 1,
                        y * (ncells + 1) + x + ncells + 2,
                    ]);
                    cells.push([
                        y * (ncells + 1) + x,
                        y * (ncells + 1) + x + ncells + 2,
                        y * (ncells + 1) + x + ncells + 1,
                    ]);
                }
            }
            if degree == 1 {
                for (i, v) in cells.iter().enumerate() {
                    b.add_cell(i, v)
                }
            } else {
                for (i, v) in resample_cells::<T, 3, 3>(degree, b, &cells, cell_type)
                    .iter()
                    .enumerate()
                {
                    b.add_cell(i, v)
                }
            }
        }
        ReferenceCellType::Quadrilateral => {
            let mut cells = vec![];
            for y in 0..ncells {
                for x in 0..ncells {
                    cells.push([
                        y * (ncells + 1) + x,
                        y * (ncells + 1) + x + 1,
                        y * (ncells + 1) + x + ncells + 1,
                        y * (ncells + 1) + x + ncells + 2,
                    ]);
                }
            }
            if degree == 1 {
                for (i, v) in cells.iter().enumerate() {
                    b.add_cell(i, v)
                }
            } else {
                for (i, v) in resample_cells::<T, 3, 4>(degree, b, &cells, cell_type)
                    .iter()
                    .enumerate()
                {
                    b.add_cell(i, v)
                }
            }
        }
        _ => {
            panic!("Unsupported cell type: {cell_type:?}");
        }
    }
}

/// Create a mesh of a square screen
///
/// Create a mesh of the square \[0,1\]^2. The input ncells is the number of cells
/// along each side of the square.
pub fn screen<T: Scalar>(
    ncells: usize,
    cell_type: ReferenceCellType,
    degree: usize,
) -> SingleElementMesh<T, CiarletElement<T, IdentityMap, T>> {
    let mut b = SingleElementMeshBuilder::new(3, (cell_type, degree));
    screen_add_points_and_cells(&mut b, ncells, cell_type, degree);
    b.create_mesh()
}

/// Create a mesh of a square screen distributed in parallel
#[cfg(feature = "mpi")]
pub fn screen_distributed<T: Scalar + Equivalence, C: Communicator>(
    comm: &C,
    partitioner: GraphPartitioner,
    ncells: usize,
    cell_type: ReferenceCellType,
    degree: usize,
) -> ParallelMeshImpl<'_, C, SingleElementMesh<T, CiarletElement<T, IdentityMap, T>>> {
    let mut b = SingleElementMeshBuilder::new(3, (cell_type, 1));
    if comm.rank() == 0 {
        screen_add_points_and_cells(&mut b, ncells, cell_type, degree);
        b.create_parallel_mesh_root(comm, partitioner)
    } else {
        b.create_parallel_mesh(comm, 0)
    }
}

#[cfg(test)]
mod test {
    use super::super::test::check_volume;
    use super::*;
    use crate::traits::{GeometryMap, Mesh};
    use approx::assert_relative_eq;
    use rlst::rlst_dynamic_array;

    #[test]
    fn test_screen_triangles() {
        let _g1 = screen::<f64>(1, ReferenceCellType::Triangle, 1);
        let _g2 = screen::<f64>(2, ReferenceCellType::Triangle, 1);
        let _g3 = screen::<f64>(3, ReferenceCellType::Triangle, 1);
    }
    #[test]
    fn test_screen_triangles_normals() {
        for i in 1..5 {
            let g = screen::<f64>(i, ReferenceCellType::Triangle, 1);
            let mut points = rlst_dynamic_array!(f64, [2, 1]);
            points[[0, 0]] = 1.0 / 3.0;
            points[[1, 0]] = 1.0 / 3.0;
            let map = g.geometry_map(ReferenceCellType::Triangle, 1, &points);
            let mut mapped_pt = rlst_dynamic_array!(f64, [3, 1]);
            let mut j = rlst_dynamic_array!(f64, [3, 2, 1]);
            let mut jinv = rlst_dynamic_array!(f64, [2, 3, 1]);
            let mut jdet = vec![0.0];
            let mut normal = rlst_dynamic_array!(f64, [3, 1]);
            for i in 0..g.entity_count(ReferenceCellType::Triangle) {
                map.physical_points(i, &mut mapped_pt);
                map.jacobians_inverses_dets_normals(i, &mut j, &mut jinv, &mut jdet, &mut normal);
                assert!(normal[[2, 0]] > 0.0);
                assert_relative_eq!(normal[[2, 0]], 1.0);
            }
        }
    }

    #[test]
    fn test_area_triangles_deg1() {
        let g = screen::<f64>(1, ReferenceCellType::Triangle, 1);
        check_volume(&g, 1, 1.0, 1e-10);
    }

    #[test]
    fn test_area_triangles_deg2() {
        let g = screen::<f64>(1, ReferenceCellType::Triangle, 2);
        check_volume(&g, 2, 1.0, 1e-10);
    }

    #[test]
    fn test_area_triangles_deg3() {
        let g = screen::<f64>(1, ReferenceCellType::Triangle, 3);
        check_volume(&g, 3, 1.0, 1e-10);
    }

    #[test]
    fn test_screen_quadrilaterals() {
        let _g1 = screen::<f64>(1, ReferenceCellType::Quadrilateral, 1);
        let _g2 = screen::<f64>(2, ReferenceCellType::Quadrilateral, 1);
        let _g3 = screen::<f64>(3, ReferenceCellType::Quadrilateral, 1);
    }

    #[test]
    fn test_screen_quadrilaterals_normals() {
        for i in 1..5 {
            let g = screen::<f64>(i, ReferenceCellType::Quadrilateral, 1);
            let mut points = rlst_dynamic_array!(f64, [2, 1]);
            points[[0, 0]] = 1.0 / 3.0;
            points[[1, 0]] = 1.0 / 3.0;
            let map = g.geometry_map(ReferenceCellType::Quadrilateral, 1, &points);
            let mut mapped_pt = rlst_dynamic_array!(f64, [3, 1]);
            let mut j = rlst_dynamic_array!(f64, [3, 2, 1]);
            let mut jinv = rlst_dynamic_array!(f64, [2, 3, 1]);
            let mut jdet = vec![0.0];
            let mut normal = rlst_dynamic_array!(f64, [3, 1]);
            for i in 0..g.entity_count(ReferenceCellType::Quadrilateral) {
                map.physical_points(i, &mut mapped_pt);
                map.jacobians_inverses_dets_normals(i, &mut j, &mut jinv, &mut jdet, &mut normal);
                assert!(normal[[2, 0]] > 0.0);
                assert_relative_eq!(normal[[2, 0]], 1.0);
            }
        }
    }

    #[test]
    fn test_area_quadrilaterals_deg1() {
        let g = screen::<f64>(1, ReferenceCellType::Quadrilateral, 1);
        check_volume(&g, 1, 1.0, 1e-10);
    }

    #[test]
    fn test_area_quadrilaterals_deg2() {
        let g = screen::<f64>(1, ReferenceCellType::Quadrilateral, 2);
        check_volume(&g, 2, 1.0, 1e-10);
    }

    #[test]
    fn test_area_quadrilaterals_deg3() {
        let g = screen::<f64>(1, ReferenceCellType::Quadrilateral, 3);
        check_volume(&g, 3, 1.0, 1e-10);
    }
}
