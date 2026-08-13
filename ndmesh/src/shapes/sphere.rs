//! Regular sphere mesh

use super::resample_cells_map_points;
#[cfg(feature = "mpi")]
use crate::{ParallelMeshImpl, traits::ParallelBuilder, types::GraphPartitioner};
use crate::{
    mesh::local_mesh::{MixedMesh, MixedMeshBuilder, SingleElementMesh, SingleElementMeshBuilder},
    traits::Builder,
    types::Scalar,
};
#[cfg(feature = "mpi")]
use mpi::traits::{Communicator, Equivalence};
use ndelement::{ciarlet::CiarletElement, map::IdentityMap, types::ReferenceCellType};
use rlst::RlstScalar;
use std::collections::{HashMap, hash_map::Entry::Vacant};

/// Normalise a point so that it lies on the sphere with centre (0,0) and radius 1
fn normalise<T: Scalar>(pt: [T; 3]) -> [T; 3] {
    let size = (pt.iter().map(|x| x.powi(2)).sum::<T>()).sqrt();
    [pt[0] / size, pt[1] / size, pt[2] / size]
}

/// Refine each triangle into four triangles
fn refine_triangles<T: Scalar>(
    b: &mut impl Builder<T = T>,
    triangles: &[[usize; 3]],
    edge_points: &mut HashMap<(usize, usize), usize>,
) -> Vec<[usize; 3]> {
    let mut new_triangles = Vec::with_capacity(4 * triangles.len());
    let mut v = [
        [T::zero(), T::zero(), T::zero()],
        [T::zero(), T::zero(), T::zero()],
        [T::zero(), T::zero(), T::zero()],
    ];

    for c in triangles {
        for (i, v_i) in v.iter_mut().enumerate() {
            for (j, v_ij) in v_i.iter_mut().enumerate() {
                *v_ij = b.points()[3 * c[i] + j];
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
                    let new_pt = normalise([
                        T::from(0.5).unwrap() * (v_i[0] + v_j[0]),
                        T::from(0.5).unwrap() * (v_i[1] + v_j[1]),
                        T::from(0.5).unwrap() * (v_i[2] + v_j[2]),
                    ]);
                    e.insert(b.point_count());
                    b.add_point(b.point_count(), &new_pt);
                }
                edge_points[&(pt_i, pt_j)]
            })
            .collect::<Vec<_>>();
        new_triangles.push([c[0], edges[2], edges[1]]);
        new_triangles.push([c[1], edges[0], edges[2]]);
        new_triangles.push([c[2], edges[1], edges[0]]);
        new_triangles.push([edges[0], edges[1], edges[2]]);
    }
    new_triangles
}

/// Refine each quadrilateral into four quadrilaterals
fn refine_quadrilaterals<T: Scalar>(
    b: &mut impl Builder<T = T>,
    quadrilaterals: &[[usize; 4]],
    edge_points: &mut HashMap<(usize, usize), usize>,
) -> Vec<[usize; 4]> {
    let mut new_quadrilaterals = Vec::with_capacity(4 * quadrilaterals.len());
    let mut v = [
        [T::zero(), T::zero(), T::zero()],
        [T::zero(), T::zero(), T::zero()],
        [T::zero(), T::zero(), T::zero()],
        [T::zero(), T::zero(), T::zero()],
    ];

    for c in quadrilaterals {
        for (i, v_i) in v.iter_mut().enumerate() {
            for (j, v_ij) in v_i.iter_mut().enumerate() {
                *v_ij = b.points()[3 * c[i] + j];
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
                    let new_pt = normalise([
                        T::from(0.5).unwrap() * (v_i[0] + v_j[0]),
                        T::from(0.5).unwrap() * (v_i[1] + v_j[1]),
                        T::from(0.5).unwrap() * (v_i[2] + v_j[2]),
                    ]);
                    e.insert(b.point_count());
                    b.add_point(b.point_count(), &new_pt);
                }
                edge_points[&(pt_i, pt_j)]
            })
            .collect::<Vec<_>>();
        new_quadrilaterals.push([c[0], edges[0], edges[1], b.point_count()]);
        new_quadrilaterals.push([edges[0], c[1], b.point_count(), edges[2]]);
        new_quadrilaterals.push([edges[1], b.point_count(), c[2], edges[3]]);
        new_quadrilaterals.push([b.point_count(), edges[2], edges[3], c[3]]);

        b.add_point(
            b.point_count(),
            &normalise([
                T::from(0.25).unwrap() * v.iter().map(|i| i[0]).sum(),
                T::from(0.25).unwrap() * v.iter().map(|i| i[1]).sum(),
                T::from(0.25).unwrap() * v.iter().map(|i| i[2]).sum(),
            ]),
        );
    }
    new_quadrilaterals
}

/// Add points and cells for regular sphere to builder
fn regular_sphere_triangle_add_points_and_cells<T: Scalar>(
    b: &mut SingleElementMeshBuilder<T>,
    refinement_level: u32,
    degree: usize,
) {
    b.add_point(0, &[T::zero(), T::zero(), T::one()]);
    b.add_point(1, &[T::one(), T::zero(), T::zero()]);
    b.add_point(2, &[T::zero(), T::one(), T::zero()]);
    b.add_point(3, &[-T::one(), T::zero(), T::zero()]);
    b.add_point(4, &[T::zero(), -T::one(), T::zero()]);
    b.add_point(5, &[T::zero(), T::zero(), -T::one()]);

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

    for _level in 0..refinement_level {
        let mut edge_points = HashMap::new();
        cells = refine_triangles(b, &cells, &mut edge_points);
    }

    if degree == 1 {
        for (i, v) in cells.iter().enumerate() {
            b.add_cell(i, v);
        }
    } else {
        for (i, v) in resample_cells_map_points::<T, 3, 3>(
            degree,
            b,
            &cells,
            ReferenceCellType::Triangle,
            normalise,
        )
        .iter()
        .enumerate()
        {
            b.add_cell(i, v);
        }
    }
}

/// Add points and cells for regular sphere to builder
fn regular_sphere_quadrilateral_add_points_and_cells<T: Scalar>(
    b: &mut SingleElementMeshBuilder<T>,
    refinement_level: u32,
    degree: usize,
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

    let mut cells = vec![
        [0, 1, 2, 3],
        [4, 6, 5, 7],
        [0, 4, 1, 5],
        [2, 3, 6, 7],
        [0, 2, 4, 6],
        [1, 5, 3, 7],
    ];

    for _level in 0..refinement_level {
        let mut edge_points = HashMap::new();
        cells = refine_quadrilaterals(b, &cells, &mut edge_points);
    }

    if degree == 1 {
        for (i, v) in cells.iter().enumerate() {
            b.add_cell(i, v);
        }
    } else {
        for (i, v) in resample_cells_map_points::<T, 3, 4>(
            degree,
            b,
            &cells,
            ReferenceCellType::Quadrilateral,
            normalise,
        )
        .iter()
        .enumerate()
        {
            b.add_cell(i, v);
        }
    }
}
/// Create a surface mesh of a regular sphere
///
/// A regular sphere is created by starting with a regular octahedron. The shape is then refined `refinement_level` times.
/// Each time the mesh is refined, each triangle is split into four triangles (by adding lines connecting the midpoints of
/// each edge). The new points are scaled so that they are a distance of 1 from the origin.
pub fn regular_sphere<T: Scalar + RlstScalar<Real = T>>(
    refinement_level: u32,
    cell_type: ReferenceCellType,
    degree: usize,
) -> SingleElementMesh<T, CiarletElement<T, IdentityMap, T>>
where
{
    let mut b = SingleElementMeshBuilder::new(3, (cell_type, degree));
    match cell_type {
        ReferenceCellType::Triangle => {
            regular_sphere_triangle_add_points_and_cells(&mut b, refinement_level, degree)
        }
        ReferenceCellType::Quadrilateral => {
            regular_sphere_quadrilateral_add_points_and_cells(&mut b, refinement_level, degree)
        }
        _ => {
            panic!("Unsupported cell type: {cell_type:?}");
        }
    }

    b.create_mesh()
}

/// Create a mesh of a regular sphere distributed in parallel
#[cfg(feature = "mpi")]
pub fn regular_sphere_distributed<T: Scalar + Equivalence, C: Communicator>(
    comm: &C,
    partitioner: GraphPartitioner,
    refinement_level: u32,
    cell_type: ReferenceCellType,
    degree: usize,
) -> ParallelMeshImpl<'_, C, SingleElementMesh<T, CiarletElement<T, IdentityMap, T>>> {
    let mut b = SingleElementMeshBuilder::new(3, (cell_type, degree));
    if comm.rank() == 0 {
        match cell_type {
            ReferenceCellType::Triangle => {
                regular_sphere_triangle_add_points_and_cells(&mut b, refinement_level, degree)
            }
            ReferenceCellType::Quadrilateral => {
                regular_sphere_quadrilateral_add_points_and_cells(&mut b, refinement_level, degree)
            }
            _ => {
                panic!("Unsupported cell type: {cell_type:?}");
            }
        }
        b.create_parallel_mesh_root(comm, partitioner)
    } else {
        b.create_parallel_mesh(comm, 0)
    }
}

/// Add points and cells for mixed sphere to builder
fn mixed_sphere_add_points_and_cells<T: Scalar>(
    b: &mut MixedMeshBuilder<T>,
    refinement_level: u32,
    degree: usize,
) {
    let k = T::from(1.0 / 2.0).unwrap().sqrt();
    b.add_point(0, &[T::zero(), -k, -k]);
    b.add_point(1, &[T::zero(), -k, k]);
    b.add_point(2, &[T::zero(), k, -k]);
    b.add_point(3, &[T::zero(), k, k]);
    b.add_point(4, &[-k, T::zero(), -k]);
    b.add_point(5, &[-k, T::zero(), k]);
    b.add_point(6, &[k, T::zero(), -k]);
    b.add_point(7, &[k, T::zero(), k]);
    b.add_point(8, &[-k, -k, T::zero()]);
    b.add_point(9, &[-k, k, T::zero()]);
    b.add_point(10, &[k, -k, T::zero()]);
    b.add_point(11, &[k, k, T::zero()]);

    let mut tris = vec![
        [0, 8, 4],
        [0, 6, 10],
        [2, 4, 9],
        [2, 11, 6],
        [7, 11, 3],
        [1, 10, 7],
        [3, 9, 5],
        [5, 8, 1],
    ];
    let mut quads = vec![
        [0, 10, 8, 1],
        [7, 10, 11, 6],
        [2, 9, 11, 3],
        [4, 8, 9, 5],
        [3, 5, 7, 1],
        [0, 4, 6, 2],
    ];

    for _level in 0..refinement_level {
        let mut edge_points = HashMap::new();
        tris = refine_triangles(b, &tris, &mut edge_points);
        quads = refine_quadrilaterals(b, &quads, &mut edge_points);
    }

    if degree == 1 {
        for (i, v) in tris.iter().enumerate() {
            b.add_cell(i, (ReferenceCellType::Triangle, 1, v));
        }
        for (i, v) in quads.iter().enumerate() {
            b.add_cell(tris.len() + i, (ReferenceCellType::Quadrilateral, 1, v));
        }
    } else {
        for (i, v) in resample_cells_map_points::<T, 3, 3>(
            degree,
            b,
            &tris,
            ReferenceCellType::Triangle,
            normalise,
        )
        .iter()
        .enumerate()
        {
            b.add_cell(i, (ReferenceCellType::Triangle, degree, v));
        }
        for (i, v) in resample_cells_map_points::<T, 3, 4>(
            degree,
            b,
            &quads,
            ReferenceCellType::Quadrilateral,
            normalise,
        )
        .iter()
        .enumerate()
        {
            b.add_cell(
                tris.len() + i,
                (ReferenceCellType::Quadrilateral, degree, v),
            );
        }
    }
}
/// Create a surface mesh of a sphere with a mixture of triangles and quadrilaterals
///
/// A mixed sphere is created by starting with a cuboctahedron. The shape is then refined `refinement_level` times.
/// Each time the mesh is refined, each triangle is split into four triangles (by adding lines connecting the midpoints of
/// each edge) and each quadrilateral is refined into four quadrilataterals (by adding lines connecting the midpoints of
/// opposite edges). The new points are scaled so that they are a distance of 1 from the origin.
pub fn mixed_sphere<T: Scalar + RlstScalar<Real = T>>(
    refinement_level: u32,
    degree: usize,
) -> MixedMesh<T, CiarletElement<T, IdentityMap, T>>
where
{
    let mut b = MixedMeshBuilder::new(3);
    mixed_sphere_add_points_and_cells(&mut b, refinement_level, degree);

    b.create_mesh()
}

/// Create a mesh of a mixed sphere distributed in parallel
#[cfg(feature = "mpi")]
pub fn mixed_sphere_distributed<T: Scalar + Equivalence, C: Communicator>(
    comm: &C,
    partitioner: GraphPartitioner,
    refinement_level: u32,
    degree: usize,
) -> ParallelMeshImpl<'_, C, MixedMesh<T, CiarletElement<T, IdentityMap, T>>> {
    let mut b = MixedMeshBuilder::new(3);
    if comm.rank() == 0 {
        mixed_sphere_add_points_and_cells(&mut b, refinement_level, degree);
        b.create_parallel_mesh_root(comm, partitioner)
    } else {
        b.create_parallel_mesh(comm, 0)
    }
}

#[cfg(test)]
mod test {
    use super::super::test::{check_volume, test_normals_are_outward, test_normals_are_unit};
    use super::*;
    use crate::traits::Mesh;
    use paste::paste;
    use rlst::rlst_dynamic_array;
    use std::f64::consts::PI;

    macro_rules! test_regular_sphere {
        ($cell:ident, $refine:expr, $degree:expr) => {
            paste! {
                #[test]
                fn [<test_normals_are_outward_ $cell:lower _ref $refine _deg $degree>]() {
                    let g = regular_sphere::<f64>([<$refine>], ReferenceCellType::[<$cell>], [<$degree>]);
                    let mut point = rlst_dynamic_array!(f64, [2, 1]);
                    point[[0, 0]] = 1.0 / 3.0;
                    point[[1, 0]] = 1.0 / 3.0;
                    let mut centre = rlst_dynamic_array!(f64, [3, 1]);
                    centre[[0, 0]] = 0.0;
                    centre[[1, 0]] = 0.0;
                    centre[[2, 0]] = 0.0;
                    test_normals_are_outward(&g, ReferenceCellType::[<$cell>], &point, &centre, [<$degree>]);
                }

                #[test]
                fn [<test_normals_are_unit_ $cell:lower _ref $refine _deg $degree>]() {
                    let g = regular_sphere::<f64>([<$refine>], ReferenceCellType::[<$cell>], [<$degree>]);
                    let mut point = rlst_dynamic_array!(f64, [2, 1]);
                    point[[0, 0]] = 1.0 / 3.0;
                    point[[1, 0]] = 1.0 / 3.0;
                    test_normals_are_unit(&g, ReferenceCellType::[<$cell>], &point, [<$degree>]);
                }
            }
        };
    }

    test_regular_sphere!(Triangle, 0, 1);
    test_regular_sphere!(Triangle, 1, 1);
    test_regular_sphere!(Triangle, 2, 1);
    test_regular_sphere!(Triangle, 3, 1);
    test_regular_sphere!(Triangle, 1, 2);
    test_regular_sphere!(Triangle, 1, 3);
    test_regular_sphere!(Quadrilateral, 0, 1);
    test_regular_sphere!(Quadrilateral, 1, 1);
    test_regular_sphere!(Quadrilateral, 2, 1);
    test_regular_sphere!(Quadrilateral, 3, 1);
    test_regular_sphere!(Quadrilateral, 1, 2);
    test_regular_sphere!(Quadrilateral, 1, 3);

    #[test]
    fn test_surface_area_regular_sphere_triangle_deg1() {
        let g = regular_sphere::<f64>(4, ReferenceCellType::Triangle, 1);
        check_volume(&g, 1, 4.0 * PI, 1e-1);
    }

    #[test]
    fn test_surface_area_regular_sphere_triangle_deg2() {
        let g = regular_sphere::<f64>(3, ReferenceCellType::Triangle, 2);
        check_volume(&g, 2, 4.0 * PI, 1e-2);
    }

    #[test]
    fn test_surface_area_regular_sphere_triangle_deg3() {
        let g = regular_sphere::<f64>(2, ReferenceCellType::Triangle, 3);
        check_volume(&g, 3, 4.0 * PI, 1e-2);
    }

    #[test]
    fn test_surface_area_regular_sphere_quadrilateral_deg1() {
        let g = regular_sphere::<f64>(4, ReferenceCellType::Quadrilateral, 1);
        check_volume(&g, 1, 4.0 * PI, 1e-1);
    }

    #[test]
    fn test_surface_area_regular_sphere_quadrilateral_deg2() {
        let g = regular_sphere::<f64>(3, ReferenceCellType::Quadrilateral, 2);
        check_volume(&g, 2, 4.0 * PI, 1e-2);
    }

    #[test]
    fn test_surface_area_regular_sphere_quadrilateral_deg3() {
        let g = regular_sphere::<f64>(2, ReferenceCellType::Quadrilateral, 3);
        check_volume(&g, 3, 4.0 * PI, 1e-2);
    }

    #[test]
    fn test_mixed_sphere_0() {
        let mesh = mixed_sphere::<f64>(0, 1);
        assert_eq!(mesh.entity_count(ReferenceCellType::Point), 12);
        assert_eq!(mesh.entity_count(ReferenceCellType::Interval), 24);
        assert_eq!(mesh.entity_count(ReferenceCellType::Quadrilateral), 6);
        assert_eq!(mesh.entity_count(ReferenceCellType::Triangle), 8);
    }

    macro_rules! test_mixed_sphere {
        ($refine:expr, $degree:expr) => {
            paste! {
                #[test]
                fn [<test_normals_are_outward_mixed_sphere_ref $refine _deg $degree>]() {
                    let g = mixed_sphere::<f64>([<$refine>], [<$degree>]);
                    let mut point = rlst_dynamic_array!(f64, [2, 1]);
                    point[[0, 0]] = 1.0 / 3.0;
                    point[[1, 0]] = 1.0 / 3.0;
                    let mut centre = rlst_dynamic_array!(f64, [3, 1]);
                    centre[[0, 0]] = 0.0;
                    centre[[1, 0]] = 0.0;
                    centre[[2, 0]] = 0.0;
                    test_normals_are_outward(&g, ReferenceCellType::Triangle, &point, &centre, [<$degree>]);
                    test_normals_are_outward(&g, ReferenceCellType::Quadrilateral, &point, &centre, [<$degree>]);
                }

                #[test]
                fn [<test_normals_are_unit_mixed_sphere_ref $refine _deg $degree>]() {
                    let g = mixed_sphere::<f64>([<$refine>], [<$degree>]);
                    let mut point = rlst_dynamic_array!(f64, [2, 1]);
                    point[[0, 0]] = 1.0 / 3.0;
                    point[[1, 0]] = 1.0 / 3.0;
                    test_normals_are_unit(&g, ReferenceCellType::Triangle, &point, [<$degree>]);
                    test_normals_are_unit(&g, ReferenceCellType::Quadrilateral, &point, [<$degree>]);
                }
            }
        };
    }

    test_mixed_sphere!(0, 1);
    test_mixed_sphere!(1, 1);
    test_mixed_sphere!(2, 1);
    test_mixed_sphere!(3, 1);
    test_mixed_sphere!(1, 2);
    test_mixed_sphere!(1, 3);

    #[test]
    fn test_surface_area_mixed_sphere_deg1() {
        let g = mixed_sphere::<f64>(4, 1);
        check_volume(&g, 1, 4.0 * PI, 1e-1);
    }

    #[test]
    fn test_surface_area_mixed_sphere_deg2() {
        let g = mixed_sphere::<f64>(3, 2);
        check_volume(&g, 2, 4.0 * PI, 1e-2);
    }
}
