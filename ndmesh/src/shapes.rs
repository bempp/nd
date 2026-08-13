//! Functions to create simple example meshes

mod cube;
mod screen;
mod sphere;

use crate::{traits::Builder, types::Scalar};
pub use cube::{
    unit_cube, unit_cube_boundary, unit_cube_edges, unit_interval, unit_square,
    unit_square_boundary,
};
#[cfg(feature = "mpi")]
pub use cube::{
    unit_cube_boundary_distributed, unit_cube_distributed, unit_cube_edges_distributed,
    unit_interval_distributed, unit_square_boundary_distributed, unit_square_distributed,
};
use itertools::izip;
use ndelement::{
    ciarlet::{LagrangeElementFamily, LagrangeVariant},
    traits::{ElementFamily, FiniteElement},
    types::Continuity,
};
use ndelement::{reference_cell, types::ReferenceCellType};
use rlst::rlst_dynamic_array;
pub use screen::screen;
#[cfg(feature = "mpi")]
pub use screen::screen_distributed;
pub use sphere::{mixed_sphere, regular_sphere};
#[cfg(feature = "mpi")]
pub use sphere::{mixed_sphere_distributed, regular_sphere_distributed};
use std::collections::HashMap;

fn resample_cells_map_points<T: Scalar, const GDIM: usize, const NPTS: usize>(
    degree: usize,
    b: &mut impl Builder<T = T, EntityDescriptor = ReferenceCellType>,
    cells: &[[usize; NPTS]],
    cell_type: ReferenceCellType,
    map_points: impl Fn([T; GDIM]) -> [T; GDIM],
) -> Vec<Vec<usize>> {
    assert_eq!(
        match cell_type {
            ReferenceCellType::Point => 1,
            ReferenceCellType::Interval => 2,
            ReferenceCellType::Triangle => 3,
            ReferenceCellType::Quadrilateral => 4,
            ReferenceCellType::Tetrahedron => 4,
            ReferenceCellType::Hexahedron => 8,
            ReferenceCellType::Prism => 6,
            ReferenceCellType::Pyramid => 5,
        },
        NPTS,
        "Incorrect cell array shape"
    );

    let p1_family =
        LagrangeElementFamily::<T>::new(1, Continuity::Standard, LagrangeVariant::Equispaced);
    let family =
        LagrangeElementFamily::<T>::new(degree, Continuity::Standard, LagrangeVariant::Equispaced);

    let tdim = reference_cell::dim(cell_type);
    let gdim = b.gdim();

    let mut elements = HashMap::new();
    for d in 1..=tdim {
        for et in &reference_cell::entity_types(cell_type)[d] {
            elements.entry(*et).or_insert_with(|| family.element(*et));
        }
    }

    let mut tables = HashMap::new();
    for (et, e) in elements {
        let points = &e.interpolation_points()[reference_cell::dim(et)][0];
        let p1_element = p1_family.element(et);
        let mut table = rlst_dynamic_array!(T, [1, points.shape()[1], p1_element.dim(), 1]);
        p1_element.tabulate(points, 0, &mut table);
        tables.insert(et, table);
    }

    let element = family.element(cell_type);
    let mut entity_additional_points = HashMap::new();
    cells
        .iter()
        .map(|cell| {
            let mut new_cell = vec![];
            new_cell.extend(cell);
            for d in 1..=tdim {
                for (et, vs) in izip!(
                    &reference_cell::entity_types(cell_type)[d],
                    &reference_cell::connectivity(cell_type)[d]
                ) {
                    let sorted_entity = vs[0].iter().map(|v| cell[*v]).collect::<Vec<_>>();
                    // sorted_entity.sort();
                    new_cell.extend(
                        &*entity_additional_points
                            .entry(*et)
                            .or_insert(HashMap::new())
                            .entry(sorted_entity)
                            .or_insert_with(|| {
                                let table = &tables[et];
                                let npts = table.shape()[1];
                                let p1_dim = table.shape()[2];

                                let mut vertices = rlst_dynamic_array!(T, [gdim, p1_dim]);
                                for (i, v) in vs[0].iter().enumerate() {
                                    for c in 0..gdim {
                                        vertices[[c, i]] = b.points()[GDIM * cell[*v] + c];
                                    }
                                }
                                let mut pt_i = vec![];
                                for p_i in 0..npts {
                                    let mut vertex = [T::zero(); GDIM];
                                    for c in 0..GDIM {
                                        vertex[c] = (0..p1_dim)
                                            .map(|v_i| vertices[[c, v_i]] * table[[0, p_i, v_i, 0]])
                                            .sum::<T>();
                                    }
                                    pt_i.push(b.point_count());
                                    b.add_point(b.point_count(), &map_points(vertex));
                                }
                                pt_i
                            }),
                    );
                }
            }
            assert_eq!(new_cell.len(), element.dim());
            new_cell
        })
        .collect::<Vec<_>>()
}

fn resample_cells<T: Scalar, const GDIM: usize, const NPTS: usize>(
    degree: usize,
    b: &mut impl Builder<T = T, EntityDescriptor = ReferenceCellType>,
    cells: &[[usize; NPTS]],
    cell_type: ReferenceCellType,
) -> Vec<Vec<usize>> {
    resample_cells_map_points::<T, GDIM, NPTS>(degree, b, cells, cell_type, |a| a)
}

#[cfg(test)]
pub mod test {
    //! Utility function used throughout the test for shapes
    use super::*;
    use crate::{
        traits::{GeometryMap, Mesh},
        types::Scalar,
    };
    use approx::*;
    use quadraturerules::{Domain, QuadratureRule, single_integral_quadrature};
    use rlst::{Array, DynArray, ValueArrayImpl, rlst_dynamic_array};

    /// Test that normals to each cell in a mesh are unit vectors
    pub fn test_normals_are_unit<Array2Impl: ValueArrayImpl<f64, 2>>(
        mesh: &impl Mesh<T = f64, EntityDescriptor = ReferenceCellType>,
        ct: ReferenceCellType,
        point: &Array<Array2Impl, 2>,
        degree: usize,
    ) {
        let tdim = point.shape()[0];
        let gdim = tdim + 1;
        let map = mesh.geometry_map(ct, degree, point);
        let mut mapped_pt = rlst_dynamic_array!(f64, [gdim, 1]);
        let mut j = rlst_dynamic_array!(f64, [gdim, tdim, 1]);
        let mut jinv = rlst_dynamic_array!(f64, [tdim, gdim, 1]);
        let mut jdet = vec![0.0];
        let mut normal = rlst_dynamic_array!(f64, [gdim, 1]);
        for i in 0..mesh.entity_count(ct) {
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

    /// Test that normals to each cell in a mesh are pointing outwards
    pub fn test_normals_are_outward<Array2Impl: ValueArrayImpl<f64, 2>>(
        mesh: &impl Mesh<T = f64, EntityDescriptor = ReferenceCellType>,
        ct: ReferenceCellType,
        point: &Array<Array2Impl, 2>,
        centre: &Array<Array2Impl, 2>,
        degree: usize,
    ) {
        let tdim = point.shape()[0];
        let gdim = tdim + 1;
        assert_eq!(centre.shape()[0], gdim);
        let map = mesh.geometry_map(ct, degree, point);
        let mut mapped_pt = rlst_dynamic_array!(f64, [gdim, 1]);
        let mut j = rlst_dynamic_array!(f64, [gdim, tdim, 1]);
        let mut jinv = rlst_dynamic_array!(f64, [tdim, gdim, 1]);
        let mut jdet = vec![0.0];
        let mut normal = rlst_dynamic_array!(f64, [gdim, 1]);
        for i in 0..mesh.entity_count(ct) {
            map.physical_points(i, &mut mapped_pt);
            map.jacobians_inverses_dets_normals(i, &mut j, &mut jinv, &mut jdet, &mut normal);
            let dot = mapped_pt
                .iter_value()
                .zip(centre.iter_value())
                .zip(normal.iter_value())
                .map(|((i, j), k)| (i - j) * k)
                .sum::<f64>();
            assert!(dot > 0.0);
        }
    }

    /// Create quadrature rule
    fn quadrature<T: Scalar>(
        cell_type: ReferenceCellType,
        degree: usize,
    ) -> (DynArray<T, 2>, Vec<T>) {
        let (pts, wts) = match cell_type {
            ReferenceCellType::Interval => {
                let (p, w) = single_integral_quadrature(
                    QuadratureRule::GaussLobattoLegendre,
                    Domain::Interval,
                    degree,
                )
                .unwrap();
                let mut pts = rlst_dynamic_array!(T, [1, w.len()]);
                for i in 0..w.len() {
                    pts[[0, i]] = T::from(p[2 * i + 1]).unwrap();
                }
                (pts, w)
            }
            ReferenceCellType::Triangle => {
                let (p, w) = single_integral_quadrature(
                    QuadratureRule::XiaoGimbutas,
                    Domain::Triangle,
                    degree,
                )
                .unwrap();
                let mut pts = rlst_dynamic_array!(T, [2, w.len()]);
                for i in 0..w.len() {
                    pts[[0, i]] = T::from(p[3 * i + 1]).unwrap();
                    pts[[1, i]] = T::from(p[3 * i + 2]).unwrap();
                }
                (pts, w.iter().map(|i| *i * 0.5).collect::<Vec<_>>())
            }
            ReferenceCellType::Quadrilateral => {
                let (p, w) = single_integral_quadrature(
                    QuadratureRule::GaussLobattoLegendre,
                    Domain::Interval,
                    degree,
                )
                .unwrap();
                let mut pts = rlst_dynamic_array!(T, [2, w.len().pow(2)]);
                let mut wts = vec![0.0; w.len().pow(2)];
                for (ix, wx) in w.iter().enumerate() {
                    for (iy, wy) in w.iter().enumerate() {
                        let i = ix * w.len() + iy;
                        pts[[0, i]] = T::from(p[2 * ix + 1]).unwrap();
                        pts[[1, i]] = T::from(p[2 * iy + 1]).unwrap();
                        wts[i] = *wx * *wy;
                    }
                }
                (pts, wts)
            }
            _ => {
                panic!("Unsupported cell type: {cell_type:?}")
            }
        };
        (
            pts,
            wts.iter().map(|w| T::from(*w).unwrap()).collect::<Vec<_>>(),
        )
    }

    /// Check that the total volume of all cells in a mesh is the expected volume
    pub fn check_volume(
        mesh: &impl Mesh<T = f64, EntityDescriptor = ReferenceCellType>,
        degree: usize,
        volume: f64,
        epsilon: f64,
    ) {
        let mut total = 0.0;
        for ct in mesh.cell_types() {
            let (points, weights) = quadrature::<f64>(*ct, degree);
            let tdim = mesh.topology_dim();
            let gdim = mesh.geometry_dim();
            let npts = weights.len();

            let map = mesh.geometry_map(*ct, degree, &points);
            let mut mapped_pt = rlst_dynamic_array!(f64, [gdim, npts]);
            let mut j = rlst_dynamic_array!(f64, [gdim, tdim, npts]);
            let mut jinv = rlst_dynamic_array!(f64, [tdim, gdim, npts]);
            let mut jdet = vec![0.0; npts];

            for i in 0..mesh.entity_count(*ct) {
                map.physical_points(i, &mut mapped_pt);
                map.jacobians_inverses_dets(i, &mut j, &mut jinv, &mut jdet);
                total += jdet.iter().zip(&weights).map(|(j, w)| *j * *w).sum::<f64>();
            }
        }
        assert_relative_eq!(total, volume, epsilon = epsilon);
    }
}
