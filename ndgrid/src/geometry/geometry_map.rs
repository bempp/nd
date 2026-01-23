//! Geometry map
use crate::{traits::GeometryMap as GeometryMapTrait, types::Scalar};
use itertools::izip;
use ndelement::{reference_cell, traits::MappedFiniteElement, types::ReferenceCellType};
use rlst::{Array, DynArray, MutableArrayImpl, RlstScalar, ValueArrayImpl};

/// Single element geometry
pub struct GeometryMap<'a, T, B2D, C2D> {
    geometry_points: &'a Array<B2D, 2>,
    entities: &'a Array<C2D, 2>,
    tdim: usize,
    gdim: usize,
    table: DynArray<T, 4>,
}

/// Dot product of two vectors
fn dot<T: RlstScalar, Array1Impl: ValueArrayImpl<T, 1>>(
    a: &Array<Array1Impl, 1>,
    b: &Array<Array1Impl, 1>,
) -> T {
    debug_assert!(a.shape()[0] == b.shape()[0]);
    izip!(a.iter_value(), b.iter_value())
        .map(|(i, j)| i * j)
        .sum::<T>()
}

/// Invert a matrix and return the determinad, or compute the Moore-Penrose left pseudoinverse
/// and return sqrt(det(A^T*A) if the matrix is not square
/// Note: the shape argument assumes column-major ordering
fn inverse_and_det<
    T: RlstScalar,
    Array2Impl: ValueArrayImpl<T, 2>,
    Array2ImplMut: MutableArrayImpl<T, 2>,
>(
    mat: &Array<Array2Impl, 2>,
    result: &mut Array<Array2ImplMut, 2>,
) -> T {
    let gdim = mat.shape()[0];
    let tdim = mat.shape()[1];

    debug_assert!(result.shape()[0] == tdim);
    debug_assert!(result.shape()[1] == gdim);
    debug_assert!(tdim <= gdim);

    match tdim {
        0 => T::one(),
        1 => {
            let det = dot(&mat.r().slice(1, 0), &mat.r().slice(1, 0));
            for (r, m) in izip!(result.iter_mut(), mat.iter_value()) {
                *r = m / det;
            }
            det
        }
        2 => {
            let col0 = &mat.r().slice(1, 0);
            let col1 = &mat.r().slice(1, 1);
            let ata = [
                dot(col0, col0),
                dot(col0, col1),
                dot(col1, col0),
                dot(col1, col1),
            ];
            let ata_det = ata[0] * ata[3] - ata[1] * ata[2];
            let ata_inv = [
                ata[3] / ata_det,
                -ata[2] / ata_det,
                -ata[1] / ata_det,
                ata[0] / ata_det,
            ];

            for i in 0..gdim {
                *result.get_mut([0, i]).unwrap() = ata_inv[0] * mat.get_value([i, 0]).unwrap()
                    + ata_inv[2] * mat.get_value([i, 1]).unwrap();
                *result.get_mut([1, i]).unwrap() = ata_inv[1] * mat.get_value([i, 0]).unwrap()
                    + ata_inv[3] * mat.get_value([i, 1]).unwrap();
            }
            ata_det
        }
        3 => {
            let col0 = &mat.r().slice(1, 0);
            let col1 = &mat.r().slice(1, 1);
            let col2 = &mat.r().slice(1, 2);
            let ata = [
                dot(col0, col0),
                dot(col0, col1),
                dot(col0, col2),
                dot(col1, col0),
                dot(col1, col1),
                dot(col1, col2),
                dot(col2, col0),
                dot(col2, col1),
                dot(col2, col2),
            ];
            let ata_det = ata[0] * (ata[4] * ata[8] - ata[5] * ata[7])
                + ata[1] * (ata[5] * ata[6] - ata[3] * ata[8])
                + ata[2] * (ata[3] * ata[7] - ata[4] * ata[6]);
            let ata_inv = [
                (ata[4] * ata[8] - ata[5] * ata[7]) / ata_det,
                (ata[2] * ata[7] - ata[1] * ata[8]) / ata_det,
                (ata[1] * ata[5] - ata[2] * ata[4]) / ata_det,
                (ata[5] * ata[6] - ata[3] * ata[8]) / ata_det,
                (ata[0] * ata[8] - ata[2] * ata[6]) / ata_det,
                (ata[2] * ata[3] - ata[0] * ata[5]) / ata_det,
                (ata[3] * ata[7] - ata[4] * ata[6]) / ata_det,
                (ata[1] * ata[6] - ata[0] * ata[7]) / ata_det,
                (ata[0] * ata[4] - ata[1] * ata[3]) / ata_det,
            ];

            for i in 0..gdim {
                *result.get_mut([0, i]).unwrap() = ata_inv[0] * mat.get_value([i, 0]).unwrap()
                    + ata_inv[3] * mat.get_value([i, 1]).unwrap()
                    + ata_inv[6] * mat.get_value([i, 2]).unwrap();
                *result.get_mut([1, i]).unwrap() = ata_inv[1] * mat.get_value([i, 0]).unwrap()
                    + ata_inv[4] * mat.get_value([i, 1]).unwrap()
                    + ata_inv[7] * mat.get_value([i, 2]).unwrap();
                *result.get_mut([2, i]).unwrap() = ata_inv[2] * mat.get_value([i, 0]).unwrap()
                    + ata_inv[5] * mat.get_value([i, 1]).unwrap()
                    + ata_inv[8] * mat.get_value([i, 2]).unwrap();
            }
            ata_det
        }
        _ => {
            panic!("Unsupported dimension");
        }
    }
    .sqrt()
}

fn cross<T: RlstScalar, Array2Impl: ValueArrayImpl<T, 2>, Array1ImplMut: MutableArrayImpl<T, 1>>(
    mat: &Array<Array2Impl, 2>,
    result: &mut Array<Array1ImplMut, 1>,
) {
    debug_assert!(mat.shape()[0] == mat.shape()[1] + 1);
    match mat.shape()[1] {
        0 => {}
        1 => {
            debug_assert!(result.shape()[0] == 2);
            *result.get_mut([0]).unwrap() = mat.get_value([1, 0]).unwrap();
            *result.get_mut([1]).unwrap() = -mat.get_value([0, 0]).unwrap();
        }
        2 => {
            debug_assert!(result.shape()[0] == 3);
            *result.get_mut([0]).unwrap() = mat.get_value([1, 0]).unwrap()
                * mat.get_value([2, 1]).unwrap()
                - mat.get_value([2, 0]).unwrap() * mat.get_value([1, 1]).unwrap();
            *result.get_mut([1]).unwrap() = mat.get_value([2, 0]).unwrap()
                * mat.get_value([0, 1]).unwrap()
                - mat.get_value([0, 0]).unwrap() * mat.get_value([2, 1]).unwrap();
            *result.get_mut([2]).unwrap() = mat.get_value([0, 0]).unwrap()
                * mat.get_value([1, 1]).unwrap()
                - mat.get_value([1, 0]).unwrap() * mat.get_value([0, 1]).unwrap();
        }
        _ => {
            unimplemented!();
        }
    }
}

impl<'a, T: Scalar, B2D: ValueArrayImpl<T, 2>, C2D: ValueArrayImpl<usize, 2>>
    GeometryMap<'a, T, B2D, C2D>
{
    /// Create new
    pub fn new<A2D: ValueArrayImpl<T, 2>>(
        element: &impl MappedFiniteElement<CellType = ReferenceCellType, T = T>,
        points: &Array<A2D, 2>,
        geometry_points: &'a Array<B2D, 2>,
        entities: &'a Array<C2D, 2>,
    ) -> Self {
        let tdim = reference_cell::dim(element.cell_type());
        debug_assert!(points.shape()[0] == tdim);
        let gdim = geometry_points.shape()[0];
        let npoints = points.shape()[1];

        let mut table = DynArray::<T, 4>::from_shape(element.tabulate_array_shape(1, npoints));
        element.tabulate(points, 1, &mut table);

        Self {
            geometry_points,
            entities,
            tdim,
            gdim,
            table,
        }
    }
}

impl<T: Scalar, B2D: ValueArrayImpl<T, 2>, C2D: ValueArrayImpl<usize, 2>> GeometryMapTrait
    for GeometryMap<'_, T, B2D, C2D>
{
    type T = T;

    fn entity_topology_dimension(&self) -> usize {
        self.tdim
    }
    fn geometry_dimension(&self) -> usize {
        self.gdim
    }
    fn point_count(&self) -> usize {
        self.table.shape()[1]
    }
    fn physical_points<Array2Impl: MutableArrayImpl<T, 2>>(
        &self,
        entity_index: usize,
        points: &mut Array<Array2Impl, 2>,
    ) {
        let npts = self.table.shape()[1];
        debug_assert!(points.shape()[0] == self.gdim);
        debug_assert!(points.shape()[1] == npts);

        points.fill_with_value(T::default());
        for i in 0..self.entities.shape()[0] {
            let v = unsafe { self.entities.get_value_unchecked([i, entity_index]) };
            for point_index in 0..npts {
                let t = unsafe { *self.table.get_unchecked([0, point_index, i, 0]) };
                for gd in 0..self.gdim {
                    unsafe {
                        *points.get_unchecked_mut([gd, point_index]) +=
                            self.geometry_points.get_value_unchecked([gd, v]) * t
                    };
                }
            }
        }
    }
    fn jacobians<Array3MutImpl: MutableArrayImpl<T, 3>>(
        &self,
        entity_index: usize,
        jacobians: &mut Array<Array3MutImpl, 3>,
    ) {
        let npts = self.table.shape()[1];
        debug_assert!(jacobians.shape()[0] == self.gdim);
        debug_assert!(jacobians.shape()[1] == self.tdim);
        debug_assert!(jacobians.shape()[2] == npts);

        jacobians.fill_with_value(T::zero());
        for i in 0..self.entities.shape()[0] {
            let v = unsafe { self.entities.get_value_unchecked([i, entity_index]) };
            for point_index in 0..npts {
                for td in 0..self.tdim {
                    let t = unsafe { self.table.get_value_unchecked([1 + td, point_index, i, 0]) };
                    for gd in 0..self.gdim {
                        unsafe {
                            *jacobians.get_unchecked_mut([gd, td, point_index]) +=
                                self.geometry_points.get_value_unchecked([gd, v]) * t
                        };
                    }
                }
            }
        }
    }

    fn jacobians_inverses_dets<Array3MutImpl: MutableArrayImpl<T, 3>>(
        &self,
        entity_index: usize,
        jacobians: &mut Array<Array3MutImpl, 3>,
        inverse_jacobians: &mut Array<Array3MutImpl, 3>,
        jdets: &mut [T],
    ) {
        let npts = self.table.shape()[1];
        debug_assert!(jacobians.shape()[0] == self.gdim);
        debug_assert!(jacobians.shape()[1] == self.tdim);
        debug_assert!(jacobians.shape()[2] == npts);
        debug_assert!(inverse_jacobians.shape()[0] == self.tdim);
        debug_assert!(inverse_jacobians.shape()[1] == self.gdim);
        debug_assert!(inverse_jacobians.shape()[2] == npts);
        debug_assert!(jdets.len() == npts);

        self.jacobians(entity_index, jacobians);

        for point_index in 0..npts {
            let j = &jacobians.r().slice(2, point_index);

            *jdets.get_mut(point_index).unwrap() =
                inverse_and_det(j, &mut inverse_jacobians.r_mut().slice(2, point_index));
        }
    }

    fn jacobians_inverses_dets_normals<
        Array2Impl: MutableArrayImpl<T, 2>,
        Array3MutImpl: MutableArrayImpl<T, 3>,
    >(
        &self,
        entity_index: usize,
        jacobians: &mut Array<Array3MutImpl, 3>,
        inverse_jacobians: &mut Array<Array3MutImpl, 3>,
        jdets: &mut [T],
        normals: &mut Array<Array2Impl, 2>,
    ) {
        if self.tdim + 1 != self.gdim {
            panic!("Can only compute normal for entities where tdim + 1 == gdim");
        }
        let npts = self.table.shape()[1];
        debug_assert!(jacobians.shape()[0] == self.gdim);
        debug_assert!(jacobians.shape()[1] == self.tdim);
        debug_assert!(jacobians.shape()[2] == npts);
        debug_assert!(inverse_jacobians.shape()[0] == self.tdim);
        debug_assert!(inverse_jacobians.shape()[1] == self.gdim);
        debug_assert!(inverse_jacobians.shape()[2] == npts);
        debug_assert!(jdets.len() == npts);
        debug_assert!(normals.shape()[0] == self.gdim);
        debug_assert!(normals.shape()[1] == npts);

        self.jacobians(entity_index, jacobians);

        for point_index in 0..npts {
            let j = &jacobians.r().slice(2, point_index);
            let jd = jdets.get_mut(point_index).unwrap();

            *jd = inverse_and_det(j, &mut inverse_jacobians.r_mut().slice(2, point_index));

            let n = &mut normals.r_mut().slice(1, point_index);
            cross(j, n);
            for n_i in n.iter_mut() {
                *n_i /= *jd;
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_relative_eq;
    use ndelement::{
        ciarlet::lagrange,
        types::{Continuity, ReferenceCellType},
    };
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use rlst::{DynArray, Lu, rlst_dynamic_array};

    fn is_singular(mat: &DynArray<f64, 2>) -> bool {
        if mat.shape()[0] == 0 || mat.shape()[1] == 0 {
            return false;
        }
        let mut mat_t = rlst_dynamic_array!(f64, [mat.shape()[1], mat.shape()[0]]);
        mat_t.fill_from(&mat.r().transpose());

        if let Ok(lu) = rlst::dot!(mat_t.r(), mat.r()).lu() {
            lu.det().abs() < 0.1
        } else {
            true
        }
    }

    fn non_singular_matrix(gdim: usize, tdim: usize) -> DynArray<f64, 2> {
        let mut mat = rlst_dynamic_array!(f64, [gdim, tdim]);
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        while is_singular(&mat) {
            for m in mat.iter_mut() {
                *m = rng.random();
            }
        }
        mat
    }

    macro_rules! tests {
        ($gdim:expr, $tdim:expr) => {
            paste::item! {
                #[test]
                fn [< test_inverse_ $gdim _ $tdim >]() {
                    let mat = non_singular_matrix($gdim, $tdim);
                    let mut inv = rlst_dynamic_array!(f64, [$tdim, $gdim]);

                    inverse_and_det(&mat, &mut inv);

                    #[allow(clippy::reversed_empty_ranges)]
                    for i in 0..$tdim {
                        #[allow(clippy::reversed_empty_ranges)]
                        for j in 0..$tdim {
                            let entry = (0..$gdim)
                                .map(|k| inv[[i, k]] * mat[[k, j]])
                                .sum::<f64>();
                            assert_relative_eq!(
                                entry,
                                if i == j { 1.0 } else { 0.0 },
                                epsilon = 1e-10
                            );
                        }
                    }

                }

                #[test]
                fn [< test_det_ $gdim _ $tdim >]() {
                    let mat = non_singular_matrix($gdim, $tdim);

                    let rlst_det = if $tdim == 0 || $gdim == 0 {
                        1.0
                    } else {
                        let mut mat_t = rlst_dynamic_array!(f64, [$tdim, $gdim]);
                        mat_t.fill_from(&mat.r().transpose());

                        if let Ok(lu) = rlst::dot!(mat_t.r(), mat.r()).lu() {
                            lu.det().sqrt()
                        } else {
                            0.0
                        }
                    };

                    let mut inv = rlst_dynamic_array!(f64, [$tdim, $gdim]);

                    assert_relative_eq!(
                        inverse_and_det(&mat, &mut inv),
                        rlst_det,
                        epsilon = 1e-10
                    );
                }
            }
        };
    }

    tests!(0, 0);
    tests!(1, 0);
    tests!(1, 1);
    tests!(2, 0);
    tests!(2, 1);
    tests!(2, 2);
    tests!(3, 0);
    tests!(3, 1);
    tests!(3, 2);
    tests!(3, 3);
    tests!(4, 0);
    tests!(4, 1);
    tests!(4, 2);
    tests!(4, 3);
    tests!(5, 0);
    tests!(5, 1);
    tests!(5, 2);
    tests!(5, 3);

    #[test]
    fn test_geometry_map_3_2() {
        let e = lagrange::create::<f64, f64>(ReferenceCellType::Triangle, 1, Continuity::Standard);
        let mut rng = ChaCha8Rng::seed_from_u64(13);

        let npts = 4;
        let mut points = rlst_dynamic_array!(f64, [2, npts]);

        for p in 0..npts {
            *points.get_mut([0, p]).unwrap() = rng.random();
            *points.get_mut([1, p]).unwrap() = rng.random::<f64>() * points.get_value([0, p]).unwrap();
        }

        *points.get_mut([0, 0]).unwrap() = 1.0;
        *points.get_mut([0, 0]).unwrap() = 1.0;
        let mut geo_points = rlst_dynamic_array!(f64, [3, 3]);
        *geo_points.get_mut([0, 0]).unwrap() = 1.0;
        *geo_points.get_mut([1, 0]).unwrap() = 0.0;
        *geo_points.get_mut([2, 0]).unwrap() = 0.0;
        *geo_points.get_mut([0, 1]).unwrap() = 2.0;
        *geo_points.get_mut([1, 1]).unwrap() = 0.0;
        *geo_points.get_mut([2, 1]).unwrap() = 1.0;
        *geo_points.get_mut([0, 2]).unwrap() = 0.0;
        *geo_points.get_mut([1, 2]).unwrap() = 1.0;
        *geo_points.get_mut([2, 2]).unwrap() = 0.0;

        let mut entities = rlst_dynamic_array!(usize, [2, 1]);
        *entities.get_mut([0, 0]).unwrap() = 2;
        *entities.get_mut([1, 0]).unwrap() = 0;

        let gmap = GeometryMap::new(&e, &points, &geo_points, &entities);

        let mut jacobians = rlst_dynamic_array!(f64, [3, 2, npts]);
        let mut jinv = rlst_dynamic_array!(f64, [2, 3, npts]);
        let mut jdets = vec![0.0; npts];

        gmap.jacobians_inverses_dets(0, &mut jacobians, &mut jinv, &mut jdets);

        for p in 0..npts {
            for i in 0..2 {
                for j in 0..2 {
                    assert_relative_eq!(
                        (0..3)
                            .map(|k| jinv[[i, k, p]] * jacobians[[k, j, p]])
                            .sum::<f64>(),
                        if i == j { 1.0 } else { 0.0 },
                        epsilon = 1e-10
                    );
                }
            }
        }
    }
}
