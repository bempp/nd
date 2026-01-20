//! Geometry map
use crate::{traits::GeometryMap as GeometryMapTrait, types::Scalar};
use itertools::izip;
use ndelement::{reference_cell, traits::MappedFiniteElement, types::ReferenceCellType};
use rlst::{Array, DynArray, RlstScalar, ValueArrayImpl};

/// Single element geometry
pub struct GeometryMap<'a, T, B2D, C2D> {
    geometry_points: &'a Array<B2D, 2>,
    entities: &'a Array<C2D, 2>,
    tdim: usize,
    gdim: usize,
    table: DynArray<T, 4>,
}

/// Dot product of two vectors
fn dot<T: RlstScalar>(a: &[T], b: &[T]) -> T {
    debug_assert!(a.len() == b.len());
    izip!(a, b).map(|(&i, &j)| i * j).sum::<T>()
}

/// Invert a matrix and return the determinad, or compute the Moore-Penrose left pseudoinverse
/// and return sqrt(det(A^T*A) if the matrix is not square
/// Note: the shape argument assumes column-major ordering
fn inverse_and_det<T: RlstScalar>(mat: &[T], shape: [usize; 2], result: &mut [T]) -> T {
    let gdim = shape[0];
    let tdim = shape[1];

    debug_assert!(mat.len() == tdim * gdim);
    debug_assert!(result.len() == tdim * gdim);
    debug_assert!(tdim <= gdim);

    match tdim {
        0 => T::one(),
        1 => {
            let det = dot(mat, mat);
            for (r, m) in izip!(result, mat) {
                *r = *m / det;
            }
            det
        }
        2 => {
            let col0 = &mat[0..gdim];
            let col1 = &mat[gdim..2 * gdim];
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
                result[2 * i] = ata_inv[0] * mat[i] + ata_inv[2] * mat[gdim + i];
                result[2 * i + 1] = ata_inv[1] * mat[i] + ata_inv[3] * mat[gdim + i];
            }
            ata_det
        }
        3 => {
            let col0 = &mat[0..gdim];
            let col1 = &mat[gdim..2 * gdim];
            let col2 = &mat[2 * gdim..3 * gdim];
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
                result[3 * i] = ata_inv[0] * mat[i]
                    + ata_inv[3] * mat[gdim + i]
                    + ata_inv[6] * mat[2 * gdim + i];
                result[3 * i + 1] = ata_inv[1] * mat[i]
                    + ata_inv[4] * mat[gdim + i]
                    + ata_inv[7] * mat[2 * gdim + i];
                result[3 * i + 2] = ata_inv[2] * mat[i]
                    + ata_inv[5] * mat[gdim + i]
                    + ata_inv[8] * mat[2 * gdim + i];
            }
            ata_det
        }
        _ => {
            panic!("Unsupported dimension");
        }
    }
    .sqrt()
}

fn cross<T: RlstScalar>(mat: &[T], result: &mut [T]) {
    match mat.len() {
        0 => {}
        2 => {
            debug_assert!(result.len() == 2);
            unsafe {
                *result.get_unchecked_mut(0) = *mat.get_unchecked(1);
                *result.get_unchecked_mut(1) = -*mat.get_unchecked(0);
            }
        }
        6 => {
            debug_assert!(result.len() == 3);
            unsafe {
                *result.get_unchecked_mut(0) = *mat.get_unchecked(1) * *mat.get_unchecked(5)
                    - *mat.get_unchecked(2) * *mat.get_unchecked(4);
                *result.get_unchecked_mut(1) = *mat.get_unchecked(2) * *mat.get_unchecked(3)
                    - *mat.get_unchecked(0) * *mat.get_unchecked(5);
                *result.get_unchecked_mut(2) = *mat.get_unchecked(0) * *mat.get_unchecked(4)
                    - *mat.get_unchecked(1) * *mat.get_unchecked(3);
            }
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
    fn physical_points(&self, entity_index: usize, points: &mut [T]) {
        let npts = self.table.shape()[1];
        debug_assert!(points.len() == self.gdim * npts);

        points.fill(T::default());
        for i in 0..self.entities.shape()[0] {
            let v = unsafe { self.entities.get_value_unchecked([i, entity_index]) };
            for point_index in 0..npts {
                let t = unsafe { *self.table.get_unchecked([0, point_index, i, 0]) };
                for gd in 0..self.gdim {
                    unsafe {
                        *points.get_unchecked_mut(gd + self.gdim * point_index) +=
                            self.geometry_points.get_value_unchecked([gd, v]) * t
                    };
                }
            }
        }
    }
    fn jacobians(&self, entity_index: usize, jacobians: &mut [T]) {
        let npts = self.table.shape()[1];
        debug_assert!(jacobians.len() == self.gdim * self.tdim * npts);

        jacobians.fill(T::zero());
        for i in 0..self.entities.shape()[0] {
            let v = unsafe { self.entities.get_value_unchecked([i, entity_index]) };
            for point_index in 0..npts {
                for td in 0..self.tdim {
                    let t = unsafe { self.table.get_value_unchecked([1 + td, point_index, i, 0]) };
                    for gd in 0..self.gdim {
                        unsafe {
                            *jacobians.get_unchecked_mut(
                                gd + self.gdim * td + self.gdim * self.tdim * point_index,
                            ) += self.geometry_points.get_value_unchecked([gd, v]) * t
                        };
                    }
                }
            }
        }
    }

    fn jacobians_inverses_dets(
        &self,
        entity_index: usize,
        jacobians: &mut [Self::T],
        inverse_jacobians: &mut [Self::T],
        jdets: &mut [Self::T],
    ) {
        let npts = self.table.shape()[1];
        debug_assert!(jacobians.len() == self.gdim * self.tdim * npts);
        debug_assert!(inverse_jacobians.len() == self.gdim * self.tdim * npts);
        debug_assert!(jdets.len() == npts);

        self.jacobians(entity_index, jacobians);

        for point_index in 0..npts {
            let j = &jacobians
                [self.gdim * self.tdim * point_index..self.gdim * self.tdim * (point_index + 1)];

            *jdets.get_mut(point_index).unwrap() = inverse_and_det(
                j,
                [self.gdim, self.tdim],
                &mut inverse_jacobians[self.gdim * self.tdim * point_index
                    ..self.gdim * self.tdim * (point_index + 1)],
            );
        }
    }

    fn jacobians_inverses_dets_normals(
        &self,
        entity_index: usize,
        jacobians: &mut [Self::T],
        inverse_jacobians: &mut [Self::T],
        jdets: &mut [Self::T],
        normals: &mut [Self::T],
    ) {
        if self.tdim + 1 != self.gdim {
            panic!("Can only compute normal for entities where tdim + 1 == gdim");
        }
        let npts = self.table.shape()[1];
        debug_assert!(jacobians.len() == self.gdim * self.tdim * npts);
        debug_assert!(inverse_jacobians.len() == self.gdim * self.tdim * npts);
        debug_assert!(jdets.len() == npts);
        debug_assert!(normals.len() == self.gdim * npts);

        self.jacobians(entity_index, jacobians);

        for point_index in 0..npts {
            let j = &jacobians
                [self.gdim * self.tdim * point_index..self.gdim * self.tdim * (point_index + 1)];
            let jd = jdets.get_mut(point_index).unwrap();

            *jd = inverse_and_det(
                j,
                [self.gdim, self.tdim],
                &mut inverse_jacobians[self.gdim * self.tdim * point_index
                    ..self.gdim * self.tdim * (point_index + 1)],
            );

            let n = &mut normals[self.gdim * point_index..self.gdim * (point_index + 1)];
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
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use rlst::{Lu, SliceArray, rlst_dynamic_array};

    fn is_singular(mat: &[f64], gdim: usize, tdim: usize) -> bool {
        if tdim == 0 || gdim == 0 {
            return false;
        }
        let a = SliceArray::from_shape(mat, [gdim, tdim]);
        let mut at = rlst_dynamic_array!(f64, [tdim, gdim]);
        at.fill_from(&a.r().transpose());

        let ata = rlst::dot!(at.r(), a.r());
        if let Ok(lu) = ata.lu() {
            lu.det().abs() < 0.1
        } else {
            true
        }
    }

    fn non_singular_matrix(gdim: usize, tdim: usize) -> Vec<f64> {
        let mut mat = vec![0.0; gdim * tdim];
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        while is_singular(&mat, gdim, tdim) {
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
                    let mut inv = vec![0.0; $tdim * $gdim];

                    inverse_and_det(&mat, [$gdim, $tdim], &mut inv);

                    #[allow(clippy::reversed_empty_ranges)]
                    for i in 0..$tdim {
                        #[allow(clippy::reversed_empty_ranges)]
                        for j in 0..$tdim {
                            let entry = (0..$gdim)
                                .map(|k| inv[$tdim * k + i] * mat[$gdim * j + k])
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
                        let a = SliceArray::from_shape(&mat, [$gdim, $tdim]);
                        let mut at = rlst_dynamic_array!(f64, [$tdim, $gdim]);
                        at.fill_from(&a.r().transpose());

                        let ata = rlst::dot!(at.r(), a.r());
                        if let Ok(lu) = ata.lu() {
                            lu.det().sqrt()
                        } else {
                            0.0
                        }
                    };

                    let mut inv = vec![0.0; $tdim * $gdim];

                    assert_relative_eq!(
                        inverse_and_det(&mat, [$gdim, $tdim], &mut inv),
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
}
