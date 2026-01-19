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

fn norm<T: RlstScalar>(vector: &[T]) -> T {
    vector.iter().map(|&i| i * i).sum::<T>().sqrt()
}

/// Invert a matrix
/// Note: the shape argument assumes column-major ordering
fn inverse<T: RlstScalar>(mat: &[T], shape: [usize; 2], result: &mut [T]) {
    let gdim = shape[0];
    let tdim = shape[1];

    debug_assert!(mat.len() == tdim * gdim);
    debug_assert!(result.len() == tdim * gdim);
    debug_assert!(tdim <= gdim);

    match tdim {
        0 => {}
        1 => {
            let det = mat.iter().map(|i| i.powi(2)).sum::<T>();
            for (r, m) in izip!(result, mat) {
                *r = *m / det;
            }
        }
        2 => {
            let ata = [
                (0..gdim).map(|i| mat[i].powi(2)).sum::<T>(),
                (0..gdim).map(|i| mat[i] * mat[gdim + i]).sum::<T>(),
                (0..gdim).map(|i| mat[i] * mat[gdim + i]).sum::<T>(),
                (0..gdim).map(|i| mat[gdim + i].powi(2)).sum::<T>(),
            ];
            let ata_det = ata[0] * ata[3] - ata[1] * ata[2];
            let ata_inv = [
                ata[3] / ata_det,
                -ata[2] / ata_det,
                -ata[1] / ata_det,
                ata[0] / ata_det,
            ];

            dbg!(&ata);
            dbg!(&ata_det);
            dbg!(&ata_inv);

            for i in 0..gdim {
                result[2 * i] = ata_inv[0] * mat[i] + ata_inv[2] * mat[gdim + i];
                result[2 * i + 1] = ata_inv[1] * mat[i] + ata_inv[3] * mat[gdim + i];
            }
        }
        3 => match gdim {
            3 => {
                let det = mat[0] * (mat[4] * mat[8] - mat[5] * mat[7])
                    + mat[1] * (mat[5] * mat[6] - mat[3] * mat[8])
                    + mat[1] * (mat[3] * mat[7] - mat[4] * mat[6]);
                result[0] = (mat[4] * mat[8] - mat[5] * mat[7]) / det;
                result[1] = (mat[2] * mat[7] - mat[1] * mat[8]) / det;
                result[2] = (mat[1] * mat[5] - mat[2] * mat[4]) / det;
                result[3] = (mat[5] * mat[6] - mat[3] * mat[8]) / det;
                result[4] = (mat[0] * mat[8] - mat[2] * mat[6]) / det;
                result[5] = (mat[2] * mat[3] - mat[0] * mat[5]) / det;
                result[6] = (mat[3] * mat[7] - mat[4] * mat[5]) / det;
                result[7] = (mat[1] * mat[6] - mat[0] * mat[7]) / det;
                result[8] = (mat[0] * mat[4] - mat[1] * mat[3]) / det;
            }
            _ => {
                panic!("Unsupported dimension");
            }
        },
        _ => {
            panic!("Unsupported dimension");
        }
    }
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

    fn jacobians_dets_normals(
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
        debug_assert!(jdets.len() == npts);
        debug_assert!(normals.len() == self.gdim * npts);

        self.jacobians(entity_index, jacobians);

        for point_index in 0..npts {
            let j = &jacobians
                [self.gdim * self.tdim * point_index..self.gdim * self.tdim * (point_index + 1)];

            inverse(
                j,
                [self.gdim, self.tdim],
                &mut inverse_jacobians[self.gdim * self.tdim * point_index
                    ..self.gdim * self.tdim * (point_index + 1)],
            );

            let n = &mut normals[self.gdim * point_index..self.gdim * (point_index + 1)];
            let jd = unsafe { jdets.get_unchecked_mut(point_index) };
            cross(j, n);
            *jd = norm(n);
            for n_i in n.iter_mut() {
                *n_i /= *jd;
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::*;
    use rand::Rng;
    use rlst::{rlst_dynamic_array, SliceArray, DynArray, ValueArrayImpl, RandomAccessByRef, Shape, Lu};

    fn det(mat: &DynArray<f64, 2>) -> f64 {
        assert_eq!(mat.shape()[0], mat.shape()[1]);
        if mat.shape()[0] == 1 {
            *mat.get([0, 0]).unwrap()
        } else {
            let mut sub = rlst_dynamic_array!(f64, [mat.shape()[0] - 1, mat.shape()[1] - 1]);
            let mut d = 0.0;
            for i in 0..mat.shape()[0] {
                for j in 0..mat.shape()[0] {
                    if j != i {
                        for k in 0..mat.shape()[0] {
                            if k != i {
                                *sub.get_mut([if k > i {k - 1} else {k}, if j > i {j - 1} else {j}]).unwrap() = *mat.get([j, k]).unwrap();
                            }
                        }
                    }
                }
                d += *mat.get([0, i]).unwrap() * det(&sub);
            }
            d
        }
    }

    fn is_singular(mat: &[f64], gdim: usize, tdim: usize) -> bool {
        let a = SliceArray::from_shape(mat, [gdim, tdim]);
        let mut at = rlst_dynamic_array!(f64, [tdim, gdim]);
        at.fill_from(&a.r().transpose());
        
        let ata = rlst::dot!(at.r(), a.r());
        det(&ata).abs() < 0.1
    }

    fn non_singular_matrix(gdim: usize, tdim: usize) -> Vec<f64> {
        let mut mat = vec![0.0; gdim * tdim];
        let mut rng = rand::rng();
        while is_singular(&mat, gdim, tdim) {
            for m in mat.iter_mut() {
                *m = rng.random();
            }
        }
        mat
    }

    fn test_inverse(gdim: usize, tdim: usize) {
        let mat = non_singular_matrix(gdim, tdim);
        let mut inv = vec![0.0; tdim * gdim];

        inverse(&mat, [gdim, tdim], &mut inv);

        dbg!(&mat);
        dbg!(&inv);

        for i in 0..tdim {
            for j in 0..tdim {
                for k in 0..gdim {
                    print!("inv[{}] * mat[{}] + ", tdim * k + i, gdim * j + k);
                }
                println!();
                for k in 0..gdim {
                    print!("{} * {} + ", inv[tdim * k + i], mat[gdim * j + k]);
                }
                println!();
                let entry = (0..gdim)
                    .map(|k| inv[tdim * k + i] * mat[gdim * j + k])
                    .sum::<f64>();
                assert_relative_eq!(entry, if i == j { 1.0 } else { 0.0 }, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_inverse_2_2() {
        test_inverse(2, 2);
    }

    #[test]
    fn test_inverse_3_2() {
        test_inverse(3, 2);
    }

    #[test]
    fn test_inverse_3_3() {
        test_inverse(3, 3);
    }
}
