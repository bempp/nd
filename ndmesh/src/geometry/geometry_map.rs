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
            panic!("Cross product should only be used for matrices of shape 0x1, 1x2 or 2x3.");
        }
    }
}

impl<'a, T: Scalar, B2D: ValueArrayImpl<T, 2>, C2D: ValueArrayImpl<usize, 2>>
    GeometryMap<'a, T, B2D, C2D>
{
    /// Create new
    pub fn new<A2D: ValueArrayImpl<T, 2>>(
        element: &impl MappedFiniteElement<CellType = ReferenceCellType, T = T, TGeo = T>,
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
        self.table.shape()[0]
    }
    fn physical_points<Array2Impl: MutableArrayImpl<T, 2>>(
        &self,
        entity_index: usize,
        points: &mut Array<Array2Impl, 2>,
    ) {
        let npts = self.table.shape()[0];
        debug_assert!(points.shape()[0] == self.gdim);
        debug_assert!(points.shape()[1] == npts);

        points.fill_with_value(T::default());
        for i in 0..self.entities.shape()[0] {
            let v = unsafe { self.entities.get_value_unchecked([i, entity_index]) };
            for point_index in 0..npts {
                let t = unsafe { *self.table.get_unchecked([point_index, 0, 0, i]) };
                for gd in 0..self.gdim {
                    unsafe {
                        *points.get_unchecked_mut([gd, point_index]) +=
                            self.geometry_points.get_value_unchecked([gd, v]) * t;
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
        let npts = self.table.shape()[0];
        debug_assert!(jacobians.shape()[0] == self.gdim);
        debug_assert!(jacobians.shape()[1] == self.tdim);
        debug_assert!(jacobians.shape()[2] == npts);

        jacobians.fill_with_value(T::zero());
        for i in 0..self.entities.shape()[0] {
            let v = unsafe { self.entities.get_value_unchecked([i, entity_index]) };
            for point_index in 0..npts {
                for td in 0..self.tdim {
                    let t = unsafe { self.table.get_value_unchecked([point_index, 0, 1 + td, i]) };
                    for gd in 0..self.gdim {
                        unsafe {
                            *jacobians.get_unchecked_mut([gd, td, point_index]) +=
                                self.geometry_points.get_value_unchecked([gd, v]) * t;
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
        let npts = self.table.shape()[0];
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
        assert!(
            (self.tdim + 1 == self.gdim),
            "Can only compute normal for entities where tdim + 1 == gdim"
        );
        let npts = self.table.shape()[0];
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
        traits::FiniteElement,
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
        let e = lagrange::create::<f64, f64>(
            ReferenceCellType::Triangle,
            1,
            Continuity::Standard,
            lagrange::LagrangeVariant::Equispaced,
        );
        let mut rng = ChaCha8Rng::seed_from_u64(13);

        let npts = 4;
        let mut points = rlst_dynamic_array!(f64, [2, npts]);

        for p in 0..npts {
            *points.get_mut([0, p]).unwrap() = rng.random();
            *points.get_mut([1, p]).unwrap() =
                rng.random::<f64>() * points.get_value([0, p]).unwrap();
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

    /// Evaluate a known test map $F$ from a reference cell into $\mathbb{R}^{gdim}$, and its
    /// Jacobian, at the reference point $\xi$.
    ///
    /// The geometry map tests use this as the exact answer. They place each geometry point of
    /// a cell at $F$ of the corresponding Lagrange interpolation point, so the cell's geometry
    /// is the Lagrange interpolant of $F$. $F$ is chosen to lie in the span of the degree
    /// `degree` Lagrange element on the cell, so the interpolant equals $F$ exactly. The
    /// physical points and Jacobians computed by [`GeometryMap`] can then be compared with
    /// $F$ and $\partial F / \partial \xi$ to rounding error.
    ///
    /// Component $g$ of the map, for $g = 0, \dots, gdim - 1$, is
    /// $$
    /// F_g(\xi) = \frac{g + 1}{2} + \sum_{t=0}^{tdim-1} A_{gt} \xi_t
    ///     + q_g \xi_0 \xi_1 + s_g \xi_{tdim-1}^2,
    /// $$
    /// with
    /// - $A_{gt} = 0.1 (g + 2t + 1)$, plus $1.5 + 0.5 g$ on the diagonal $g = t$. The
    ///   diagonal term makes the leading $tdim \times tdim$ block of $A$ diagonally
    ///   dominant, so the affine part has full rank. All entries are distinct, so a Jacobian
    ///   that is transposed or has its rows or columns permuted does not match.
    /// - $q_g = 0.2 (g + 1)$ if `tdim >= 2` and either `tensor_product` is set or
    ///   `degree >= 2`, and $q_g = 0$ otherwise. The bilinear term $\xi_0 \xi_1$ is in the
    ///   degree 1 space on quadrilaterals and hexahedra, but not on triangles or tetrahedra.
    /// - $s_g = 0.15 (g + 1)$ if `degree >= 2`, and $s_g = 0$ otherwise. This makes the
    ///   map genuinely curved for higher degree cells.
    ///
    /// So on a degree 1 simplex $F$ is affine, and in every other case the Jacobian varies
    /// with $\xi$.
    ///
    /// # Arguments
    /// - `tdim`: The topological dimension of the reference cell, which is the length of `xi`.
    /// - `gdim`: The geometric dimension, which is the number of components of $F$.
    /// - `degree`: The degree of the Lagrange element used for the geometry.
    /// - `tensor_product`: Whether the cell is a quadrilateral or hexahedron.
    /// - `xi`: The reference point at which to evaluate the map.
    ///
    /// Returns `(f, jac)`, where `f[g]` is $F_g(\xi)$ and `jac[g][t]` is
    /// $\partial F_g / \partial \xi_t (\xi)$.
    fn polynomial_map(
        tdim: usize,
        gdim: usize,
        degree: usize,
        tensor_product: bool,
        xi: &[f64],
    ) -> (Vec<f64>, Vec<Vec<f64>>) {
        let mut f = vec![0.0; gdim];
        let mut jac = vec![vec![0.0; tdim]; gdim];
        for g in 0..gdim {
            let gf = g as f64;
            // Constant term
            f[g] = 0.5 * (gf + 1.0);
            // Affine term A xi
            for t in 0..tdim {
                let a =
                    if g == t { 1.5 + 0.5 * gf } else { 0.0 } + 0.1 * (gf + 2.0 * t as f64 + 1.0);
                f[g] += a * xi[t];
                jac[g][t] += a;
            }
            // Bilinear term q xi_0 xi_1
            if tdim >= 2 && (tensor_product || degree >= 2) {
                let q = 0.2 * (gf + 1.0);
                f[g] += q * xi[0] * xi[1];
                jac[g][0] += q * xi[1];
                jac[g][1] += q * xi[0];
            }
            // Quadratic term s xi_last^2 in the last reference coordinate
            if degree >= 2 {
                let s = 0.15 * (gf + 1.0);
                let last = tdim - 1;
                f[g] += s * xi[last] * xi[last];
                jac[g][last] += 2.0 * s * xi[last];
            }
        }
        (f, jac)
    }

    /// Determinant of a square matrix of size at most 3
    fn det(m: &[Vec<f64>]) -> f64 {
        match m.len() {
            1 => m[0][0],
            2 => m[0][0] * m[1][1] - m[0][1] * m[1][0],
            3 => {
                m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                    - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                    + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
            }
            _ => panic!("Unsupported size"),
        }
    }

    /// Check a geometry map for a single cell whose geometry is given by `polynomial_map`.
    fn check_polynomial_geometry(cell: ReferenceCellType, degree: usize, gdim: usize) {
        let tdim = reference_cell::dim(cell);
        let tensor_product = matches!(
            cell,
            ReferenceCellType::Quadrilateral | ReferenceCellType::Hexahedron
        );
        let e = lagrange::create::<f64, f64>(
            cell,
            degree,
            Continuity::Standard,
            lagrange::LagrangeVariant::Equispaced,
        );

        // Put the geometry point for each DOF at the image of its interpolation point
        let mut geo_points = rlst_dynamic_array!(f64, [gdim, e.dim()]);
        for d in 0..=tdim {
            for (i, pts) in e.interpolation_points()[d].iter().enumerate() {
                for (col, dof) in e.entity_dofs(d, i).unwrap().iter().enumerate() {
                    let xi = (0..tdim).map(|t| pts[[t, col]]).collect::<Vec<_>>();
                    let (f, _) = polynomial_map(tdim, gdim, degree, tensor_product, &xi);
                    for g in 0..gdim {
                        geo_points[[g, *dof]] = f[g];
                    }
                }
            }
        }
        let mut entities = rlst_dynamic_array!(usize, [e.dim(), 1]);
        for i in 0..e.dim() {
            entities[[i, 0]] = i;
        }

        let coords: &[&[f64]] = match cell {
            ReferenceCellType::Interval => &[&[0.15], &[0.4], &[0.85]],
            ReferenceCellType::Triangle => &[&[0.2, 0.3], &[0.1, 0.6], &[0.55, 0.15]],
            ReferenceCellType::Quadrilateral => &[&[0.2, 0.7], &[0.6, 0.1], &[0.85, 0.45]],
            ReferenceCellType::Tetrahedron => {
                &[&[0.1, 0.2, 0.3], &[0.3, 0.1, 0.25], &[0.15, 0.45, 0.2]]
            }
            ReferenceCellType::Hexahedron => {
                &[&[0.2, 0.7, 0.4], &[0.6, 0.1, 0.8], &[0.35, 0.55, 0.15]]
            }
            _ => panic!("Unsupported cell type: {cell:?}"),
        };
        let npts = coords.len();
        let mut points = rlst_dynamic_array!(f64, [tdim, npts]);
        for (p, c) in coords.iter().enumerate() {
            for t in 0..tdim {
                points[[t, p]] = c[t];
            }
        }

        let gmap = GeometryMap::new(&e, &points, &geo_points, &entities);
        assert_eq!(gmap.point_count(), npts);

        let mut physical_points = rlst_dynamic_array!(f64, [gdim, npts]);
        let mut jacobians = rlst_dynamic_array!(f64, [gdim, tdim, npts]);
        let mut jinv = rlst_dynamic_array!(f64, [tdim, gdim, npts]);
        let mut jdets = vec![0.0; npts];
        let mut normals = rlst_dynamic_array!(f64, [gdim, npts]);

        gmap.physical_points(0, &mut physical_points);
        if gdim == tdim + 1 {
            gmap.jacobians_inverses_dets_normals(
                0,
                &mut jacobians,
                &mut jinv,
                &mut jdets,
                &mut normals,
            );
        } else {
            gmap.jacobians_inverses_dets(0, &mut jacobians, &mut jinv, &mut jdets);
        }

        for (p, xi) in coords.iter().enumerate() {
            let (f, jac) = polynomial_map(tdim, gdim, degree, tensor_product, xi);
            for g in 0..gdim {
                assert_relative_eq!(physical_points[[g, p]], f[g], epsilon = 1e-12);
                for t in 0..tdim {
                    assert_relative_eq!(jacobians[[g, t, p]], jac[g][t], epsilon = 1e-12);
                }
            }

            // sqrt(det(J^T J))
            let jtj = (0..tdim)
                .map(|s| {
                    (0..tdim)
                        .map(|t| (0..gdim).map(|g| jac[g][s] * jac[g][t]).sum::<f64>())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            assert_relative_eq!(jdets[p], det(&jtj).sqrt(), epsilon = 1e-12);

            // J^{-1} J is the identity
            let jinv_jac = (0..tdim)
                .map(|s| {
                    (0..tdim)
                        .map(|t| (0..gdim).map(|g| jinv[[s, g, p]] * jac[g][t]).sum::<f64>())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            for (s, row) in jinv_jac.iter().enumerate() {
                for (t, value) in row.iter().enumerate() {
                    assert_relative_eq!(*value, if s == t { 1.0 } else { 0.0 }, epsilon = 1e-10);
                }
            }

            if gdim == tdim + 1 {
                let n = match tdim {
                    1 => vec![jac[1][0], -jac[0][0]],
                    2 => vec![
                        jac[1][0] * jac[2][1] - jac[2][0] * jac[1][1],
                        jac[2][0] * jac[0][1] - jac[0][0] * jac[2][1],
                        jac[0][0] * jac[1][1] - jac[1][0] * jac[0][1],
                    ],
                    _ => panic!("Unsupported dimension"),
                };
                for g in 0..gdim {
                    assert_relative_eq!(normals[[g, p]], n[g] / jdets[p], epsilon = 1e-12);
                }
            }
        }
    }

    #[test]
    fn test_geometry_map_polynomial_interval() {
        for degree in 1..=2 {
            for gdim in 1..=3 {
                check_polynomial_geometry(ReferenceCellType::Interval, degree, gdim);
            }
        }
    }

    #[test]
    fn test_geometry_map_polynomial_triangle() {
        for degree in 1..=2 {
            for gdim in 2..=3 {
                check_polynomial_geometry(ReferenceCellType::Triangle, degree, gdim);
            }
        }
    }

    #[test]
    fn test_geometry_map_polynomial_quadrilateral() {
        for degree in 1..=2 {
            for gdim in 2..=3 {
                check_polynomial_geometry(ReferenceCellType::Quadrilateral, degree, gdim);
            }
        }
    }

    #[test]
    fn test_geometry_map_polynomial_tetrahedron() {
        for degree in 1..=2 {
            check_polynomial_geometry(ReferenceCellType::Tetrahedron, degree, 3);
        }
    }

    #[test]
    fn test_geometry_map_polynomial_hexahedron() {
        for degree in 1..=2 {
            check_polynomial_geometry(ReferenceCellType::Hexahedron, degree, 3);
        }
    }
}
