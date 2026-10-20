//! Orthonormal polynomials.
//!
//! The orthonormal polynomials in ndelement span the degree $k$ natural polynomial (or rationomial
//! for a pyramid) space on a cell. As given at <https://defelement.org/ciarlet.html#The+degree+of+a+finite+element>,
//! these natural spaces are defined for an interval, triangle, quadrilateral, tetrahedron,
//! hexahedron, triangular prism and square-based pyramid by
//!
//! - $\mathbb{P}^{\text{interval}}_k=\operatorname{span}\left\{x^{p_0}\,\middle|\,p_0\in\mathbb{N},\,p_0\leqslant k\right\},$
//! - $\mathbb{P}^{\text{triangle}}_k=\operatorname{span}\left\{x^{p_0}y^{p_1}\,\middle|\,p_0,p_1\in\mathbb{N}_0,\,p_0+p_1\leqslant k\right\},$
//! - $\mathbb{P}^{\text{quadrilateral}}_k=\operatorname{span}\left\{x^{p_0}y^{p_1}\,\middle|\,p_0,p_1\in\mathbb{N}_0,\,p_0\leqslant k,\,p_1\leqslant k\right\},$
//! - $\mathbb{P}^{\text{tetrahedron}}_k=\operatorname{span}\left\{x^{p_0}y^{p_1}z^{p_2}\,\middle|\,p_0,p_1,p_2\in\mathbb{N}_0,\,p_0+p_1+p_2\leqslant k\right\},$
//! - $\mathbb{P}^{\text{hexahedron}}_k=\operatorname{span}\left\{x^{p_0}y^{p_1}z^{p_2}\,\middle|\,p_0,p_1,p_2\in\mathbb{N}_0,\,p_0\leqslant k,\,p_1\leqslant k,\,p_2\leqslant k\right\},$
//! - $\mathbb{P}^{\text{prism}}_k=\operatorname{span}\left\{x^{p_0}y^{p_1}z^{p_2}\,\middle|\,p_0,p_1,p_2\in\mathbb{N}_0,\,p_0+p_1\leqslant k,\,p_2\leqslant k\right\},$
//! - $\mathbb{P}^{\text{pyramid}}_k=\operatorname{span}\left\{\frac{x^{p_0}y^{p_1}z^{p_2}}{(1-z)^{p_0+p_1}}\,\middle|\,p_0,p_1,p_2\in\mathbb{N}_0,\,p_0\leqslant k,\,p_1\leqslant k,\,p_2\leqslant k\right\}.$
//!
//! Note that for non-pyramid cells, these coincide with the polynomial space for the degree $k$
//! Lagrange element on the cell.

mod legendre;
pub use legendre::{shape as legendre_shape, tabulate as tabulate_legendre_polynomials};

use crate::reference_cell;
use crate::types::ReferenceCellType;

/// Return the number of polynomials in the natural polynomial set for a given cell type and degree.
#[must_use]
pub fn polynomial_count(cell_type: ReferenceCellType, degree: usize) -> usize {
    match cell_type {
        ReferenceCellType::Interval => degree + 1,
        ReferenceCellType::Triangle => (degree + 1) * (degree + 2) / 2,
        ReferenceCellType::Quadrilateral => (degree + 1).pow(2),
        ReferenceCellType::Tetrahedron => (degree + 1) * (degree + 2) * (degree + 3) / 6,
        ReferenceCellType::Hexahedron => (degree + 1).pow(3),
        _ => {
            panic!("Unsupported cell type: {cell_type:?}");
        }
    }
}

/// Return the total number of partial derivatives up to a given degree.
#[must_use]
pub fn derivative_count(cell_type: ReferenceCellType, degree: usize) -> usize {
    let mut num = 1;
    let mut denom = 1;
    for i in 0..reference_cell::dim(cell_type) {
        num *= degree + i + 1;
        denom *= i + 1;
    }
    num / denom
}

#[cfg(test)]
pub(crate) mod test {
    use super::*;
    use crate::quadrature::gauss_jacobi_rule;
    use crate::types::ReferenceCellType;
    use approx::*;
    use paste::paste;
    use rlst::{DynArray, SliceArray, rlst_dynamic_array};

    /// The index of the partial derivative with multi-index `alpha` in the derivative axis of
    /// tabulated data.
    ///
    /// `alpha[i]` is the number of derivatives taken with respect to the `i`th reference
    /// coordinate. All tests that depend on the ordering of derivatives use this function.
    pub(crate) fn derivative_index(cell: ReferenceCellType, alpha: &[usize]) -> usize {
        assert_eq!(alpha.len(), reference_cell::dim(cell));
        match alpha.len() {
            0 => 0,
            1 => alpha[0],
            2 => {
                let n = alpha[0] + alpha[1];
                n * (n + 1) / 2 + alpha[1]
            }
            3 => {
                let n = alpha[0] + alpha[1] + alpha[2];
                let m = alpha[1] + alpha[2];
                n * (n + 1) * (n + 2) / 6 + m * (m + 1) / 2 + alpha[2]
            }
            _ => panic!("Unsupported dimension"),
        }
    }

    /// All multi-indices in `tdim` variables with total order at most `nderivs`.
    pub(crate) fn multi_indices(tdim: usize, nderivs: usize) -> Vec<Vec<usize>> {
        if tdim == 0 {
            return vec![vec![]];
        }
        let mut indices = vec![];
        for first in 0..=nderivs {
            for mut rest in multi_indices(tdim - 1, nderivs - first) {
                rest.insert(0, first);
                indices.push(rest);
            }
        }
        indices
    }

    #[test]
    fn test_derivative_index_is_bijection() {
        for cell in [
            ReferenceCellType::Interval,
            ReferenceCellType::Triangle,
            ReferenceCellType::Quadrilateral,
            ReferenceCellType::Tetrahedron,
            ReferenceCellType::Hexahedron,
        ] {
            let tdim = reference_cell::dim(cell);
            for nderivs in 0..5 {
                let mut indices = multi_indices(tdim, nderivs)
                    .iter()
                    .map(|alpha| derivative_index(cell, alpha))
                    .collect::<Vec<_>>();
                indices.sort_unstable();
                assert_eq!(
                    indices,
                    (0..derivative_count(cell, nderivs)).collect::<Vec<_>>(),
                    "{cell:?}, nderivs = {nderivs}"
                );
            }
        }
    }

    #[test]
    fn test_derivative_index_values() {
        use ReferenceCellType::{Hexahedron, Interval, Quadrilateral, Tetrahedron, Triangle};
        assert_eq!(derivative_index(Interval, &[2]), 2);
        for cell in [Triangle, Quadrilateral] {
            for (alpha, index) in [
                ([0, 0], 0),
                ([1, 0], 1),
                ([0, 1], 2),
                ([2, 0], 3),
                ([1, 1], 4),
                ([0, 2], 5),
                ([3, 0], 6),
                ([0, 3], 9),
            ] {
                assert_eq!(derivative_index(cell, &alpha), index, "{cell:?} {alpha:?}");
            }
        }
        for cell in [Tetrahedron, Hexahedron] {
            for (alpha, index) in [
                ([0, 0, 0], 0),
                ([1, 0, 0], 1),
                ([0, 1, 0], 2),
                ([0, 0, 1], 3),
                ([2, 0, 0], 4),
                ([1, 1, 0], 5),
                ([1, 0, 1], 6),
                ([0, 2, 0], 7),
                ([0, 1, 1], 8),
                ([0, 0, 2], 9),
                ([3, 0, 0], 10),
                ([0, 0, 3], 19),
            ] {
                assert_eq!(derivative_index(cell, &alpha), index, "{cell:?} {alpha:?}");
            }
        }
    }

    macro_rules! test_orthogonal {
        ($cell:ident, $degree:expr) => {
            paste! {
                #[test]
                fn [<test_orthogonal_ $cell:lower _ $degree>]() {
                    let rule = gauss_jacobi_rule(
                        ReferenceCellType::[<$cell>],
                        2 * [<$degree>],
                    ).unwrap();


                    let points = SliceArray::<f64, 2>::from_shape(&rule.points, [rule.dim, rule.npoints]);

                    let mut data = DynArray::<f64, 3>::from_shape(
                        legendre_shape(ReferenceCellType::[<$cell>], &points, [<$degree>], 0,)
                    );
                    tabulate_legendre_polynomials(ReferenceCellType::[<$cell>], &points, [<$degree>], 0, &mut data);

                    for i in 0..[<$degree>] + 1 {
                        for j in 0..[<$degree>] + 1 {
                            let mut product = 0.0;
                            for k in 0..rule.npoints {
                                product += data.get([0, i, k]).unwrap()
                                    * data.get([0, j, k]).unwrap()
                                    * rule.weights[k];
                            }
                            if i == j {
                                assert_relative_eq!(product, 1.0, epsilon = 1e-12);
                            } else {
                                assert_relative_eq!(product, 0.0, epsilon = 1e-12);
                            }
                        }
                    }
                }
            }
        };
    }

    test_orthogonal!(Interval, 2);
    test_orthogonal!(Interval, 3);
    test_orthogonal!(Interval, 4);
    test_orthogonal!(Interval, 5);
    test_orthogonal!(Interval, 6);
    test_orthogonal!(Triangle, 2);
    test_orthogonal!(Triangle, 3);
    test_orthogonal!(Triangle, 4);
    test_orthogonal!(Triangle, 5);
    test_orthogonal!(Triangle, 6);
    test_orthogonal!(Quadrilateral, 2);
    test_orthogonal!(Quadrilateral, 3);
    test_orthogonal!(Quadrilateral, 4);
    test_orthogonal!(Quadrilateral, 5);
    test_orthogonal!(Quadrilateral, 6);
    test_orthogonal!(Tetrahedron, 2);
    test_orthogonal!(Tetrahedron, 3);
    test_orthogonal!(Tetrahedron, 4);
    test_orthogonal!(Tetrahedron, 5);
    test_orthogonal!(Tetrahedron, 6);
    test_orthogonal!(Hexahedron, 2);
    test_orthogonal!(Hexahedron, 3);
    test_orthogonal!(Hexahedron, 4);
    test_orthogonal!(Hexahedron, 5);
    test_orthogonal!(Hexahedron, 6);

    fn generate_points(cell: ReferenceCellType, epsilon: f64) -> DynArray<f64, 2> {
        let mut points = match cell {
            ReferenceCellType::Interval => {
                let mut points = rlst_dynamic_array!(f64, [1, 20]);
                for i in 0..10 {
                    *points.get_mut([0, 2 * i]).unwrap() = i as f64 / 10.0;
                }
                points
            }
            ReferenceCellType::Triangle => {
                let mut points = rlst_dynamic_array!(f64, [2, 165]);
                let mut index = 0;
                for i in 0..10 {
                    for j in 0..10 - i {
                        *points.get_mut([0, 3 * index]).unwrap() = i as f64 / 10.0;
                        *points.get_mut([1, 3 * index]).unwrap() = j as f64 / 10.0;
                        index += 1;
                    }
                }
                points
            }
            ReferenceCellType::Quadrilateral => {
                let mut points = rlst_dynamic_array!(f64, [2, 300]);
                for i in 0..10 {
                    for j in 0..10 {
                        let index = 10 * i + j;
                        *points.get_mut([0, 3 * index]).unwrap() = i as f64 / 10.0;
                        *points.get_mut([1, 3 * index]).unwrap() = j as f64 / 10.0;
                    }
                }
                points
            }
            ReferenceCellType::Tetrahedron => {
                let mut points = rlst_dynamic_array!(f64, [3, 140]);
                let mut index = 0;
                for i in 0..5 {
                    for j in 0..5 - i {
                        for k in 0..5 - i - j {
                            *points.get_mut([0, 4 * index]).unwrap() = i as f64 / 5.0;
                            *points.get_mut([1, 4 * index]).unwrap() = j as f64 / 5.0;
                            *points.get_mut([2, 4 * index]).unwrap() = k as f64 / 5.0;
                            index += 1;
                        }
                    }
                }
                points
            }
            ReferenceCellType::Hexahedron => {
                let mut points = rlst_dynamic_array!(f64, [3, 500]);
                for i in 0..5 {
                    for j in 0..5 {
                        for k in 0..5 {
                            let index = 25 * i + 5 * j + k;
                            *points.get_mut([0, 4 * index]).unwrap() = i as f64 / 5.0;
                            *points.get_mut([1, 4 * index]).unwrap() = j as f64 / 5.0;
                            *points.get_mut([2, 4 * index]).unwrap() = k as f64 / 5.0;
                        }
                    }
                }
                points
            }
            _ => {
                panic!("Unsupported cell type: {cell:?}");
            }
        };
        let dim = reference_cell::dim(cell);
        for index in 0..points.shape()[1] / (dim + 1) {
            for d in 0..dim {
                for i in 0..dim {
                    *points.get_mut([i, (dim + 1) * index + d + 1]).unwrap() =
                        *points.get([i, (dim + 1) * index]).unwrap()
                            + if i == d { epsilon } else { 0.0 };
                }
            }
        }
        points
    }
    macro_rules! test_derivatives {
        ($cell:ident, $degree:expr) => {
            paste! {
                #[test]
                fn [<test_legendre_derivatives_ $cell:lower _ $degree>]() {
                    let dim = reference_cell::dim(ReferenceCellType::[<$cell>]);
                    let epsilon = 1e-10;
                    let points = generate_points(ReferenceCellType::[<$cell>], epsilon);

                    let mut data = DynArray::<f64, 3>::from_shape(
                        legendre_shape(ReferenceCellType::[<$cell>], &points, [<$degree>], 1,)
                    );
                    tabulate_legendre_polynomials(ReferenceCellType::[<$cell>], &points, [<$degree>], 1, &mut data);

                    for i in 0..data.shape()[1] {
                        for k in 0..points.shape()[1] / (dim + 1) {
                            for d in 1..dim + 1 {
                                assert_relative_eq!(
                                    *data.get([d, i, (dim + 1) * k]).unwrap(),
                                    (data.get([0, i, (dim + 1) * k + d]).unwrap() - data.get([0, i, (dim + 1) * k]).unwrap())
                                        / epsilon,
                                    epsilon = 1e-3
                                );
                            }
                        }
                    }
                }
            }
        };
    }

    test_derivatives!(Interval, 1);
    test_derivatives!(Interval, 2);
    test_derivatives!(Interval, 3);
    test_derivatives!(Interval, 4);
    test_derivatives!(Interval, 5);
    test_derivatives!(Interval, 6);
    test_derivatives!(Triangle, 1);
    test_derivatives!(Triangle, 2);
    test_derivatives!(Triangle, 3);
    test_derivatives!(Triangle, 4);
    test_derivatives!(Triangle, 5);
    test_derivatives!(Triangle, 6);
    test_derivatives!(Quadrilateral, 1);
    test_derivatives!(Quadrilateral, 2);
    test_derivatives!(Quadrilateral, 3);
    test_derivatives!(Quadrilateral, 4);
    test_derivatives!(Quadrilateral, 5);
    test_derivatives!(Quadrilateral, 6);
    test_derivatives!(Tetrahedron, 1);
    test_derivatives!(Tetrahedron, 2);
    test_derivatives!(Tetrahedron, 3);
    test_derivatives!(Tetrahedron, 4);
    test_derivatives!(Tetrahedron, 5);
    test_derivatives!(Tetrahedron, 6);
    test_derivatives!(Hexahedron, 1);
    test_derivatives!(Hexahedron, 2);
    test_derivatives!(Hexahedron, 3);
    test_derivatives!(Hexahedron, 4);
    test_derivatives!(Hexahedron, 5);
    test_derivatives!(Hexahedron, 6);

    /// A few points in the interior of a reference cell, stored as columns.
    pub(crate) fn interior_points(cell: ReferenceCellType) -> DynArray<f64, 2> {
        let coords = match cell {
            ReferenceCellType::Interval => vec![vec![0.15], vec![0.4], vec![0.85]],
            ReferenceCellType::Triangle => vec![vec![0.2, 0.3], vec![0.1, 0.6], vec![0.55, 0.15]],
            ReferenceCellType::Quadrilateral => {
                vec![vec![0.2, 0.7], vec![0.6, 0.1], vec![0.85, 0.45]]
            }
            ReferenceCellType::Tetrahedron => {
                vec![
                    vec![0.1, 0.2, 0.3],
                    vec![0.3, 0.1, 0.25],
                    vec![0.15, 0.45, 0.2],
                ]
            }
            ReferenceCellType::Hexahedron => {
                vec![
                    vec![0.2, 0.7, 0.4],
                    vec![0.6, 0.1, 0.8],
                    vec![0.35, 0.55, 0.15],
                ]
            }
            _ => panic!("Unsupported cell type: {cell:?}"),
        };
        let mut points = rlst_dynamic_array!(f64, [coords[0].len(), coords.len()]);
        for (p, c) in coords.iter().enumerate() {
            for (i, x) in c.iter().enumerate() {
                points[[i, p]] = *x;
            }
        }
        points
    }

    /// Copy of `points` with `h` added to coordinate `direction` of every point.
    pub(crate) fn shift_points(
        points: &DynArray<f64, 2>,
        direction: usize,
        h: f64,
    ) -> DynArray<f64, 2> {
        let mut shifted = rlst_dynamic_array!(f64, points.shape());
        shifted.fill_from(points);
        for p in 0..points.shape()[1] {
            shifted[[direction, p]] += h;
        }
        shifted
    }

    macro_rules! test_higher_derivatives {
        ($cell:ident, $degree:expr) => {
            paste! {
                #[test]
                fn [<test_legendre_higher_derivatives_ $cell:lower _ $degree>]() {
                    let cell = ReferenceCellType::[<$cell>];
                    let tdim = reference_cell::dim(cell);
                    let nderivs = 3;
                    let h = 1e-5;
                    let points = interior_points(cell);
                    let npts = points.shape()[1];

                    let mut data = DynArray::<f64, 3>::from_shape(
                        legendre_shape(cell, &points, $degree, nderivs)
                    );
                    tabulate_legendre_polynomials(cell, &points, $degree, nderivs, &mut data);
                    assert_eq!(data.shape()[0], derivative_count(cell, nderivs));
                    assert_eq!(data.shape()[1], polynomial_count(cell, $degree));
                    assert_eq!(data.shape()[2], npts);

                    // The derivative with multi-index alpha + e_m is the derivative in direction m
                    // of the derivative with multi-index alpha.
                    for m in 0..tdim {
                        let mut plus = DynArray::<f64, 3>::from_shape(data.shape());
                        let mut minus = DynArray::<f64, 3>::from_shape(data.shape());
                        tabulate_legendre_polynomials(
                            cell, &shift_points(&points, m, h), $degree, nderivs, &mut plus
                        );
                        tabulate_legendre_polynomials(
                            cell, &shift_points(&points, m, -h), $degree, nderivs, &mut minus
                        );
                        for alpha in multi_indices(tdim, nderivs - 1) {
                            let mut beta = alpha.clone();
                            beta[m] += 1;
                            let a = derivative_index(cell, &alpha);
                            let b = derivative_index(cell, &beta);
                            for i in 0..data.shape()[1] {
                                for p in 0..npts {
                                    assert_relative_eq!(
                                        data[[b, i, p]],
                                        (plus[[a, i, p]] - minus[[a, i, p]]) / (2.0 * h),
                                        epsilon = 1e-5,
                                        max_relative = 1e-6
                                    );
                                }
                            }
                        }
                    }
                }
            }
        };
    }

    test_higher_derivatives!(Interval, 4);
    test_higher_derivatives!(Triangle, 1);
    test_higher_derivatives!(Triangle, 4);
    test_higher_derivatives!(Quadrilateral, 1);
    test_higher_derivatives!(Quadrilateral, 3);
    test_higher_derivatives!(Tetrahedron, 1);
    test_higher_derivatives!(Tetrahedron, 3);
    test_higher_derivatives!(Hexahedron, 1);
    test_higher_derivatives!(Hexahedron, 2);

    #[test]
    fn test_legendre_interval_against_known_polynomials() {
        let degree = 3;

        let mut points = rlst_dynamic_array!(f64, [1, 11]);
        for i in 0..11 {
            *points.get_mut([0, i]).unwrap() = i as f64 / 10.0;
        }

        let mut data = DynArray::<f64, 3>::from_shape(legendre_shape(
            ReferenceCellType::Interval,
            &points,
            degree,
            3,
        ));
        tabulate_legendre_polynomials(ReferenceCellType::Interval, &points, degree, 3, &mut data);

        for k in 0..points.shape()[0] {
            let x = *points.get([0, k]).unwrap();

            // 0 => 1
            assert_relative_eq!(*data.get([0, 0, k]).unwrap(), 1.0, epsilon = 1e-12);
            assert_relative_eq!(*data.get([1, 0, k]).unwrap(), 0.0, epsilon = 1e-12);
            assert_relative_eq!(*data.get([2, 0, k]).unwrap(), 0.0, epsilon = 1e-12);
            assert_relative_eq!(*data.get([3, 0, k]).unwrap(), 0.0, epsilon = 1e-12);

            // 1 => sqrt(3)*(2x - 1)
            assert_relative_eq!(
                *data.get([0, 1, k]).unwrap(),
                f64::sqrt(3.0) * (2.0 * x - 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([1, 1, k]).unwrap(),
                2.0 * f64::sqrt(3.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(*data.get([2, 1, k]).unwrap(), 0.0, epsilon = 1e-12);
            assert_relative_eq!(*data.get([3, 1, k]).unwrap(), 0.0, epsilon = 1e-12);

            // 2 => sqrt(5)*(6x^2 - 6x + 1)
            assert_relative_eq!(
                *data.get([0, 2, k]).unwrap(),
                f64::sqrt(5.0) * (6.0 * x * x - 6.0 * x + 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([1, 2, k]).unwrap(),
                f64::sqrt(5.0) * (12.0 * x - 6.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([2, 2, k]).unwrap(),
                f64::sqrt(5.0) * 12.0,
                epsilon = 1e-12
            );
            assert_relative_eq!(*data.get([3, 2, k]).unwrap(), 0.0, epsilon = 1e-12);

            // 3 => sqrt(7)*(20x^3 - 30x^2 + 12x - 1)
            assert_relative_eq!(
                *data.get([0, 3, k]).unwrap(),
                f64::sqrt(7.0) * (20.0 * x * x * x - 30.0 * x * x + 12.0 * x - 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([1, 3, k]).unwrap(),
                f64::sqrt(7.0) * (60.0 * x * x - 60.0 * x + 12.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([2, 3, k]).unwrap(),
                f64::sqrt(7.0) * (120.0 * x - 60.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([3, 3, k]).unwrap(),
                f64::sqrt(7.0) * 120.0,
                epsilon = 1e-12
            );
        }
    }

    #[test]
    fn test_legendre_quadrilateral_against_known_polynomials() {
        let degree = 2;

        let mut points = rlst_dynamic_array!(f64, [2, 121]);
        for i in 0..11 {
            for j in 0..11 {
                *points.get_mut([0, 11 * i + j]).unwrap() = i as f64 / 10.0;
                *points.get_mut([1, 11 * i + j]).unwrap() = j as f64 / 10.0;
            }
        }

        let mut data = DynArray::<f64, 3>::from_shape(legendre_shape(
            ReferenceCellType::Quadrilateral,
            &points,
            degree,
            1,
        ));
        tabulate_legendre_polynomials(
            ReferenceCellType::Quadrilateral,
            &points,
            degree,
            1,
            &mut data,
        );

        for k in 0..points.shape()[1] {
            let x = *points.get([0, k]).unwrap();
            let y = *points.get([1, k]).unwrap();

            // 0 => 1
            assert_relative_eq!(*data.get([0, 0, k]).unwrap(), 1.0, epsilon = 1e-12);
            assert_relative_eq!(*data.get([1, 0, k]).unwrap(), 0.0, epsilon = 1e-12);
            assert_relative_eq!(*data.get([2, 0, k]).unwrap(), 0.0, epsilon = 1e-12);

            // 3 => sqrt(3)*(2x - 1)
            assert_relative_eq!(
                *data.get([0, 3, k]).unwrap(),
                f64::sqrt(3.0) * (2.0 * x - 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([1, 3, k]).unwrap(),
                2.0 * f64::sqrt(3.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(*data.get([2, 3, k]).unwrap(), 0.0, epsilon = 1e-12);

            // 6 => sqrt(5)*(6x^2 - 6x + 1)
            assert_relative_eq!(
                *data.get([0, 6, k]).unwrap(),
                f64::sqrt(5.0) * (6.0 * x * x - 6.0 * x + 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([1, 6, k]).unwrap(),
                f64::sqrt(5.0) * (12.0 * x - 6.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(*data.get([2, 6, k]).unwrap(), 0.0, epsilon = 1e-12);

            // 1 => sqrt(3)*(2y - 1)
            assert_relative_eq!(
                *data.get([0, 1, k]).unwrap(),
                f64::sqrt(3.0) * (2.0 * y - 1.0),
                epsilon = 1e-12
            );

            assert_relative_eq!(*data.get([1, 1, k]).unwrap(), 0.0, epsilon = 1e-12);
            assert_relative_eq!(
                *data.get([2, 1, k]).unwrap(),
                2.0 * f64::sqrt(3.0),
                epsilon = 1e-12
            );

            // 4 => 3*(2x - 1)*(2y - 1)
            assert_relative_eq!(
                *data.get([0, 4, k]).unwrap(),
                3.0 * (2.0 * x - 1.0) * (2.0 * y - 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([1, 4, k]).unwrap(),
                6.0 * (2.0 * y - 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([2, 4, k]).unwrap(),
                6.0 * (2.0 * x - 1.0),
                epsilon = 1e-12
            );

            // 7 => sqrt(15)*(6x^2 - 6x + 1)*(2y - 1)
            assert_relative_eq!(
                *data.get([0, 7, k]).unwrap(),
                f64::sqrt(15.0) * (6.0 * x * x - 6.0 * x + 1.0) * (2.0 * y - 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([1, 7, k]).unwrap(),
                f64::sqrt(15.0) * (12.0 * x - 6.0) * (2.0 * y - 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([2, 7, k]).unwrap(),
                2.0 * f64::sqrt(15.0) * (6.0 * x * x - 6.0 * x + 1.0),
                epsilon = 1e-12
            );

            // 2 => sqrt(5)*(6y^2 - 6y + 1)
            assert_relative_eq!(
                *data.get([0, 2, k]).unwrap(),
                f64::sqrt(5.0) * (6.0 * y * y - 6.0 * y + 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(*data.get([1, 2, k]).unwrap(), 0.0, epsilon = 1e-12);
            assert_relative_eq!(
                *data.get([2, 2, k]).unwrap(),
                f64::sqrt(5.0) * (12.0 * y - 6.0),
                epsilon = 1e-12
            );

            // 5 => sqrt(15)*(2x - 1)*(6y^2 - 6y + 1)
            assert_relative_eq!(
                *data.get([0, 5, k]).unwrap(),
                f64::sqrt(15.0) * (2.0 * x - 1.0) * (6.0 * y * y - 6.0 * y + 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([1, 5, k]).unwrap(),
                2.0 * f64::sqrt(15.0) * (6.0 * y * y - 6.0 * y + 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([2, 5, k]).unwrap(),
                f64::sqrt(15.0) * (2.0 * x - 1.0) * (12.0 * y - 6.0),
                epsilon = 1e-12
            );

            // 8 => 5*(6x^2 - 6x + 1)*(6y^2 - 6y + 1)
            assert_relative_eq!(
                *data.get([0, 8, k]).unwrap(),
                5.0 * (6.0 * x * x - 6.0 * x + 1.0) * (6.0 * y * y - 6.0 * y + 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([1, 8, k]).unwrap(),
                5.0 * (12.0 * x - 6.0) * (6.0 * y * y - 6.0 * y + 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([2, 8, k]).unwrap(),
                5.0 * (12.0 * y - 6.0) * (6.0 * x * x - 6.0 * x + 1.0),
                epsilon = 1e-12
            );
        }
    }
}
