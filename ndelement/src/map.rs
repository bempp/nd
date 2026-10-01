//! Maps from the reference cell to/from physical cells.
use crate::traits::Map;
use rlst::{Array, MutableArrayImpl, RlstScalar, ValueArrayImpl};

/// Identity map
///
/// An identity map pushes values from the reference to the physical
/// cell without modifying them.
pub struct IdentityMap {}

impl Map for IdentityMap {
    fn push_forward<
        T: RlstScalar,
        TGeo: RlstScalar,
        Array3GeoImpl: ValueArrayImpl<TGeo, 3>,
        Array4Impl: ValueArrayImpl<T, 4>,
        Array4MutImpl: MutableArrayImpl<T, 4>,
    >(
        &self,
        reference_values: &Array<Array4Impl, 4>,
        nderivs: usize,
        _jacobians: &Array<Array3GeoImpl, 3>,
        _jacobian_determinants: &[TGeo],
        _inverse_jacobians: &Array<Array3GeoImpl, 3>,
        physical_values: &mut Array<Array4MutImpl, 4>,
    ) {
        assert!(nderivs == 0, "Maps with derivatives not supported.");

        physical_values.fill_from(reference_values);
    }
    fn pull_back<
        T: RlstScalar,
        TGeo: RlstScalar,
        Array3GeoImpl: ValueArrayImpl<TGeo, 3>,
        Array4Impl: ValueArrayImpl<T, 4>,
        Array4MutImpl: MutableArrayImpl<T, 4>,
    >(
        &self,
        physical_values: &Array<Array4Impl, 4>,
        nderivs: usize,
        _jacobians: &Array<Array3GeoImpl, 3>,
        _jacobian_determinants: &[TGeo],
        _inverse_jacobians: &Array<Array3GeoImpl, 3>,
        reference_values: &mut Array<Array4MutImpl, 4>,
    ) {
        assert!(nderivs == 0, "Maps with derivatives not supported.");

        reference_values.fill_from(physical_values);
    }
    fn physical_value_shape(&self, _gdim: usize) -> Vec<usize> {
        vec![1]
    }
}

/// Covariant Piola map.
///
/// Let $F$ be the map from the reference cell to the physical cell
/// and let $J$ be its Jacobian. Let $\Phi$ be the function values
/// on the reference cell.  The covariant Piola map is defined by
/// $$
/// J^{-T}\Phi\circ F^{-1}
/// $$
/// The covariant Piola map preserves tangential continuity. If $J$
/// is a rectangular matrix then the pseudo-inverse is used instead of
/// the inverse.
pub struct CovariantPiolaMap {}

impl Map for CovariantPiolaMap {
    fn push_forward<
        T: RlstScalar,
        TGeo: RlstScalar,
        Array3GeoImpl: ValueArrayImpl<TGeo, 3>,
        Array4Impl: ValueArrayImpl<T, 4>,
        Array4MutImpl: MutableArrayImpl<T, 4>,
    >(
        &self,
        reference_values: &Array<Array4Impl, 4>,
        nderivs: usize,
        _jacobians: &Array<Array3GeoImpl, 3>,
        _jacobian_determinants: &[TGeo],
        inverse_jacobians: &Array<Array3GeoImpl, 3>,
        physical_values: &mut Array<Array4MutImpl, 4>,
    ) {
        let tdim = inverse_jacobians.shape()[0];
        let gdim = inverse_jacobians.shape()[1];
        assert_eq!(reference_values.shape()[0], physical_values.shape()[0]);
        assert_eq!(reference_values.shape()[3], physical_values.shape()[3]);
        assert_eq!(reference_values.shape()[2], physical_values.shape()[2]);
        assert_eq!(reference_values.shape()[1], tdim);
        assert_eq!(physical_values.shape()[1], gdim);
        assert!(nderivs == 0, "Maps with derivatives not supported.");
        for b in 0..reference_values.shape()[3] {
            for p in 0..reference_values.shape()[0] {
                for gd in 0..gdim {
                    unsafe {
                        *physical_values.get_unchecked_mut([p, gd, 0, b]) = (0..tdim)
                            .map(|td| {
                                T::from(inverse_jacobians.get_value_unchecked([td, gd, p])).unwrap()
                                    * reference_values.get_value_unchecked([p, td, 0, b])
                            })
                            .sum::<T>();
                    }
                }
            }
        }
    }
    fn pull_back<
        T: RlstScalar,
        TGeo: RlstScalar,
        Array3GeoImpl: ValueArrayImpl<TGeo, 3>,
        Array4Impl: ValueArrayImpl<T, 4>,
        Array4MutImpl: MutableArrayImpl<T, 4>,
    >(
        &self,
        physical_values: &Array<Array4Impl, 4>,
        nderivs: usize,
        jacobians: &Array<Array3GeoImpl, 3>,
        _jacobian_determinants: &[TGeo],
        _inverse_jacobians: &Array<Array3GeoImpl, 3>,
        reference_values: &mut Array<Array4MutImpl, 4>,
    ) {
        let gdim = jacobians.shape()[0];
        let tdim = jacobians.shape()[1];
        assert_eq!(reference_values.shape()[0], physical_values.shape()[0]);
        assert_eq!(reference_values.shape()[3], physical_values.shape()[3]);
        assert_eq!(reference_values.shape()[2], physical_values.shape()[2]);
        assert_eq!(reference_values.shape()[1], tdim);
        assert_eq!(physical_values.shape()[1], gdim);
        assert!(nderivs == 0, "Maps with derivatives not supported.");
        for b in 0..physical_values.shape()[3] {
            for p in 0..physical_values.shape()[0] {
                for td in 0..tdim {
                    unsafe {
                        *reference_values.get_unchecked_mut([p, td, 0, b]) = (0..gdim)
                            .map(|gd| {
                                T::from(jacobians.get_value_unchecked([gd, td, p])).unwrap()
                                    * physical_values.get_value_unchecked([p, gd, 0, b])
                            })
                            .sum::<T>();
                    }
                }
            }
        }
    }
    fn physical_value_shape(&self, gdim: usize) -> Vec<usize> {
        vec![gdim]
    }
}

/// Contravariant Piola map.
///
/// Let $F$ be the map from the reference cell to the physical cell
/// and let $J$ be its Jacobian. Let $\Phi$ be the function values
/// on the reference cell.  The contravariant Piola map is defined by
/// $$
/// \frac{1}{\det{J}}J\Phi\circ F^{-1}
/// $$
/// The contravariant Piola map preserves normal continuity. If $J$
/// is a rectangular matrix then $\sqrt{\det{J^TJ}}$ is used instead
/// of $\det{J}$.
pub struct ContravariantPiolaMap {}

impl Map for ContravariantPiolaMap {
    fn push_forward<
        T: RlstScalar,
        TGeo: RlstScalar,
        Array3GeoImpl: ValueArrayImpl<TGeo, 3>,
        Array4Impl: ValueArrayImpl<T, 4>,
        Array4MutImpl: MutableArrayImpl<T, 4>,
    >(
        &self,
        reference_values: &Array<Array4Impl, 4>,
        nderivs: usize,
        jacobians: &Array<Array3GeoImpl, 3>,
        jacobian_determinants: &[TGeo],
        _inverse_jacobians: &Array<Array3GeoImpl, 3>,
        physical_values: &mut Array<Array4MutImpl, 4>,
    ) {
        let gdim = jacobians.shape()[0];
        let tdim = jacobians.shape()[1];

        assert_eq!(reference_values.shape()[0], physical_values.shape()[0]);
        assert_eq!(reference_values.shape()[3], physical_values.shape()[3]);
        assert_eq!(reference_values.shape()[2], physical_values.shape()[2]);
        assert_eq!(reference_values.shape()[1], tdim);
        assert_eq!(physical_values.shape()[1], gdim);
        assert!(nderivs == 0, "Maps with derivatives not supported.");
        for b in 0..physical_values.shape()[3] {
            for p in 0..physical_values.shape()[0] {
                for gd in 0..gdim {
                    unsafe {
                        *physical_values.get_unchecked_mut([p, gd, 0, b]) = (0..tdim)
                            .map(|td| {
                                T::from(jacobians.get_value_unchecked([gd, td, p])).unwrap()
                                    * reference_values.get_value_unchecked([p, td, 0, b])
                            })
                            .sum::<T>()
                            / T::from(*jacobian_determinants.get_unchecked(p)).unwrap();
                    }
                }
            }
        }
    }
    fn pull_back<
        T: RlstScalar,
        TGeo: RlstScalar,
        Array3GeoImpl: ValueArrayImpl<TGeo, 3>,
        Array4Impl: ValueArrayImpl<T, 4>,
        Array4MutImpl: MutableArrayImpl<T, 4>,
    >(
        &self,
        physical_values: &Array<Array4Impl, 4>,
        nderivs: usize,
        _jacobians: &Array<Array3GeoImpl, 3>,
        jacobian_determinants: &[TGeo],
        inverse_jacobians: &Array<Array3GeoImpl, 3>,
        reference_values: &mut Array<Array4MutImpl, 4>,
    ) {
        let tdim = inverse_jacobians.shape()[0];
        let gdim = inverse_jacobians.shape()[1];
        assert_eq!(reference_values.shape()[0], physical_values.shape()[0]);
        assert_eq!(reference_values.shape()[3], physical_values.shape()[3]);
        assert_eq!(reference_values.shape()[2], physical_values.shape()[2]);
        assert_eq!(reference_values.shape()[1], tdim);
        assert_eq!(physical_values.shape()[1], gdim);
        assert!(nderivs == 0, "Maps with derivatives not supported.");
        for b in 0..physical_values.shape()[3] {
            for p in 0..physical_values.shape()[0] {
                for td in 0..tdim {
                    unsafe {
                        *reference_values.get_unchecked_mut([p, td, 0, b]) = (0..gdim)
                            .map(|gd| {
                                T::from(inverse_jacobians.get_value_unchecked([td, gd, p])).unwrap()
                                    * physical_values.get_value_unchecked([p, gd, 0, b])
                            })
                            .sum::<T>()
                            * T::from(*jacobian_determinants.get_unchecked(p)).unwrap();
                    }
                }
            }
        }
    }
    fn physical_value_shape(&self, gdim: usize) -> Vec<usize> {
        vec![gdim]
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::*;
    use rlst::{Array, DynArray, rlst_dynamic_array};

    fn fill_jacobians<T: RlstScalar, Array3MutImpl: MutableArrayImpl<T, 3>>(
        j: &mut Array<Array3MutImpl, 3>,
        jdet: &mut [T],
        jinv: &mut Array<Array3MutImpl, 3>,
    ) {
        *j.get_mut([0, 0, 0]).unwrap() = T::from(1.0).unwrap();
        *j.get_mut([0, 1, 0]).unwrap() = T::from(1.0).unwrap();
        *j.get_mut([1, 0, 0]).unwrap() = T::from(0.0).unwrap();
        *j.get_mut([1, 1, 0]).unwrap() = T::from(1.0).unwrap();
        *j.get_mut([0, 0, 1]).unwrap() = T::from(2.0).unwrap();
        *j.get_mut([0, 1, 1]).unwrap() = T::from(0.0).unwrap();
        *j.get_mut([1, 0, 1]).unwrap() = T::from(0.0).unwrap();
        *j.get_mut([1, 1, 1]).unwrap() = T::from(3.0).unwrap();
        jdet[0] = T::from(1.0).unwrap();
        jdet[1] = T::from(6.0).unwrap();
        *jinv.get_mut([0, 0, 0]).unwrap() = T::from(1.0).unwrap();
        *jinv.get_mut([0, 1, 0]).unwrap() = T::from(-1.0).unwrap();
        *jinv.get_mut([1, 0, 0]).unwrap() = T::from(0.0).unwrap();
        *jinv.get_mut([1, 1, 0]).unwrap() = T::from(1.0).unwrap();
        *jinv.get_mut([0, 0, 1]).unwrap() = T::from(0.5).unwrap();
        *jinv.get_mut([0, 1, 1]).unwrap() = T::from(0.0).unwrap();
        *jinv.get_mut([1, 0, 1]).unwrap() = T::from(0.0).unwrap();
        *jinv.get_mut([1, 1, 1]).unwrap() = T::from(1.0 / 3.0).unwrap();
    }

    #[test]
    fn test_identity() {
        let map = IdentityMap {};
        let mut values = rlst_dynamic_array!(f64, [2, 1, 1, 2]);
        *values.get_mut([0, 0, 0, 0]).unwrap() = 1.0;
        *values.get_mut([1, 0, 0, 0]).unwrap() = 0.0;
        *values.get_mut([0, 0, 0, 1]).unwrap() = 0.5;
        *values.get_mut([1, 0, 0, 1]).unwrap() = 2.0;

        let mut j = rlst_dynamic_array!(f64, [2, 2, 2]);
        let mut jdet = vec![0.0; 2];
        let mut jinv = rlst_dynamic_array!(f64, [2, 2, 2]);
        fill_jacobians(&mut j, &mut jdet, &mut jinv);

        let mut physical_values = rlst_dynamic_array!(f64, [2, 1, 1, 2]);

        map.push_forward(&values, 0, &j, &jdet, &jinv, &mut physical_values);

        assert_relative_eq!(
            *physical_values.get([0, 0, 0, 0]).unwrap(),
            1.0,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            *physical_values.get([1, 0, 0, 0]).unwrap(),
            0.0,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            *physical_values.get([0, 0, 0, 1]).unwrap(),
            0.5,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            *physical_values.get([1, 0, 0, 1]).unwrap(),
            2.0,
            epsilon = 1e-14
        );

        values.set_zero();

        map.pull_back(&physical_values, 0, &j, &jdet, &jinv, &mut values);

        assert_relative_eq!(*values.get([0, 0, 0, 0]).unwrap(), 1.0, epsilon = 1e-14);
        assert_relative_eq!(*values.get([1, 0, 0, 0]).unwrap(), 0.0, epsilon = 1e-14);
        assert_relative_eq!(*values.get([0, 0, 0, 1]).unwrap(), 0.5, epsilon = 1e-14);
        assert_relative_eq!(*values.get([1, 0, 0, 1]).unwrap(), 2.0, epsilon = 1e-14);
    }

    #[test]
    fn test_covariant_piola() {
        let map = CovariantPiolaMap {};
        let mut values = rlst_dynamic_array!(f64, [2, 2, 1, 2]);
        *values.get_mut([0, 0, 0, 0]).unwrap() = 1.0;
        *values.get_mut([0, 1, 0, 0]).unwrap() = 0.0;
        *values.get_mut([1, 0, 0, 0]).unwrap() = 0.0;
        *values.get_mut([1, 1, 0, 0]).unwrap() = 1.0;
        *values.get_mut([0, 0, 0, 1]).unwrap() = 0.5;
        *values.get_mut([0, 1, 0, 1]).unwrap() = 1.5;
        *values.get_mut([1, 0, 0, 1]).unwrap() = 2.0;
        *values.get_mut([1, 1, 0, 1]).unwrap() = 2.0;

        let mut j = rlst_dynamic_array!(f64, [2, 2, 2]);
        let mut jdet = vec![0.0; 2];
        let mut jinv = rlst_dynamic_array!(f64, [2, 2, 2]);
        fill_jacobians(&mut j, &mut jdet, &mut jinv);

        let mut physical_values = rlst_dynamic_array!(f64, [2, 2, 1, 2]);

        map.push_forward(&values, 0, &j, &jdet, &jinv, &mut physical_values);

        assert_relative_eq!(
            *physical_values.get([0, 0, 0, 0]).unwrap(),
            1.0,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            *physical_values.get([0, 1, 0, 0]).unwrap(),
            -1.0,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            *physical_values.get([1, 0, 0, 0]).unwrap(),
            0.0,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            *physical_values.get([1, 1, 0, 0]).unwrap(),
            1.0 / 3.0,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            *physical_values.get([0, 0, 0, 1]).unwrap(),
            0.5,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            *physical_values.get([0, 1, 0, 1]).unwrap(),
            1.0,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            *physical_values.get([1, 0, 0, 1]).unwrap(),
            1.0,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            *physical_values.get([1, 1, 0, 1]).unwrap(),
            2.0 / 3.0,
            epsilon = 1e-14
        );

        values.set_zero();
        map.pull_back(&physical_values, 0, &j, &jdet, &jinv, &mut values);

        assert_relative_eq!(*values.get([0, 0, 0, 0]).unwrap(), 1.0, epsilon = 1e-14);
        assert_relative_eq!(*values.get([0, 1, 0, 0]).unwrap(), 0.0, epsilon = 1e-14);
        assert_relative_eq!(*values.get([1, 0, 0, 0]).unwrap(), 0.0, epsilon = 1e-14);
        assert_relative_eq!(*values.get([1, 1, 0, 0]).unwrap(), 1.0, epsilon = 1e-14);
        assert_relative_eq!(*values.get([0, 0, 0, 1]).unwrap(), 0.5, epsilon = 1e-14);
        assert_relative_eq!(*values.get([0, 1, 0, 1]).unwrap(), 1.5, epsilon = 1e-14);
        assert_relative_eq!(*values.get([1, 0, 0, 1]).unwrap(), 2.0, epsilon = 1e-14);
        assert_relative_eq!(*values.get([1, 1, 0, 1]).unwrap(), 2.0, epsilon = 1e-14);
    }

    #[test]
    fn test_contravariant_piola() {
        let map = ContravariantPiolaMap {};
        let mut values = rlst_dynamic_array!(f64, [2, 2, 1, 2]);
        *values.get_mut([0, 0, 0, 0]).unwrap() = 1.0;
        *values.get_mut([0, 1, 0, 0]).unwrap() = 0.0;
        *values.get_mut([1, 0, 0, 0]).unwrap() = 0.0;
        *values.get_mut([1, 1, 0, 0]).unwrap() = 1.0;
        *values.get_mut([0, 0, 0, 1]).unwrap() = 0.5;
        *values.get_mut([0, 1, 0, 1]).unwrap() = 1.5;
        *values.get_mut([1, 0, 0, 1]).unwrap() = 2.0;
        *values.get_mut([1, 1, 0, 1]).unwrap() = 2.0;

        let mut j = rlst_dynamic_array!(f64, [2, 2, 2]);
        let mut jdet = vec![0.0; 2];
        let mut jinv = rlst_dynamic_array!(f64, [2, 2, 2]);
        fill_jacobians(&mut j, &mut jdet, &mut jinv);

        let mut physical_values = rlst_dynamic_array!(f64, [2, 2, 1, 2]);

        map.push_forward(&values, 0, &j, &jdet, &jinv, &mut physical_values);

        assert_relative_eq!(
            *physical_values.get([0, 0, 0, 0]).unwrap(),
            1.0,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            *physical_values.get([0, 1, 0, 0]).unwrap(),
            0.0,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            *physical_values.get([1, 0, 0, 0]).unwrap(),
            0.0,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            *physical_values.get([1, 1, 0, 0]).unwrap(),
            0.5,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            *physical_values.get([0, 0, 0, 1]).unwrap(),
            2.0,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            *physical_values.get([0, 1, 0, 1]).unwrap(),
            1.5,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            *physical_values.get([1, 0, 0, 1]).unwrap(),
            2.0 / 3.0,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            *physical_values.get([1, 1, 0, 1]).unwrap(),
            1.0,
            epsilon = 1e-14
        );

        values.set_zero();
        map.pull_back(&physical_values, 0, &j, &jdet, &jinv, &mut values);

        assert_relative_eq!(*values.get([0, 0, 0, 0]).unwrap(), 1.0, epsilon = 1e-14);
        assert_relative_eq!(*values.get([0, 1, 0, 0]).unwrap(), 0.0, epsilon = 1e-14);
        assert_relative_eq!(*values.get([1, 0, 0, 0]).unwrap(), 0.0, epsilon = 1e-14);
        assert_relative_eq!(*values.get([1, 1, 0, 0]).unwrap(), 1.0, epsilon = 1e-14);
        assert_relative_eq!(*values.get([0, 0, 0, 1]).unwrap(), 0.5, epsilon = 1e-14);
        assert_relative_eq!(*values.get([0, 1, 0, 1]).unwrap(), 1.5, epsilon = 1e-14);
        assert_relative_eq!(*values.get([1, 0, 0, 1]).unwrap(), 2.0, epsilon = 1e-14);
        assert_relative_eq!(*values.get([1, 1, 0, 1]).unwrap(), 2.0, epsilon = 1e-14);
    }

    /// Jacobians of three maps from a 2D reference cell into 3D, with their pseudo-inverses
    /// and the values of $\sqrt{\det(J^TJ)}$.
    ///
    /// The columns of the Jacobians are $(e_0, 2e_1)$, $(e_1, e_0)$ and $(e_0, 3e_2)$.
    fn rectangular_jacobians() -> (DynArray<f64, 3>, Vec<f64>, DynArray<f64, 3>) {
        let mut j = rlst_dynamic_array!(f64, [3, 2, 3]);
        let mut jinv = rlst_dynamic_array!(f64, [2, 3, 3]);
        j[[0, 0, 0]] = 1.0;
        j[[1, 1, 0]] = 2.0;
        jinv[[0, 0, 0]] = 1.0;
        jinv[[1, 1, 0]] = 0.5;
        j[[1, 0, 1]] = 1.0;
        j[[0, 1, 1]] = 1.0;
        jinv[[0, 1, 1]] = 1.0;
        jinv[[1, 0, 1]] = 1.0;
        j[[0, 0, 2]] = 1.0;
        j[[2, 1, 2]] = 3.0;
        jinv[[0, 0, 2]] = 1.0;
        jinv[[1, 2, 2]] = 1.0 / 3.0;
        (j, vec![2.0, 1.0, 3.0], jinv)
    }

    /// Reference values with 3 points, 2 basis functions and 2 components that are all distinct.
    fn rectangular_reference_values() -> DynArray<f64, 4> {
        let mut values = rlst_dynamic_array!(f64, [3, 2, 1, 2]);
        for p in 0..3 {
            for b in 0..2 {
                for c in 0..2 {
                    values[[p, c, 0, b]] = 1.0 + p as f64 + 10.0 * b as f64 + 100.0 * c as f64;
                }
            }
        }
        values
    }

    /// Check that pushing forward `map` gives `expected(p, [a, b])` at point `p` for the
    /// reference value `[a, b]`, and that pulling back recovers the reference values.
    fn check_rectangular(map: impl Map, expected: impl Fn(usize, [f64; 2]) -> [f64; 3]) {
        let (j, jdet, jinv) = rectangular_jacobians();
        let values = rectangular_reference_values();
        let mut physical_values = rlst_dynamic_array!(f64, [3, 3, 1, 2]);
        map.push_forward(&values, 0, &j, &jdet, &jinv, &mut physical_values);

        for p in 0..3 {
            for b in 0..2 {
                let e = expected(p, [values[[p, 0, 0, b]], values[[p, 1, 0, b]]]);
                for (c, e_c) in e.iter().enumerate() {
                    assert_relative_eq!(physical_values[[p, c, 0, b]], *e_c, epsilon = 1e-12);
                }
            }
        }

        let mut pulled_back = rlst_dynamic_array!(f64, [3, 2, 1, 2]);
        map.pull_back(&physical_values, 0, &j, &jdet, &jinv, &mut pulled_back);
        for (v, w) in values.iter_value().zip(pulled_back.iter_value()) {
            assert_relative_eq!(v, w, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_covariant_piola_rectangular() {
        // J^{+T} applied to the reference values
        check_rectangular(CovariantPiolaMap {}, |p, [a, b]| match p {
            0 => [a, b / 2.0, 0.0],
            1 => [b, a, 0.0],
            _ => [a, 0.0, b / 3.0],
        });
    }

    #[test]
    fn test_contravariant_piola_rectangular() {
        // J applied to the reference values, divided by sqrt(det(J^T J))
        check_rectangular(ContravariantPiolaMap {}, |p, [a, b]| match p {
            0 => [a / 2.0, b, 0.0],
            1 => [b, a, 0.0],
            _ => [a / 3.0, 0.0, b],
        });
    }
}
