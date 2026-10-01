//! Lagrange elements.

use super::CiarletElement;
use crate::map::IdentityMap;
use crate::polynomials::polynomial_count;
use crate::reference_cell;
use crate::traits::ElementFamily;
use crate::types::{Continuity, ReferenceCellType};
use quadraturerules::{Domain, QuadratureRule, single_integral_quadrature};
use rlst::dense::linalg::lapack::interface::{getrf::Getrf, getri::Getri};
use rlst::{RlstScalar, rlst_dynamic_array};
use std::marker::PhantomData;

/// Variant of a Lagrange element
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
#[repr(u8)]
pub enum LagrangeVariant {
    /// Element defined using equispaced points
    Equispaced,
    /// Element defined using Gauss-Lobatto-Legendre (GLL) points
    Gll,
}

/// Create a Lagrange element.
#[must_use]
pub fn create<T: RlstScalar + Getrf + Getri, TGeo: RlstScalar>(
    cell_type: ReferenceCellType,
    degree: usize,
    continuity: Continuity,
    variant: LagrangeVariant,
) -> CiarletElement<T, IdentityMap, TGeo> {
    let dim = polynomial_count(cell_type, degree);
    let tdim = reference_cell::dim(cell_type);
    let mut wcoeffs = rlst_dynamic_array!(T, [dim, 1, dim]);
    for i in 0..dim {
        *wcoeffs.get_mut([i, 0, i]).unwrap() = T::from(1.0).unwrap();
    }

    let mut x = [vec![], vec![], vec![], vec![]];
    let mut m = [vec![], vec![], vec![], vec![]];
    let entity_counts = reference_cell::entity_counts(cell_type);
    let vertices = reference_cell::vertices::<TGeo>(cell_type);
    if degree == 0 {
        assert!(
            continuity == Continuity::Discontinuous,
            "Cannot create continuous degree 0 Lagrange element"
        );
        for (d, counts) in entity_counts.iter().enumerate() {
            for _e in 0..*counts {
                x[d].push(rlst_dynamic_array!(TGeo, [tdim, 0]));
                m[d].push(rlst_dynamic_array!(T, [0, 1, 0]));
            }
        }
        let mut midp = rlst_dynamic_array!(TGeo, [tdim, 1]);
        let nvertices = entity_counts[0];
        for i in 0..tdim {
            for vertex in &vertices {
                *midp.get_mut([i, 0]).unwrap() += num::cast::<_, TGeo>(vertex[i]).unwrap();
            }
            *midp.get_mut([i, 0]).unwrap() /= num::cast::<_, TGeo>(nvertices).unwrap();
        }
        x[tdim].push(midp);
        let mut mentry = rlst_dynamic_array!(T, [1, 1, 1]);
        *mentry.get_mut([0, 0, 0]).unwrap() = T::from(1.0).unwrap();
        m[tdim].push(mentry);
    } else {
        let edges = reference_cell::edges(cell_type);
        let faces = reference_cell::faces(cell_type);
        let volumes = reference_cell::volumes(cell_type);
        for vertex in &vertices {
            let mut pts = rlst_dynamic_array!(TGeo, [tdim, 1]);
            for (i, v) in vertex.iter().enumerate() {
                *pts.get_mut([i, 0]).unwrap() = num::cast::<_, TGeo>(*v).unwrap();
            }
            x[0].push(pts);
            let mut mentry = rlst_dynamic_array!(T, [1, 1, 1]);
            *mentry.get_mut([0, 0, 0]).unwrap() = T::from(1.0).unwrap();
            m[0].push(mentry);
        }

        let pts1d = match variant {
            LagrangeVariant::Equispaced => (1..degree)
                .map(|i| num::cast::<_, TGeo>(i).unwrap() / num::cast::<_, TGeo>(degree).unwrap())
                .collect::<Vec<_>>(),
            LagrangeVariant::Gll => {
                let (points, _weights) = single_integral_quadrature(
                    QuadratureRule::GaussLobattoLegendre,
                    Domain::Interval,
                    degree - 1,
                )
                .unwrap();
                (1..degree)
                    .map(|i| num::cast::<_, TGeo>(points[2 * i + 1]).unwrap())
                    .collect::<Vec<_>>()
            }
        };

        for e in &edges {
            let mut pts = rlst_dynamic_array!(TGeo, [tdim, degree - 1]);
            let [vn0, vn1] = e[..] else {
                panic!("Invalid edge");
            };
            let v0 = &vertices[vn0];
            let v1 = &vertices[vn1];
            let mut ident = rlst_dynamic_array!(T, [degree - 1, 1, degree - 1]);

            for (i, p) in pts1d.iter().enumerate() {
                *ident.get_mut([i, 0, i]).unwrap() = T::from(1.0).unwrap();
                for j in 0..tdim {
                    *pts.get_mut([j, i]).unwrap() = num::cast::<_, TGeo>(v0[j]).unwrap()
                        + *p * num::cast::<_, TGeo>(v1[j] - v0[j]).unwrap();
                }
            }
            x[1].push(pts);
            m[1].push(ident);
        }
        for (e, face_type) in reference_cell::entity_types(cell_type)[2]
            .iter()
            .enumerate()
        {
            let npts = match face_type {
                ReferenceCellType::Triangle => {
                    if degree > 2 {
                        (degree - 1) * (degree - 2) / 2
                    } else {
                        0
                    }
                }
                ReferenceCellType::Quadrilateral => (degree - 1).pow(2),
                _ => {
                    panic!("Unsupported face type");
                }
            };
            let mut pts = rlst_dynamic_array!(TGeo, [tdim, npts]);

            let [vn0, vn1, vn2] = faces[e][..3] else {
                panic!("Invalid face");
            };
            let v0 = &vertices[vn0];
            let v1 = &vertices[vn1];
            let v2 = &vertices[vn2];

            match face_type {
                ReferenceCellType::Triangle => {
                    assert!(
                        variant == LagrangeVariant::Equispaced,
                        "Only equispaced Lagrange elements are supported on triangular cells."
                    );
                    let mut n = 0;
                    for i1 in 1..degree {
                        for i0 in 1..degree - i1 {
                            for j in 0..tdim {
                                *pts.get_mut([j, n]).unwrap() = num::cast::<_, TGeo>(v0[j])
                                    .unwrap()
                                    + num::cast::<_, TGeo>(i0).unwrap()
                                        / num::cast::<_, TGeo>(degree).unwrap()
                                        * num::cast::<_, TGeo>(v1[j] - v0[j]).unwrap()
                                    + num::cast::<_, TGeo>(i1).unwrap()
                                        / num::cast::<_, TGeo>(degree).unwrap()
                                        * num::cast::<_, TGeo>(v2[j] - v0[j]).unwrap();
                            }
                            n += 1;
                        }
                    }
                }
                ReferenceCellType::Quadrilateral => {
                    let mut n = 0;
                    for p1 in &pts1d {
                        for p0 in &pts1d {
                            for j in 0..tdim {
                                *pts.get_mut([j, n]).unwrap() = num::cast::<_, TGeo>(v0[j])
                                    .unwrap()
                                    + *p0 * num::cast::<_, TGeo>(v1[j] - v0[j]).unwrap()
                                    + *p1 * num::cast::<_, TGeo>(v2[j] - v0[j]).unwrap();
                            }
                            n += 1;
                        }
                    }
                }
                _ => {
                    panic!("Unsupported face type.");
                }
            }

            let mut ident = rlst_dynamic_array!(T, [npts, 1, npts]);
            for i in 0..npts {
                *ident.get_mut([i, 0, i]).unwrap() = T::from(1.0).unwrap();
            }
            x[2].push(pts);
            m[2].push(ident);
        }
        for (e, volume_type) in reference_cell::entity_types(cell_type)[3]
            .iter()
            .enumerate()
        {
            let npts = match volume_type {
                ReferenceCellType::Tetrahedron => {
                    if degree > 2 {
                        (degree - 1) * (degree - 2) * (degree - 3) / 6
                    } else {
                        0
                    }
                }
                ReferenceCellType::Hexahedron => (degree - 1).pow(3),
                _ => {
                    panic!("Unsupported face type");
                }
            };
            let mut pts = rlst_dynamic_array!(TGeo, [tdim, npts]);

            match volume_type {
                ReferenceCellType::Tetrahedron => {
                    assert!(
                        variant == LagrangeVariant::Equispaced,
                        "Only equispaced Lagrange elements are supported on tetrahedral cells."
                    );
                    let [vn0, vn1, vn2, vn3] = volumes[e][..4] else {
                        panic!("invalid tetrahedron");
                    };
                    let v0 = &vertices[vn0];
                    let v1 = &vertices[vn1];
                    let v2 = &vertices[vn2];
                    let v3 = &vertices[vn3];

                    let mut n = 0;
                    for i2 in 1..degree {
                        for i1 in 1..degree - i2 {
                            for i0 in 1..degree - i2 - i1 {
                                for j in 0..tdim {
                                    *pts.get_mut([j, n]).unwrap() = num::cast::<_, TGeo>(v0[j])
                                        .unwrap()
                                        + num::cast::<_, TGeo>(i0).unwrap()
                                            / num::cast::<_, TGeo>(degree).unwrap()
                                            * num::cast::<_, TGeo>(v1[j] - v0[j]).unwrap()
                                        + num::cast::<_, TGeo>(i1).unwrap()
                                            / num::cast::<_, TGeo>(degree).unwrap()
                                            * num::cast::<_, TGeo>(v2[j] - v0[j]).unwrap()
                                        + num::cast::<_, TGeo>(i2).unwrap()
                                            / num::cast::<_, TGeo>(degree).unwrap()
                                            * num::cast::<_, TGeo>(v3[j] - v0[j]).unwrap();
                                }
                                n += 1;
                            }
                        }
                    }
                }
                ReferenceCellType::Hexahedron => {
                    let [vn0, vn1, vn2, _, vn3] = volumes[e][..5] else {
                        panic!("Invalid hexahedron");
                    };
                    let v0 = &vertices[vn0];
                    let v1 = &vertices[vn1];
                    let v2 = &vertices[vn2];
                    let v3 = &vertices[vn3];

                    let mut n = 0;
                    for p2 in &pts1d {
                        for p1 in &pts1d {
                            for p0 in &pts1d {
                                for j in 0..tdim {
                                    *pts.get_mut([j, n]).unwrap() = num::cast::<_, TGeo>(v0[j])
                                        .unwrap()
                                        + *p0 * num::cast::<_, TGeo>(v1[j] - v0[j]).unwrap()
                                        + *p1 * num::cast::<_, TGeo>(v2[j] - v0[j]).unwrap()
                                        + *p2 * num::cast::<_, TGeo>(v3[j] - v0[j]).unwrap();
                                }
                                n += 1;
                            }
                        }
                    }
                }
                _ => {
                    panic!("Unsupported face type.");
                }
            }

            let mut ident = rlst_dynamic_array!(T, [npts, 1, npts]);
            for i in 0..npts {
                *ident.get_mut([i, 0, i]).unwrap() = T::from(1.0).unwrap();
            }
            x[3].push(pts);
            m[3].push(ident);
        }
    }
    CiarletElement::<T, IdentityMap, TGeo>::create(
        "Lagrange".to_string(),
        cell_type,
        degree,
        vec![],
        wcoeffs,
        x,
        m,
        continuity,
        degree,
        IdentityMap {},
    )
}

/// Lagrange element family.
///
/// A family of Lagrange elements on multiple cell types with appropriate
/// continuity across different cell types.
pub struct LagrangeElementFamily<T: RlstScalar + Getrf + Getri = f64, TGeo: RlstScalar = f64> {
    degree: usize,
    continuity: Continuity,
    variant: LagrangeVariant,
    _t: PhantomData<T>,
    _tgeo: PhantomData<TGeo>,
}

impl<T: RlstScalar + Getrf + Getri, TGeo: RlstScalar> LagrangeElementFamily<T, TGeo> {
    /// Create new family with given `degree` and `continuity`.
    #[must_use]
    pub fn new(degree: usize, continuity: Continuity, variant: LagrangeVariant) -> Self {
        Self {
            degree,
            continuity,
            variant,
            _t: PhantomData,
            _tgeo: PhantomData,
        }
    }
}

impl<T: RlstScalar + Getrf + Getri, TGeo: RlstScalar> ElementFamily
    for LagrangeElementFamily<T, TGeo>
{
    type T = T;
    type FiniteElement = CiarletElement<T, IdentityMap, TGeo>;
    type CellType = ReferenceCellType;
    fn element(&self, cell_type: ReferenceCellType) -> CiarletElement<T, IdentityMap, TGeo> {
        create::<T, TGeo>(cell_type, self.degree, self.continuity, self.variant)
    }
}

#[cfg(test)]
mod test {
    use super::super::test::{check_dofs, check_tabulate_layout, tabulate};
    use super::*;
    use crate::polynomials::test::{derivative_index, interior_points, multi_indices};
    use crate::traits::FiniteElement;
    use approx::*;
    use paste::paste;
    use rlst::DynArray;

    #[test]
    fn test_lagrange_1() {
        let e = create::<f64, f64>(
            ReferenceCellType::Triangle,
            1,
            Continuity::Standard,
            LagrangeVariant::Equispaced,
        );
        assert_eq!(e.value_size(), 1);
    }

    #[test]
    fn test_lagrange_0_interval() {
        let e = create::<f64, f64>(
            ReferenceCellType::Interval,
            0,
            Continuity::Discontinuous,
            LagrangeVariant::Equispaced,
        );
        assert_eq!(e.value_size(), 1);
        let mut data = DynArray::<f64, 4>::from_shape(e.tabulate_array_shape(0, 4));
        let mut points = rlst_dynamic_array!(f64, [1, 4]);
        *points.get_mut([0, 0]).unwrap() = 0.0;
        *points.get_mut([0, 1]).unwrap() = 0.2;
        *points.get_mut([0, 2]).unwrap() = 0.4;
        *points.get_mut([0, 3]).unwrap() = 1.0;
        e.tabulate(&points, 0, &mut data);

        for pt in 0..4 {
            assert_relative_eq!(*data.get([pt, 0, 0, 0]).unwrap(), 1.0);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_1_interval() {
        let e = create::<f64, f64>(
            ReferenceCellType::Interval,
            1,
            Continuity::Standard,
            LagrangeVariant::Equispaced,
        );
        assert_eq!(e.value_size(), 1);
        let mut data = DynArray::<f64, 4>::from_shape(e.tabulate_array_shape(0, 4));
        let mut points = rlst_dynamic_array!(f64, [1, 4]);
        *points.get_mut([0, 0]).unwrap() = 0.0;
        *points.get_mut([0, 1]).unwrap() = 0.2;
        *points.get_mut([0, 2]).unwrap() = 0.4;
        *points.get_mut([0, 3]).unwrap() = 1.0;
        e.tabulate(&points, 0, &mut data);

        for pt in 0..4 {
            let x = *points.get([0, pt]).unwrap();
            assert_relative_eq!(*data.get([pt, 0, 0, 0]).unwrap(), 1.0 - x);
            assert_relative_eq!(*data.get([pt, 0, 0, 1]).unwrap(), x);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_0_triangle() {
        let e = create::<f64, f64>(
            ReferenceCellType::Triangle,
            0,
            Continuity::Discontinuous,
            LagrangeVariant::Equispaced,
        );
        assert_eq!(e.value_size(), 1);
        let mut data = DynArray::<f64, 4>::from_shape(e.tabulate_array_shape(0, 6));

        let mut points = rlst_dynamic_array!(f64, [2, 6]);
        *points.get_mut([0, 0]).unwrap() = 0.0;
        *points.get_mut([1, 0]).unwrap() = 0.0;
        *points.get_mut([0, 1]).unwrap() = 1.0;
        *points.get_mut([1, 1]).unwrap() = 0.0;
        *points.get_mut([0, 2]).unwrap() = 0.0;
        *points.get_mut([1, 2]).unwrap() = 1.0;
        *points.get_mut([0, 3]).unwrap() = 0.5;
        *points.get_mut([1, 3]).unwrap() = 0.0;
        *points.get_mut([0, 4]).unwrap() = 0.0;
        *points.get_mut([1, 4]).unwrap() = 0.5;
        *points.get_mut([0, 5]).unwrap() = 0.5;
        *points.get_mut([1, 5]).unwrap() = 0.5;

        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            assert_relative_eq!(*data.get([pt, 0, 0, 0]).unwrap(), 1.0);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_1_triangle() {
        let e = create::<f64, f64>(
            ReferenceCellType::Triangle,
            1,
            Continuity::Standard,
            LagrangeVariant::Equispaced,
        );
        assert_eq!(e.value_size(), 1);
        let mut data = DynArray::<f64, 4>::from_shape(e.tabulate_array_shape(0, 6));
        let mut points = rlst_dynamic_array!(f64, [2, 6]);
        *points.get_mut([0, 0]).unwrap() = 0.0;
        *points.get_mut([1, 0]).unwrap() = 0.0;
        *points.get_mut([0, 1]).unwrap() = 1.0;
        *points.get_mut([1, 1]).unwrap() = 0.0;
        *points.get_mut([0, 2]).unwrap() = 0.0;
        *points.get_mut([1, 2]).unwrap() = 1.0;
        *points.get_mut([0, 3]).unwrap() = 0.5;
        *points.get_mut([1, 3]).unwrap() = 0.0;
        *points.get_mut([0, 4]).unwrap() = 0.0;
        *points.get_mut([1, 4]).unwrap() = 0.5;
        *points.get_mut([0, 5]).unwrap() = 0.5;
        *points.get_mut([1, 5]).unwrap() = 0.5;
        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            let x = *points.get([0, pt]).unwrap();
            let y = *points.get([1, pt]).unwrap();
            assert_relative_eq!(*data.get([pt, 0, 0, 0]).unwrap(), 1.0 - x - y);
            assert_relative_eq!(*data.get([pt, 0, 0, 1]).unwrap(), x);
            assert_relative_eq!(*data.get([pt, 0, 0, 2]).unwrap(), y);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_0_quadrilateral() {
        let e = create::<f64, f64>(
            ReferenceCellType::Quadrilateral,
            0,
            Continuity::Discontinuous,
            LagrangeVariant::Equispaced,
        );
        assert_eq!(e.value_size(), 1);
        let mut data = DynArray::<f64, 4>::from_shape(e.tabulate_array_shape(0, 6));
        let mut points = rlst_dynamic_array!(f64, [2, 6]);
        *points.get_mut([0, 0]).unwrap() = 0.0;
        *points.get_mut([1, 0]).unwrap() = 0.0;
        *points.get_mut([0, 1]).unwrap() = 1.0;
        *points.get_mut([1, 1]).unwrap() = 0.0;
        *points.get_mut([0, 2]).unwrap() = 0.0;
        *points.get_mut([1, 2]).unwrap() = 1.0;
        *points.get_mut([0, 3]).unwrap() = 0.5;
        *points.get_mut([1, 3]).unwrap() = 0.0;
        *points.get_mut([0, 4]).unwrap() = 0.0;
        *points.get_mut([1, 4]).unwrap() = 0.5;
        *points.get_mut([0, 5]).unwrap() = 0.5;
        *points.get_mut([1, 5]).unwrap() = 0.5;
        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            assert_relative_eq!(*data.get([pt, 0, 0, 0]).unwrap(), 1.0);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_1_quadrilateral() {
        let e = create::<f64, f64>(
            ReferenceCellType::Quadrilateral,
            1,
            Continuity::Standard,
            LagrangeVariant::Equispaced,
        );
        assert_eq!(e.value_size(), 1);
        let mut data = DynArray::<f64, 4>::from_shape(e.tabulate_array_shape(0, 6));
        let mut points = rlst_dynamic_array!(f64, [2, 6]);
        *points.get_mut([0, 0]).unwrap() = 0.0;
        *points.get_mut([1, 0]).unwrap() = 0.0;
        *points.get_mut([0, 1]).unwrap() = 1.0;
        *points.get_mut([1, 1]).unwrap() = 0.0;
        *points.get_mut([0, 2]).unwrap() = 0.0;
        *points.get_mut([1, 2]).unwrap() = 1.0;
        *points.get_mut([0, 3]).unwrap() = 1.0;
        *points.get_mut([1, 3]).unwrap() = 1.0;
        *points.get_mut([0, 4]).unwrap() = 0.25;
        *points.get_mut([1, 4]).unwrap() = 0.5;
        *points.get_mut([0, 5]).unwrap() = 0.3;
        *points.get_mut([1, 5]).unwrap() = 0.2;

        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            let x = *points.get([0, pt]).unwrap();
            let y = *points.get([1, pt]).unwrap();
            assert_relative_eq!(*data.get([pt, 0, 0, 0]).unwrap(), (1.0 - x) * (1.0 - y));
            assert_relative_eq!(*data.get([pt, 0, 0, 1]).unwrap(), x * (1.0 - y));
            assert_relative_eq!(*data.get([pt, 0, 0, 2]).unwrap(), (1.0 - x) * y);
            assert_relative_eq!(*data.get([pt, 0, 0, 3]).unwrap(), x * y);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_2_quadrilateral() {
        let e = create::<f64, f64>(
            ReferenceCellType::Quadrilateral,
            2,
            Continuity::Standard,
            LagrangeVariant::Equispaced,
        );
        assert_eq!(e.value_size(), 1);
        let mut data = DynArray::<f64, 4>::from_shape(e.tabulate_array_shape(0, 6));
        let mut points = rlst_dynamic_array!(f64, [2, 6]);
        *points.get_mut([0, 0]).unwrap() = 0.0;
        *points.get_mut([1, 0]).unwrap() = 0.0;
        *points.get_mut([0, 1]).unwrap() = 1.0;
        *points.get_mut([1, 1]).unwrap() = 0.0;
        *points.get_mut([0, 2]).unwrap() = 0.0;
        *points.get_mut([1, 2]).unwrap() = 1.0;
        *points.get_mut([0, 3]).unwrap() = 1.0;
        *points.get_mut([1, 3]).unwrap() = 1.0;
        *points.get_mut([0, 4]).unwrap() = 0.25;
        *points.get_mut([1, 4]).unwrap() = 0.5;
        *points.get_mut([0, 5]).unwrap() = 0.3;
        *points.get_mut([1, 5]).unwrap() = 0.2;

        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            let x = *points.get([0, pt]).unwrap();
            let y = *points.get([1, pt]).unwrap();
            assert_relative_eq!(
                *data.get([pt, 0, 0, 0]).unwrap(),
                (1.0 - x) * (1.0 - 2.0 * x) * (1.0 - y) * (1.0 - 2.0 * y),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([pt, 0, 0, 1]).unwrap(),
                x * (2.0 * x - 1.0) * (1.0 - y) * (1.0 - 2.0 * y),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([pt, 0, 0, 2]).unwrap(),
                (1.0 - x) * (1.0 - 2.0 * x) * y * (2.0 * y - 1.0),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([pt, 0, 0, 3]).unwrap(),
                x * (2.0 * x - 1.0) * y * (2.0 * y - 1.0),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([pt, 0, 0, 4]).unwrap(),
                4.0 * x * (1.0 - x) * (1.0 - y) * (1.0 - 2.0 * y),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([pt, 0, 0, 5]).unwrap(),
                (1.0 - x) * (1.0 - 2.0 * x) * 4.0 * y * (1.0 - y),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([pt, 0, 0, 6]).unwrap(),
                x * (2.0 * x - 1.0) * 4.0 * y * (1.0 - y),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([pt, 0, 0, 7]).unwrap(),
                4.0 * x * (1.0 - x) * y * (2.0 * y - 1.0),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([pt, 0, 0, 8]).unwrap(),
                4.0 * x * (1.0 - x) * 4.0 * y * (1.0 - y),
                epsilon = 1e-14
            );
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_0_tetrahedron() {
        let e = create::<f64, f64>(
            ReferenceCellType::Tetrahedron,
            0,
            Continuity::Discontinuous,
            LagrangeVariant::Equispaced,
        );
        assert_eq!(e.value_size(), 1);
        let mut data = DynArray::<f64, 4>::from_shape(e.tabulate_array_shape(0, 6));
        let mut points = rlst_dynamic_array!(f64, [3, 6]);
        *points.get_mut([0, 0]).unwrap() = 0.0;
        *points.get_mut([1, 0]).unwrap() = 0.0;
        *points.get_mut([2, 0]).unwrap() = 0.0;
        *points.get_mut([0, 1]).unwrap() = 1.0;
        *points.get_mut([1, 1]).unwrap() = 0.0;
        *points.get_mut([2, 1]).unwrap() = 0.0;
        *points.get_mut([0, 2]).unwrap() = 0.0;
        *points.get_mut([1, 2]).unwrap() = 1.0;
        *points.get_mut([2, 2]).unwrap() = 0.0;
        *points.get_mut([0, 3]).unwrap() = 0.5;
        *points.get_mut([1, 3]).unwrap() = 0.0;
        *points.get_mut([2, 3]).unwrap() = 0.5;
        *points.get_mut([0, 4]).unwrap() = 0.0;
        *points.get_mut([1, 4]).unwrap() = 0.5;
        *points.get_mut([2, 4]).unwrap() = 0.5;
        *points.get_mut([0, 5]).unwrap() = 0.5;
        *points.get_mut([1, 5]).unwrap() = 0.2;
        *points.get_mut([2, 5]).unwrap() = 0.3;
        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            assert_relative_eq!(*data.get([pt, 0, 0, 0]).unwrap(), 1.0);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_1_tetrahedron() {
        let e = create::<f64, f64>(
            ReferenceCellType::Tetrahedron,
            1,
            Continuity::Standard,
            LagrangeVariant::Equispaced,
        );
        assert_eq!(e.value_size(), 1);
        let mut data = DynArray::<f64, 4>::from_shape(e.tabulate_array_shape(2, 6));
        let mut points = rlst_dynamic_array!(f64, [3, 6]);
        *points.get_mut([0, 0]).unwrap() = 0.0;
        *points.get_mut([1, 0]).unwrap() = 0.0;
        *points.get_mut([2, 0]).unwrap() = 0.0;
        *points.get_mut([0, 1]).unwrap() = 1.0;
        *points.get_mut([1, 1]).unwrap() = 0.0;
        *points.get_mut([2, 1]).unwrap() = 0.0;
        *points.get_mut([0, 2]).unwrap() = 0.0;
        *points.get_mut([1, 2]).unwrap() = 0.8;
        *points.get_mut([2, 2]).unwrap() = 0.2;
        *points.get_mut([0, 3]).unwrap() = 0.0;
        *points.get_mut([1, 3]).unwrap() = 0.0;
        *points.get_mut([2, 3]).unwrap() = 0.8;
        *points.get_mut([0, 4]).unwrap() = 0.25;
        *points.get_mut([1, 4]).unwrap() = 0.5;
        *points.get_mut([2, 4]).unwrap() = 0.1;
        *points.get_mut([0, 5]).unwrap() = 0.2;
        *points.get_mut([1, 5]).unwrap() = 0.1;
        *points.get_mut([2, 5]).unwrap() = 0.15;

        e.tabulate(&points, 2, &mut data);

        for pt in 0..6 {
            let x = *points.get([0, pt]).unwrap();
            let y = *points.get([1, pt]).unwrap();
            let z = *points.get([2, pt]).unwrap();
            assert_relative_eq!(
                *data.get([pt, 0, 0, 0]).unwrap(),
                1.0 - x - y - z,
                epsilon = 1e-14
            );
            assert_relative_eq!(*data.get([pt, 0, 0, 1]).unwrap(), x, epsilon = 1e-14);
            assert_relative_eq!(*data.get([pt, 0, 0, 2]).unwrap(), y, epsilon = 1e-14);
            assert_relative_eq!(*data.get([pt, 0, 0, 3]).unwrap(), z, epsilon = 1e-14);
        }

        let cell = ReferenceCellType::Tetrahedron;
        let gradients = [
            [-1.0, -1.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        for pt in 0..6 {
            for (b, gradient) in gradients.iter().enumerate() {
                for (alpha, value) in [[1, 0, 0], [0, 1, 0], [0, 0, 1]].iter().zip(gradient) {
                    assert_relative_eq!(
                        *data.get([pt, 0, derivative_index(cell, alpha), b]).unwrap(),
                        *value,
                        epsilon = 1e-12
                    );
                }
                // All second derivatives vanish
                for alpha in multi_indices(3, 2)
                    .iter()
                    .filter(|a| a.iter().sum::<usize>() == 2)
                {
                    assert_relative_eq!(
                        *data.get([pt, 0, derivative_index(cell, alpha), b]).unwrap(),
                        0.0,
                        epsilon = 1e-12
                    );
                }
            }
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_0_hexahedron() {
        let e = create::<f64, f64>(
            ReferenceCellType::Hexahedron,
            0,
            Continuity::Discontinuous,
            LagrangeVariant::Equispaced,
        );
        assert_eq!(e.value_size(), 1);
        let mut data = DynArray::<f64, 4>::from_shape(e.tabulate_array_shape(0, 6));
        let mut points = rlst_dynamic_array!(f64, [3, 6]);
        *points.get_mut([0, 0]).unwrap() = 0.0;
        *points.get_mut([1, 0]).unwrap() = 0.0;
        *points.get_mut([2, 0]).unwrap() = 0.0;
        *points.get_mut([0, 1]).unwrap() = 1.0;
        *points.get_mut([1, 1]).unwrap() = 0.0;
        *points.get_mut([2, 1]).unwrap() = 0.0;
        *points.get_mut([0, 2]).unwrap() = 0.0;
        *points.get_mut([1, 2]).unwrap() = 1.0;
        *points.get_mut([2, 2]).unwrap() = 0.0;
        *points.get_mut([0, 3]).unwrap() = 0.5;
        *points.get_mut([1, 3]).unwrap() = 0.0;
        *points.get_mut([2, 3]).unwrap() = 0.5;
        *points.get_mut([0, 4]).unwrap() = 0.0;
        *points.get_mut([1, 4]).unwrap() = 0.5;
        *points.get_mut([2, 4]).unwrap() = 0.5;
        *points.get_mut([0, 5]).unwrap() = 0.5;
        *points.get_mut([1, 5]).unwrap() = 0.5;
        *points.get_mut([2, 5]).unwrap() = 0.5;
        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            assert_relative_eq!(*data.get([pt, 0, 0, 0]).unwrap(), 1.0);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_1_hexahedron() {
        let e = create::<f64, f64>(
            ReferenceCellType::Hexahedron,
            1,
            Continuity::Standard,
            LagrangeVariant::Equispaced,
        );
        assert_eq!(e.value_size(), 1);
        let mut data = DynArray::<f64, 4>::from_shape(e.tabulate_array_shape(0, 6));
        let mut points = rlst_dynamic_array!(f64, [3, 6]);
        *points.get_mut([0, 0]).unwrap() = 0.0;
        *points.get_mut([1, 0]).unwrap() = 0.0;
        *points.get_mut([2, 0]).unwrap() = 0.0;
        *points.get_mut([0, 1]).unwrap() = 1.0;
        *points.get_mut([1, 1]).unwrap() = 0.0;
        *points.get_mut([2, 1]).unwrap() = 0.0;
        *points.get_mut([0, 2]).unwrap() = 0.0;
        *points.get_mut([1, 2]).unwrap() = 1.0;
        *points.get_mut([2, 2]).unwrap() = 1.0;
        *points.get_mut([0, 3]).unwrap() = 1.0;
        *points.get_mut([1, 3]).unwrap() = 1.0;
        *points.get_mut([2, 3]).unwrap() = 1.0;
        *points.get_mut([0, 4]).unwrap() = 0.25;
        *points.get_mut([1, 4]).unwrap() = 0.5;
        *points.get_mut([2, 4]).unwrap() = 0.1;
        *points.get_mut([0, 5]).unwrap() = 0.3;
        *points.get_mut([1, 5]).unwrap() = 0.2;
        *points.get_mut([2, 5]).unwrap() = 0.4;

        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            let x = *points.get([0, pt]).unwrap();
            let y = *points.get([1, pt]).unwrap();
            let z = *points.get([2, pt]).unwrap();
            assert_relative_eq!(
                *data.get([pt, 0, 0, 0]).unwrap(),
                (1.0 - x) * (1.0 - y) * (1.0 - z),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([pt, 0, 0, 1]).unwrap(),
                x * (1.0 - y) * (1.0 - z),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([pt, 0, 0, 2]).unwrap(),
                (1.0 - x) * y * (1.0 - z),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([pt, 0, 0, 3]).unwrap(),
                x * y * (1.0 - z),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([pt, 0, 0, 4]).unwrap(),
                (1.0 - x) * (1.0 - y) * z,
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([pt, 0, 0, 5]).unwrap(),
                x * (1.0 - y) * z,
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([pt, 0, 0, 6]).unwrap(),
                (1.0 - x) * y * z,
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([pt, 0, 0, 7]).unwrap(),
                x * y * z,
                epsilon = 1e-14
            );
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_higher_degree_triangle() {
        let _ = create::<f64, f64>(
            ReferenceCellType::Triangle,
            2,
            Continuity::Standard,
            LagrangeVariant::Equispaced,
        );
        let _ = create::<f64, f64>(
            ReferenceCellType::Triangle,
            3,
            Continuity::Standard,
            LagrangeVariant::Equispaced,
        );
        let _ = create::<f64, f64>(
            ReferenceCellType::Triangle,
            4,
            Continuity::Standard,
            LagrangeVariant::Equispaced,
        );
        let _ = create::<f64, f64>(
            ReferenceCellType::Triangle,
            5,
            Continuity::Standard,
            LagrangeVariant::Equispaced,
        );

        let _ = create::<f64, f64>(
            ReferenceCellType::Triangle,
            2,
            Continuity::Discontinuous,
            LagrangeVariant::Equispaced,
        );
        let _ = create::<f64, f64>(
            ReferenceCellType::Triangle,
            3,
            Continuity::Discontinuous,
            LagrangeVariant::Equispaced,
        );
        let _ = create::<f64, f64>(
            ReferenceCellType::Triangle,
            4,
            Continuity::Discontinuous,
            LagrangeVariant::Equispaced,
        );
        let _ = create::<f64, f64>(
            ReferenceCellType::Triangle,
            5,
            Continuity::Discontinuous,
            LagrangeVariant::Equispaced,
        );
    }

    #[test]
    fn test_lagrange_higher_degree_interval() {
        let _ = create::<f64, f64>(
            ReferenceCellType::Interval,
            2,
            Continuity::Standard,
            LagrangeVariant::Equispaced,
        );
        let _ = create::<f64, f64>(
            ReferenceCellType::Interval,
            3,
            Continuity::Standard,
            LagrangeVariant::Equispaced,
        );
        let _ = create::<f64, f64>(
            ReferenceCellType::Interval,
            4,
            Continuity::Standard,
            LagrangeVariant::Equispaced,
        );
        let _ = create::<f64, f64>(
            ReferenceCellType::Interval,
            5,
            Continuity::Standard,
            LagrangeVariant::Equispaced,
        );

        let _ = create::<f64, f64>(
            ReferenceCellType::Interval,
            2,
            Continuity::Discontinuous,
            LagrangeVariant::Equispaced,
        );
        let _ = create::<f64, f64>(
            ReferenceCellType::Interval,
            3,
            Continuity::Discontinuous,
            LagrangeVariant::Equispaced,
        );
        let _ = create::<f64, f64>(
            ReferenceCellType::Interval,
            4,
            Continuity::Discontinuous,
            LagrangeVariant::Equispaced,
        );
        let _ = create::<f64, f64>(
            ReferenceCellType::Interval,
            5,
            Continuity::Discontinuous,
            LagrangeVariant::Equispaced,
        );
    }

    #[test]
    fn test_lagrange_higher_degree_quadrilateral() {
        let _ = create::<f64, f64>(
            ReferenceCellType::Quadrilateral,
            2,
            Continuity::Standard,
            LagrangeVariant::Equispaced,
        );
        let _ = create::<f64, f64>(
            ReferenceCellType::Quadrilateral,
            3,
            Continuity::Standard,
            LagrangeVariant::Equispaced,
        );
        let _ = create::<f64, f64>(
            ReferenceCellType::Quadrilateral,
            4,
            Continuity::Standard,
            LagrangeVariant::Equispaced,
        );
        let _ = create::<f64, f64>(
            ReferenceCellType::Quadrilateral,
            5,
            Continuity::Standard,
            LagrangeVariant::Equispaced,
        );

        let _ = create::<f64, f64>(
            ReferenceCellType::Quadrilateral,
            2,
            Continuity::Discontinuous,
            LagrangeVariant::Equispaced,
        );
        let _ = create::<f64, f64>(
            ReferenceCellType::Quadrilateral,
            3,
            Continuity::Discontinuous,
            LagrangeVariant::Equispaced,
        );
        let _ = create::<f64, f64>(
            ReferenceCellType::Quadrilateral,
            4,
            Continuity::Discontinuous,
            LagrangeVariant::Equispaced,
        );
        let _ = create::<f64, f64>(
            ReferenceCellType::Quadrilateral,
            5,
            Continuity::Discontinuous,
            LagrangeVariant::Equispaced,
        );
    }

    #[test]
    fn test_lagrange_interval_equispaced() {
        let e = create::<f64, f64>(
            ReferenceCellType::Interval,
            4,
            Continuity::Discontinuous,
            LagrangeVariant::Equispaced,
        );

        let mut data = DynArray::<f64, 4>::from_shape(e.tabulate_array_shape(0, 2));
        let mut points = rlst_dynamic_array!(f64, [1, 2]);
        *points.get_mut([0, 0]).unwrap() = 0.25;
        *points.get_mut([0, 1]).unwrap() = 0.75;

        e.tabulate(&points, 0, &mut data);

        for pt in 0..data.shape()[0] {
            for basis in 0..data.shape()[3] {
                assert!(
                    data.get([pt, 0, 0, basis]).unwrap().abs() < 1e-10
                        || (1.0 - data.get([pt, 0, 0, basis]).unwrap()).abs() < 1e-14
                );
            }
        }
    }

    #[test]
    fn test_lagrange_interval_gll() {
        let e = create::<f64, f64>(
            ReferenceCellType::Interval,
            4,
            Continuity::Discontinuous,
            LagrangeVariant::Gll,
        );

        let mut data = DynArray::<f64, 4>::from_shape(e.tabulate_array_shape(0, 2));
        let mut points = rlst_dynamic_array!(f64, [1, 2]);
        *points.get_mut([0, 0]).unwrap() = 0.25;
        *points.get_mut([0, 1]).unwrap() = 0.75;

        e.tabulate(&points, 0, &mut data);

        for pt in 0..data.shape()[0] {
            for basis in 0..data.shape()[3] {
                assert!(data.get([pt, 0, 0, basis]).unwrap().abs() > 1e-10);
            }
        }
    }

    #[test]
    fn test_lagrange_quadrilateral_equispaced() {
        let e = create::<f64, f64>(
            ReferenceCellType::Quadrilateral,
            4,
            Continuity::Discontinuous,
            LagrangeVariant::Equispaced,
        );

        let mut data = DynArray::<f64, 4>::from_shape(e.tabulate_array_shape(0, 4));
        let mut points = rlst_dynamic_array!(f64, [2, 4]);
        *points.get_mut([0, 0]).unwrap() = 0.25;
        *points.get_mut([1, 0]).unwrap() = 0.25;
        *points.get_mut([0, 1]).unwrap() = 0.25;
        *points.get_mut([1, 1]).unwrap() = 0.75;
        *points.get_mut([0, 2]).unwrap() = 0.75;
        *points.get_mut([1, 2]).unwrap() = 0.25;
        *points.get_mut([0, 3]).unwrap() = 0.75;
        *points.get_mut([1, 3]).unwrap() = 0.75;

        e.tabulate(&points, 0, &mut data);

        for pt in 0..data.shape()[0] {
            for basis in 0..data.shape()[3] {
                assert!(
                    data.get([pt, 0, 0, basis]).unwrap().abs() < 1e-10
                        || (1.0 - data.get([pt, 0, 0, basis]).unwrap()).abs() < 1e-14
                );
            }
        }
    }

    #[test]
    fn test_lagrange_quadrilateral_gll() {
        let e = create::<f64, f64>(
            ReferenceCellType::Quadrilateral,
            4,
            Continuity::Discontinuous,
            LagrangeVariant::Gll,
        );

        let mut data = DynArray::<f64, 4>::from_shape(e.tabulate_array_shape(0, 4));
        let mut points = rlst_dynamic_array!(f64, [2, 4]);
        *points.get_mut([0, 0]).unwrap() = 0.25;
        *points.get_mut([1, 0]).unwrap() = 0.25;
        *points.get_mut([0, 1]).unwrap() = 0.25;
        *points.get_mut([1, 1]).unwrap() = 0.75;
        *points.get_mut([0, 2]).unwrap() = 0.75;
        *points.get_mut([1, 2]).unwrap() = 0.25;
        *points.get_mut([0, 3]).unwrap() = 0.75;
        *points.get_mut([1, 3]).unwrap() = 0.75;

        e.tabulate(&points, 0, &mut data);

        for pt in 0..data.shape()[0] {
            for basis in 0..data.shape()[3] {
                assert!(data.get([pt, 0, 0, basis]).unwrap().abs() > 1e-10);
            }
        }
    }

    fn ordered(x: &[f64], y: &[f64]) -> bool {
        assert_eq!(x.len(), y.len());
        if x.is_empty() {
            false
        } else if (x[x.len() - 1] - y[y.len() - 1]).abs() < 1e-8 {
            ordered(&x[..x.len() - 1], &y[..y.len() - 1])
        } else {
            x[x.len() - 1] < y[y.len() - 1]
        }
    }

    macro_rules! test_point_ordering {
        ($cell:ident, $max_degree:expr) => {
            paste! {
                #[test]
                fn [<test_point_ordering_ $cell:lower>]() {
                    for degree in 3..=[<$max_degree>] {
                        let e = create::<f64, f64>(
                            ReferenceCellType::[<$cell>],
                            degree,
                            Continuity::Standard,
                            LagrangeVariant::Equispaced,
                        );
                        for pt_sets in &e.interpolation_points {
                            for pts in pt_sets {
                                if pts.shape()[1] > 0 {
                                    for i in 0..pts.shape()[1] - 1 {
                                        assert!(ordered(&pts.r().col(i).data().unwrap(), &pts.r().col(i + 1).data().unwrap()));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };
    }

    test_point_ordering!(Triangle, 7);
    test_point_ordering!(Quadrilateral, 7);
    test_point_ordering!(Tetrahedron, 6);
    test_point_ordering!(Hexahedron, 3);

    #[test]
    fn test_tabulate_array_shape() {
        use ReferenceCellType::{Hexahedron, Interval, Quadrilateral, Tetrahedron, Triangle};
        let npts = 5;

        for (cell, degree, nderivs, shape) in [
            (Interval, 3, 0, [npts, 1, 1, 4]),
            (Interval, 3, 2, [npts, 1, 3, 4]),
            (Triangle, 2, 0, [npts, 1, 1, 6]),
            (Triangle, 2, 1, [npts, 1, 3, 6]),
            (Triangle, 2, 2, [npts, 1, 6, 6]),
            (Triangle, 2, 3, [npts, 1, 10, 6]),
            (Quadrilateral, 2, 1, [npts, 1, 3, 9]),
            (Quadrilateral, 2, 2, [npts, 1, 6, 9]),
            (Tetrahedron, 2, 1, [npts, 1, 4, 10]),
            (Tetrahedron, 2, 2, [npts, 1, 10, 10]),
            (Hexahedron, 1, 1, [npts, 1, 4, 8]),
            (Hexahedron, 1, 2, [npts, 1, 10, 8]),
        ] {
            assert_eq!(
                create::<f64, f64>(
                    cell,
                    degree,
                    Continuity::Standard,
                    LagrangeVariant::Equispaced
                )
                .tabulate_array_shape(nderivs, npts),
                shape,
                "Lagrange {cell:?} {degree} {nderivs}"
            );
        }
    }

    #[test]
    fn test_tabulate_lagrange_2_triangle_derivatives() {
        let cell = ReferenceCellType::Triangle;
        let e = create::<f64, f64>(cell, 2, Continuity::Standard, LagrangeVariant::Equispaced);
        let points = interior_points(cell);
        let data = tabulate(&e, &points, 3);

        for p in 0..points.shape()[1] {
            let x = points[[0, p]];
            let y = points[[1, p]];
            // Basis functions from DefElement
            // (https://defelement.org/elements/examples/triangle-lagrange-equispaced-2.html)
            // listed as [f, df/dx, df/dy, d2f/dx2, d2f/dxdy, d2f/dy2]
            let expected = [
                [
                    (x + y - 1.0) * (2.0 * x + 2.0 * y - 1.0),
                    4.0 * x + 4.0 * y - 3.0,
                    4.0 * x + 4.0 * y - 3.0,
                    4.0,
                    4.0,
                    4.0,
                ],
                [x * (2.0 * x - 1.0), 4.0 * x - 1.0, 0.0, 4.0, 0.0, 0.0],
                [y * (2.0 * y - 1.0), 0.0, 4.0 * y - 1.0, 0.0, 0.0, 4.0],
                [
                    4.0 * x * (1.0 - x - y),
                    4.0 - 8.0 * x - 4.0 * y,
                    -4.0 * x,
                    -8.0,
                    -4.0,
                    0.0,
                ],
                [
                    4.0 * y * (1.0 - x - y),
                    -4.0 * y,
                    4.0 - 4.0 * x - 8.0 * y,
                    0.0,
                    -4.0,
                    -8.0,
                ],
                [4.0 * x * y, 4.0 * y, 4.0 * x, 0.0, 4.0, 0.0],
            ];
            let alphas = [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]];
            for (b, values) in expected.iter().enumerate() {
                for (alpha, value) in alphas.iter().zip(values) {
                    assert_relative_eq!(
                        data[[p, 0, derivative_index(cell, alpha), b]],
                        *value,
                        epsilon = 1e-12
                    );
                }
                // All third derivatives vanish
                for alpha in [[3, 0], [2, 1], [1, 2], [0, 3]] {
                    assert_relative_eq!(
                        data[[p, 0, derivative_index(cell, &alpha), b]],
                        0.0,
                        epsilon = 1e-10
                    );
                }
            }
        }
    }

    /// Check the degree 1 Lagrange element on an interval, quadrilateral or hexahedron.
    ///
    /// The basis function for the vertex $v$ is the product over the coordinates $x_i$ of
    /// $x_i$ (if $v_i = 1$) or $1 - x_i$ (if $v_i = 0$). This checks the mixed derivatives,
    /// which are the only non-zero higher order derivatives.
    fn check_tensor_product_lagrange_1(cell: ReferenceCellType) {
        let tdim = reference_cell::dim(cell);
        let e = create::<f64, f64>(cell, 1, Continuity::Standard, LagrangeVariant::Equispaced);
        let points = interior_points(cell);
        let nderivs = tdim + 1;
        let data = tabulate(&e, &points, nderivs);
        let vertices = reference_cell::vertices::<f64>(cell);
        assert_eq!(e.dim(), vertices.len());

        for p in 0..points.shape()[1] {
            for (v, vertex) in vertices.iter().enumerate() {
                for alpha in multi_indices(tdim, nderivs) {
                    let expected = (0..tdim)
                        .map(|i| {
                            let x = points[[i, p]];
                            match (alpha[i], vertex[i] == 1.0) {
                                (0, true) => x,
                                (0, false) => 1.0 - x,
                                (1, true) => 1.0,
                                (1, false) => -1.0,
                                _ => 0.0,
                            }
                        })
                        .product::<f64>();
                    assert_relative_eq!(
                        data[[p, 0, derivative_index(cell, &alpha), v]],
                        expected,
                        epsilon = 1e-12
                    );
                }
            }
        }
    }

    #[test]
    fn test_tabulate_lagrange_1_interval_derivatives() {
        check_tensor_product_lagrange_1(ReferenceCellType::Interval);
    }

    #[test]
    fn test_tabulate_lagrange_1_quadrilateral_derivatives() {
        check_tensor_product_lagrange_1(ReferenceCellType::Quadrilateral);
    }

    #[test]
    fn test_tabulate_lagrange_1_hexahedron_derivatives() {
        check_tensor_product_lagrange_1(ReferenceCellType::Hexahedron);
    }

    #[test]
    fn test_tabulate_layout_lagrange() {
        for (cell, max_degree) in [
            (ReferenceCellType::Interval, 4),
            (ReferenceCellType::Triangle, 3),
            (ReferenceCellType::Quadrilateral, 3),
            (ReferenceCellType::Tetrahedron, 3),
            (ReferenceCellType::Hexahedron, 2),
        ] {
            for degree in 1..=max_degree {
                let e = create::<f64, f64>(
                    cell,
                    degree,
                    Continuity::Standard,
                    LagrangeVariant::Equispaced,
                );
                check_tabulate_layout(&e, 3);
            }
        }
    }

    #[test]
    fn test_tabulate_layout_discontinuous_lagrange_0() {
        for cell in [
            ReferenceCellType::Interval,
            ReferenceCellType::Triangle,
            ReferenceCellType::Quadrilateral,
            ReferenceCellType::Tetrahedron,
            ReferenceCellType::Hexahedron,
        ] {
            let e = create::<f64, f64>(
                cell,
                0,
                Continuity::Discontinuous,
                LagrangeVariant::Equispaced,
            );
            check_tabulate_layout(&e, 2);
        }
    }
}
