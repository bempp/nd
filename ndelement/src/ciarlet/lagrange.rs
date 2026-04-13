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
pub enum Variant {
    /// Element defined using equispaced points
    Equispaced,
    /// Element defined using Guass-Lobatto-Legendre (GLL) points
    GLL,
}

/// Create a Lagrange element.
pub fn create<T: RlstScalar + Getrf + Getri, TGeo: RlstScalar>(
    cell_type: ReferenceCellType,
    degree: usize,
    continuity: Continuity,
    variant: Variant,
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
        if continuity == Continuity::Standard {
            panic!("Cannot create continuous degree 0 Lagrange element");
        }
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
            Variant::Equispaced => (1..degree)
                .map(|i| num::cast::<_, TGeo>(i).unwrap() / num::cast::<_, TGeo>(degree).unwrap())
                .collect::<Vec<_>>(),
            Variant::GLL => {
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
                panic!();
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
                panic!();
            };
            let v0 = &vertices[vn0];
            let v1 = &vertices[vn1];
            let v2 = &vertices[vn2];

            match face_type {
                ReferenceCellType::Triangle => {
                    if variant != Variant::Equispaced {
                        unimplemented!();
                    }
                    let mut n = 0;
                    for i0 in 1..degree {
                        for i1 in 1..degree - i0 {
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
                    for p0 in &pts1d {
                        for p1 in &pts1d {
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
            };

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
                    if variant != Variant::Equispaced {
                        unimplemented!();
                    }
                    let [vn0, vn1, vn2, vn3] = volumes[e][..4] else {
                        panic!();
                    };
                    let v0 = &vertices[vn0];
                    let v1 = &vertices[vn1];
                    let v2 = &vertices[vn2];
                    let v3 = &vertices[vn3];

                    let mut n = 0;
                    for i0 in 1..degree {
                        for i1 in 1..degree - i0 {
                            for i2 in 1..degree - i0 - i1 {
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
                        panic!();
                    };
                    let v0 = &vertices[vn0];
                    let v1 = &vertices[vn1];
                    let v2 = &vertices[vn2];
                    let v3 = &vertices[vn3];

                    let mut n = 0;
                    for p0 in &pts1d {
                        for p1 in &pts1d {
                            for p2 in &pts1d {
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
            };

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
    variant: Variant,
    _t: PhantomData<T>,
    _tgeo: PhantomData<TGeo>,
}

impl<T: RlstScalar + Getrf + Getri, TGeo: RlstScalar> LagrangeElementFamily<T, TGeo> {
    /// Create new family with given `degree` and `continuity`.
    pub fn new(degree: usize, continuity: Continuity, variant: Variant) -> Self {
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
    use super::super::test::check_dofs;
    use super::*;
    use crate::traits::FiniteElement;
    use approx::*;
    use rlst::DynArray;

    #[test]
    fn test_lagrange_1() {
        let e = create::<f64, f64>(
            ReferenceCellType::Triangle,
            1,
            Continuity::Standard,
            Variant::Equispaced,
        );
        assert_eq!(e.value_size(), 1);
    }

    #[test]
    fn test_lagrange_0_interval() {
        let e = create::<f64, f64>(
            ReferenceCellType::Interval,
            0,
            Continuity::Discontinuous,
            Variant::Equispaced,
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
            assert_relative_eq!(*data.get([0, pt, 0, 0]).unwrap(), 1.0);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_1_interval() {
        let e = create::<f64, f64>(
            ReferenceCellType::Interval,
            1,
            Continuity::Standard,
            Variant::Equispaced,
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
            assert_relative_eq!(*data.get([0, pt, 0, 0]).unwrap(), 1.0 - x);
            assert_relative_eq!(*data.get([0, pt, 1, 0]).unwrap(), x);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_0_triangle() {
        let e = create::<f64, f64>(
            ReferenceCellType::Triangle,
            0,
            Continuity::Discontinuous,
            Variant::Equispaced,
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
            assert_relative_eq!(*data.get([0, pt, 0, 0]).unwrap(), 1.0);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_1_triangle() {
        let e = create::<f64, f64>(
            ReferenceCellType::Triangle,
            1,
            Continuity::Standard,
            Variant::Equispaced,
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
            assert_relative_eq!(*data.get([0, pt, 0, 0]).unwrap(), 1.0 - x - y);
            assert_relative_eq!(*data.get([0, pt, 1, 0]).unwrap(), x);
            assert_relative_eq!(*data.get([0, pt, 2, 0]).unwrap(), y);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_0_quadrilateral() {
        let e = create::<f64, f64>(
            ReferenceCellType::Quadrilateral,
            0,
            Continuity::Discontinuous,
            Variant::Equispaced,
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
            assert_relative_eq!(*data.get([0, pt, 0, 0]).unwrap(), 1.0);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_1_quadrilateral() {
        let e = create::<f64, f64>(
            ReferenceCellType::Quadrilateral,
            1,
            Continuity::Standard,
            Variant::Equispaced,
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
            assert_relative_eq!(*data.get([0, pt, 0, 0]).unwrap(), (1.0 - x) * (1.0 - y));
            assert_relative_eq!(*data.get([0, pt, 1, 0]).unwrap(), x * (1.0 - y));
            assert_relative_eq!(*data.get([0, pt, 2, 0]).unwrap(), (1.0 - x) * y);
            assert_relative_eq!(*data.get([0, pt, 3, 0]).unwrap(), x * y);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_2_quadrilateral() {
        let e = create::<f64, f64>(
            ReferenceCellType::Quadrilateral,
            2,
            Continuity::Standard,
            Variant::Equispaced,
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
                *data.get([0, pt, 0, 0]).unwrap(),
                (1.0 - x) * (1.0 - 2.0 * x) * (1.0 - y) * (1.0 - 2.0 * y),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 1, 0]).unwrap(),
                x * (2.0 * x - 1.0) * (1.0 - y) * (1.0 - 2.0 * y),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 2, 0]).unwrap(),
                (1.0 - x) * (1.0 - 2.0 * x) * y * (2.0 * y - 1.0),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 3, 0]).unwrap(),
                x * (2.0 * x - 1.0) * y * (2.0 * y - 1.0),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 4, 0]).unwrap(),
                4.0 * x * (1.0 - x) * (1.0 - y) * (1.0 - 2.0 * y),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 5, 0]).unwrap(),
                (1.0 - x) * (1.0 - 2.0 * x) * 4.0 * y * (1.0 - y),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 6, 0]).unwrap(),
                x * (2.0 * x - 1.0) * 4.0 * y * (1.0 - y),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 7, 0]).unwrap(),
                4.0 * x * (1.0 - x) * y * (2.0 * y - 1.0),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 8, 0]).unwrap(),
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
            Variant::Equispaced,
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
            assert_relative_eq!(*data.get([0, pt, 0, 0]).unwrap(), 1.0);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_1_tetrahedron() {
        let e = create::<f64, f64>(
            ReferenceCellType::Tetrahedron,
            1,
            Continuity::Standard,
            Variant::Equispaced,
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

        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            let x = *points.get([0, pt]).unwrap();
            let y = *points.get([1, pt]).unwrap();
            let z = *points.get([2, pt]).unwrap();
            assert_relative_eq!(
                *data.get([0, pt, 0, 0]).unwrap(),
                1.0 - x - y - z,
                epsilon = 1e-14
            );
            assert_relative_eq!(*data.get([0, pt, 1, 0]).unwrap(), x, epsilon = 1e-14);
            assert_relative_eq!(*data.get([0, pt, 2, 0]).unwrap(), y, epsilon = 1e-14);
            assert_relative_eq!(*data.get([0, pt, 3, 0]).unwrap(), z, epsilon = 1e-14);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_0_hexahedron() {
        let e = create::<f64, f64>(
            ReferenceCellType::Hexahedron,
            0,
            Continuity::Discontinuous,
            Variant::Equispaced,
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
            assert_relative_eq!(*data.get([0, pt, 0, 0]).unwrap(), 1.0);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_1_hexahedron() {
        let e = create::<f64, f64>(
            ReferenceCellType::Hexahedron,
            1,
            Continuity::Standard,
            Variant::Equispaced,
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
                *data.get([0, pt, 0, 0]).unwrap(),
                (1.0 - x) * (1.0 - y) * (1.0 - z),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 1, 0]).unwrap(),
                x * (1.0 - y) * (1.0 - z),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 2, 0]).unwrap(),
                (1.0 - x) * y * (1.0 - z),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 3, 0]).unwrap(),
                x * y * (1.0 - z),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 4, 0]).unwrap(),
                (1.0 - x) * (1.0 - y) * z,
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 5, 0]).unwrap(),
                x * (1.0 - y) * z,
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 6, 0]).unwrap(),
                (1.0 - x) * y * z,
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 7, 0]).unwrap(),
                x * y * z,
                epsilon = 1e-14
            );
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_higher_degree_triangle() {
        create::<f64, f64>(
            ReferenceCellType::Triangle,
            2,
            Continuity::Standard,
            Variant::Equispaced,
        );
        create::<f64, f64>(
            ReferenceCellType::Triangle,
            3,
            Continuity::Standard,
            Variant::Equispaced,
        );
        create::<f64, f64>(
            ReferenceCellType::Triangle,
            4,
            Continuity::Standard,
            Variant::Equispaced,
        );
        create::<f64, f64>(
            ReferenceCellType::Triangle,
            5,
            Continuity::Standard,
            Variant::Equispaced,
        );

        create::<f64, f64>(
            ReferenceCellType::Triangle,
            2,
            Continuity::Discontinuous,
            Variant::Equispaced,
        );
        create::<f64, f64>(
            ReferenceCellType::Triangle,
            3,
            Continuity::Discontinuous,
            Variant::Equispaced,
        );
        create::<f64, f64>(
            ReferenceCellType::Triangle,
            4,
            Continuity::Discontinuous,
            Variant::Equispaced,
        );
        create::<f64, f64>(
            ReferenceCellType::Triangle,
            5,
            Continuity::Discontinuous,
            Variant::Equispaced,
        );
    }

    #[test]
    fn test_lagrange_higher_degree_interval() {
        create::<f64, f64>(
            ReferenceCellType::Interval,
            2,
            Continuity::Standard,
            Variant::Equispaced,
        );
        create::<f64, f64>(
            ReferenceCellType::Interval,
            3,
            Continuity::Standard,
            Variant::Equispaced,
        );
        create::<f64, f64>(
            ReferenceCellType::Interval,
            4,
            Continuity::Standard,
            Variant::Equispaced,
        );
        create::<f64, f64>(
            ReferenceCellType::Interval,
            5,
            Continuity::Standard,
            Variant::Equispaced,
        );

        create::<f64, f64>(
            ReferenceCellType::Interval,
            2,
            Continuity::Discontinuous,
            Variant::Equispaced,
        );
        create::<f64, f64>(
            ReferenceCellType::Interval,
            3,
            Continuity::Discontinuous,
            Variant::Equispaced,
        );
        create::<f64, f64>(
            ReferenceCellType::Interval,
            4,
            Continuity::Discontinuous,
            Variant::Equispaced,
        );
        create::<f64, f64>(
            ReferenceCellType::Interval,
            5,
            Continuity::Discontinuous,
            Variant::Equispaced,
        );
    }

    #[test]
    fn test_lagrange_higher_degree_quadrilateral() {
        create::<f64, f64>(
            ReferenceCellType::Quadrilateral,
            2,
            Continuity::Standard,
            Variant::Equispaced,
        );
        create::<f64, f64>(
            ReferenceCellType::Quadrilateral,
            3,
            Continuity::Standard,
            Variant::Equispaced,
        );
        create::<f64, f64>(
            ReferenceCellType::Quadrilateral,
            4,
            Continuity::Standard,
            Variant::Equispaced,
        );
        create::<f64, f64>(
            ReferenceCellType::Quadrilateral,
            5,
            Continuity::Standard,
            Variant::Equispaced,
        );

        create::<f64, f64>(
            ReferenceCellType::Quadrilateral,
            2,
            Continuity::Discontinuous,
            Variant::Equispaced,
        );
        create::<f64, f64>(
            ReferenceCellType::Quadrilateral,
            3,
            Continuity::Discontinuous,
            Variant::Equispaced,
        );
        create::<f64, f64>(
            ReferenceCellType::Quadrilateral,
            4,
            Continuity::Discontinuous,
            Variant::Equispaced,
        );
        create::<f64, f64>(
            ReferenceCellType::Quadrilateral,
            5,
            Continuity::Discontinuous,
            Variant::Equispaced,
        );
    }

    #[test]
    fn test_lagrange_interval_equispaced() {
        let e = create::<f64, f64>(
            ReferenceCellType::Interval,
            4,
            Continuity::Discontinuous,
            Variant::Equispaced,
        );

        let mut data = DynArray::<f64, 4>::from_shape(e.tabulate_array_shape(0, 2));
        let mut points = rlst_dynamic_array!(f64, [1, 2]);
        *points.get_mut([0, 0]).unwrap() = 0.25;
        *points.get_mut([0, 1]).unwrap() = 0.75;

        e.tabulate(&points, 0, &mut data);

        for pt in 0..data.shape()[1] {
            for basis in 0..data.shape()[1] {
                assert!(
                    data.get([0, pt, basis, 0]).unwrap().abs() < 1e-10
                        || (1.0 - data.get([0, pt, basis, 0]).unwrap()).abs() < 1e-14
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
            Variant::GLL,
        );

        let mut data = DynArray::<f64, 4>::from_shape(e.tabulate_array_shape(0, 2));
        let mut points = rlst_dynamic_array!(f64, [1, 2]);
        *points.get_mut([0, 0]).unwrap() = 0.25;
        *points.get_mut([0, 1]).unwrap() = 0.75;

        e.tabulate(&points, 0, &mut data);

        for pt in 0..data.shape()[1] {
            for basis in 0..data.shape()[1] {
                assert!(data.get([0, pt, basis, 0]).unwrap().abs() > 1e-10);
            }
        }
    }

    #[test]
    fn test_lagrange_quadrilateral_equispaced() {
        let e = create::<f64, f64>(
            ReferenceCellType::Quadrilateral,
            4,
            Continuity::Discontinuous,
            Variant::Equispaced,
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

        for pt in 0..data.shape()[1] {
            for basis in 0..data.shape()[1] {
                assert!(
                    data.get([0, pt, basis, 0]).unwrap().abs() < 1e-10
                        || (1.0 - data.get([0, pt, basis, 0]).unwrap()).abs() < 1e-14
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
            Variant::GLL,
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

        for pt in 0..data.shape()[1] {
            for basis in 0..data.shape()[1] {
                assert!(data.get([0, pt, basis, 0]).unwrap().abs() > 1e-10);
            }
        }
    }
}
