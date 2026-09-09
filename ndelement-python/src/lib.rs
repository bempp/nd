//! ndelement Python wrapper

#![cfg_attr(feature = "strict", deny(warnings), deny(unused_crate_dependencies))]
#![warn(missing_docs)]

#[cfg(any(target_os = "macos", target_os = "linux"))]
extern crate blas_src;
#[cfg(any(target_os = "macos", target_os = "linux"))]
extern crate lapack_src;
#[cfg(target_os = "linux")]
extern crate openblas_src;
use pyo3::pymodule;
mod element_wrapper;
mod utils;

/// A Python module implemented in Rust. The name of this module must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule(name = "_ndelementrs")]
mod _ndelementrs {
    use pyo3::prelude::pymodule;

    #[pymodule]
    mod types {
        use pyo3::prelude::pyclass;

        use ndelement::ciarlet::LagrangeVariant as LV;
        use ndelement::types::Continuity as Cty;
        use ndelement::types::ReferenceCellType as RCT;

        #[pyclass(from_py_object)]
        #[derive(Clone)]
        pub enum ReferenceCellType {
            Point,
            Interval,
            Triangle,
            Quadrilateral,
            Tetrahedron,
            Hexahedron,
            Prism,
            Pyramid,
        }
        impl ReferenceCellType {
            pub fn as_nd(&self) -> RCT {
                match self {
                    ReferenceCellType::Point => RCT::Point,
                    ReferenceCellType::Interval => RCT::Interval,
                    ReferenceCellType::Triangle => RCT::Triangle,
                    ReferenceCellType::Quadrilateral => RCT::Quadrilateral,
                    ReferenceCellType::Tetrahedron => RCT::Tetrahedron,
                    ReferenceCellType::Hexahedron => RCT::Hexahedron,
                    ReferenceCellType::Prism => RCT::Prism,
                    ReferenceCellType::Pyramid => RCT::Pyramid,
                }
            }
            pub fn from(rct: RCT) -> Self {
                match rct {
                    RCT::Point => ReferenceCellType::Point,
                    RCT::Interval => ReferenceCellType::Interval,
                    RCT::Triangle => ReferenceCellType::Triangle,
                    RCT::Quadrilateral => ReferenceCellType::Quadrilateral,
                    RCT::Tetrahedron => ReferenceCellType::Tetrahedron,
                    RCT::Hexahedron => ReferenceCellType::Hexahedron,
                    RCT::Prism => ReferenceCellType::Prism,
                    RCT::Pyramid => ReferenceCellType::Pyramid,
                }
            }
        }

        #[pyclass(from_py_object)]
        #[derive(Clone)]
        pub enum Continuity {
            Standard,
            Discontinuous,
        }

        impl Continuity {
            pub fn as_nd(&self) -> Cty {
                match self {
                    Continuity::Standard => Cty::Standard,
                    Continuity::Discontinuous => Cty::Discontinuous,
                }
            }
            pub fn from(c: Cty) -> Self {
                match c {
                    Cty::Standard => Continuity::Standard,
                    Cty::Discontinuous => Continuity::Discontinuous,
                }
            }
        }

        #[pyclass(from_py_object)]
        #[derive(Clone)]
        pub enum LagrangeVariant {
            Equispaced,
            Gll,
        }

        impl LagrangeVariant {
            pub fn as_nd(&self) -> LV {
                match self {
                    LagrangeVariant::Equispaced => LV::Equispaced,
                    LagrangeVariant::Gll => LV::Gll,
                }
            }
        }
    }

    #[pymodule]
    mod reference_cell {
        use super::types::ReferenceCellType;
        use pyo3::prelude::pyfunction;

        use ndelement::{orientation, reference_cell};

        #[pyfunction]
        fn dim(cell: ReferenceCellType) -> usize {
            reference_cell::dim(cell.as_nd())
        }

        #[pyfunction]
        fn is_simplex(cell: ReferenceCellType) -> bool {
            reference_cell::is_simplex(cell.as_nd())
        }

        #[pyfunction]
        fn vertices(cell: ReferenceCellType) -> Vec<Vec<f64>> {
            reference_cell::vertices::<f64>(cell.as_nd())
        }

        #[pyfunction]
        fn midpoint(cell: ReferenceCellType) -> Vec<f64> {
            reference_cell::midpoint::<f64>(cell.as_nd())
        }

        #[pyfunction]
        fn edges(cell: ReferenceCellType) -> Vec<Vec<usize>> {
            reference_cell::edges(cell.as_nd())
        }

        #[pyfunction]
        fn faces(cell: ReferenceCellType) -> Vec<Vec<usize>> {
            reference_cell::faces(cell.as_nd())
        }

        #[pyfunction]
        fn volumes(cell: ReferenceCellType) -> Vec<Vec<usize>> {
            reference_cell::volumes(cell.as_nd())
        }

        #[pyfunction]
        fn entity_types(cell: ReferenceCellType) -> Vec<Vec<ReferenceCellType>> {
            reference_cell::entity_types(cell.as_nd())
                .iter()
                .map(|ts| {
                    ts.iter()
                        .map(|t| ReferenceCellType::from(*t))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        }

        #[pyfunction]
        fn entity_counts(cell: ReferenceCellType) -> Vec<usize> {
            reference_cell::entity_counts(cell.as_nd())
        }

        #[pyfunction]
        fn connectivity(cell: ReferenceCellType) -> Vec<Vec<Vec<Vec<usize>>>> {
            reference_cell::connectivity(cell.as_nd())
        }

        #[pyfunction]
        fn compute_orientation(cell: ReferenceCellType, vertices: Vec<usize>) -> i32 {
            orientation::compute_orientation(cell.as_nd(), &vertices)
        }
    }

    #[pymodule]
    mod quadrature {
        use pyo3::pyfunction;

        use super::types::ReferenceCellType;
        use ndelement::quadrature;

        #[pyfunction]
        fn make_gauss_jacobi_quadrature(cell: ReferenceCellType, m: usize) -> (Vec<f64>, Vec<f64>) {
            let rule = quadrature::gauss_jacobi_rule(cell.as_nd(), m).unwrap();
            (rule.points, rule.weights)
        }
    }

    #[pymodule]
    mod polynomials {
        use numpy::{IntoPyArray, PyArray3, PyReadonlyArray2, ndarray::Array};
        use pyo3::{Bound, Python, pyfunction};

        use super::types::ReferenceCellType;
        use crate::utils::from_py2;
        use ndelement::polynomials;
        use rlst::SliceArrayMut;

        #[pyfunction]
        fn tabulate_legendre_polynomials<'py>(
            py: Python<'py>,
            cell: ReferenceCellType,
            points: PyReadonlyArray2<'py, f64>,
            degree: usize,
            derivatives: usize,
        ) -> Bound<'py, PyArray3<f64>> {
            let points = from_py2(&points);

            let npts = points.shape()[1];

            let npoly = polynomials::polynomial_count(cell.as_nd(), degree);
            let nderiv = polynomials::derivative_count(cell.as_nd(), derivatives);
            let mut result_array = Array::zeros((npts, npoly, nderiv));
            let mut result = SliceArrayMut::<f64, 3>::from_shape(
                result_array.as_slice_mut().unwrap(),
                [nderiv, npoly, npts],
            );
            polynomials::tabulate_legendre_polynomials(
                cell.as_nd(),
                &points,
                degree,
                derivatives,
                &mut result,
            );
            result_array.into_pyarray(py)
        }
    }

    #[pymodule]
    pub mod ciarlet {
        use numpy::{
            IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyArrayMethods, PyReadonlyArray1,
            PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4, ndarray::Array,
        };
        use pyo3::{Bound, Python, pyclass, pyfunction, pymethods};

        use super::types::{Continuity, LagrangeVariant, ReferenceCellType};
        use crate::element_wrapper::{WrappedElement, WrappedElementFamily};
        use crate::utils::{from_py2, from_py3, from_py4, to_py2, to_py3};
        use ndelement::ciarlet::{
            LagrangeElementFamily, NedelecFirstKindElementFamily, RaviartThomasElementFamily,
        };
        use ndelement::traits::ElementFamily as EF;
        use ndelement::traits::{FiniteElement, MappedFiniteElement};
        use rlst::SliceArrayMut;

        #[pyclass]
        struct ElementFamily {
            ef: WrappedElementFamily,
        }

        impl ElementFamily {
            fn new(ef: WrappedElementFamily) -> Self {
                Self { ef }
            }
        }

        #[pymethods]
        impl ElementFamily {
            fn element(&self, cell: ReferenceCellType) -> CiarletElement {
                CiarletElement::new(self.ef.element(cell.as_nd()))
            }
        }

        #[pyclass]
        struct CiarletElement {
            e: WrappedElement,
        }

        impl CiarletElement {
            fn new(e: WrappedElement) -> Self {
                Self { e }
            }
        }

        #[pymethods]
        impl CiarletElement {
            fn value_size(&self) -> usize {
                self.e.value_size()
            }

            fn value_shape(&self) -> &[usize] {
                self.e.value_shape()
            }

            fn degree(&self) -> usize {
                self.e.degree()
            }

            fn lagrange_superdegree(&self) -> usize {
                self.e.lagrange_superdegree()
            }

            fn dim(&self) -> usize {
                self.e.dim()
            }

            fn continuity(&self) -> Continuity {
                Continuity::from(self.e.continuity())
            }

            fn cell_type(&self) -> ReferenceCellType {
                ReferenceCellType::from(self.e.cell_type())
            }

            fn entity_dofs(&self, entity_dim: usize, entity_number: usize) -> &[usize] {
                if let Some(e) = self.e.entity_dofs(entity_dim, entity_number) {
                    e
                } else {
                    &[]
                }
            }

            fn entity_closure_dofs(&self, entity_dim: usize, entity_number: usize) -> &[usize] {
                if let Some(e) = self.e.entity_closure_dofs(entity_dim, entity_number) {
                    e
                } else {
                    &[]
                }
            }

            fn physical_value_size(&self, gdim: usize) -> usize {
                self.e.physical_value_size(gdim)
            }

            fn physical_value_shape(&self, gdim: usize) -> Vec<usize> {
                self.e.physical_value_shape(gdim)
            }

            fn tabulate_array_shape(&self, nderivs: usize, npoints: usize) -> [usize; 4] {
                self.e.tabulate_array_shape(nderivs, npoints)
            }

            fn tabulate<'py>(
                &self,
                py: Python<'py>,
                points: PyReadonlyArray2<'py, f64>,
                nderivs: usize,
            ) -> Bound<'py, PyArray4<f64>> {
                let points = from_py2(&points);
                let npts = points.shape()[1];
                let shape = self.tabulate_array_shape(nderivs, npts);
                let mut result_array = Array::zeros((shape[3], shape[2], shape[1], shape[0]));
                let mut result = SliceArrayMut::<f64, 4>::from_shape(
                    result_array.as_slice_mut().unwrap(),
                    shape,
                );
                self.e.tabulate(&points, nderivs, &mut result);
                result_array.into_pyarray(py)
            }

            fn interpolation_points<'py>(
                &self,
                py: Python<'py>,
            ) -> [Vec<Bound<'py, PyArray2<f64>>>; 4] {
                let ip = self.e.interpolation_points();
                [
                    ip[0].iter().map(|a| to_py2(py, a)).collect::<Vec<_>>(),
                    ip[1].iter().map(|a| to_py2(py, a)).collect::<Vec<_>>(),
                    ip[2].iter().map(|a| to_py2(py, a)).collect::<Vec<_>>(),
                    ip[3].iter().map(|a| to_py2(py, a)).collect::<Vec<_>>(),
                ]
            }

            fn interpolation_weights<'py>(
                &self,
                py: Python<'py>,
            ) -> [Vec<Bound<'py, PyArray3<f64>>>; 4] {
                let ip = self.e.interpolation_weights();
                [
                    ip[0].iter().map(|a| to_py3(py, a)).collect::<Vec<_>>(),
                    ip[1].iter().map(|a| to_py3(py, a)).collect::<Vec<_>>(),
                    ip[2].iter().map(|a| to_py3(py, a)).collect::<Vec<_>>(),
                    ip[3].iter().map(|a| to_py3(py, a)).collect::<Vec<_>>(),
                ]
            }

            fn push_forward<'py>(
                &self,
                py: Python<'py>,
                reference_values: PyReadonlyArray4<'py, f64>,
                nderivs: usize,
                jacobians: PyReadonlyArray3<'py, f64>,
                jacobian_determinants: PyReadonlyArray1<'py, f64>,
                inverse_jacobians: PyReadonlyArray3<'py, f64>,
            ) -> Bound<'py, PyArray4<f64>> {
                let reference_values = from_py4(&reference_values);
                let jacobians = from_py3(&jacobians);
                let inverse_jacobians = from_py3(&inverse_jacobians);

                let gdim = jacobians.shape()[0];
                let p_vs = self.physical_value_size(gdim);

                let rshape = reference_values.shape();
                let mut physical_values_array =
                    Array::zeros((p_vs, rshape[2], rshape[1], rshape[0]));
                let mut physical_values = SliceArrayMut::<f64, 4>::from_shape(
                    physical_values_array.as_slice_mut().unwrap(),
                    [rshape[0], rshape[1], rshape[2], p_vs],
                );

                self.e.push_forward(
                    &reference_values,
                    nderivs,
                    &jacobians,
                    jacobian_determinants.as_slice().unwrap(),
                    &inverse_jacobians,
                    &mut physical_values,
                );

                physical_values_array.into_pyarray(py)
            }

            fn pull_back<'py>(
                &self,
                py: Python<'py>,
                physical_values: PyReadonlyArray4<'py, f64>,
                nderivs: usize,
                jacobians: PyReadonlyArray3<'py, f64>,
                jacobian_determinants: PyReadonlyArray1<'py, f64>,
                inverse_jacobians: PyReadonlyArray3<'py, f64>,
            ) -> Bound<'py, PyArray4<f64>> {
                let physical_values = from_py4(&physical_values);
                let jacobians = from_py3(&jacobians);
                let inverse_jacobians = from_py3(&inverse_jacobians);

                let r_vs = self.value_size();

                let pshape = physical_values.shape();
                let mut reference_values_array =
                    Array::zeros((r_vs, pshape[2], pshape[1], pshape[0]));
                let mut reference_values = SliceArrayMut::<f64, 4>::from_shape(
                    reference_values_array.as_slice_mut().unwrap(),
                    [pshape[0], pshape[1], pshape[2], r_vs],
                );

                self.e.pull_back(
                    &physical_values,
                    nderivs,
                    &jacobians,
                    jacobian_determinants.as_slice().unwrap(),
                    &inverse_jacobians,
                    &mut reference_values,
                );

                reference_values_array.into_pyarray(py)
            }

            fn apply_dof_permutations_i64(
                &self,
                data: &Bound<'_, PyArray1<i64>>,
                orientation: i32,
            ) {
                self.e
                    .apply_dof_permutations(unsafe { data.as_slice_mut().unwrap() }, orientation)
            }

            fn apply_dof_permutations_u64(
                &self,
                data: &Bound<'_, PyArray1<u64>>,
                orientation: i32,
            ) {
                self.e
                    .apply_dof_permutations(unsafe { data.as_slice_mut().unwrap() }, orientation)
            }

            fn apply_dof_permutations_f64(
                &self,
                data: &Bound<'_, PyArray1<f64>>,
                orientation: i32,
            ) {
                self.e
                    .apply_dof_permutations(unsafe { data.as_slice_mut().unwrap() }, orientation)
            }

            fn apply_dof_transformations(&self, data: &Bound<'_, PyArray1<f64>>, orientation: i32) {
                self.e
                    .apply_dof_transformations(unsafe { data.as_slice_mut().unwrap() }, orientation)
            }

            fn apply_dof_permutations_and_transformations(
                &self,
                data: &Bound<'_, PyArray1<f64>>,
                orientation: i32,
            ) {
                self.e.apply_dof_permutations_and_transformations(
                    unsafe { data.as_slice_mut().unwrap() },
                    orientation,
                )
            }
        }

        #[pyfunction]
        fn create_lagrange_family(
            degree: usize,
            continuity: Continuity,
            variant: LagrangeVariant,
        ) -> ElementFamily {
            ElementFamily::new(WrappedElementFamily::Lagrange(
                LagrangeElementFamily::<f64>::new(degree, continuity.as_nd(), variant.as_nd()),
            ))
        }

        #[pyfunction]
        fn create_raviart_thomas_family(degree: usize, continuity: Continuity) -> ElementFamily {
            ElementFamily {
                ef: WrappedElementFamily::RaviartThomas(RaviartThomasElementFamily::<f64>::new(
                    degree,
                    continuity.as_nd(),
                )),
            }
        }

        #[pyfunction]
        fn create_nedelec_family(degree: usize, continuity: Continuity) -> ElementFamily {
            ElementFamily {
                ef: WrappedElementFamily::NedelecFirstKind(
                    NedelecFirstKindElementFamily::<f64>::new(degree, continuity.as_nd()),
                ),
            }
        }
    }
}
