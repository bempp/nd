//! Wrappers for finite elements and families

use ndelement::ciarlet::CiarletElement;
use ndelement::traits::{ElementFamily, FiniteElement, MappedFiniteElement};
use ndelement::types::{
    Continuity, DofTransformation, EntityPoints, EntityWeights, ReferenceCellType, Transformation,
};
use ndelement::{ciarlet, map};
use rlst::{Array, MutableArrayImpl, ValueArrayImpl};

/// A finite element family wrapped for Python
pub enum WrappedElementFamily {
    Lagrange(ciarlet::LagrangeElementFamily<f64>),
    RaviartThomas(ciarlet::RaviartThomasElementFamily<f64>),
    NedelecFirstKind(ciarlet::NedelecFirstKindElementFamily<f64>),
}

impl ElementFamily for WrappedElementFamily {
    type T = f64;
    type CellType = ReferenceCellType;
    type FiniteElement = WrappedElement;

    fn element(&self, cell_type: ReferenceCellType) -> WrappedElement {
        match self {
            WrappedElementFamily::Lagrange(a) => WrappedElement::Identity(a.element(cell_type)),
            WrappedElementFamily::RaviartThomas(a) => {
                WrappedElement::Contravariant(a.element(cell_type))
            }
            WrappedElementFamily::NedelecFirstKind(a) => {
                WrappedElement::Covariant(a.element(cell_type))
            }
        }
    }
}

/// A finite element wrapped for Python
pub enum WrappedElement {
    Identity(CiarletElement<f64, map::IdentityMap, f64>),
    Covariant(CiarletElement<f64, map::CovariantPiolaMap, f64>),
    Contravariant(CiarletElement<f64, map::ContravariantPiolaMap, f64>),
}

impl WrappedElement {
    pub fn degree(&self) -> usize {
        match self {
            WrappedElement::Identity(a) => a.degree(),
            WrappedElement::Covariant(a) => a.degree(),
            WrappedElement::Contravariant(a) => a.degree(),
        }
    }

    pub fn continuity(&self) -> Continuity {
        match self {
            WrappedElement::Identity(a) => a.continuity(),
            WrappedElement::Covariant(a) => a.continuity(),
            WrappedElement::Contravariant(a) => a.continuity(),
        }
    }
}

impl FiniteElement for WrappedElement {
    type T = f64;
    type TGeo = f64;
    type CellType = ReferenceCellType;

    fn cell_type(&self) -> ReferenceCellType {
        match self {
            WrappedElement::Identity(a) => a.cell_type(),
            WrappedElement::Covariant(a) => a.cell_type(),
            WrappedElement::Contravariant(a) => a.cell_type(),
        }
    }

    fn dim(&self) -> usize {
        match self {
            WrappedElement::Identity(a) => a.dim(),
            WrappedElement::Covariant(a) => a.dim(),
            WrappedElement::Contravariant(a) => a.dim(),
        }
    }

    fn value_shape(&self) -> &[usize] {
        match self {
            WrappedElement::Identity(a) => a.value_shape(),
            WrappedElement::Covariant(a) => a.value_shape(),
            WrappedElement::Contravariant(a) => a.value_shape(),
        }
    }

    fn value_size(&self) -> usize {
        match self {
            WrappedElement::Identity(a) => a.value_size(),
            WrappedElement::Covariant(a) => a.value_size(),
            WrappedElement::Contravariant(a) => a.value_size(),
        }
    }

    fn tabulate<Array2Impl: ValueArrayImpl<f64, 2>, Array4MutImpl: MutableArrayImpl<f64, 4>>(
        &self,
        points: &Array<Array2Impl, 2>,
        nderivs: usize,
        data: &mut Array<Array4MutImpl, 4>,
    ) {
        match self {
            WrappedElement::Identity(a) => a.tabulate(points, nderivs, data),
            WrappedElement::Covariant(a) => a.tabulate(points, nderivs, data),
            WrappedElement::Contravariant(a) => a.tabulate(points, nderivs, data),
        }
    }

    fn tabulate_array_shape(&self, nderivs: usize, npoints: usize) -> [usize; 4] {
        match self {
            WrappedElement::Identity(a) => a.tabulate_array_shape(nderivs, npoints),
            WrappedElement::Covariant(a) => a.tabulate_array_shape(nderivs, npoints),
            WrappedElement::Contravariant(a) => a.tabulate_array_shape(nderivs, npoints),
        }
    }

    fn entity_dofs(&self, entity_dim: usize, entity_number: usize) -> Option<&[usize]> {
        match self {
            WrappedElement::Identity(a) => a.entity_dofs(entity_dim, entity_number),
            WrappedElement::Covariant(a) => a.entity_dofs(entity_dim, entity_number),
            WrappedElement::Contravariant(a) => a.entity_dofs(entity_dim, entity_number),
        }
    }

    fn entity_closure_dofs(&self, entity_dim: usize, entity_number: usize) -> Option<&[usize]> {
        match self {
            WrappedElement::Identity(a) => a.entity_closure_dofs(entity_dim, entity_number),
            WrappedElement::Covariant(a) => a.entity_closure_dofs(entity_dim, entity_number),
            WrappedElement::Contravariant(a) => a.entity_closure_dofs(entity_dim, entity_number),
        }
    }
}

impl MappedFiniteElement for WrappedElement {
    type TransformationType = Transformation;

    fn lagrange_superdegree(&self) -> usize {
        match self {
            WrappedElement::Identity(a) => a.lagrange_superdegree(),
            WrappedElement::Covariant(a) => a.lagrange_superdegree(),
            WrappedElement::Contravariant(a) => a.lagrange_superdegree(),
        }
    }

    fn push_forward<
        Array3GeoImpl: ValueArrayImpl<f64, 3>,
        Array4Impl: ValueArrayImpl<f64, 4>,
        Array4MutImpl: MutableArrayImpl<f64, 4>,
    >(
        &self,
        reference_values: &Array<Array4Impl, 4>,
        nderivs: usize,
        jacobians: &Array<Array3GeoImpl, 3>,
        jacobian_determinants: &[f64],
        inverse_jacobians: &Array<Array3GeoImpl, 3>,
        physical_values: &mut Array<Array4MutImpl, 4>,
    ) {
        match self {
            WrappedElement::Identity(a) => a.push_forward(
                reference_values,
                nderivs,
                jacobians,
                jacobian_determinants,
                inverse_jacobians,
                physical_values,
            ),
            WrappedElement::Covariant(a) => a.push_forward(
                reference_values,
                nderivs,
                jacobians,
                jacobian_determinants,
                inverse_jacobians,
                physical_values,
            ),
            WrappedElement::Contravariant(a) => a.push_forward(
                reference_values,
                nderivs,
                jacobians,
                jacobian_determinants,
                inverse_jacobians,
                physical_values,
            ),
        }
    }

    fn pull_back<
        Array3GeoImpl: ValueArrayImpl<f64, 3>,
        Array4Impl: ValueArrayImpl<f64, 4>,
        Array4MutImpl: MutableArrayImpl<f64, 4>,
    >(
        &self,
        physical_values: &Array<Array4Impl, 4>,
        nderivs: usize,
        jacobians: &Array<Array3GeoImpl, 3>,
        jacobian_determinants: &[f64],
        inverse_jacobians: &Array<Array3GeoImpl, 3>,
        reference_values: &mut Array<Array4MutImpl, 4>,
    ) {
        match self {
            WrappedElement::Identity(a) => a.pull_back(
                physical_values,
                nderivs,
                jacobians,
                jacobian_determinants,
                inverse_jacobians,
                reference_values,
            ),
            WrappedElement::Covariant(a) => a.pull_back(
                physical_values,
                nderivs,
                jacobians,
                jacobian_determinants,
                inverse_jacobians,
                reference_values,
            ),
            WrappedElement::Contravariant(a) => a.pull_back(
                physical_values,
                nderivs,
                jacobians,
                jacobian_determinants,
                inverse_jacobians,
                reference_values,
            ),
        }
    }

    fn physical_value_shape(&self, gdim: usize) -> Vec<usize> {
        match self {
            WrappedElement::Identity(a) => a.physical_value_shape(gdim),
            WrappedElement::Covariant(a) => a.physical_value_shape(gdim),
            WrappedElement::Contravariant(a) => a.physical_value_shape(gdim),
        }
    }

    fn dof_transformation(
        &self,
        entity: ReferenceCellType,
        transformation: Transformation,
    ) -> Option<&DofTransformation<f64>> {
        match self {
            WrappedElement::Identity(a) => a.dof_transformation(entity, transformation),
            WrappedElement::Covariant(a) => a.dof_transformation(entity, transformation),
            WrappedElement::Contravariant(a) => a.dof_transformation(entity, transformation),
        }
    }

    fn apply_dof_permutations<T>(&self, data: &mut [T], cell_orientation: i32) {
        match self {
            WrappedElement::Identity(a) => a.apply_dof_permutations(data, cell_orientation),
            WrappedElement::Covariant(a) => a.apply_dof_permutations(data, cell_orientation),
            WrappedElement::Contravariant(a) => a.apply_dof_permutations(data, cell_orientation),
        }
    }

    fn apply_dof_transformations(&self, data: &mut [f64], cell_orientation: i32) {
        match self {
            WrappedElement::Identity(a) => a.apply_dof_transformations(data, cell_orientation),
            WrappedElement::Covariant(a) => a.apply_dof_transformations(data, cell_orientation),
            WrappedElement::Contravariant(a) => a.apply_dof_transformations(data, cell_orientation),
        }
    }

    /// The interpolation points
    fn interpolation_points(&self) -> &EntityPoints<f64> {
        match self {
            WrappedElement::Identity(a) => a.interpolation_points(),
            WrappedElement::Covariant(a) => a.interpolation_points(),
            WrappedElement::Contravariant(a) => a.interpolation_points(),
        }
    }

    /// The interpolation weights
    fn interpolation_weights(&self) -> &EntityWeights<f64> {
        match self {
            WrappedElement::Identity(a) => a.interpolation_weights(),
            WrappedElement::Covariant(a) => a.interpolation_weights(),
            WrappedElement::Contravariant(a) => a.interpolation_weights(),
        }
    }
}
