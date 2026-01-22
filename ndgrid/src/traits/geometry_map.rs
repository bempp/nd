//! Map from reference to physical space.

use crate::types::Scalar;
use rlst::{Array, MutableArrayImpl};

/// A geometry map allows the computation of maps from reference to physical space and their derivatives.
///
/// A geometry map is typically initialised with a number of points on a reference entity.
/// We can then for each physical entity compute
/// - The associated physical points as maps from the reference points.
/// - The jacobian of the map at the physical points.
/// - The jacobians, transformation determinants and the normals of the physical entity.
pub trait GeometryMap {
    /// Scalar type
    type T: Scalar;

    /// The topoloical dimension of the entity being mapped.
    ///
    /// The topological dimension is e.g. two for a triangle, independent
    /// of whether it is embedded in two or three dimensional space.
    fn entity_topology_dimension(&self) -> usize;

    /// The geometric dimension of the physical space.
    fn geometry_dimension(&self) -> usize;

    /// The number of reference points that this map uses.
    fn point_count(&self) -> usize;

    /// Write the physical points for the entity with index `entity_index` into `points`
    ///
    /// `points` should have shape [geometry_dimension, npts] and use column-major ordering.
    fn physical_points<Array2Impl: MutableArrayImpl<Self::T, 2>>(&self, entity_index: usize, points: &mut Array<Array2Impl, 2>);

    /// Write the jacobians at the physical points for the entity with index `entity_index` into `jacobians`
    ///
    /// `jacobians` should have shape [geometry_dimension, entity_topology_dimension, npts] and use column-major ordering
    fn jacobians<Array3MutImpl: MutableArrayImpl<Self::T, 3>>(&self, entity_index: usize, jacobians: &mut Array<Array3MutImpl, 3>);

    /// Write the jacobians, their inverses and their determinants for the entity with
    /// index `entity_index` into `jacobians`, `inverse_jacobians` and `jdets`.
    ///
    /// `jacobians` should have shape [geometry_dimension, entity_topology_dimension, npts] and use column-major ordering;
    /// `inverse_jacobians` should have shape [entity_topology_dimension, geometry_dimension, npts] and use column-major ordering;
    /// `jdets` should have shape \[npts\];
    fn jacobians_inverses_dets<Array3MutImpl: MutableArrayImpl<Self::T, 3>>(
        &self,
        entity_index: usize,
        jacobians: &mut Array<Array3MutImpl, 3>,
        inverse_jacobians: &mut Array<Array3MutImpl, 3>,
        jdets: &mut [Self::T],
    );

    /// Write the jacobians, their inverses, their determinants, and the normals at the physical points for the entity with
    /// index `entity_index` into `jacobians`, `inverse_jacobians`, `jdets` and `normals`.
    ///
    /// `jacobians` should have shape [geometry_dimension, entity_topology_dimension, npts] and use column-major ordering;
    /// `inverse_jacobians` should have shape [entity_topology_dimension, geometry_dimension, npts] and use column-major ordering;
    /// `jdets` should have shape \[npts\];
    /// `normals` should have shape [geometry_dimension, npts] and use column-major ordering
    fn jacobians_inverses_dets_normals<Array2Impl: MutableArrayImpl<Self::T, 2>, Array3MutImpl: MutableArrayImpl<Self::T, 3>>(
        &self,
        entity_index: usize,
        jacobians: &mut Array<Array3MutImpl, 3>,
        inverse_jacobians: &mut Array<Array3MutImpl, 3>,
        jdets: &mut [Self::T],
        normals: &mut Array<Array2Impl, 2>,
    );
}
