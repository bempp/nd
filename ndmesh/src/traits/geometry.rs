//! Entity geometry
use itertools::Itertools;

use crate::types::Scalar;
use ndelement::traits::MappedFiniteElement;

/// A point
pub trait Point {
    /// Scalar type
    type T: Scalar;

    /// The point's index
    fn index(&self) -> usize;

    /// Return the dimension of the point.
    fn dim(&self) -> usize;

    /// Get the coordinates of the point.
    fn coords(&self, data: &mut [Self::T]);
}

/// The geometry of an entity
///
/// The geometry contains information about all points that make up the entity.
pub trait Geometry {
    /// Scalar type
    type T: Scalar;

    /// Finite element type
    type E: MappedFiniteElement<T = Self::T, TGeo = Self::T>;

    /// Point Type
    type Point<'a>: Point<T = Self::T>
    where
        Self: 'a;

    /// Point iterator
    type PointIter<'a>: Iterator<Item = Self::Point<'a>>
    where
        Self: 'a;

    /// Points
    ///
    /// These are the points that define the geometry of the cell, and are equal to the
    /// result of mapping the interpolation points of the element to the physical cell
    fn points(&self) -> Self::PointIter<'_>;

    /// Indices of the points
    fn point_indices(&self) -> Vec<usize> {
        self.points().map(|p| p.index()).collect_vec()
    }

    /// Point count
    fn point_count(&self) -> usize;

    /// Finite element that defines the geometry
    fn element(&self) -> &Self::E;
}
