//! Functions in function spaces

use crate::traits::FunctionSpace;

/// A function in a function space
pub struct FunctionImpl<'a, S: FunctionSpace> {
    /// The function space
    #[allow(unused)]
    space: &'a S,
    /// The coefficients that define this function
    #[allow(unused)]
    coefficients: Vec<f64>,
}

impl<'a, S: FunctionSpace> FunctionImpl<'a, S> {
    /// Create a new function
    pub fn new(space: &'a S, coefficients: Vec<f64>) -> Self {
        Self {
            space,
            coefficients,
        }
    }
    /// Create a zero function
    pub fn zero(space: &'a S) -> Self {
        let coefficients = vec![0.0; space.local_size()];
        Self {
            space,
            coefficients,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::function_space::FunctionSpaceImpl;
    use ndelement::{
        ciarlet::{LagrangeElementFamily, LagrangeVariant},
        types::{Continuity, ReferenceCellType},
    };
    use ndmesh::shapes::{unit_cube, unit_square};

    #[test]
    fn test_function_with_unit_square() {
        let mesh = unit_square::<f64>(1, 1, ReferenceCellType::Triangle);
        let family = LagrangeElementFamily::<f64>::new(
            0,
            Continuity::Discontinuous,
            LagrangeVariant::Equispaced,
        );

        let space = FunctionSpaceImpl::new(&mesh, &family);

        let function = FunctionImpl::zero(&space);

        assert_eq!(function.coefficients.len(), 2);
    }
    #[test]
    fn test_function_with_unit_cube() {
        let mesh = unit_cube::<f64>(1, 1, 1, ReferenceCellType::Tetrahedron);
        let family = LagrangeElementFamily::<f64>::new(
            0,
            Continuity::Discontinuous,
            LagrangeVariant::Equispaced,
        );

        let space = FunctionSpaceImpl::new(&mesh, &family);

        let function = FunctionImpl::zero(&space);

        assert_eq!(function.coefficients.len(), 6);
    }
}
