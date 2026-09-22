//! This module contains functions to reorder degrees of freedom from the nd ordering to other ordering formats.

use itertools::Itertools;

use crate::{
    reorderings::vtk::{vtk_hexahedron, vtk_quadrilateral, vtk_tetrahedron, vtk_triangle},
    types::ReferenceCellType,
};

mod vtk;

/// Return a permutation vector `perm` that defines the VTK ordering of the dofs
///
/// If `dof_values` is a vector of dof values associated with the element in
/// the ordering that ND uses, then the vector of dof values in the corresponding
/// VTK ordering is given as `[dof_values[perm[0]], dof_values[perm[1]], ...]`.
pub fn vtk_ordering(cell_type: ReferenceCellType, npts: usize) -> Vec<usize> {
    match cell_type {
        ReferenceCellType::Point => vec![0],
        ReferenceCellType::Interval => (0..npts).collect_vec(),
        ReferenceCellType::Triangle => vtk_triangle(npts),
        ReferenceCellType::Quadrilateral => vtk_quadrilateral(npts),
        ReferenceCellType::Tetrahedron => vtk_tetrahedron(npts),
        ReferenceCellType::Hexahedron => vtk_hexahedron(npts),
        _ => {
            panic!("Cell type not supported.");
        }
    }
}
