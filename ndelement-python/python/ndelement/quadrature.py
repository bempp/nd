"""Quadrature."""

import numpy as np
import numpy.typing as npt

from ndelement import _ndelementrs
from ndelement.reference_cell import ReferenceCellType, dim


def make_gauss_jacobi_quadrature(
    cell: ReferenceCellType, degree: int
) -> tuple[npt.NDArray[np.float64], list[float]]:
    """Get the points and weights of a Gauss-Jacobi quadrature rule."""
    tdim = dim(cell)
    pts, wts = _ndelementrs.quadrature.make_gauss_jacobi_quadrature(
        cell._rs,
        degree,
    )
    pts_array = np.zeros((len(wts), tdim))
    for i, p in enumerate(pts):
        pts_array[i // tdim, i % tdim] = p
    return pts_array, wts
