"""Polynomials."""

import numpy as np
import numpy.typing as npt

from ndelement import _ndelementrs
from ndelement.reference_cell import ReferenceCellType


def tabulate_legendre_polynomials(
    cell: ReferenceCellType,
    points: npt.NDArray[np.float64],
    degree: int,
    derivatives: int,
) -> npt.NDArray[np.float64]:
    """Tabulate Legendre orthonormal polynomials."""
    return _ndelementrs.polynomials.tabulate_legendre_polynomials(
        cell._rs,
        points,
        degree,
        derivatives,
    )
