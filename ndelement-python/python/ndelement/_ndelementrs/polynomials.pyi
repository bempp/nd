import numpy as np
import numpy.typing as npt

from .types import ReferenceCellType

def tabulate_legendre_polynomials(
    cell: ReferenceCellType,
    points: npt.NDArray[np.float64],
    degree: int,
    derivatives: int,
) -> npt.NDArray[np.float64]: ...
