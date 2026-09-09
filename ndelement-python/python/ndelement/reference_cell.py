"""Reference cell information."""

from enum import Enum

import numpy as np
import numpy.typing as npt

from ndelement import _ndelementrs


class ReferenceCellType(Enum):
    """Reference cell type."""

    Point = 0
    Interval = 1
    Triangle = 2
    Quadrilateral = 3
    Tetrahedron = 4
    Hexahedron = 5
    Prism = 6
    Pyramid = 7

    @property
    def _rs(self) -> _ndelementrs.types.ReferenceCellType:
        return getattr(_ndelementrs.types.ReferenceCellType, self.name)


def dim(cell: ReferenceCellType) -> int:
    """Get the topological dimension of a reference cell."""
    return _ndelementrs.reference_cell.dim(cell._rs)


def is_simplex(cell: ReferenceCellType) -> bool:
    """Check if a reference cell is a simplex."""
    return _ndelementrs.reference_cell.is_simplex(cell._rs)


def vertices(cell: ReferenceCellType) -> npt.NDArray[np.float64]:
    """Get the vertices of a reference cell."""
    return np.array(_ndelementrs.reference_cell.vertices(cell._rs))


def midpoint(cell: ReferenceCellType) -> npt.NDArray[np.float64]:
    """Get the midpoint of a reference cell."""
    return np.array(_ndelementrs.reference_cell.midpoint(cell._rs))


def edges(cell: ReferenceCellType) -> list[list[int]]:
    """Get the edges of a reference cell."""
    return _ndelementrs.reference_cell.edges(cell._rs)


def faces(cell: ReferenceCellType) -> list[list[int]]:
    """Get the faces of a reference cell."""
    return _ndelementrs.reference_cell.faces(cell._rs)


def volumes(cell: ReferenceCellType) -> list[list[int]]:
    """Get the volumes of a reference cell."""
    return _ndelementrs.reference_cell.volumes(cell._rs)


def entity_types(cell: ReferenceCellType) -> list[list[ReferenceCellType]]:
    """Get the types of the sub-entities of a reference cell."""
    return [
        [getattr(ReferenceCellType, f"{t!r}".split(".")[-1]) for t in types]
        for types in _ndelementrs.reference_cell.entity_types(cell._rs)
    ]


def entity_counts(cell: ReferenceCellType) -> list[int]:
    """Get the number of the sub-entities of each dimension for a reference cell."""
    return _ndelementrs.reference_cell.entity_counts(cell._rs)


def connectivity(
    cell: ReferenceCellType,
) -> list[list[list[list[int]]]]:
    """Get the connectivity of a reference cell."""
    return _ndelementrs.reference_cell.connectivity(cell._rs)


def compute_orientation(
    cell: ReferenceCellType,
    vertices: list[int],
) -> np.int32:
    """Compute a 32-bit integer encoding the orientation differences between the reference cell and a cell."""
    return _ndelementrs.reference_cell.compute_orientation(cell._rs, vertices)
