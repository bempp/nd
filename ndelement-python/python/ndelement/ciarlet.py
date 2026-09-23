"""Ciarlet elements."""

from enum import Enum

import numpy as np
import numpy.typing as npt

from ndelement import _ndelementrs
from ndelement.reference_cell import ReferenceCellType


class Continuity(Enum):
    """Continuity."""

    Standard = 0
    Discontinuous = 1

    @property
    def _rs(self) -> _ndelementrs.types.Continuity:
        return getattr(_ndelementrs.types.Continuity, self.name)


class Family(Enum):
    """Element family."""

    Lagrange = 0
    RaviartThomas = 1
    NedelecFirstKind = 2


class LagrangeVariant(Enum):
    """Variant of Lagrange element."""

    Equispaced = 0
    Gll = 1

    @property
    def _rs(self) -> _ndelementrs.types.LagrangeVariant:
        return getattr(_ndelementrs.types.LagrangeVariant, self.name)


class CiarletElement:
    """Ciarlet element."""

    def __init__(
        self,
        rs_element: _ndelementrs.ciarlet.CiarletElement,
        continuity: Continuity,
        cell_type: ReferenceCellType,
    ):
        """Initialise."""
        self._rs = rs_element
        self._continuity = continuity
        self._cell_type = cell_type

    @property
    def value_size(self) -> int:
        """Value size of the element."""
        return self._rs.value_size()

    @property
    def value_shape(self) -> tuple[int, ...]:
        """Value size of the element."""
        return tuple(self._rs.value_shape())

    @property
    def degree(self) -> int:
        """Degree of the element."""
        return self._rs.degree()

    @property
    def lagrange_superdegree(self) -> int:
        """lagrange superdegree of the element."""
        return self._rs.lagrange_superdegree()

    @property
    def dim(self) -> int:
        """Dimension (number of basis functions) of the element."""
        return self._rs.dim()

    @property
    def continuity(self) -> Continuity:
        """Continuity of the element."""
        return self._continuity

    @property
    def cell_type(self) -> ReferenceCellType:
        """Cell type of the element."""
        return self._cell_type

    def entity_dofs(self, entity_dim: int, entity_index: int) -> list[int]:
        """Get the DOFs associated with an entity."""
        return self._rs.entity_dofs(entity_dim, entity_index)

    def entity_closure_dofs(self, entity_dim: int, entity_index: int) -> list[int]:
        """Get the DOFs associated with the closure of an entity."""
        return self._rs.entity_closure_dofs(entity_dim, entity_index)

    def interpolation_points(self) -> list[list[npt.NDArray[np.float64]]]:
        """Interpolation points."""
        return self._rs.interpolation_points()

    def interpolation_weights(self) -> list[list[npt.NDArray[np.float64]]]:
        """Interpolation weights."""
        return self._rs.interpolation_weights()

    def tabulate(self, points: npt.NDArray[np.float64], nderivs: int) -> npt.NDArray[np.float64]:
        """Tabulate the basis functions at a set of points."""
        return self._rs.tabulate(points, nderivs)

    def physical_value_size(self, gdim: int) -> int:
        """The physical value size of the element on a cell."""
        return self._rs.physical_value_size(gdim)

    def physical_value_shape(self, gdim: int) -> tuple[int, ...]:
        """The physical value shape of the element on a cell."""
        return tuple(self._rs.physical_value_shape(gdim))

    def push_forward(
        self,
        reference_values: npt.NDArray[np.float64],
        nderivs: int,
        jacobians: npt.NDArray[np.float64],
        jacobian_determinants: npt.NDArray[np.float64],
        inverse_jacobians: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Push values forward to a physical cell."""
        return self._rs.push_forward(
            reference_values, nderivs, jacobians, jacobian_determinants, inverse_jacobians
        )

    def pull_back(
        self,
        physical_values: npt.NDArray[np.float64],
        nderivs: int,
        jacobians: npt.NDArray[np.float64],
        jacobian_determinants: npt.NDArray[np.float64],
        inverse_jacobians: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Push values back from a physical cell."""
        return self._rs.pull_back(
            physical_values, nderivs, jacobians, jacobian_determinants, inverse_jacobians
        )

    def apply_dof_permutations(self, data: npt.NDArray, orientation: np.int32):
        """Apply permutation parts of DOF transformations."""
        if data.dtype == np.float64:
            self._rs.apply_dof_permutations_f64(data.reshape(-1), orientation)
        elif data.dtype == np.int64:
            self._rs.apply_dof_permutations_i64(data.reshape(-1), orientation)
        elif data.dtype == np.uint64:
            self._rs.apply_dof_permutations_u64(data.reshape(-1), orientation)
        else:
            raise TypeError(f"Unsupported data type: {data.dtype}")

    def apply_dof_transformations(self, data: npt.NDArray[np.float64], orientation: np.int32):
        """Apply non-permutation parts of DOF transformations."""
        self._rs.apply_dof_transformations(data.reshape(-1), orientation)

    def apply_dof_permutations_and_transformations(
        self, data: npt.NDArray[np.float64], orientation: np.int32
    ):
        """Apply DOF transformations."""
        self._rs.apply_dof_permutations_and_transformations(data.reshape(-1), orientation)


class ElementFamily:
    """Ciarlet element."""

    def __init__(
        self,
        family: Family,
        degree: int,
        continuity: Continuity,
        lagrange_variant: LagrangeVariant | None,
        rs_family: _ndelementrs.ciarlet.ElementFamily,
    ):
        """Initialise."""
        self._rs = rs_family
        self._family = family
        self._degree = degree
        self._continuity = continuity
        self._lagrange_variant = lagrange_variant

    @property
    def family(self) -> Family:
        """The family."""
        return self._family

    @property
    def degree(self) -> int:
        """The degree."""
        return self._degree

    @property
    def continuity(self) -> Continuity:
        """Continuity."""
        return self._continuity

    @property
    def lagrange_variant(self) -> LagrangeVariant | None:
        """Lagrange variant."""
        return self._lagrange_variant

    def element(self, cell: ReferenceCellType) -> CiarletElement:
        """Create an element."""
        return CiarletElement(self._rs.element(cell._rs), self._continuity, cell)


def create_family(
    family: Family,
    degree: int,
    continuity: Continuity = Continuity.Standard,
    lagrange_variant: LagrangeVariant | None = None,
) -> ElementFamily:
    """Create a new element family."""
    match family:
        case Family.Lagrange:
            if lagrange_variant is None:
                lagrange_variant = LagrangeVariant.Equispaced
            return ElementFamily(
                family,
                degree,
                continuity,
                lagrange_variant,
                _ndelementrs.ciarlet.create_lagrange_family(
                    degree, continuity._rs, lagrange_variant._rs
                ),
            )
        case Family.RaviartThomas:
            if lagrange_variant is not None:
                raise ValueError("Lagrange variant cannot be used with this element type")
            return ElementFamily(
                family,
                degree,
                continuity,
                None,
                _ndelementrs.ciarlet.create_raviart_thomas_family(degree, continuity._rs),
            )
        case Family.NedelecFirstKind:
            if lagrange_variant is not None:
                raise ValueError("Lagrange variant cannot be used with this element type")
            return ElementFamily(
                family,
                degree,
                continuity,
                None,
                _ndelementrs.ciarlet.create_nedelec_family(degree, continuity._rs),
            )
        case _:
            raise ValueError(f"Unsupported family: {family}")
