"""Tests of the layout of tabulated data."""

import itertools
import math

import numpy as np
import pytest
from ndelement.ciarlet import Family, create_family
from ndelement.polynomials import tabulate_legendre_polynomials
from ndelement.reference_cell import ReferenceCellType, dim, vertices

cells = [
    ReferenceCellType.Interval,
    ReferenceCellType.Triangle,
    ReferenceCellType.Quadrilateral,
    ReferenceCellType.Tetrahedron,
    ReferenceCellType.Hexahedron,
]

interior_points = {
    ReferenceCellType.Interval: np.array([[0.15], [0.4], [0.85]]),
    ReferenceCellType.Triangle: np.array([[0.2, 0.3], [0.1, 0.6], [0.55, 0.15]]),
    ReferenceCellType.Quadrilateral: np.array([[0.2, 0.7], [0.6, 0.1], [0.85, 0.45]]),
    ReferenceCellType.Tetrahedron: np.array([[0.1, 0.2, 0.3], [0.3, 0.1, 0.25], [0.15, 0.45, 0.2]]),
    ReferenceCellType.Hexahedron: np.array([[0.2, 0.7, 0.4], [0.6, 0.1, 0.8], [0.35, 0.55, 0.15]]),
}


def derivative_index(alpha: tuple[int, ...]) -> int:
    """Index of the derivative with multi-index alpha in the derivative axis."""
    if len(alpha) == 1:
        return alpha[0]
    if len(alpha) == 2:
        n = alpha[0] + alpha[1]
        return n * (n + 1) // 2 + alpha[1]
    if len(alpha) == 3:
        n = alpha[0] + alpha[1] + alpha[2]
        m = alpha[1] + alpha[2]
        return n * (n + 1) * (n + 2) // 6 + m * (m + 1) // 2 + alpha[2]
    raise ValueError(f"Unsupported dimension: {len(alpha)}")


def multi_indices(tdim: int, nderivs: int) -> list[tuple[int, ...]]:
    """All multi-indices in tdim variables with total order at most nderivs."""
    return [a for a in itertools.product(range(nderivs + 1), repeat=tdim) if sum(a) <= nderivs]


@pytest.mark.parametrize("cell", cells)
@pytest.mark.parametrize("nderivs", range(5))
def test_derivative_index_is_bijection(cell, nderivs):
    tdim = dim(cell)
    indices = sorted(derivative_index(a) for a in multi_indices(tdim, nderivs))
    assert indices == list(range(math.comb(nderivs + tdim, tdim)))


element_parameters = [
    (Family.Lagrange, cell, degree) for cell in cells for degree in range(1, 4)
] + [
    (family, cell, degree)
    for family in [Family.RaviartThomas, Family.NedelecFirstKind]
    for cell in cells[1:]
    for degree in range(1, 3)
]


@pytest.mark.parametrize(("family", "cell", "degree"), element_parameters)
@pytest.mark.parametrize("nderivs", range(3))
def test_tabulate_shape(family, cell, degree, nderivs):
    element = create_family(family, degree).element(cell)
    points = interior_points[cell]
    data = element.tabulate(points, nderivs)
    tdim = dim(cell)
    assert data.shape == (
        element.dim,
        math.comb(nderivs + tdim, tdim),
        element.value_size,
        points.shape[0],
    )


@pytest.mark.parametrize(("family", "cell", "degree"), element_parameters)
def test_tabulate_layout(family, cell, degree):
    element = create_family(family, degree).element(cell)
    points = interior_points[cell]
    tdim = dim(cell)
    nderivs = 2
    data = element.tabulate(points, nderivs)

    # Tabulating at each point on its own gives the same values
    for p, point in enumerate(points):
        assert np.allclose(data[..., p], element.tabulate(point.reshape(1, -1), nderivs)[..., 0])

    # Tabulating fewer derivatives gives the same values for the derivatives that are included
    for k in range(nderivs):
        lower = element.tabulate(points, k)
        for alpha in multi_indices(tdim, k):
            d = derivative_index(alpha)
            assert np.allclose(data[:, d], lower[:, d])

    # The derivative with multi-index alpha + e_m is the derivative in direction m of the
    # derivative with multi-index alpha
    h = 1e-5
    for m in range(tdim):
        shift = np.zeros(tdim)
        shift[m] = h
        plus = element.tabulate(points + shift, nderivs)
        minus = element.tabulate(points - shift, nderivs)
        for alpha in multi_indices(tdim, nderivs - 1):
            beta = tuple(a + 1 if i == m else a for i, a in enumerate(alpha))
            a = derivative_index(alpha)
            b = derivative_index(beta)
            assert np.allclose(data[:, b], (plus[:, a] - minus[:, a]) / (2 * h), atol=1e-5)


@pytest.mark.parametrize(
    "cell",
    [ReferenceCellType.Interval, ReferenceCellType.Quadrilateral, ReferenceCellType.Hexahedron],
)
def test_lagrange_1_tensor_product_derivatives(cell):
    # The basis function for vertex v is the product over coordinates x_i of x_i (if v_i = 1)
    # or 1 - x_i (if v_i = 0)
    tdim = dim(cell)
    element = create_family(Family.Lagrange, 1).element(cell)
    points = interior_points[cell]
    nderivs = tdim + 1
    data = element.tabulate(points, nderivs)
    cell_vertices = vertices(cell)
    assert element.dim == len(cell_vertices)

    for p, point in enumerate(points):
        for v, vertex in enumerate(cell_vertices):
            for alpha in multi_indices(tdim, nderivs):
                expected = 1.0
                for i in range(tdim):
                    if alpha[i] == 0:
                        expected *= point[i] if vertex[i] == 1 else 1 - point[i]
                    elif alpha[i] == 1:
                        expected *= 1.0 if vertex[i] == 1 else -1.0
                    else:
                        expected = 0.0
                assert np.isclose(data[v, derivative_index(alpha), 0, p], expected)


@pytest.mark.parametrize(
    ("family", "basis"),
    [
        (
            Family.RaviartThomas,
            lambda x, y: [
                [[-x, 1 - y], [-1, 0], [0, -1]],
                [[x - 1, y], [1, 0], [0, 1]],
                [[-x, -y], [-1, 0], [0, -1]],
            ],
        ),
        (
            Family.NedelecFirstKind,
            lambda x, y: [
                [[1 - y, x], [0, 1], [-1, 0]],
                [[y, 1 - x], [0, -1], [1, 0]],
                [[-y, x], [0, 1], [-1, 0]],
            ],
        ),
    ],
)
def test_vector_triangle_derivatives(family, basis):
    element = create_family(family, 1).element(ReferenceCellType.Triangle)
    points = interior_points[ReferenceCellType.Triangle]
    data = element.tabulate(points, 1)

    for p, (x, y) in enumerate(points):
        for b, (value, dx, dy) in enumerate(basis(x, y)):
            for c in range(2):
                assert np.isclose(data[b, derivative_index((0, 0)), c, p], value[c])
                assert np.isclose(data[b, derivative_index((1, 0)), c, p], dx[c])
                assert np.isclose(data[b, derivative_index((0, 1)), c, p], dy[c])


@pytest.mark.parametrize("cell", cells)
@pytest.mark.parametrize("degree", range(1, 4))
def test_legendre_derivatives(cell, degree):
    points = interior_points[cell]
    tdim = dim(cell)
    nderivs = 3
    data = tabulate_legendre_polynomials(cell, points, degree, nderivs)
    assert data.shape[0] == points.shape[0]
    assert data.shape[2] == math.comb(nderivs + tdim, tdim)

    h = 1e-5
    for m in range(tdim):
        shift = np.zeros(tdim)
        shift[m] = h
        plus = tabulate_legendre_polynomials(cell, points + shift, degree, nderivs)
        minus = tabulate_legendre_polynomials(cell, points - shift, degree, nderivs)
        for alpha in multi_indices(tdim, nderivs - 1):
            beta = tuple(a + 1 if i == m else a for i, a in enumerate(alpha))
            a = derivative_index(alpha)
            b = derivative_index(beta)
            assert np.allclose(data[:, :, b], (plus[:, :, a] - minus[:, :, a]) / (2 * h), atol=1e-4)
