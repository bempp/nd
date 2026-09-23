import ndelement
import numpy as np
import pytest


@pytest.mark.parametrize(
    ("family", "family_kwargs"),
    [
        (
            ndelement.ciarlet.Family.Lagrange,
            {"lagrange_variant": ndelement.ciarlet.LagrangeVariant.Gll},
        ),
        (ndelement.ciarlet.Family.RaviartThomas, {}),
        (ndelement.ciarlet.Family.NedelecFirstKind, {}),
    ],
)
@pytest.mark.parametrize("degree", range(1, 6))
def test_transformation_of_tabulated_data_quadrilateral(family, degree, family_kwargs):
    family = ndelement.ciarlet.create_family(family, degree, **family_kwargs)
    element = family.element(ndelement.reference_cell.ReferenceCellType.Quadrilateral)

    N = 5
    points = np.array([[i / N, j / N] for i in range(N + 1) for j in range(N + 1)])

    dofs = element.entity_dofs(1, 0)
    ndofs = len(dofs)

    if ndofs > 0:
        start = dofs[0]
        values = element.tabulate(points, 0)
        reflected_points = np.array([[1 - p[0], p[1]] for p in points])
        reflected_values = element.tabulate(reflected_points, 0)

        _J = np.array([[-1.0, 0.0], [0.0, 1.0]])
        J = np.array([_J for p in points])
        detJ = np.array([np.linalg.det(_J) for p in points])
        K = np.array([np.linalg.inv(_J) for p in points])
        mapped_values = element.push_forward(reflected_values, 0, J, detJ, K)

        # TODO: Change this to the following once #93 is resolved
        # element.apply_dof_permutations_and_transformations(values, np.int32(1))
        for d in range(values.shape[0]):
            element.apply_dof_permutations_and_transformations(values[d], np.int32(1))

        # TODO: Remove the `:, ` at the start of each access once #93 is resolved
        assert np.allclose(
            values[:, start : start + ndofs], mapped_values[:, start : start + ndofs]
        )


@pytest.mark.parametrize(
    ("family", "family_kwargs"),
    [
        (
            ndelement.ciarlet.Family.Lagrange,
            {"lagrange_variant": ndelement.ciarlet.LagrangeVariant.Equispaced},
        ),
        (ndelement.ciarlet.Family.RaviartThomas, {}),
        (ndelement.ciarlet.Family.NedelecFirstKind, {}),
    ],
)
@pytest.mark.parametrize("degree", range(1, 6))
def test_transformation_of_tabulated_data_triangle(family, degree, family_kwargs):
    family = ndelement.ciarlet.create_family(family, degree, **family_kwargs)
    element = family.element(ndelement.reference_cell.ReferenceCellType.Triangle)

    N = 5
    points = np.array([[i / N, j / N] for i in range(N + 1) for j in range(N + 1 - i)])

    dofs = element.entity_dofs(1, 0)
    ndofs = len(dofs)

    if ndofs > 0:
        start = dofs[0]
        values = element.tabulate(points, 0)
        reflected_points = np.array([[1 - p[0] - p[1], p[1]] for p in points])
        reflected_values = element.tabulate(reflected_points, 0)

        _J = np.array([[-1.0, 0.0], [-1.0, 1.0]])
        J = np.array([_J for p in points])
        detJ = np.array([np.linalg.det(_J) for p in points])
        K = np.array([np.linalg.inv(_J) for p in points])
        mapped_values = element.push_forward(reflected_values, 0, J, detJ, K)

        # TODO: Change this to the following once #93 is resolved
        # element.apply_dof_permutations_and_transformations(values, np.int32(1))
        for d in range(values.shape[0]):
            element.apply_dof_permutations_and_transformations(values[d], np.int32(1))

        # TODO: Remove the `:, ` at the start of each access once #93 is resolved
        assert np.allclose(
            values[:, start : start + ndofs], mapped_values[:, start : start + ndofs]
        )


@pytest.mark.parametrize(
    ("family", "family_kwargs"),
    [
        (
            ndelement.ciarlet.Family.Lagrange,
            {"lagrange_variant": ndelement.ciarlet.LagrangeVariant.Gll},
        ),
        (ndelement.ciarlet.Family.RaviartThomas, {}),
        (ndelement.ciarlet.Family.NedelecFirstKind, {}),
    ],
)
@pytest.mark.parametrize("degree", range(1, 4))
def test_transformation_of_tabulated_data_hexahedron(family, degree, family_kwargs):
    family = ndelement.ciarlet.create_family(family, degree, **family_kwargs)
    element = family.element(ndelement.reference_cell.ReferenceCellType.Hexahedron)

    N = 5
    points = np.array(
        [[i / N, j / N, k / N] for i in range(N + 1) for j in range(N + 1) for k in range(N + 1)]
    )

    dofs = element.entity_dofs(1, 0)
    ndofs = len(dofs)

    if ndofs > 0:
        start = dofs[0]
        values = element.tabulate(points, 0)
        reflected_points = np.array([[1 - p[0], p[1], p[2]] for p in points])
        reflected_values = element.tabulate(reflected_points, 0)

        _J = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        J = np.array([_J for p in points])
        detJ = np.array([np.linalg.det(_J) for p in points])
        K = np.array([np.linalg.inv(_J) for p in points])
        mapped_values = element.push_forward(reflected_values, 0, J, detJ, K)

        # TODO: Change this to the following once #93 is resolved
        # element.apply_dof_permutations_and_transformations(values, np.int32(1))
        for d in range(values.shape[0]):
            element.apply_dof_permutations_and_transformations(values[d], np.int32(1))

        # TODO: Remove the `:, ` at the start of each access once #93 is resolved
        assert np.allclose(
            values[:, start : start + ndofs], mapped_values[:, start : start + ndofs]
        )

    dofs = element.entity_dofs(2, 0)
    ndofs = len(dofs)

    if ndofs > 0:
        start = dofs[0]

        # Reflection of face
        values = element.tabulate(points, 0)
        reflected_points = np.array([[p[1], p[0], p[2]] for p in points])
        reflected_values = element.tabulate(reflected_points, 0)

        _J = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        J = np.array([_J for p in points])
        detJ = np.array([np.linalg.det(_J) for p in points])
        K = np.array([np.linalg.inv(_J) for p in points])
        mapped_values = element.push_forward(reflected_values, 0, J, detJ, K)

        # TODO: Change this to the following once #93 is resolved
        # element.apply_dof_permutations_and_transformations(values, np.int32(1))
        for d in range(values.shape[0]):
            element.apply_dof_permutations_and_transformations(values[d], np.int32(1 << (12 + 2)))

        # TODO: Remove the `:, ` at the start of each access once #93 is resolved
        assert np.allclose(
            values[:, start : start + ndofs], mapped_values[:, start : start + ndofs]
        )

        # Rotation of face
        values = element.tabulate(points, 0)
        rotated_points = np.array([[1 - p[1], p[0], p[2]] for p in points])
        rotated_values = element.tabulate(rotated_points, 0)

        _J = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        J = np.array([_J for p in points])
        detJ = np.array([np.linalg.det(_J) for p in points])
        K = np.array([np.linalg.inv(_J) for p in points])
        mapped_values = element.push_forward(rotated_values, 0, J, detJ, K)

        # TODO: Change this to the following once #93 is resolved
        # element.apply_dof_permutations_and_transformations(values, np.int32(1))
        for d in range(values.shape[0]):
            element.apply_dof_permutations_and_transformations(values[d], np.int32(1 << 12))

        # TODO: Remove the `:, ` at the start of each access once #93 is resolved
        assert np.allclose(
            values[:, start : start + ndofs], mapped_values[:, start : start + ndofs]
        )


@pytest.mark.parametrize(
    ("family", "family_kwargs"),
    [
        (
            ndelement.ciarlet.Family.Lagrange,
            {"lagrange_variant": ndelement.ciarlet.LagrangeVariant.Equispaced},
        ),
        (ndelement.ciarlet.Family.RaviartThomas, {}),
        (ndelement.ciarlet.Family.NedelecFirstKind, {}),
    ],
)
@pytest.mark.parametrize("degree", range(1, 4))
def test_transformation_of_tabulated_data_tetrahedron(family, degree, family_kwargs):
    family = ndelement.ciarlet.create_family(family, degree, **family_kwargs)
    element = family.element(ndelement.reference_cell.ReferenceCellType.Tetrahedron)

    N = 5
    points = np.array(
        [
            [i / N, j / N, k / N]
            for i in range(N + 1)
            for j in range(N + 1 - i)
            for k in range(N + 1 - i - j)
        ]
    )

    dofs = element.entity_dofs(1, 0)
    ndofs = len(dofs)

    if ndofs > 0:
        start = dofs[0]
        values = element.tabulate(points, 0)
        reflected_points = np.array([[1 - p[0] - p[1] - p[2], p[1], p[2]] for p in points])
        reflected_values = element.tabulate(reflected_points, 0)

        _J = np.array([[-1.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
        J = np.array([_J for p in points])
        detJ = np.array([np.linalg.det(_J) for p in points])
        K = np.array([np.linalg.inv(_J) for p in points])
        mapped_values = element.push_forward(reflected_values, 0, J, detJ, K)

        # TODO: Change this to the following once #93 is resolved
        # element.apply_dof_permutations_and_transformations(values, np.int32(1))
        for d in range(values.shape[0]):
            element.apply_dof_permutations_and_transformations(values[d], np.int32(1))

        # TODO: Remove the `:, ` at the start of each access once #93 is resolved
        assert np.allclose(
            values[:, start : start + ndofs], mapped_values[:, start : start + ndofs]
        )

    dofs = element.entity_dofs(2, 0)
    ndofs = len(dofs)

    if ndofs > 0:
        start = dofs[0]

        # Reflection of face
        values = element.tabulate(points, 0)
        reflected_points = np.array([[p[1], p[0], p[2]] for p in points])
        reflected_values = element.tabulate(reflected_points, 0)

        _J = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        J = np.array([_J for p in points])
        detJ = np.array([np.linalg.det(_J) for p in points])
        K = np.array([np.linalg.inv(_J) for p in points])
        mapped_values = element.push_forward(reflected_values, 0, J, detJ, K)

        # TODO: Change this to the following once #93 is resolved
        # element.apply_dof_permutations_and_transformations(values, np.int32(1))
        for d in range(values.shape[0]):
            element.apply_dof_permutations_and_transformations(values[d], np.int32(1 << (6 + 2)))

        # TODO: Remove the `:, ` at the start of each access once #93 is resolved
        assert np.allclose(
            values[:, start : start + ndofs], mapped_values[:, start : start + ndofs]
        )

        # Rotation of face
        values = element.tabulate(points, 0)
        rotated_points = np.array([[1 - p[0] - p[1] - p[2], p[0], p[2]] for p in points])
        rotated_values = element.tabulate(rotated_points, 0)

        _J = np.array([[0.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, -1.0, 1.0]])
        J = np.array([_J for p in points])
        detJ = np.array([np.linalg.det(_J) for p in points])
        K = np.array([np.linalg.inv(_J) for p in points])
        mapped_values = element.push_forward(rotated_values, 0, J, detJ, K)

        # TODO: Change this to the following once #93 is resolved
        # element.apply_dof_permutations_and_transformations(values, np.int32(1))
        for d in range(values.shape[0]):
            element.apply_dof_permutations_and_transformations(values[d], np.int32(1 << 6))

        # TODO: Remove the `:, ` at the start of each access once #93 is resolved
        assert np.allclose(
            values[:, start : start + ndofs], mapped_values[:, start : start + ndofs]
        )
