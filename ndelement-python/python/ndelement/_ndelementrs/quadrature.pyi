from .types import ReferenceCellType

def make_gauss_jacobi_quadrature(
    cell: ReferenceCellType, m: int
) -> tuple[list[float], list[float]]: ...
