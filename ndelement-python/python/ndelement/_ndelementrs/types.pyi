class ReferenceCellType:
    Point: ReferenceCellType
    Interval: ReferenceCellType
    Triangle: ReferenceCellType
    Quadrilateral: ReferenceCellType
    Tetrahedron: ReferenceCellType
    Hexahedron: ReferenceCellType
    Prism: ReferenceCellType
    Pyramid: ReferenceCellType

class Continuity:
    Standard: Continuity
    Discontinuous: Continuity

class LagrangeVariant:
    Equispaced: LagrangeVariant
    Gll: LagrangeVariant
