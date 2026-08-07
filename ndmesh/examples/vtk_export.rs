use ndelement::types::ReferenceCellType;
use ndmesh::{shapes, traits::Mesh};

fn main() {
    let triangle_mesh = shapes::regular_sphere::<f64>(1, ReferenceCellType::Triangle, 4);
    let quadrilateral_mesh = shapes::regular_sphere::<f64>(1, ReferenceCellType::Quadrilateral, 4);
    let tetra_mesh = shapes::unit_cube::<f64>(2, 1, 1, ReferenceCellType::Tetrahedron, 4);
    let hexa_mesh = shapes::unit_cube::<f64>(2, 1, 1, ReferenceCellType::Hexahedron, 3);

    triangle_mesh.as_vtk().export("triangle.vtu").unwrap();
    quadrilateral_mesh
        .as_vtk()
        .export("quadrilateral.vtu")
        .unwrap();
    tetra_mesh.as_vtk().export("tetra.vtu").unwrap();
    hexa_mesh.as_vtk().export("hexa.vtu").unwrap();
}
