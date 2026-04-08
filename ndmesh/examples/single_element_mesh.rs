use ndelement::types::ReferenceCellType;
use ndmesh::mesh::local_mesh::SingleElementMeshBuilder;
use ndmesh::traits::{Builder, Entity, Geometry, Mesh, Point, Topology};

/// Creating a (serial) single element mesh
///
/// In a single element mesh, the same finite element will be used to represent the geometry
/// of each cell. For example, a mesh of bilinear quadrilaterals can be created by using a degree 1
/// element on a quadrilateral
fn main() {
    // When creating the mesh builder, we give the physical/geometric dimension (3) and the cell type
    // and degree of the element
    let mut b = SingleElementMeshBuilder::<f64>::new(3, (ReferenceCellType::Quadrilateral, 1));
    // Add six points with ids 0 to 5
    b.add_point(0, &[0.0, 0.0, 0.0]);
    b.add_point(1, &[1.0, 0.0, 0.0]);
    b.add_point(2, &[2.0, 0.0, 0.2]);
    b.add_point(3, &[0.0, 1.0, 0.0]);
    b.add_point(4, &[1.0, 1.0, -0.2]);
    b.add_point(5, &[2.0, 1.0, 0.0]);
    // Add two cells
    b.add_cell(0, &[0, 1, 3, 4]);
    b.add_cell(1, &[1, 2, 4, 5]);
    // Create the mesh
    let mesh = b.create_mesh();

    // Print the coordinates or each point in the mesh
    let mut coords = vec![0.0; mesh.geometry_dim()];
    for point in mesh.entity_iter(ReferenceCellType::Point) {
        point.geometry().points().collect::<Vec<_>>()[0].coords(coords.as_mut_slice());
        println!("point {}: {:#?}", point.local_index(), coords);
    }

    // Print the vertices of each cell
    for cell in mesh.entity_iter(ReferenceCellType::Quadrilateral) {
        println!(
            "cell {}: {:?} ",
            cell.local_index(),
            cell.topology()
                .sub_entity_iter(ReferenceCellType::Point)
                .collect::<Vec<_>>()
        );
    }
}
