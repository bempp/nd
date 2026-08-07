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
    let mut b = SingleElementMeshBuilder::<f64>::new(3, (ReferenceCellType::Quadrilateral, 2));
    b.add_point(0, &[0.0, 0.0, 0.0]);
    b.add_point(1, &[0.5, 0.0, 0.0]);
    b.add_point(2, &[1.0, 0.0, 0.0]);
    b.add_point(3, &[1.5, 0.0, 0.0]);
    b.add_point(4, &[2.0, 0.0, 0.0]);
    b.add_point(5, &[0.0, 0.5, 0.0]);
    b.add_point(6, &[0.5, 0.5, 0.0]);
    b.add_point(7, &[1.0, 0.5, 0.0]);
    b.add_point(8, &[1.5, 0.5, 0.0]);
    b.add_point(9, &[2.0, 0.5, 0.0]);
    b.add_point(10, &[0.0, 1.0, 0.0]);
    b.add_point(11, &[0.5, 1.0, 0.0]);
    b.add_point(12, &[1.0, 1.0, 0.0]);
    b.add_point(13, &[1.5, 1.0, 0.0]);
    b.add_point(14, &[2.0, 1.0, 0.0]);

    // Add two cells
    b.add_cell(0, &[0, 2, 10, 12, 1, 5, 7, 11, 6]);
    b.add_cell(1, &[2, 4, 12, 14, 3, 7, 9, 13, 8]);
    // Create the mesh
    let mesh = b.create_mesh();

    println!("Mesh generated.");

    // Print the coordinates or each point in the mesh
    let mut coords = vec![0.0; mesh.geometry_dim()];
    for point in mesh.entity_iter(ReferenceCellType::Point) {
        point.geometry().points().collect::<Vec<_>>()[0].coords(coords.as_mut_slice());
        println!("point {}: {:#?}", point.local_index(), coords);
    }

    // Print the vertices of each cell
    for cell in mesh.entity_iter(ReferenceCellType::Quadrilateral) {
        println!(
            "Degree: {}. Point count: {}",
            cell.geometry().degree(),
            cell.geometry().point_count()
        );
        println!(
            "cell {}: {:?} ",
            cell.local_index(),
            cell.topology()
                .sub_entity_iter(ReferenceCellType::Point)
                .collect::<Vec<_>>()
        );
    }
}
