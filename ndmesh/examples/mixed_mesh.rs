use ndelement::types::ReferenceCellType;
use ndmesh::mesh::local_mesh::MixedMeshBuilder;
use ndmesh::traits::{Builder, Entity, Geometry, GmshExport, Mesh, Point, Topology};

/// Creating a (serial) mixed mesh
///
/// In a mixed mesh, multiple cell types can be present and mutiple finite elements can be used to
/// represent the geometry of cells.
fn main() {
    // When creating the mesh builder, we give the physical/geometric dimension (3)
    let mut b = MixedMeshBuilder::<f64>::new(2);
    // Add nine points with ids 0 to 8
    b.add_point(0, &[0.0, 0.0]);
    b.add_point(1, &[0.5, -0.3]);
    b.add_point(2, &[1.0, 0.0]);
    b.add_point(3, &[2.0, 0.0]);
    b.add_point(4, &[-0.3, 0.5]);
    b.add_point(5, &[0.5, 0.5]);
    b.add_point(6, &[0.0, 1.0]);
    b.add_point(7, &[1.0, 1.0]);
    b.add_point(8, &[2.0, 1.0]);
    // Add a linear triangle cell. The inputs to add_cell are
    // (id, (cell type, cell degree, vertices))
    b.add_cell(0, (ReferenceCellType::Triangle, 1, &[2, 7, 6]));
    // Add a quadratic triangle cell. The edge that is shared with cell 0 is straight to ensure
    // that there are no discontinuities in the mesh.
    b.add_cell(1, (ReferenceCellType::Triangle, 2, &[0, 2, 6, 5, 4, 1]));
    // Add a quadrilateral cell
    b.add_cell(2, (ReferenceCellType::Quadrilateral, 1, &[2, 3, 7, 8]));
    // Create the mesh
    let mesh = b.create_mesh();

    // Print the vertices of each triangle cell: this will only include the three points
    // for each cell as these are the topological vertices (ie the 0-dimensional entities at
    // each corner of the cell), not the geometric points
    for cell in mesh.entity_iter(ReferenceCellType::Triangle) {
        println!(
            "triangle cell {}: {:?}",
            cell.local_index(),
            cell.topology()
                .sub_entity_iter(ReferenceCellType::Point)
                .collect::<Vec<_>>(),
        );
    }
    // Print the geometric points that definie each cells position in space. Note that
    // The indexing of these points may differ from the indexing used for the topological vertices.
    let mut coords = vec![0.0; 2];
    for cell in mesh.entity_iter(ReferenceCellType::Triangle) {
        println!("triangle cell {}:", cell.local_index());
        for p in cell.geometry().points() {
            p.coords(&mut coords);
            println!("  point {}: {:?}", p.index(), coords);
        }
    }
    // Print the vertices of each quadrilateral cell
    for cell in mesh.entity_iter(ReferenceCellType::Quadrilateral) {
        println!(
            "quadrilateral cell {}: {:?} ",
            cell.local_index(),
            cell.topology()
                .sub_entity_iter(ReferenceCellType::Point)
                .collect::<Vec<_>>()
        );
    }

    // Export the mesh in Gmsh format
    mesh.export_as_gmsh("_mixed.msh");
}
