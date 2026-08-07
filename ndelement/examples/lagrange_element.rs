use itertools::{self, izip};
use ndelement::ciarlet::lagrange;
use ndelement::reference_cell::faces;
use ndelement::{
    traits::FiniteElement,
    types::{Continuity, ReferenceCellType},
};

fn main() {
    // Create a hexahedron of degree 4
    let element = lagrange::create::<f64, f64>(
        ReferenceCellType::Hexahedron,
        4,
        Continuity::Standard,
        lagrange::LagrangeVariant::Equispaced,
    );

    // We want the dofs and dof position associated with the sixth face
    let entity_number = 5; // Entity 5 is the sixth face

    // Print all topological vertex indices associated with this face
    let face_indices = faces(ReferenceCellType::Hexahedron);
    println!(
        "Vertices of face {}:{:#?}",
        entity_number, face_indices[entity_number]
    );

    // We now print the dof indices and the associated points for that face
    let dofs = element.entity_dofs(2, entity_number).unwrap();
    let int_points = element.interpolation_points();
    for (dof, int_point) in izip!(dofs, int_points[2][entity_number].col_iter()) {
        println!(
            "{}: [{}, {}, {}]",
            dof,
            int_point[[0]],
            int_point[[1]],
            int_point[[2]]
        );
    }
}
