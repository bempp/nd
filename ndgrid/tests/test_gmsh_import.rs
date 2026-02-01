use ndelement::types::ReferenceCellType;
use ndgrid::{
    SingleElementGridBuilder,
    traits::{Builder, GmshImport},
};

fn relative_file(filename: &str) -> String {
    format!("tests/{filename}")
}

#[test]
fn test_gmsh_import_v1() {
    let mut b = SingleElementGridBuilder::<f64>::new(3, (ReferenceCellType::Triangle, 1));
    b.import_from_gmsh(&relative_file("test_mesh_v1.msh"));
    let _g = b.create_grid();
}

#[test]
fn test_gmsh_import_ascii_v2() {
    let mut b = SingleElementGridBuilder::<f64>::new(3, (ReferenceCellType::Triangle, 1));
    b.import_from_gmsh(&relative_file("test_mesh_ascii_v2.msh"));
    let _g = b.create_grid();
}

#[test]
fn test_gmsh_import_binary_v2() {
    let mut b = SingleElementGridBuilder::<f64>::new(3, (ReferenceCellType::Triangle, 1));
    b.import_from_gmsh(&relative_file("test_mesh_binary_v2.msh"));
    let _g = b.create_grid();
}

#[test]
fn test_gmsh_import_ascii_v4() {
    let mut b = SingleElementGridBuilder::<f64>::new(3, (ReferenceCellType::Triangle, 1));
    b.import_from_gmsh(&relative_file("test_mesh_ascii_v4.msh"));
    let _g = b.create_grid();
}

#[test]
fn test_gmsh_import_binary_v4() {
    let mut b = SingleElementGridBuilder::<f64>::new(3, (ReferenceCellType::Triangle, 1));
    b.import_from_gmsh(&relative_file("test_mesh_binary_v4.msh"));
    let _g = b.create_grid();
}

#[test]
fn test_gmsh_import_ascii_v4_parametric() {
    let mut b = SingleElementGridBuilder::<f64>::new(3, (ReferenceCellType::Triangle, 1));
    b.import_from_gmsh(&relative_file("test_mesh_parametric_ascii_v4.msh"));

    // Verify that some points have parametric coordinates stored
    // The mesh has nodes on curves (dim=1) and surfaces (dim=2) with parametric coordinates
    let has_parametric = (0..b.point_count()).any(|i| b.point_parametric_coords(i).is_some());
    assert!(
        has_parametric,
        "Expected some points to have parametric coordinates"
    );

    // Verify parametric coords have correct length for their entity dimension
    for i in 0..b.point_count() {
        if let Some((entity_dim, coords)) = b.point_parametric_coords(i) {
            assert!(
                (1..=3).contains(&entity_dim),
                "entity_dim should be 1, 2, or 3"
            );
            assert_eq!(
                coords.len(),
                entity_dim.min(3),
                "Parametric coords length should match entity_dim"
            );
        }
    }

    let _g = b.create_grid();
}

#[test]
fn test_gmsh_import_binary_v4_parametric() {
    let mut b = SingleElementGridBuilder::<f64>::new(3, (ReferenceCellType::Triangle, 1));
    b.import_from_gmsh(&relative_file("test_mesh_parametric_binary_v4.msh"));

    // Verify that some points have parametric coordinates stored
    let has_parametric = (0..b.point_count()).any(|i| b.point_parametric_coords(i).is_some());
    assert!(
        has_parametric,
        "Expected some points to have parametric coordinates"
    );

    // Verify parametric coords have correct length for their entity dimension
    for i in 0..b.point_count() {
        if let Some((entity_dim, coords)) = b.point_parametric_coords(i) {
            assert!(
                (1..=3).contains(&entity_dim),
                "entity_dim should be 1, 2, or 3"
            );
            assert_eq!(
                coords.len(),
                entity_dim.min(3),
                "Parametric coords length should match entity_dim"
            );
        }
    }

    let _g = b.create_grid();
}
