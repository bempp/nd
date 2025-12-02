use ndelement::types::ReferenceCellType;
use ndgrid::{
    SingleElementGridBuilder,
    traits::{Builder, GmshImport},
};
use std::path::{PathBuf, absolute};

fn relative_file(filename: &str) -> String {
    let file = PathBuf::from(file!());
    // TODO: Revert to using these two lines once https://github.com/rust-lang/rust/issues/149536 is fixed
    // let dir = absolute(file.parent().unwrap()).unwrap();
    // format!("{}/{filename}", dir.display())
    let mut dir = absolute(file).unwrap();
    dir = dir.parent().unwrap().to_path_buf();
    while dir.display().to_string().ends_with("/ndgrid") || dir.display().to_string().ends_with("/tests") {
        dir = dir.parent().unwrap().to_path_buf();
    }
    format!("{}/ndgrid/tests/{filename}", dir.display())
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
