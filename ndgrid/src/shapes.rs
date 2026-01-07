//! Functions to create simple example grids

mod cube;
mod regular_sphere;
mod screen;

pub use cube::{
    unit_cube, unit_cube_boundary, unit_cube_edges, unit_interval, unit_square,
    unit_square_boundary,
};
pub use regular_sphere::regular_sphere;
pub use screen::{screen_quadrilaterals, screen_triangles};
#[cfg(feature = "mpi")]
pub use cube::{
    unit_cube_distributed, unit_cube_boundary_distributed, unit_cube_edges_distributed, unit_interval_distributed, unit_square_distributed,
    unit_square_boundary_distributed,
};
/*
#[cfg(feature = "mpi")]
pub use regular_sphere::regular_sphere_distributed;
#[cfg(feature = "mpi")]
pub use screen::{screen_quadrilaterals_distributed, screen_triangles_distributed};
*/
