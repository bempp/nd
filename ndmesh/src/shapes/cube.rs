//! Cube meshes

use super::resample_cells;
#[cfg(feature = "mpi")]
use crate::{ParallelMeshImpl, traits::ParallelBuilder, types::GraphPartitioner};
use crate::{
    mesh::local_mesh::{SingleElementMesh, SingleElementMeshBuilder},
    traits::Builder,
    types::Scalar,
};
#[cfg(feature = "mpi")]
use mpi::traits::{Communicator, Equivalence};
use ndelement::{ciarlet::CiarletElement, map::IdentityMap, types::ReferenceCellType};

/// Add points and cells for unit interval to builder
fn unit_interval_add_points_and_cells<T: Scalar>(
    b: &mut SingleElementMeshBuilder<T>,
    nx: usize,
    degree: usize,
) {
    for i in 0..nx + 1 {
        b.add_point(i, &[T::from(i).unwrap() / T::from(nx).unwrap()]);
    }

    let mut cells = vec![];
    for i in 0..nx {
        cells.push([i, i + 1]);
    }
    if degree == 1 {
        for (i, v) in cells.iter().enumerate() {
            b.add_cell(i, v);
        }
    } else {
        for (i, v) in resample_cells::<T, 1, 2>(degree, b, &cells, ReferenceCellType::Interval)
            .iter()
            .enumerate()
        {
            b.add_cell(i, v);
        }
    }
}

/// Add points and cells for unit square to builder
fn unit_square_add_points_and_cells<T: Scalar>(
    b: &mut SingleElementMeshBuilder<T>,
    nx: usize,
    ny: usize,
    cell_type: ReferenceCellType,
    degree: usize,
) {
    for i in 0..nx + 1 {
        for j in 0..ny + 1 {
            b.add_point(
                i * (ny + 1) + j,
                &[
                    T::from(i).unwrap() / T::from(nx).unwrap(),
                    T::from(j).unwrap() / T::from(ny).unwrap(),
                ],
            );
        }
    }

    let dx = ny + 1;
    let dy = 1;
    match cell_type {
        ReferenceCellType::Triangle => {
            let mut cells = vec![];
            for i in 0..nx {
                for j in 0..ny {
                    let origin = i * dx + j * dy;
                    cells.push([origin, origin + dx, origin + dx + dy]);
                    cells.push([origin, origin + dx + dy, origin + dy]);
                }
            }
            if degree == 1 {
                for (i, v) in cells.iter().enumerate() {
                    b.add_cell(i, v);
                }
            } else {
                for (i, v) in
                    resample_cells::<T, 2, 3>(degree, b, &cells, ReferenceCellType::Triangle)
                        .iter()
                        .enumerate()
                {
                    b.add_cell(i, v);
                }
            }
        }
        ReferenceCellType::Quadrilateral => {
            let mut cells = vec![];
            for i in 0..nx {
                for j in 0..ny {
                    let origin = i * dx + j * dy;
                    cells.push([origin, origin + dx, origin + dy, origin + dx + dy]);
                }
            }
            if degree == 1 {
                for (i, v) in cells.iter().enumerate() {
                    b.add_cell(i, v);
                }
            } else {
                for (i, v) in
                    resample_cells::<T, 2, 4>(degree, b, &cells, ReferenceCellType::Quadrilateral)
                        .iter()
                        .enumerate()
                {
                    b.add_cell(i, v);
                }
            }
        }
        _ => {
            panic!("Unsupported cell type: {cell_type:?}")
        }
    }
}

/// Add points and cells for unit square boundary to builder
fn unit_square_boundary_add_points_and_cells<T: Scalar>(
    b: &mut SingleElementMeshBuilder<T>,
    nx: usize,
    ny: usize,
    degree: usize,
) {
    let dx = ny + 1;
    let dy = 1;

    for i in 0..nx + 1 {
        b.add_point(
            i * dx,
            &[T::from(i).unwrap() / T::from(nx).unwrap(), T::zero()],
        );
        b.add_point(
            i * dx + ny * dy,
            &[T::from(i).unwrap() / T::from(nx).unwrap(), T::one()],
        );
    }
    for j in 1..ny {
        b.add_point(
            j * dy,
            &[T::zero(), T::from(j).unwrap() / T::from(ny).unwrap()],
        );
        b.add_point(
            nx * dx + j * dy,
            &[T::one(), T::from(j).unwrap() / T::from(ny).unwrap()],
        );
    }

    let mut cells = vec![];
    for i in 0..nx {
        let origin = i * dx;
        cells.push([origin, origin + dx]);
        let origin = i * dx + ny * dy;
        cells.push([origin + dx, origin]);
    }
    for j in 0..ny {
        let origin = j * dy;
        cells.push([origin + dy, origin]);
        let origin = nx * dx + j * dy;
        cells.push([origin, origin + dy]);
    }
    if degree == 1 {
        for (i, v) in cells.iter().enumerate() {
            b.add_cell(i, v);
        }
    } else {
        for (i, v) in resample_cells::<T, 2, 2>(degree, b, &cells, ReferenceCellType::Interval)
            .iter()
            .enumerate()
        {
            b.add_cell(i, v);
        }
    }
}

/// Add points and cells for unit cube to builder
fn unit_cube_add_points_and_cells<T: Scalar>(
    b: &mut SingleElementMeshBuilder<T>,
    nx: usize,
    ny: usize,
    nz: usize,
    cell_type: ReferenceCellType,
    degree: usize,
) {
    for i in 0..=nx {
        for j in 0..=ny {
            for k in 0..=nz {
                b.add_point(
                    (i * (ny + 1) + j) * (nz + 1) + k,
                    &[
                        T::from(i).unwrap() / T::from(nx).unwrap(),
                        T::from(j).unwrap() / T::from(ny).unwrap(),
                        T::from(k).unwrap() / T::from(nz).unwrap(),
                    ],
                );
            }
        }
    }

    let dx = (ny + 1) * (nz + 1);
    let dy = nz + 1;
    let dz = 1;
    match cell_type {
        ReferenceCellType::Tetrahedron => {
            let mut cells = vec![];
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        let origin = i * dx + j * dy + k * dz;
                        cells.push([origin, origin + dx, origin + dx + dy, origin + dx + dy + dz]);
                        cells.push([origin, origin + dy, origin + dx + dy, origin + dx + dy + dz]);
                        cells.push([origin, origin + dx, origin + dx + dz, origin + dx + dy + dz]);
                        cells.push([origin, origin + dz, origin + dx + dz, origin + dx + dy + dz]);
                        cells.push([origin, origin + dy, origin + dy + dz, origin + dx + dy + dz]);
                        cells.push([origin, origin + dz, origin + dy + dz, origin + dx + dy + dz]);
                    }
                }
            }
            if degree == 1 {
                for (i, v) in cells.iter().enumerate() {
                    b.add_cell(i, v);
                }
            } else {
                for (i, v) in
                    resample_cells::<T, 3, 4>(degree, b, &cells, ReferenceCellType::Tetrahedron)
                        .iter()
                        .enumerate()
                {
                    b.add_cell(i, v);
                }
            }
        }
        ReferenceCellType::Hexahedron => {
            let mut cells = vec![];
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        let origin = i * dx + j * dy + k * dz;
                        cells.push([
                            origin,
                            origin + dx,
                            origin + dy,
                            origin + dx + dy,
                            origin + dz,
                            origin + dx + dz,
                            origin + dy + dz,
                            origin + dx + dy + dz,
                        ]);
                    }
                }
            }
            if degree == 1 {
                for (i, v) in cells.iter().enumerate() {
                    b.add_cell(i, v);
                }
            } else {
                for (i, v) in
                    resample_cells::<T, 3, 8>(degree, b, &cells, ReferenceCellType::Hexahedron)
                        .iter()
                        .enumerate()
                {
                    b.add_cell(i, v);
                }
            }
        }
        _ => {
            panic!("Unsupported cell type: {cell_type:?}")
        }
    }
}

/// Add points and cells for unit cube boundary to builder
fn unit_cube_boundary_add_points_and_cells<T: Scalar>(
    b: &mut SingleElementMeshBuilder<T>,
    nx: usize,
    ny: usize,
    nz: usize,
    cell_type: ReferenceCellType,
    degree: usize,
) {
    for i in 0..nx + 1 {
        for j in 0..ny + 1 {
            for k in if i == 0 || i == nx || j == 0 || j == ny {
                (0..nz + 1).collect::<Vec<_>>()
            } else {
                vec![0, nz]
            } {
                b.add_point(
                    (i * (ny + 1) + j) * (nz + 1) + k,
                    &[
                        T::from(i).unwrap() / T::from(nx).unwrap(),
                        T::from(j).unwrap() / T::from(ny).unwrap(),
                        T::from(k).unwrap() / T::from(nz).unwrap(),
                    ],
                );
            }
        }
    }

    let dx = (ny + 1) * (nz + 1);
    let dy = nz + 1;
    let dz = 1;
    match cell_type {
        ReferenceCellType::Triangle => {
            let mut cells = vec![];
            for i in 0..nx {
                for j in 0..ny {
                    let origin = i * dx + j * dy;
                    cells.push([origin, origin + dx + dy, origin + dx]);
                    cells.push([origin, origin + dy, origin + dx + dy]);
                    let origin = i * dx + j * dy + nz * dz;
                    cells.push([origin, origin + dx, origin + dx + dy]);
                    cells.push([origin, origin + dx + dy, origin + dy]);
                }
            }
            for i in 0..nx {
                for k in 0..nz {
                    let origin = i * dx + k * dz;
                    cells.push([origin, origin + dx, origin + dx + dz]);
                    cells.push([origin, origin + dx + dz, origin + dz]);
                    let origin = i * dx + ny * dy + k * dz;
                    cells.push([origin, origin + dx + dz, origin + dx]);
                    cells.push([origin, origin + dz, origin + dx + dz]);
                }
            }
            for j in 0..ny {
                for k in 0..nz {
                    let origin = j * dy + k * dz;
                    cells.push([origin, origin + dy + dz, origin + dy]);
                    cells.push([origin, origin + dz, origin + dy + dz]);
                    let origin = nx * dx + j * dy + k * dz;
                    cells.push([origin, origin + dy, origin + dy + dz]);
                    cells.push([origin, origin + dy + dz, origin + dz]);
                }
            }
            if degree == 1 {
                for (i, v) in cells.iter().enumerate() {
                    b.add_cell(i, v);
                }
            } else {
                for (i, v) in
                    resample_cells::<T, 3, 3>(degree, b, &cells, ReferenceCellType::Triangle)
                        .iter()
                        .enumerate()
                {
                    b.add_cell(i, v);
                }
            }
        }
        ReferenceCellType::Quadrilateral => {
            let mut cells = vec![];
            for i in 0..nx {
                for j in 0..ny {
                    let origin = i * dx + j * dy;
                    cells.push([origin, origin + dy, origin + dx, origin + dx + dy]);
                    let origin = i * dx + j * dy + nz * dz;
                    cells.push([origin, origin + dx, origin + dy, origin + dx + dy]);
                }
            }
            for i in 0..nx {
                for k in 0..nz {
                    let origin = i * dx + k * dz;
                    cells.push([origin, origin + dx, origin + dz, origin + dx + dz]);
                    let origin = i * dx + ny * dy + k * dz;
                    cells.push([origin, origin + dz, origin + dx, origin + dx + dz]);
                }
            }
            for j in 0..ny {
                for k in 0..nz {
                    let origin = j * dy + k * dz;
                    cells.push([origin, origin + dz, origin + dy, origin + dy + dz]);
                    let origin = nx * dx + j * dy + k * dz;
                    cells.push([origin, origin + dy, origin + dz, origin + dy + dz]);
                }
            }
            if degree == 1 {
                for (i, v) in cells.iter().enumerate() {
                    b.add_cell(i, v);
                }
            } else {
                for (i, v) in
                    resample_cells::<T, 3, 4>(degree, b, &cells, ReferenceCellType::Quadrilateral)
                        .iter()
                        .enumerate()
                {
                    b.add_cell(i, v);
                }
            }
        }
        _ => {
            panic!("Unsupported cell type: {cell_type:?}")
        }
    }
}

/// Add points and cells for unit cube edges to builder
fn unit_cube_edges_add_points_and_cells<T: Scalar>(
    b: &mut SingleElementMeshBuilder<T>,
    nx: usize,
    ny: usize,
    nz: usize,
    degree: usize,
) {
    for i in 0..nx + 1 {
        for j in if i == 0 || i == nx {
            (0..ny + 1).collect::<Vec<_>>()
        } else {
            vec![0, ny]
        } {
            for k in if (i == 0 || i == nx) && (j == 0 || j == ny) {
                (0..nz + 1).collect::<Vec<_>>()
            } else {
                vec![0, nz]
            } {
                b.add_point(
                    (i * (ny + 1) + j) * (nz + 1) + k,
                    &[
                        T::from(i).unwrap() / T::from(nx).unwrap(),
                        T::from(j).unwrap() / T::from(ny).unwrap(),
                        T::from(k).unwrap() / T::from(nz).unwrap(),
                    ],
                );
            }
        }
    }

    let dx = (ny + 1) * (nz + 1);
    let dy = nz + 1;
    let dz = 1;
    let mut cells = vec![];
    for i in 0..nx {
        for origin in [
            i * dx,
            i * dx + ny * dy,
            i * dx + nz * dz,
            i * dx + ny * dy + nz * dz,
        ] {
            cells.push([origin, origin + dx]);
        }
    }
    for j in 0..ny {
        for origin in [
            j * dy,
            j * dy + nx * dx,
            j * dy + nz * dz,
            j * dy + nx * dx + nz * dz,
        ] {
            cells.push([origin, origin + dy]);
        }
    }
    for k in 0..nz {
        for origin in [
            k * dz,
            k * dz + nx * dx,
            k * dz + ny * dy,
            k * dz + nx * dx + ny * dy,
        ] {
            cells.push([origin, origin + dz]);
        }
    }
    if degree == 1 {
        for (i, v) in cells.iter().enumerate() {
            b.add_cell(i, v);
        }
    } else {
        for (i, v) in resample_cells::<T, 3, 2>(degree, b, &cells, ReferenceCellType::Interval)
            .iter()
            .enumerate()
        {
            b.add_cell(i, v);
        }
    }
}

/// Create a unit interval mesh
///
/// The unit interval is the interval between (0,) and (1,)
pub fn unit_interval<T: Scalar>(
    nx: usize,
    degree: usize,
) -> SingleElementMesh<T, CiarletElement<T, IdentityMap, T>> {
    let mut b = SingleElementMeshBuilder::new(1, (ReferenceCellType::Interval, degree));
    unit_interval_add_points_and_cells(&mut b, nx, degree);
    b.create_mesh()
}

/// Create a unit interval mesh distributed in parallel
#[cfg(feature = "mpi")]
pub fn unit_interval_distributed<T: Scalar + Equivalence, C: Communicator>(
    comm: &C,
    partitioner: GraphPartitioner,
    nx: usize,
    degree: usize,
) -> ParallelMeshImpl<'_, C, SingleElementMesh<T, CiarletElement<T, IdentityMap, T>>> {
    let mut b = SingleElementMeshBuilder::new(1, (ReferenceCellType::Interval, 1));
    if comm.rank() == 0 {
        unit_interval_add_points_and_cells(&mut b, nx, degree);
        b.create_parallel_mesh_root(comm, partitioner)
    } else {
        b.create_parallel_mesh(comm, 0)
    }
}

/// Create a unit square mesh
///
/// The unit square is the square with corners at (0,0), (1,0), (0,1) and (1,1)
pub fn unit_square<T: Scalar>(
    nx: usize,
    ny: usize,
    cell_type: ReferenceCellType,
    degree: usize,
) -> SingleElementMesh<T, CiarletElement<T, IdentityMap, T>> {
    let mut b = SingleElementMeshBuilder::new(2, (cell_type, degree));
    unit_square_add_points_and_cells(&mut b, nx, ny, cell_type, degree);
    b.create_mesh()
}

/// Create a unit square mesh distributed in parallel
#[cfg(feature = "mpi")]
pub fn unit_square_distributed<T: Scalar + Equivalence, C: Communicator>(
    comm: &C,
    partitioner: GraphPartitioner,
    nx: usize,
    ny: usize,
    cell_type: ReferenceCellType,
    degree: usize,
) -> ParallelMeshImpl<'_, C, SingleElementMesh<T, CiarletElement<T, IdentityMap, T>>> {
    let mut b = SingleElementMeshBuilder::new(1, (cell_type, 1));
    if comm.rank() == 0 {
        unit_square_add_points_and_cells(&mut b, nx, ny, cell_type, degree);
        b.create_parallel_mesh_root(comm, partitioner)
    } else {
        b.create_parallel_mesh(comm, 0)
    }
}

/// Create a mesh of the boundary of a unit square
///
/// The unit square is the square with corners at (0,0), (1,0), (0,1) and (1,1)
pub fn unit_square_boundary<T: Scalar>(
    nx: usize,
    ny: usize,
    degree: usize,
) -> SingleElementMesh<T, CiarletElement<T, IdentityMap, T>> {
    let mut b = SingleElementMeshBuilder::new(2, (ReferenceCellType::Interval, degree));
    unit_square_boundary_add_points_and_cells(&mut b, nx, ny, degree);
    b.create_mesh()
}

/// Create a mesh of the boundary distributed in parallel
#[cfg(feature = "mpi")]
pub fn unit_square_boundary_distributed<T: Scalar + Equivalence, C: Communicator>(
    comm: &C,
    partitioner: GraphPartitioner,
    nx: usize,
    ny: usize,
    cell_type: ReferenceCellType,
    degree: usize,
) -> ParallelMeshImpl<'_, C, SingleElementMesh<T, CiarletElement<T, IdentityMap, T>>> {
    let mut b = SingleElementMeshBuilder::new(1, (cell_type, 1));
    if comm.rank() == 0 {
        unit_square_boundary_add_points_and_cells(&mut b, nx, ny, degree);
        b.create_parallel_mesh_root(comm, partitioner)
    } else {
        b.create_parallel_mesh(comm, 0)
    }
}

/// Create a unit cube mesh
///
/// The unit cube is the cube with corners at (0,0,0), (1,0,0), (0,1,0), (1,1,0), (0,0,1),
/// (1,0,1), (0,1,1) and (1,1,1)
pub fn unit_cube<T: Scalar>(
    nx: usize,
    ny: usize,
    nz: usize,
    cell_type: ReferenceCellType,
    degree: usize,
) -> SingleElementMesh<T, CiarletElement<T, IdentityMap, T>> {
    let mut b = SingleElementMeshBuilder::new(3, (cell_type, degree));

    unit_cube_add_points_and_cells(&mut b, nx, ny, nz, cell_type, degree);
    b.create_mesh()
}

/// Create a unit cube mesh distributed in parallel
#[cfg(feature = "mpi")]
pub fn unit_cube_distributed<T: Scalar + Equivalence, C: Communicator>(
    comm: &C,
    partitioner: GraphPartitioner,
    nx: usize,
    ny: usize,
    nz: usize,
    cell_type: ReferenceCellType,
    degree: usize,
) -> ParallelMeshImpl<'_, C, SingleElementMesh<T, CiarletElement<T, IdentityMap, T>>> {
    let mut b = SingleElementMeshBuilder::new(3, (cell_type, 1));
    if comm.rank() == 0 {
        unit_cube_add_points_and_cells(&mut b, nx, ny, nz, cell_type, degree);
        b.create_parallel_mesh_root(comm, partitioner)
    } else {
        b.create_parallel_mesh(comm, 0)
    }
}

/// Create a mesh of the boundary of a unit cube
///
/// The unit cube is the cube with corners at (0,0,0), (1,0,0), (0,1,0), (1,1,0), (0,0,1),
/// (1,0,1), (0,1,1) and (1,1,1)
pub fn unit_cube_boundary<T: Scalar>(
    nx: usize,
    ny: usize,
    nz: usize,
    cell_type: ReferenceCellType,
    degree: usize,
) -> SingleElementMesh<T, CiarletElement<T, IdentityMap, T>> {
    let mut b = SingleElementMeshBuilder::new(3, (cell_type, degree));
    unit_cube_boundary_add_points_and_cells(&mut b, nx, ny, nz, cell_type, degree);
    b.create_mesh()
}

/// Create a mesh of the boundary of a unit cube distributed in parallel
#[cfg(feature = "mpi")]
pub fn unit_cube_boundary_distributed<T: Scalar + Equivalence, C: Communicator>(
    comm: &C,
    partitioner: GraphPartitioner,
    nx: usize,
    ny: usize,
    nz: usize,
    cell_type: ReferenceCellType,
    degree: usize,
) -> ParallelMeshImpl<'_, C, SingleElementMesh<T, CiarletElement<T, IdentityMap, T>>> {
    let mut b = SingleElementMeshBuilder::new(3, (cell_type, 1));
    if comm.rank() == 0 {
        unit_cube_boundary_add_points_and_cells(&mut b, nx, ny, nz, cell_type, degree);
        b.create_parallel_mesh_root(comm, partitioner)
    } else {
        b.create_parallel_mesh(comm, 0)
    }
}

/// Create a mesh of the edges of a unit cube
///
/// The unit cube is the cube with corners at (0,0,0), (1,0,0), (0,1,0), (1,1,0), (0,0,1),
/// (1,0,1), (0,1,1) and (1,1,1)
pub fn unit_cube_edges<T: Scalar>(
    nx: usize,
    ny: usize,
    nz: usize,
    degree: usize,
) -> SingleElementMesh<T, CiarletElement<T, IdentityMap, T>> {
    let mut b = SingleElementMeshBuilder::new(3, (ReferenceCellType::Interval, degree));
    unit_cube_edges_add_points_and_cells(&mut b, nx, ny, nz, degree);
    b.create_mesh()
}

/// Create a mesh of the edges of a unit cube distributed in parallel
#[cfg(feature = "mpi")]
pub fn unit_cube_edges_distributed<T: Scalar + Equivalence, C: Communicator>(
    comm: &C,
    partitioner: GraphPartitioner,
    nx: usize,
    ny: usize,
    nz: usize,
    degree: usize,
) -> ParallelMeshImpl<'_, C, SingleElementMesh<T, CiarletElement<T, IdentityMap, T>>> {
    let mut b = SingleElementMeshBuilder::new(3, (ReferenceCellType::Interval, 1));
    if comm.rank() == 0 {
        unit_cube_edges_add_points_and_cells(&mut b, nx, ny, nz, degree);
        b.create_parallel_mesh_root(comm, partitioner)
    } else {
        b.create_parallel_mesh(comm, 0)
    }
}

#[cfg(test)]
mod test {
    use super::super::test::{test_normals_are_outward, test_normals_are_unit};
    use super::*;
    use crate::traits::{Entity, Geometry, Mesh, Point};
    use approx::*;
    use itertools::izip;
    use rlst::rlst_dynamic_array;

    fn max(values: &[f64]) -> f64 {
        let mut out = values[0];
        for i in &values[1..] {
            if *i > out {
                out = *i;
            }
        }
        out
    }

    fn check_volume(
        mesh: &impl Mesh<T = f64, EntityDescriptor = ReferenceCellType>,
        expected_volume: f64,
    ) {
        let mut volume = 0.0;
        let gdim = mesh.geometry_dim();
        for t in mesh.cell_types() {
            for cell in mesh.entity_iter(*t) {
                let g = cell.geometry();
                let mut point = vec![0.0; gdim];
                let mut min_p = vec![10.0; gdim];
                let mut max_p = vec![-10.0; gdim];
                for p in g.points() {
                    p.coords(&mut point);
                    for (j, v) in point.iter().enumerate() {
                        if *v < min_p[j] {
                            min_p[j] = *v;
                        }
                        if *v > max_p[j] {
                            max_p[j] = *v;
                        }
                    }
                }
                volume += match cell.entity_type() {
                    ReferenceCellType::Interval => {
                        max(&izip!(min_p, max_p).map(|(i, j)| j - i).collect::<Vec<_>>())
                    }
                    ReferenceCellType::Triangle => match gdim {
                        2 => (max_p[0] - min_p[0]) * (max_p[1] - min_p[1]) / 2.0,
                        3 => max(&[
                            (max_p[0] - min_p[0]) * (max_p[1] - min_p[1]) / 2.0,
                            (max_p[0] - min_p[0]) * (max_p[2] - min_p[2]) / 2.0,
                            (max_p[1] - min_p[1]) * (max_p[2] - min_p[2]) / 2.0,
                        ]),
                        _ => {
                            panic!("Unsupported dimension");
                        }
                    },
                    ReferenceCellType::Quadrilateral => match gdim {
                        2 => (max_p[0] - min_p[0]) * (max_p[1] - min_p[1]),
                        3 => max(&[
                            (max_p[0] - min_p[0]) * (max_p[1] - min_p[1]),
                            (max_p[0] - min_p[0]) * (max_p[2] - min_p[2]),
                            (max_p[1] - min_p[1]) * (max_p[2] - min_p[2]),
                        ]),
                        _ => {
                            panic!("Unsupported dimension");
                        }
                    },
                    ReferenceCellType::Tetrahedron => {
                        (max_p[0] - min_p[0]) * (max_p[1] - min_p[1]) * (max_p[2] - min_p[2]) / 6.0
                    }
                    ReferenceCellType::Hexahedron => {
                        (max_p[0] - min_p[0]) * (max_p[1] - min_p[1]) * (max_p[2] - min_p[2])
                    }
                    _ => {
                        panic!("Unsupported cell");
                    }
                };
            }
        }
        assert_relative_eq!(volume, expected_volume, epsilon = 1e-10);
    }

    #[test]
    fn test_unit_interval() {
        check_volume(&unit_interval::<f64>(1, 1), 1.0);
        check_volume(&unit_interval::<f64>(2, 1), 1.0);
        check_volume(&unit_interval::<f64>(4, 1), 1.0);
        check_volume(&unit_interval::<f64>(7, 1), 1.0);
    }

    #[test]
    fn test_unit_square_triangle() {
        check_volume(
            &unit_square::<f64>(1, 1, ReferenceCellType::Triangle, 1),
            1.0,
        );
        check_volume(
            &unit_square::<f64>(2, 2, ReferenceCellType::Triangle, 1),
            1.0,
        );
        check_volume(
            &unit_square::<f64>(4, 5, ReferenceCellType::Triangle, 1),
            1.0,
        );
        check_volume(
            &unit_square::<f64>(7, 6, ReferenceCellType::Triangle, 1),
            1.0,
        );
    }

    #[test]
    fn test_unit_square_quadrilateral() {
        check_volume(
            &unit_square::<f64>(1, 1, ReferenceCellType::Quadrilateral, 1),
            1.0,
        );
        check_volume(
            &unit_square::<f64>(2, 2, ReferenceCellType::Quadrilateral, 1),
            1.0,
        );
        check_volume(
            &unit_square::<f64>(4, 5, ReferenceCellType::Quadrilateral, 1),
            1.0,
        );
        check_volume(
            &unit_square::<f64>(7, 6, ReferenceCellType::Quadrilateral, 1),
            1.0,
        );
    }

    #[test]
    fn test_unit_square_boundary() {
        check_volume(&unit_square_boundary::<f64>(1, 1, 1), 4.0);
        check_volume(&unit_square_boundary::<f64>(2, 2, 1), 4.0);
        check_volume(&unit_square_boundary::<f64>(4, 5, 1), 4.0);
        check_volume(&unit_square_boundary::<f64>(7, 6, 1), 4.0);
    }

    #[test]
    fn test_unit_cube_boundary_triangle() {
        check_volume(
            &unit_cube_boundary::<f64>(1, 1, 1, ReferenceCellType::Triangle, 1),
            6.0,
        );
        check_volume(
            &unit_cube_boundary::<f64>(2, 2, 2, ReferenceCellType::Triangle, 1),
            6.0,
        );
        check_volume(
            &unit_cube_boundary::<f64>(4, 5, 5, ReferenceCellType::Triangle, 1),
            6.0,
        );
        check_volume(
            &unit_cube_boundary::<f64>(7, 6, 4, ReferenceCellType::Triangle, 1),
            6.0,
        );
    }

    #[test]
    fn test_unit_cube_boundary_quadrilateral() {
        check_volume(
            &unit_cube_boundary::<f64>(1, 1, 1, ReferenceCellType::Quadrilateral, 1),
            6.0,
        );
        check_volume(
            &unit_cube_boundary::<f64>(2, 2, 2, ReferenceCellType::Quadrilateral, 1),
            6.0,
        );
        check_volume(
            &unit_cube_boundary::<f64>(4, 5, 5, ReferenceCellType::Quadrilateral, 1),
            6.0,
        );
        check_volume(
            &unit_cube_boundary::<f64>(7, 6, 4, ReferenceCellType::Quadrilateral, 1),
            6.0,
        );
    }

    #[test]
    fn test_unit_cube_tetrahedron() {
        check_volume(
            &unit_cube::<f64>(1, 1, 1, ReferenceCellType::Tetrahedron, 1),
            1.0,
        );
        check_volume(
            &unit_cube::<f64>(2, 2, 2, ReferenceCellType::Tetrahedron, 1),
            1.0,
        );
        check_volume(
            &unit_cube::<f64>(4, 5, 5, ReferenceCellType::Tetrahedron, 1),
            1.0,
        );
        check_volume(
            &unit_cube::<f64>(7, 6, 4, ReferenceCellType::Tetrahedron, 1),
            1.0,
        );
    }
    #[test]
    fn test_unit_cube_hexahedron() {
        check_volume(
            &unit_cube::<f64>(1, 1, 1, ReferenceCellType::Hexahedron, 1),
            1.0,
        );
        check_volume(
            &unit_cube::<f64>(2, 2, 2, ReferenceCellType::Hexahedron, 1),
            1.0,
        );
        check_volume(
            &unit_cube::<f64>(4, 5, 5, ReferenceCellType::Hexahedron, 1),
            1.0,
        );
        check_volume(
            &unit_cube::<f64>(7, 6, 4, ReferenceCellType::Hexahedron, 1),
            1.0,
        );
    }

    #[test]
    fn test_unit_cube_edges() {
        check_volume(&unit_cube_edges::<f64>(1, 1, 1, 1), 12.0);
        check_volume(&unit_cube_edges::<f64>(2, 2, 2, 1), 12.0);
        check_volume(&unit_cube_edges::<f64>(4, 5, 5, 1), 12.0);
        check_volume(&unit_cube_edges::<f64>(7, 6, 4, 1), 12.0);
    }

    #[test]
    fn test_normals_are_unit_unit_square_boundary() {
        let mut point = rlst_dynamic_array!(f64, [1, 1]);
        point[[0, 0]] = 0.5;
        test_normals_are_unit(
            &unit_square_boundary(1, 1, 1),
            ReferenceCellType::Interval,
            &point,
            1,
        );
        test_normals_are_unit(
            &unit_square_boundary(2, 2, 1),
            ReferenceCellType::Interval,
            &point,
            1,
        );
        test_normals_are_unit(
            &unit_square_boundary(3, 4, 1),
            ReferenceCellType::Interval,
            &point,
            1,
        );
    }

    #[test]
    fn test_normals_are_unit_unit_cube_boundary_triangles() {
        let mut point = rlst_dynamic_array!(f64, [2, 1]);
        point[[0, 0]] = 0.2;
        point[[1, 0]] = 0.2;
        test_normals_are_unit(
            &unit_cube_boundary(1, 1, 1, ReferenceCellType::Triangle, 1),
            ReferenceCellType::Triangle,
            &point,
            1,
        );
        test_normals_are_unit(
            &unit_cube_boundary(2, 2, 2, ReferenceCellType::Triangle, 1),
            ReferenceCellType::Triangle,
            &point,
            1,
        );
        test_normals_are_unit(
            &unit_cube_boundary(3, 4, 5, ReferenceCellType::Triangle, 1),
            ReferenceCellType::Triangle,
            &point,
            1,
        );
    }

    #[test]
    fn test_normals_are_unit_unit_cube_boundary_quadrilaterals() {
        let mut point = rlst_dynamic_array!(f64, [2, 1]);
        point[[0, 0]] = 0.2;
        point[[1, 0]] = 0.2;
        test_normals_are_unit(
            &unit_cube_boundary(1, 1, 1, ReferenceCellType::Quadrilateral, 1),
            ReferenceCellType::Quadrilateral,
            &point,
            1,
        );
        test_normals_are_unit(
            &unit_cube_boundary(2, 2, 2, ReferenceCellType::Quadrilateral, 1),
            ReferenceCellType::Quadrilateral,
            &point,
            1,
        );
        test_normals_are_unit(
            &unit_cube_boundary(3, 4, 5, ReferenceCellType::Quadrilateral, 1),
            ReferenceCellType::Quadrilateral,
            &point,
            1,
        );
    }

    #[test]
    fn test_normals_are_outward_unit_square_boundary() {
        let mut point = rlst_dynamic_array!(f64, [1, 1]);
        point[[0, 0]] = 0.5;
        let mut centre = rlst_dynamic_array!(f64, [2, 1]);
        centre[[0, 0]] = 0.5;
        centre[[1, 0]] = 0.5;
        test_normals_are_outward(
            &unit_square_boundary(1, 1, 1),
            ReferenceCellType::Interval,
            &point,
            &centre,
            1,
        );
        test_normals_are_outward(
            &unit_square_boundary(2, 2, 1),
            ReferenceCellType::Interval,
            &point,
            &centre,
            1,
        );
        test_normals_are_outward(
            &unit_square_boundary(3, 4, 1),
            ReferenceCellType::Interval,
            &point,
            &centre,
            1,
        );
    }

    #[test]
    fn test_normals_are_outward_unit_cube_boundary_triangles() {
        let mut point = rlst_dynamic_array!(f64, [2, 1]);
        point[[0, 0]] = 0.2;
        point[[1, 0]] = 0.2;
        let mut centre = rlst_dynamic_array!(f64, [3, 1]);
        centre[[0, 0]] = 0.5;
        centre[[1, 0]] = 0.5;
        centre[[2, 0]] = 0.5;
        test_normals_are_outward(
            &unit_cube_boundary(1, 1, 1, ReferenceCellType::Triangle, 1),
            ReferenceCellType::Triangle,
            &point,
            &centre,
            1,
        );
        test_normals_are_outward(
            &unit_cube_boundary(2, 2, 2, ReferenceCellType::Triangle, 1),
            ReferenceCellType::Triangle,
            &point,
            &centre,
            1,
        );
        test_normals_are_outward(
            &unit_cube_boundary(3, 4, 5, ReferenceCellType::Triangle, 1),
            ReferenceCellType::Triangle,
            &point,
            &centre,
            1,
        );
    }

    #[test]
    fn test_normals_are_outward_unit_cube_boundary_quadrilaterals() {
        let mut point = rlst_dynamic_array!(f64, [2, 1]);
        point[[0, 0]] = 0.2;
        point[[1, 0]] = 0.2;
        let mut centre = rlst_dynamic_array!(f64, [3, 1]);
        centre[[0, 0]] = 0.5;
        centre[[1, 0]] = 0.5;
        centre[[2, 0]] = 0.5;
        test_normals_are_outward(
            &unit_cube_boundary(1, 1, 1, ReferenceCellType::Quadrilateral, 1),
            ReferenceCellType::Quadrilateral,
            &point,
            &centre,
            1,
        );
        test_normals_are_outward(
            &unit_cube_boundary(2, 2, 2, ReferenceCellType::Quadrilateral, 1),
            ReferenceCellType::Quadrilateral,
            &point,
            &centre,
            1,
        );
        test_normals_are_outward(
            &unit_cube_boundary(3, 4, 5, ReferenceCellType::Quadrilateral, 1),
            ReferenceCellType::Quadrilateral,
            &point,
            &centre,
            1,
        );
    }
}
