//! Routines to create VTK orderings from ND orderings.
//!
//! This file is adapted from https://github.com/FEniCS/dolfinx/blob/main/cpp/dolfinx/io/cells.cpp

//! A signed remove function.
use itertools::{Itertools, izip};
use num::integer::cbrt;

use crate::types::ReferenceCellType;

/// Negative indices remove from the right. Non-negative indices remove from the left.
fn remove_signed<T: Sized>(v: &mut Vec<T>, index: i64) -> T {
    let n = v.len() as i64;
    assert!(
        -n <= index && index < n,
        "Value {index} must be smaller in magnitue than {n}",
    );
    let index = if index < 0 { n + index } else { index };

    v.remove(index as usize)
}

/// Reorder triangle dofs given in inverse lexicographic
/// ordering into the standard triangle layout of ND
#[allow(dead_code)]
fn inverse_lexicographic_to_triangle(dofs: &[usize]) -> Vec<usize> {
    let npts = dofs.len();

    // First get the degree.

    let degree = cell_degree(ReferenceCellType::Triangle, npts);

    let mut map = Vec::<usize>::with_capacity(npts);

    // We now store the points of the triangle

    map.push(dofs[0]);
    map.push(dofs[degree]);
    map.push(dofs[npts - 1]);

    // Now we push through the edge dofs

    let edge_dofs = degree - 1;

    if map.len() == npts {
        // No further dofs
        return map;
    }

    // Edge (0, 1)
    map.extend_from_slice(&dofs[1..1 + edge_dofs]);
    // Edge (0, 2)
    let mut count = degree + 1;
    for index in 0..edge_dofs {
        map.push(dofs[count]);
        count += degree - index;
    }

    // Now Edge (1, 2)
    let mut count = 2 * degree;
    for index in 0..edge_dofs {
        map.push(dofs[count]);
        count += degree - 1 - index;
    }

    if map.len() == npts {
        return map;
    }

    // Finally, we need to reorder the interior dofs

    let mut count = degree + 2;
    for row_index in 1..edge_dofs {
        for _ in 0..edge_dofs - row_index {
            map.push(dofs[count]);
            count += 1;
        }
        count += 2;
    }

    map
}

/// Rotate the interior triangle dofs in ND counter-clockwise
///
/// The output are interior dofs rotated counter clock-wise.
fn rotate_nd_triangle_interior_counter_clockwise(int_dofs: &[usize]) -> Vec<usize> {
    let n_int_dofs = int_dofs.len();

    let mut rotated_dofs = Vec::<usize>::with_capacity(n_int_dofs);

    let int_cols = cell_degree(ReferenceCellType::Triangle, n_int_dofs) + 1;

    let triangle_it_1 = TriangleIterator::default();

    for (col, start) in izip!(0..int_cols, triangle_it_1) {
        let start = 1 + start;
        let mut i = 0;
        for step in 2 + col..2 + int_cols {
            rotated_dofs.push(int_dofs[n_int_dofs - start - i]);
            i += step;
        }
    }

    rotated_dofs
}

/// Rotate triangle dofs in ND counter-clockwise
///
/// The output are the triangle dofs with respect to the counter-clockwise
/// rotated triangle.
#[allow(dead_code)]
fn rotate_nd_triangle_counter_clockwise(dofs: &[usize]) -> Vec<usize> {
    let npts = dofs.len();
    let degree = cell_degree(ReferenceCellType::Triangle, npts);

    let mut rotated_dofs = Vec::<usize>::with_capacity(npts);

    // First we take the 3 vertex dofs.
    rotated_dofs.push(dofs[2]);
    rotated_dofs.push(dofs[0]);
    rotated_dofs.push(dofs[1]);

    if npts == 3 {
        return rotated_dofs;
    }

    // Now we take all the edge dofs

    let edge_dofs = degree - 1;

    // First take the dofs from edge (0, 2) in reverse order
    for &dof in dofs[3 + edge_dofs..3 + 2 * edge_dofs].iter().rev() {
        rotated_dofs.push(dof);
    }

    // Then the dofs in (1, 2) but reverse the order.
    for &dof in dofs[3 + 2 * edge_dofs..3 + 3 * edge_dofs].iter().rev() {
        rotated_dofs.push(dof);
    }

    // Finally, the dofs from (0, 1)
    for &dof in &dofs[3..3 + edge_dofs] {
        rotated_dofs.push(dof);
    }

    // We now deal with the interior degrees of freedom

    let int_dofs = npts - 3 * degree;

    if int_dofs == 0 {
        rotated_dofs
    } else {
        rotated_dofs.extend_from_slice(
            rotate_nd_triangle_interior_counter_clockwise(&dofs[3 * degree..npts]).as_slice(),
        );
        rotated_dofs
    }
}

/// Reflect the interior dofs of an ND triangle.
///
/// If the interior dofs in the original triangle are those of a triangle with vertices (0, 1, 2),
/// the reflected ones are the interior dofs of a tringle with vertices (0, 2, 1).
pub fn reflect_nd_triangle_interior(int_dofs: &[usize]) -> Vec<usize> {
    let n_int_dofs = int_dofs.len();
    let int_cols = cell_degree(ReferenceCellType::Triangle, n_int_dofs) + 1;

    let mut mirror_dofs = Vec::<usize>::with_capacity(n_int_dofs);

    for col in 0..int_cols {
        let mut pos = col;
        for step in (1..=int_cols).rev().take(int_cols - col) {
            mirror_dofs.push(int_dofs[pos]);
            pos += step;
        }
    }

    mirror_dofs
}

/// Reflect an ND triangle
///
/// If the ordering of vertices in the original triangle is (0, 1, 2),
/// the reflected triangle has nodes (0, 2, 1).
#[allow(dead_code)]
pub fn reflect_nd_triangle(dofs: &[usize]) -> Vec<usize> {
    let npts = dofs.len();
    let degree = cell_degree(ReferenceCellType::Triangle, npts);

    let mut mirror_dofs = Vec::<usize>::with_capacity(npts);

    // First we take the 3 vertex dofs.
    mirror_dofs.push(dofs[0]);
    mirror_dofs.push(dofs[2]);
    mirror_dofs.push(dofs[1]);

    if npts == 3 {
        return mirror_dofs;
    }

    // Now we take all the edge dofs

    let edge_dofs = degree - 1;

    // First take the dofs from edge (0, 2)
    for &dof in dofs[3 + edge_dofs..3 + 2 * edge_dofs].iter() {
        mirror_dofs.push(dof);
    }

    // Then the dofs in (0, 1)
    for &dof in dofs[3..3 + edge_dofs].iter() {
        mirror_dofs.push(dof);
    }

    // Finally, the dofs from (1, 2) but reversed
    for &dof in dofs[3 + 2 * edge_dofs..3 + 3 * edge_dofs].iter().rev() {
        mirror_dofs.push(dof);
    }

    let int_dofs = npts - 3 * degree;

    if int_dofs == 0 {
        return mirror_dofs;
    }

    mirror_dofs.extend_from_slice(&reflect_nd_triangle_interior(&dofs[3 * degree..npts]));
    mirror_dofs

    // We now deal with the interior degrees of freedom
}

/// Permute interior dofs of the standard triangle into another triangle
///
/// If permutation is e.g. [0, 2, 1], the interior dofs of the standard triangle with vertices [0, 1, 2] are reflected
/// so that they follow the memory order of a triangle with first vertex being zero, second two, and third one.
pub fn permute_interior_triangle_dofs(int_dofs: &[usize], permutation: [usize; 3]) -> Vec<usize> {
    match permutation {
        [0, 1, 2] => int_dofs.to_owned(),
        [0, 2, 1] => reflect_nd_triangle_interior(int_dofs),
        [2, 0, 1] => rotate_nd_triangle_interior_counter_clockwise(int_dofs),
        [2, 1, 0] => {
            reflect_nd_triangle_interior(&rotate_nd_triangle_interior_counter_clockwise(int_dofs))
        }
        [1, 2, 0] => rotate_nd_triangle_interior_counter_clockwise(
            &rotate_nd_triangle_interior_counter_clockwise(int_dofs),
        ),
        [1, 0, 2] => reflect_nd_triangle_interior(&rotate_nd_triangle_interior_counter_clockwise(
            &rotate_nd_triangle_interior_counter_clockwise(int_dofs),
        )),
        _ => panic!("Unknown permutation."),
    }
}

fn tetrahedron_remainders(mut remainders: Vec<usize>) -> Vec<usize> {
    let mut map = Vec::<usize>::new();
    map.reserve_exact(remainders.len());

    while !remainders.is_empty() {
        if remainders.len() == 1 {
            map.push(remove_signed(&mut remainders, 0));
            break;
        }
        let deg = cell_degree(ReferenceCellType::Tetrahedron, remainders.len()) as i64 + 1;

        map.push(remove_signed(&mut remainders, 0));
        map.push(remove_signed(&mut remainders, deg - 2));
        map.push(remove_signed(&mut remainders, deg * (deg + 1) / 2 - 3));
        map.push(remove_signed(&mut remainders, -1));

        if deg > 2 {
            for _ in 0..deg - 2 {
                map.push(remove_signed(&mut remainders, 0));
            }
            let mut d = deg - 2;
            for i in 0..deg - 2 {
                map.push(remove_signed(&mut remainders, d));
                d += deg - 3 - i;
            }

            let mut d = (deg - 2) * (deg - 1) / 2 - 1;
            for i in 0..deg - 2 {
                map.push(remove_signed(&mut remainders, d));
                d -= 2 + i;
            }

            let mut d = (deg - 3) * (deg - 2) / 2;
            for i in 0..deg - 2 {
                map.push(remove_signed(&mut remainders, d));
                d += (deg - i) * (deg - i - 1) / 2 - 1;
            }

            let mut d = (deg - 3) * (deg - 2) / 2 + deg - 3;
            for i in 0..deg - 2 {
                map.push(remove_signed(&mut remainders, d));
                d += (deg - 2 - i) * (deg - 1 - i) / 2 + deg - 4 - i;
            }

            let mut d = (deg - 3) * (deg - 2) / 2 + deg - 3 + (deg - 2) * (deg - 1) / 2 - 1;
            for i in 0..deg - 2 {
                map.push(remove_signed(&mut remainders, d));
                d += (deg - 3 - i) * (deg - 2 - i) / 2 + deg - i - 5;
            }
        }

        if deg > 3 {
            let mut dofs = Vec::<usize>::new();
            let mut d = (deg - 3) * (deg - 2) / 2;
            for i in 0..deg - 3 {
                for _ in 0..deg - 3 - i {
                    dofs.push(remove_signed(&mut remainders, d));
                    d += (deg - 2 - i) * (deg - 1 - i) / 2 - 1;
                }
            }
            map.extend(triangle_remainders(dofs).iter());

            let mut dofs = Vec::<usize>::new();
            let mut start = deg * deg - 4 * deg + 2;
            let sub_i_start = deg - 3;
            for i in 0..deg - 3 {
                let mut d = start;
                let mut sub_i = sub_i_start;
                for _ in 0..deg - 3 - i {
                    dofs.push(remove_signed(&mut remainders, d));
                    d += sub_i * (sub_i + 1) / 2 - 1 - 2 * i;
                    sub_i -= 1;
                }
                start += deg - 4 - i;
            }
            map.extend(triangle_remainders(dofs).iter());

            let mut dofs = Vec::<usize>::new();
            for (add_start, i) in (0..=deg - 4).rev().zip(0..deg - 3) {
                let mut d = 0;
                let mut add = add_start;
                for _ in 0..deg - 3 - i {
                    dofs.push(remove_signed(&mut remainders, d));
                    d += add;
                    add -= 1;
                }
            }
            map.extend(triangle_remainders(dofs).iter());
        }
    }
    map
}

fn triangle_remainders(mut remainders: Vec<usize>) -> Vec<usize> {
    let mut map = Vec::<usize>::new();
    map.reserve_exact(remainders.len());

    while !remainders.is_empty() {
        if remainders.len() == 1 {
            map.push(remove_signed(&mut remainders, 0));
            break;
        }
        let degree = cell_degree(ReferenceCellType::Triangle, remainders.len()) as i64;

        map.push(remove_signed(&mut remainders, 0));
        map.push(remove_signed(&mut remainders, degree - 1));
        map.push(remove_signed(&mut remainders, -1));

        for _ in 0..degree - 1 {
            map.push(remove_signed(&mut remainders, 0))
        }

        let mut k = degree * (degree - 1) / 2;
        for i in 1..degree {
            map.push(remove_signed(&mut remainders, -k));
            k -= degree - i;
        }

        let mut k = 1;
        for i in 1..degree {
            map.push(remove_signed(&mut remainders, -k));
            k += i;
        }
    }

    map
}

/// Return the VTK permutation vector for the hexahedron
pub fn vtk_hexahedron(num_nodes: usize) -> Vec<usize> {
    let mut map = Vec::<usize>::new();
    map.reserve_exact(num_nodes);

    let degree = cell_degree(ReferenceCellType::Hexahedron, num_nodes);

    // Insert the vertices
    map.extend([0, 1, 3, 2, 4, 5, 7, 6].iter());

    let mut base: usize = 8;
    let edge_nodes = degree - 1;

    // Insert the edges
    for e in [0, 3, 5, 1, 8, 10, 11, 9, 2, 4, 7, 6] {
        map.extend(base + edge_nodes * e..base + edge_nodes * (1 + e));
    }

    // Insert the faces
    base += 12 * edge_nodes;
    let face_nodes = edge_nodes * edge_nodes;

    for f in [2, 3, 1, 4, 0, 5] {
        map.extend(base + face_nodes * f..base + face_nodes * (1 + f));
    }

    base += 6 * face_nodes;
    map.extend(base..num_nodes);

    map
}

pub fn vtk_triangle(num_nodes: usize) -> Vec<usize> {
    let mut map = Vec::<usize>::new();
    map.reserve_exact(num_nodes);

    let degree = cell_degree(ReferenceCellType::Triangle, num_nodes);

    // The three nodal dofs are identical in ND and VTK ordering
    map.push(0);
    map.push(1);
    map.push(2);

    for k in 1..degree {
        map.push(3 + k - 1);
    }
    for k in 1..degree {
        map.push(2 * degree + k);
    }
    for k in 1..degree {
        map.push(2 * degree - (k - 1));
    }

    if degree < 3 {
        map
    } else {
        let nremainders = num_nodes - map.len();
        let mut remainders = triangle_remainders((3 * degree..nremainders + 3 * degree).collect());
        map.append(&mut remainders);
        map
    }
}

pub fn vtk_quadrilateral(num_nodes: usize) -> Vec<usize> {
    let mut map = Vec::<usize>::with_capacity(num_nodes);

    map.extend_from_slice(&[0, 1, 3, 2]);
    let degree = cell_degree(ReferenceCellType::Quadrilateral, num_nodes);

    let edge_nodes = degree - 1;

    // Bottom edge
    for k in 0..edge_nodes {
        map.push(4 + k);
    }

    // Right edge
    for k in 0..edge_nodes {
        map.push(4 + 2 * edge_nodes + k);
    }

    // Top edge
    for k in 0..edge_nodes {
        map.push(4 + 3 * edge_nodes + k);
    }

    // Left edge
    for k in 0..edge_nodes {
        map.push(4 + edge_nodes + k);
    }
    if degree < 2 {
        map
    } else {
        let nremainders = num_nodes - map.len();
        map.extend(4 * degree..4 * degree + nremainders);
        map
    }
}

pub fn vtk_tetrahedron(num_nodes: usize) -> Vec<usize> {
    let mut map = Vec::<usize>::with_capacity(num_nodes);

    let degree = cell_degree(ReferenceCellType::Tetrahedron, num_nodes);
    map.extend_from_slice(&[0, 1, 2, 3]);

    if degree < 2 {
        return map;
    }

    let base = 4;
    let edge_dofs = degree - 1;

    // Edge (0, 1)
    for index in 0..edge_dofs {
        map.push(base + index);
    }
    // Edge (1, 2)
    for index in 0..edge_dofs {
        map.push(base + 3 * edge_dofs + index);
    }
    // Edge (2, 0)
    for index in 1..=edge_dofs {
        map.push(base + 2 * edge_dofs - index);
    }
    // Edge (0, 3)
    for index in 0..edge_dofs {
        map.push(base + 2 * edge_dofs + index);
    }
    // Edge (1, 3)
    for index in 0..edge_dofs {
        map.push(base + 4 * edge_dofs + index);
    }
    // Edge (2, 3)
    for index in 0..edge_dofs {
        map.push(base + 5 * edge_dofs + index);
    }

    if num_nodes == map.len() {
        // No more dofs to process
        return map;
    }

    // We now have to do the faces
    let face_dofs = (degree - 1) * (degree - 2) / 2;
    let base = base + 6 * edge_dofs;

    // Face (0, 1, 3) - Second face in ND ordering
    map.extend_from_slice(&triangle_remainders(
        (base + face_dofs..base + 2 * face_dofs).collect_vec(),
    ));

    // Face (2, 3, 1). Permuted 4th face in ND ordering
    map.extend_from_slice(&triangle_remainders(permute_interior_triangle_dofs(
        &(base + 3 * face_dofs..base + 4 * face_dofs).collect_vec(),
        [1, 2, 0],
    )));

    // Face (0, 3, 2). Permuted 3rd face in ND ordering
    map.extend_from_slice(&triangle_remainders(permute_interior_triangle_dofs(
        &(base + 2 * face_dofs..base + 3 * face_dofs).collect_vec(),
        [0, 2, 1],
    )));

    // Face (0, 2, 1). Permuted first face in ND ordering
    map.extend_from_slice(&triangle_remainders(permute_interior_triangle_dofs(
        &(base..base + face_dofs).collect_vec(),
        [0, 2, 1],
    )));

    let base = base + 4 * face_dofs;

    if num_nodes == base {
        return map;
    }

    map.extend_from_slice(&tetrahedron_remainders((base..num_nodes).collect_vec()));

    map
}

fn cell_degree(cell_type: ReferenceCellType, num_nodes: usize) -> usize {
    match cell_type {
        ReferenceCellType::Point => 1,
        ReferenceCellType::Interval => num_nodes - 1,
        ReferenceCellType::Triangle => {
            let n = ((f64::sqrt(1.0 + 8.0 * num_nodes as f64) - 1.0) / 2.0) as usize;
            assert_eq!(2 * num_nodes, n * (n + 1), "Unknown triangle layout");
            n - 1
        }
        ReferenceCellType::Quadrilateral => {
            let n = num_nodes.isqrt();
            assert_eq!(num_nodes, n * n, "Unknown quadrilateral layout");
            n - 1
        }
        ReferenceCellType::Tetrahedron => {
            let mut n = 0;
            while n * (n + 1) * (n + 2) < 6 * num_nodes {
                n += 1;
            }
            assert_eq!(
                n * (n + 1) * (n + 2),
                6 * num_nodes,
                "Unknown tetrahedron layout"
            );
            n - 1
        }
        ReferenceCellType::Hexahedron => {
            let n = cbrt(num_nodes);
            assert_eq!(n * n * n, num_nodes, "Unknown hexahedron layout");
            n - 1
        }
        _ => unimplemented!(),
    }
}

/// This iterator returns the sequence 0, 1, 3, 6, 10, 15, ...
///
/// It is useful to go up array elements ordered in a triange,
/// where each row has one more element than the previous row.
#[derive(Default)]
struct TriangleIterator {
    count: usize,
    step: usize,
}

impl Iterator for TriangleIterator {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.count += self.step;
        self.step += 1;
        Some(self.count)
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use crate::reorderings::vtk::{
        inverse_lexicographic_to_triangle, permute_interior_triangle_dofs, reflect_nd_triangle,
        rotate_nd_triangle_counter_clockwise, triangle_remainders, vtk_hexahedron,
        vtk_quadrilateral, vtk_tetrahedron, vtk_triangle,
    };

    #[test]
    fn test_triangle_ordering() {
        // A degree 5 triangle in ND has the following DOFS

        //        2
        //
        //        10  14
        //
        //        9   20  13
        //
        //        8   18  19  12
        //
        //        7   15  16   17   11
        //
        //        0   3    4    5    6    1
        //
        //
        //        In VTK it has the following DOFS
        //
        //        2
        //
        //        11  10
        //
        //        12   17  9
        //
        //        13   20  19  8
        //
        //        14   15  18   16  7
        //
        //        0   3    4    5    6    1

        // The permutation vector should therefore have the following form
        //
        // [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 10, 9, 8, 7, 15, 17, 20, 16, 19, 18]

        let expected = [
            0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 10, 9, 8, 7, 15, 17, 20, 16, 19, 18,
        ]
        .to_vec();

        let actual = vtk_triangle(21);

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_quadrilateral_ordering() {
        // A degree 5 quadrilateral in ND has the following ordering.
        //
        //  2  13  14   15  3
        //
        //  9  22  23  24   12
        //
        //  8  19  20  21   11
        //
        //  7  16  17  18   10
        //
        //  0   4   5   6   1
        //
        //
        //  In VTK it has the following ordering
        //
        //  3  10  11   12  2
        //
        //  15 22  23  24    9
        //
        //  14 19  20  21    8
        //
        //  13 16  17  18    7
        //
        //  0   4   5   6   1
        //
        // The permutation vector should therefore have the following form
        //
        // [0, 1, 3, 2, 4, 5, 6, 10, 11, 12, 13, 14, 15, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 24]

        let expected = [
            0, 1, 3, 2, 4, 5, 6, 10, 11, 12, 13, 14, 15, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23,
            24,
        ]
        .to_vec();

        let actual = vtk_quadrilateral(25);

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_tetrahedron() {
        // A tetrahedron in nd has the following layout:
        //
        //
        //     3
        //     |
        //     |   2
        //     |  /
        //     | /
        //     |/
        //     0 ---------  1
        //
        //  with the vertex positions being
        //  0: (0, 0, 0)
        //  1: (1, 0, 0)
        //  2: (0, 1, 0)
        //  3: (0, 0, 1)
        //
        //
        // We now describe the dofs for a degree 5 tetrahedron. They are distributed as follows
        //
        // Edges (The numbers in edges are the associated vertices).
        // A vertex tuple (2, 3) means that dofs start close to vertex 2 and
        // are counted towards vertex 3.
        //  0 (0, 1): 4, 5, 6, 7
        //  1 (0, 2): 8, 9, 10, 11
        //  2 (0, 3): 12, 13, 14, 15
        //  3 (1, 2): 16, 17, 18, 19
        //  4 (1, 3): 20, 21, 22, 23
        //  5 (2, 3): 24, 25, 26, 27
        //
        //  Faces: The faces are sorted in the same way as triangles defined through the order of the corresponding vertices
        //  0 (0, 1, 2): 28, 29, 30, 31, 32, 33
        //  1 (0, 1, 3): 34, 35, 36, 37, 38, 39
        //  2 (0, 2, 3): 40, 41, 42, 43, 44, 45
        //  3 (1, 2, 3): 46, 47, 48, 49, 50, 51
        //
        //  Volume:
        //  The volume dofs iterate along the (x,y,z) axis in reverse lexicographic order. For the given example this means the dofs are:
        //
        // 52: [0.2, 0.2, 0.2]
        // 53: [0.4, 0.2, 0.2]
        // 54: [0.2, 0.4, 0.2]
        // 55: [0.2, 0.2, 0.4]
        //
        //  0 (0, 1, 2, 3): 52, 53, 54, 55
        //
        //  Now the corresponding VTK element
        //
        //  Vertices (same as nd):
        //
        //     3
        //     |
        //     |   2
        //     |  /
        //     | /
        //     |/
        //     0 ---------  1
        //
        //  with the vertex positions being
        //  0: (0, 0, 0)
        //  1: (1, 0, 0)
        //  2: (0, 1, 0)
        //  3: (0, 0, 1)
        //
        // Edges:
        // 0 (0, 1): 4, 5, 6, 7
        // 1 (1, 2): 8, 9, 10, 11
        // 2 (2, 0): 12, 13, 14, 15
        // 3 (0, 3): 16, 17, 18, 19
        // 4 (1, 3): 20, 21, 22, 23

        //
        // Faces: The interior face dofs are arranged in the same way as the interior dofs of triangles,
        // that is, they are enumerated layer by layer from exterior to interior.
        //
        // 0 (0, 1, 3): 28, 29, 30, 31, 32, 33
        // 1 (2, 3, 1): 34, 35, 36, 37, 38, 39
        // 2 (0, 3, 2): 40, 41, 42, 43, 44, 45
        // 3 (0, 2, 1): 46, 47, 48, 49, 50, 51
        //
        // Volume
        //
        // 0 (0, 1, 2, 3): 52, 53, 54, 55

        // The expected permutation vector is as follows:

        let expected = [
            0, 1, 2, 3, // Points are not perturbed
            4, 5, 6, 7, // Edge (0, 1)
            16, 17, 18, 19, // Edge (1, 2)
            11, 10, 9, 8, // Edge (2, 0)
            12, 13, 14, 15, // Edge (0, 3)
            20, 21, 22, 23, // Edge (1, 3)
            24, 25, 26, 27, // Edge (2, 3)
            34, 36, 39, 35, 38, 37, // Face (0, 1, 3)
            48, 51, 46, 50, 49, 47, // Face (2, 3, 1)
            40, 45, 42, 43, 44, 41, // Face (0, 3, 2)
            28, 33, 30, 31, 32, 29, // Face (0, 2, 1)
            52, 53, 54, 55, // Volume (0, 1, 2, 3)
        ];

        let tet = vtk_tetrahedron(56);

        assert_eq!(expected.as_slice(), tet);
    }

    #[test]
    fn test_rotate_nd_triangle() {
        // Check a single rotation

        let expected = [2, 0, 1, 8, 7, 6, 11, 10, 9, 3, 4, 5, 14, 12, 13];

        let actual = rotate_nd_triangle_counter_clockwise(&[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
        ]);

        assert_eq!(actual, expected);

        let npts = 21;
        let expected: Vec<usize> = (0..npts).collect_vec();

        // Rotating three times should give the same result back

        let first = rotate_nd_triangle_counter_clockwise(&expected);
        let second = rotate_nd_triangle_counter_clockwise(&first);
        let third = rotate_nd_triangle_counter_clockwise(&second);

        assert_eq!(expected, third);
    }

    #[test]
    fn test_reflect_triangle() {
        let npts = 21;

        let expected: [usize; 21] = [
            0, 2, 1, 7, 8, 9, 10, 3, 4, 5, 6, 14, 13, 12, 11, 15, 18, 20, 16, 19, 17,
        ];

        let actual = reflect_nd_triangle(&(0..npts).collect_vec());

        assert_eq!(expected.as_slice(), &actual);

        let npts = 21;
        let expected: Vec<usize> = (0..npts).collect_vec();

        let first = reflect_nd_triangle(&expected);
        let second = reflect_nd_triangle(&first);

        assert_eq!(expected, second);
    }

    #[test]
    fn test_permute_triangle() {
        // Consider the triangle
        //
        //   2
        //   10  14
        //   9   20  13
        //   8   18  19   12
        //   7   15  16   17   11
        //   0    3   4    5    6   1
        //
        // We want to check that the possible permutations of its interior dofs are handled correctly.

        let original_dofs = [15, 16, 17, 18, 19, 20];

        // Perm: (0, 1, 2) (All vertices stay the same).
        // Should return the same triangle

        let permuted_dofs = original_dofs;

        let actual = permute_interior_triangle_dofs(&original_dofs, [0, 1, 2]);

        assert_eq!(permuted_dofs.as_slice(), actual);

        // Perm (0, 2, 1).
        // Expected interior dofs
        //
        //  17
        //  16  19
        //  15  18   20
        //
        let permuted_dofs = [15, 18, 20, 16, 19, 17];
        let actual = permute_interior_triangle_dofs(&original_dofs, [0, 2, 1]);

        assert_eq!(permuted_dofs.as_slice(), actual);

        // Perm (2, 0, 1).
        // Expected interior dofs
        //
        //  17
        //  19  16
        //  20  18  15
        //
        let permuted_dofs = [20, 18, 15, 19, 16, 17];
        let actual = permute_interior_triangle_dofs(&original_dofs, [2, 0, 1]);

        assert_eq!(permuted_dofs.as_slice(), actual);

        // Perm (2, 1, 0).
        // Expected interior dofs
        //
        //  15
        //  18  16
        //  20  19  17
        //
        let permuted_dofs = [20, 19, 17, 18, 16, 15];
        let actual = permute_interior_triangle_dofs(&original_dofs, [2, 1, 0]);

        assert_eq!(permuted_dofs.as_slice(), actual);

        // Perm (1, 2, 0).
        // Expected interior dofs
        //
        //  15
        //  16  18
        //  17  19 20
        //
        let permuted_dofs = [17, 19, 20, 16, 18, 15];
        let actual = permute_interior_triangle_dofs(&original_dofs, [1, 2, 0]);

        assert_eq!(permuted_dofs.as_slice(), actual);

        // Perm (1, 0, 2).
        // Expected interior dofs
        //
        //  20
        //  19  18
        //  17  16 15
        //
        let permuted_dofs = [17, 16, 15, 19, 18, 20];
        let actual = permute_interior_triangle_dofs(&original_dofs, [1, 0, 2]);

        assert_eq!(permuted_dofs.as_slice(), actual);
    }

    #[test]
    fn test_vtk_remainder_of_permuted_triangle() {
        // Consider the triangle
        //
        //   2
        //   10  14
        //   9   20  13
        //   8   18  19   12
        //   7   15  16   17   11
        //   0    3   4    5    6   1
        //
        //  We want to permute the interior dofs and then use the vtk_triangle_remainder function
        //  to convert the permuted triangle into the right order.
        //
        //  We use the (1, 0, 2) DOF Permutation. The corresponding interior dofs are
        //
        //  20
        //  19  18
        //  17  16 15
        //
        //  The vtk_triangle_remainder function should therefore return a vector of the form
        //  [17, 15, 20, 16, 18, 19]. Let's test this.
        //

        let original_dofs = [15, 16, 17, 18, 19, 20];
        let permuted_dofs = permute_interior_triangle_dofs(&original_dofs, [1, 0, 2]);
        let vtk_remainders = triangle_remainders(permuted_dofs);

        let expected = [17, 15, 20, 16, 18, 19];
        assert_eq!(expected.as_slice(), vtk_remainders);
    }

    #[test]
    fn test_inverse_lexicographic_to_triangle() {
        // We use the following test triangle
        //
        // 14
        // 12  13
        // 9   10  11
        // 5   6   7   8
        // 0   1   2   3   4
        //
        // This should give the following ND ordering
        // [0, 4, 14, 1, 2, 3, 5, 9, 12, 8, 11, 13, 6, 7, 10]

        let npts = 15;

        let dofs = (0..npts).collect_vec();
        let expected = [0, 4, 14, 1, 2, 3, 5, 9, 12, 8, 11, 13, 6, 7, 10];
        let actual = inverse_lexicographic_to_triangle(&dofs);
        assert_eq!(expected.as_slice(), &actual);
    }

    #[test]
    fn test_vtk_hexahedron() {
        // Consider the 8 vertices of a hexadron.
        //
        //
        //      6  ---   7
        //     /|       /|
        //    / |      / |
        //   4--------5  |
        //   |  2 --- |- 3
        //   | /      | /
        //   |/       |/
        //   0 -----  1
        //
        //
        //   The ND ordering is as follows (assuming a degree 4 tetrahedron)
        //
        //   Vertices: [0, 1, 2, 3, 4, 5, 6, 7]
        //
        //   Edges:
        //
        //   (0, 1): 8, 9, 10
        //   (0, 2): 11, 12, 13
        //   (0, 4): 14, 15, 16
        //   (1, 3): 17, 18, 19
        //   (1, 5): 20, 21, 22
        //   (2, 3): 23, 24, 25
        //   (2, 6): 26, 27, 28
        //   (3, 7): 29, 30, 31
        //   (4, 5): 32, 33, 34
        //   (4, 6): 35, 36, 37
        //   (5, 7): 38, 39, 40
        //   (6, 7): 41, 42, 43
        //
        //   Faces
        //   (0, 1, 2, 3): 44, 45, 46, 47, 48, 49, 50, 51, 52
        //   (0, 1, 4, 5): 53, 54, 55, 56, 57, 58, 59, 60, 61
        //   (0, 2, 4, 6): 62, 63, 64, 65, 66, 67, 68, 69, 70
        //   (1, 3, 5, 7): 71, 72, 73, 74, 75, 76, 77, 78, 79
        //   (2, 3, 6, 7): 80, 81, 82, 83, 84, 85, 86, 87, 88
        //   (4, 5, 6, 7): 89, 90, 91, 92, 93, 94, 95, 96, 97
        //
        //   Volume: The volume dofs proceed along the (x, y, z)
        //           indexing with x the fastest index and z the slowest
        //           index.
        //
        //   VTK ordering
        //
        //
        //      7  ---   6
        //     /|       /|
        //    / |      / |
        //   4 -| ---  5 |
        //   |  3     |  2
        //   | /      | /
        //   |/       |/
        //   0 -----  1
        //
        //   Edges: The order of the points in an edge denote
        //          the direction of the dofs
        //
        //   (0, 1): 8, 9, 10   (8, 9, 10)
        //   (1, 2): 17, 18, 19 (11, 12, 13)
        //   (3, 2): 23, 24, 25 (14, 15, 16)
        //   (0, 3): 11, 12, 13 (17, 18, 19)
        //   (4, 5): 32, 33, 34 (20, 21, 22)
        //   (5, 6): 38, 39, 40 (23, 24, 25)
        //   (7, 6): 41, 42, 43 (26, 27, 28)
        //   (4, 7): 35, 36, 37 (29, 30, 31)
        //   (0, 4): 14, 15, 16 (32, 33, 34)
        //   (1, 5): 20, 21, 22 (35, 36, 37)
        //   (2, 6): 29, 30, 31 (38, 39, 40)
        //   (3, 7): 26, 27, 28, (41, 42, 43)
        //
        //   Faces
        //   (0, 3, 4, 7): 62, 63, 64, 65, 66, 67, 68, 69, 70 (44-52)
        //   (1, 2, 5, 6): 71, 72, 73, 74, 75, 76, 77, 78, 79 (53-61)
        //   (0, 1, 4, 5): 53, 54, 55, 56, 57, 58, 59, 60, 61 (62-70)
        //   (3, 2, 7, 6): 80, 81, 82, 83, 84, 85, 86, 87, 88 (71-79)
        //   (0, 1, 3, 2): 44, 45, 46, 47, 48, 49, 50, 51, 52 (80-88)
        //   (4, 5, 7, 6): 89, 90, 91, 92, 93, 94, 95, 96, 97 (89-97)
        //
        //   Volume: The volume dofs proceed along the (x, y, z)
        //           indexing with x the fastest index and z the slowest
        //           index. They are ordered the same way as the ND dofs

        let expected = [
            0, 1, 3, 2, 4, 5, 7, 6, 8, 9, 10, 17, 18, 19, 23, 24, 25, 11, 12, 13, 32, 33, 34, 38,
            39, 40, 41, 42, 43, 35, 36, 37, 14, 15, 16, 20, 21, 22, 29, 30, 31, 26, 27, 28, 62, 63,
            64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 53, 54, 55, 56, 57, 58,
            59, 60, 61, 80, 81, 82, 83, 84, 85, 86, 87, 88, 44, 45, 46, 47, 48, 49, 50, 51, 52, 89,
            90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
            109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
        ];

        let hex = vtk_hexahedron(125);

        assert_eq!(expected.as_slice(), hex);
    }
}
