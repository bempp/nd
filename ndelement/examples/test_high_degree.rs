//! Finite element definitions

use ndelement::ciarlet::{lagrange, nedelec, raviart_thomas};
use ndelement::types::{Continuity, ReferenceCellType};
use paste::paste;

fn main() {
    macro_rules! construct_lagrange {
        ($cell:ident, $degree:expr) => {
            paste! {
                println!("Constructing Lagrange(degree={}, cell={:?})", [<$degree>], ReferenceCellType::[<$cell>]);
                let _e = lagrange::create::<f64, f64>(ReferenceCellType::[<$cell>], [<$degree>], Continuity::Standard, lagrange::Variant::Equispaced);
            }
        };
    }

    macro_rules! construct_raviart_thomas {
        ($cell:ident, $degree:expr) => {
            paste! {
                println!("Constructing RaviartThomas(degree={}, cell={:?})", [<$degree>], ReferenceCellType::[<$cell>]);
                let _e = raviart_thomas::create::<f64, f64>(ReferenceCellType::[<$cell>], [<$degree>], Continuity::Standard);
            }
        };
    }

    macro_rules! construct_nedelec {
        ($cell:ident, $degree:expr) => {
            paste! {
                println!("Constructing Nedelec(degree={}, cell={:?})", [<$degree>], ReferenceCellType::[<$cell>]);
                let _e = nedelec::create::<f64, f64>(ReferenceCellType::[<$cell>], [<$degree>], Continuity::Standard);
            }
        };
    }

    construct_lagrange!(Interval, 3);
    construct_lagrange!(Interval, 8);
    construct_lagrange!(Interval, 14);
    construct_lagrange!(Triangle, 3);
    construct_lagrange!(Triangle, 7);
    construct_lagrange!(Quadrilateral, 3);
    construct_lagrange!(Quadrilateral, 7);
    construct_lagrange!(Tetrahedron, 3);
    construct_lagrange!(Hexahedron, 5);
    construct_lagrange!(Tetrahedron, 3);
    construct_lagrange!(Hexahedron, 5);

    construct_raviart_thomas!(Triangle, 3);
    construct_raviart_thomas!(Triangle, 7);
    construct_raviart_thomas!(Quadrilateral, 3);
    construct_raviart_thomas!(Quadrilateral, 7);
    construct_raviart_thomas!(Tetrahedron, 3);
    construct_raviart_thomas!(Hexahedron, 5);
    construct_raviart_thomas!(Tetrahedron, 3);
    construct_raviart_thomas!(Hexahedron, 5);

    construct_nedelec!(Triangle, 3);
    construct_nedelec!(Triangle, 7);
    construct_nedelec!(Quadrilateral, 3);
    construct_nedelec!(Quadrilateral, 7);
    construct_nedelec!(Tetrahedron, 3);
    construct_nedelec!(Hexahedron, 5);
    construct_nedelec!(Tetrahedron, 3);
    construct_nedelec!(Hexahedron, 5);
}
