//! Utility functions
use numpy::{
    Element, IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4,
    ndarray::Array,
};
use pyo3::{Bound, Python};

use rlst::{DynArray, RlstScalar, SliceArray};

/// Convert a 2D RLST array to a Python array
pub fn to_py2<'py, T: Element + RlstScalar>(
    py: Python<'py>,
    a: &DynArray<T, 2>,
) -> Bound<'py, PyArray2<T>> {
    let d0 = a.shape()[0];
    let d1 = a.shape()[1];
    let mut result_array = Array::zeros((d1, d0));
    for i in 0..d1 {
        for j in 0..d0 {
            result_array[[i, j]] = a[[j, i]];
        }
    }
    result_array.into_pyarray(py)
}

/// Convert a 3D RLST array to a Python array
pub fn to_py3<'py, T: Element + RlstScalar>(
    py: Python<'py>,
    a: &DynArray<T, 3>,
) -> Bound<'py, PyArray3<T>> {
    let d0 = a.shape()[0];
    let d1 = a.shape()[1];
    let d2 = a.shape()[2];
    let mut result_array = Array::zeros((d2, d1, d0));
    for i in 0..d2 {
        for j in 0..d1 {
            for k in 0..d0 {
                result_array[[i, j, k]] = a[[k, j, i]];
            }
        }
    }
    result_array.into_pyarray(py)
}

/// Convert a 2D Python array to a 2D RLST array
pub fn from_py2<'py, T: Element + RlstScalar>(
    a: &'py PyReadonlyArray2<'py, T>,
) -> SliceArray<'py, T, 2> {
    let a_array = a.as_array();
    let shape = a_array.shape();
    SliceArray::<T, 2>::from_shape(a.as_slice().unwrap(), [shape[1], shape[0]])
}

/// Convert a 3D Python array to a 3D RLST array
pub fn from_py3<'py, T: Element + RlstScalar>(
    a: &'py PyReadonlyArray3<'py, T>,
) -> SliceArray<'py, T, 3> {
    let a_array = a.as_array();
    let shape = a_array.shape();
    SliceArray::<T, 3>::from_shape(a.as_slice().unwrap(), [shape[2], shape[1], shape[0]])
}

/// Convert a 4D Python array to a 4D RLST array
pub fn from_py4<'py, T: Element + RlstScalar>(
    a: &'py PyReadonlyArray4<'py, T>,
) -> SliceArray<'py, T, 4> {
    let a_array = a.as_array();
    let shape = a_array.shape();
    SliceArray::<T, 4>::from_shape(
        a.as_slice().unwrap(),
        [shape[3], shape[2], shape[1], shape[0]],
    )
}
