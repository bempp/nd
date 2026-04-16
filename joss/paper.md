---
title: 'ndelement and ndmesh: numerical discretisation in Rust'
tags:
  - Rust
  - finite element method
  - partial differential equations
  - numerical analysis
authors:
  - name: Timo Betcke
    orcid: 0000-0002-3323-2110
    affiliation: 1
  - name: Matthew W. Scroggs
    orcid: 0000-0002-4658-2443
    affiliation: 2
affiliations:
 - name: Department of Mathematics, University College London
   index: 1
 - name: Advanced Research Computing Centre, University College London
   index: 2
date: xx March 2026
bibliography: paper.bib
---

# Summary

When working with partial differential equations (PDEs) that arise from problems in science or
engineering, it is common to use numerical methods to obtain approximate solutions.
Many numerical methods for PDEs, such as the finite element method [@ciarlet;@johnson] and
boundary element method [@steinbach], involve the discretisation of a domain into a mesh of
polygonal or polyhedral cells, and
the definition of a set of (usually) polynomial basis functions on each cell. A library implementing
one of these methods will typically be built around two core pieces of functionality:
the creation and storage of a mesh, and the definition of basis functions on a cell.

The Rust programming language [@rust] has recieved significant attention in recent years, and
the community using it for scientific applications is steadily growing. Rust is a compiled language,
with strict typing and memory safety enforced by the compiler, supported by high quality modern tooling
and package management. There are a number of features of Rust that make it a highly powerful
language for writing scientific applications.

Using Rust's concept of a trait, a developer can define abstract interfaces. A trait can then be
implemented for one or many structures, with the Rust compiler ensuring at compile time that
all necessary traits are implemented. By allowing many traits to
all be separately implemented for the same structure, Rust allows for the implementation of
complex behaviours without any need for complicated hierarchical class inheritance. Traits
also make Rust libraries naturally extensible, as if another developer implements your trait
on their structure, then their structure can in many cases be seamlessly used with the functionality of your
library.

Rust is supported by modern and powerful tooling via the cargo build system and package manager,
with code linting, code quality and testing frameworks included by default. Releases of Rust
crates are hosted online at [crates.io](https://crated.io), and these crates can be easily added
as dependencies of your project by adding them to your crate's Cargo.toml configuration file.

In order to enable the implementation of various discretisation methods using Rust, we have
developed the libraries ndelement and ndmesh, with both libraries developed as sub-crates within
the nd (**n**umerical **d**iscretisation) repository [@nd]. The libraries provide core functionality to
define basis functions and handle meshes. These libraries make use of the Rust linear algebra
library rlst [@rlst] and the parallisation library rsmpi [@rsmpi].

# Software design

In this section, we outline the key features of the ndelement and ndmesh libraries.

## ndelement

The ndelement library includes traits for a finite element and a reference-mapped finite element,
as defined in definitions 2 and 3 of @defelement.
By providing these traits, we allow potential users to build on top of ndelement by implementing these traits for their new
element implementations.
Using some pseudo-code for simplicity and with documentation hidden, the trait for a finite
element looks like this:

\bgroup
\footnotesize
```rust
pub trait FiniteElement {
    type T;
    type CellType;

    fn cell_type(&self) -> Self::CellType;

    fn dim(&self) -> usize;

    fn value_shape(&self) -> &[usize];

    fn value_size(&self) -> usize;

    fn tabulate(
        &self,
        points: &Array2D<Self::T>,
        nderivs: usize,
        data: &mut Array4D<Self::T>,
    );

    fn tabulate_array_shape(
        &self,
        nderivs: usize,
        npoints: usize
    ) -> [usize; 4];

    fn entity_dofs(
        &self,
        entity_dim: usize,
        entity_number: usize,
    ) -> Option<&[usize]>;

    fn entity_closure_dofs(
        &self,
        entity_dim: usize,
        entity_number: usize,
    ) -> Option<&[usize]>;
}
```
\egroup

This trait defines the most general finite element, with no unnecessary assumptions made.
It includes functions to:

- see what cell type the finite element is defined on (`cell_type`). 
  Importantly, the return type of the function `cell_type` is a defined by the implementation of the trait, allowing developers
  of other libraries to return the correct cell type even when it is not included in the `ReferenceCellType` enum that we use
  for the elements we have implemented, allowing us to avoid any assumption that elements can only be defined on certain cell types.
- get the number of basis functions of the finite element (`dim`);
- obtain the shape of the (possibly vector-, matrix- or tensor-valued) basis functions (`value_shape` and `value_size`);
- evaluate the basis functions at a set of points (`tabulate` and the utility function `tabulate_array_shape`); and
- find out which basis functions are associated with each sub-entity of the cell (`entity_dofs` and `entity_closure_dofs`).

The trait for a reference-mapped finite element looks like this:

\bgroup
\footnotesize
```rust
pub trait MappedFiniteElement: FiniteElement {
    type TransformationType;

    fn lagrange_superdegree(&self) -> usize;

    fn push_forward(
        &self,
        reference_values: &Array4D<Self::T>,
        nderivs: usize,
        jacobians: &Array3D<Self::T>,
        jacobian_determinants: &[Self::T],
        inverse_jacobians: &Array3D<Self::T>,
        physical_values: &mut Array4D<Self::T>,
    );

    fn pull_back(
        &self,
        physical_values: &Array4D<Self::T>,
        nderivs: usize,
        jacobians: &Array3D<Self::T>,
        jacobian_determinants: &[Self::T],
        inverse_jacobians: &Array3D<Self::T>,
        reference_values: &mut Array4D<Self::T>,
    );

    fn physical_value_shape(&self, gdim: usize) -> Vec<usize>;

    fn physical_value_size(&self, gdim: usize) -> usize;

    fn dof_transformation(
        &self,
        entity: Self::CellType,
        transformation: Self::TransformationType,
    ) -> Option<&DofTransformation>;

    fn apply_dof_permutations(
        &self,
        data: &mut [Self::T],
        cell_orientation: i32,
    );

    fn apply_dof_transformations(
        &self,
        data: &mut [Self::T],
        cell_orientation: i32,
    );

    fn apply_dof_permutations_and_transformations(
        &self,
        data: &mut [Self::T],
        cell_orientation: i32,
    );
}

```
\egroup

In addition to the functions that make up a finite element, reference-mapped finite elements additionally
include functions to:

- get the maximum Lagrange degree of the finite element's basis functions (as defined in section 2.4 of @defelement) (`lagrange_superdegree`);
- map function values between the reference cell and a physical cell (`push_forward` and `pull_back`);
- obtain the shape of the basis functions after the mapping to a physical cell (`physical_value_shape` and `physical_value_size`); and
- apply the degree-of-freedom (DOF) permutation and transformation method introduced in @dof-transformations and @dof-transformations-algorithm
  (`dof_transformation`, `apply_dof_*`).

ndelement includes implementations of arbitrary degree
Lagrange, Raviart--Thomas [@rt], and N&eacute;d&eacute;lec [@nedelec] finite elements, with
the above traits implemented for these elements. The
correctness of these implementations is verified using DefElement's element verificiation feature
[@defelement]; details of this verification are available at
\href{https://defelement.org/verification/ndelement.html}{defelement.org/verification/ndelement.html}.
The currently available elements in ndelement are summarised in the following table:

Cell type     | Supported element types
------------- | -----------------------
Interval      | Lagrange
Triangle      | Lagrange, Raviart--Thomas, N&eacute;d&eacute;lec (first kind)
Quadrilateral | Lagrange, Raviart--Thomas, N&eacute;d&eacute;lec (first kind)
Tetrahedron   | Lagrange, Raviart--Thomas, N&eacute;d&eacute;lec (first kind)
Hexahedron    | Lagrange, Raviart--Thomas, N&eacute;d&eacute;lec (first kind)

## ndmesh

The aim of the ndmesh library is to handle meshes of polygonal or polyhedral cells. ndmesh
can load and save meshes from/to gmsh format files [@gmsh] and create uniform meshes of some
built-in shapes such as unit squares and cubes. If the library is compiled with the mpi feature
activated, meshes can be distributed over multiple MPI processes, using either
Scotch [@scotch] or Coupe [@coupe] to partition the cells, or by providing a custom partition.

ndmesh includes traits that define a minimal user interface to a mesh. The trait for a mesh includes
functions to iterate through sub-entities. The entities given by this iterator must implement the entity trait
that includes functions to obtain structures implementing the topology and geometry traits that can be used to
obtain information about the connectivity between entities and the physical position of the entity (respectively).

A key strength of this
library is that implementing these traits for any implementation of a mesh (either within ndmesh
or defined in an external library) can allow users to (for example) use functions to assemble finite element matrices
built on top of ndmesh with any of these mesh implementations. Within ndmesh, implementations
of a *single element mesh* (where every cell has the same cell type is represented using the same
degree finite element) and a *mixed mesh* (where cells can be a mixture of cell types and represented
using a mixture of polynomial degrees) are available.

The meshes in ndmesh are implemented with a strict separation between the topology and geometry of
the mesh. This approach is heavily inspired by the representation of meshes in DOLFINx [@dolfinx].

# State of the field

A number of libraries that provide similar functionality to ndelement
already exist. These include the Python library FIAT [@fiat] that is used as part of the Firedrake
framework [@firedrake] and the C++/Python library Basix [@basix] that is one of the components of
FEniCSx [@dolfinx]. The Python library Symfem [@symfem] allows users to find the basis functions
of finite elements symbolically, which is useful for debugging and prototyping but too slow to be used for real
applications.

There are a large number of open source libraries available that implement the finite element
method---including, but not limited to,
deal.II [@deal.ii]
FEniCSx [@dolfinx],
Ferrite.jl [@ferrite],
Firedrake [@firedrake],
MFEM [@mfem], and
ngsolve [@ngsolve].
Each of these libraries necessarily either includes the implementation of finite elements and
a finite element mesh, or uses an implemention of these from another library.

A few implementations of finite elements in Rust are available.
The majority of these implmentations are for a limited set of problems, such as
FEM_2D [@fem2d], which only looks to solve problems in two spatial dimensions.
The most general purpose Rust FEM library that we are aware of is
Fenris [@fenris]. However, the project's readme file states that
Fenris is currently "not recommended for general usage" and it appears that development ceased in
2023.

We do not know of any implementation of boundary element methods in Rust
except for our own experimental implementation built on top of older versions of
ndelement and ndmesh [@bempp-rs].

# Statement of need

While a number of finite element method libraries exist that include implementations of
meshes and finite element basis functions, there is currently no well-maintained general-purpose
library in Rust. We believe that the Rust ecosystem presents an excellent opportunity for very
powerful finite and boundary element libraries, and that the development of solid foundations to
build numerical discretisation Rust libraries on top of is very important.

# Research impact

ndelement and ndmesh are the first two components that we have developed in our work towards
developing a new Rust-based version of the Bempp boundary element method library. Previous versions
of Bempp were implemented in C++ with a Python interface [@bempp] and in pure Python with
just-in-time compiled OpenCL assembly kernels [@bempp-cl].
These previous versions have been used in a wide range of research applications
(see e.g.
@2024-coupling,
@2024-inverse,
@2015-hifu,
@2024-through-wall).
The ultimate goal of this development is that a new Rust library with a Python user interface
that closely resembles Bempp-cl. This will make it easy for the userbase of Bempp to move
to the new version and take advantage of the wide range of new features due to ndelement and ndmesh,
such as the availability of quadrilateral and curved cells in meshes, MPI-distributed meshes, and a wider
range of finite elements.

New research is already being enabled by ndelement and ndmesh. This includes work on
improvements to the fast multiple method for accelerating matrix multiplication [@sri-thesis],
the randomized strong recursive skeletonisation matrix algorithm [@rsrsrs],
and an investigation into function spaces for operator preconditioning on quadrilateral cells
[@bc_quads_repo].

By developing our libraries in a modular way, we aim to allow a variety of discretisation methods
to be developed on top of a set of shared libraries, removing the need for any other groups looking
to develop numerical discretisation methods in Rust to first implement all of the core fundamentals
included in ndelement and ndmesh. We have started developing the crate ndfunctionspace, which uses
ndelement and ndgrid to create function spaces of basis functions on a mesh by assigning global
degree-of-freedom (DOF) numbers to each cell. These function spaces can be used for both finite and
boundary element methods, and so this development is our next step towards implementing these methods
in Rust.

# AI usage
AI inline suggestions from Github Copilot were used for code documentation and as general coding
support tool. No agentic tools were used in the development of the libraries. All AI suggestions
were thoroughly checked before being merged into the main branch.

# References
