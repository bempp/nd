//! Serial function space
use crate::traits::FunctionSpace;
use itertools::izip;
use ndelement::{
    reference_cell,
    traits::{ElementFamily, MappedFiniteElement},
    types::ReferenceCellType,
};
use ndmesh::{
    traits::{Entity, Mesh, Topology},
    types::Ownership,
};
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

/// Function space.
pub struct FunctionSpaceImpl<
    'a,
    E: Debug + PartialEq + Eq + Clone + Copy + Hash,
    G: Mesh<EntityDescriptor = E>,
    F: MappedFiniteElement<CellType = E>,
> {
    mesh: &'a G,
    elements: Vec<F>,
    entity_dofs: HashMap<E, Vec<Vec<usize>>>,
    entity_closure_dofs: HashMap<E, Vec<Vec<usize>>>,
    ndofs: usize,
    entities_by_element: HashMap<E, Vec<usize>>,
}

impl<
    'a,
    G: Mesh<EntityDescriptor = ReferenceCellType>,
    F: MappedFiniteElement<CellType = ReferenceCellType>,
> FunctionSpaceImpl<'a, ReferenceCellType, G, F>
{
    /// Create a new serial function space
    pub fn new<EF: ElementFamily<FiniteElement = F, CellType = ReferenceCellType>>(
        mesh: &'a G,
        family: &EF,
    ) -> Self {
        let elements = mesh
            .entity_types(mesh.topology_dim())
            .iter()
            .map(|e| family.element(*e))
            .collect::<Vec<_>>();
        let mut entity_dofs = HashMap::new();
        let mut entity_closure_dofs = HashMap::new();
        let mut entities_by_element = HashMap::new();

        for d in 0..=mesh.topology_dim() {
            for e in mesh.entity_types(d) {
                entity_dofs.insert(*e, vec![vec![]; mesh.entity_count(*e)]);
                entity_closure_dofs.insert(*e, vec![vec![]; mesh.entity_count(*e)]);
            }
        }

        let mut ndofs = 0;

        for (cell_type, element) in izip!(mesh.entity_types(mesh.topology_dim()), &elements) {
            let sub_entity_types = reference_cell::entity_types(*cell_type);
            let sub_entity_indices = reference_cell::entity_indices(*cell_type);
            let mut cell_indices = vec![];
            for cell in mesh.entity_iter(*cell_type) {
                cell_indices.push(cell.local_index());
                let mut cell_dofs = vec![]; // &mut entity_closure_dofs.get_mut(cell_type).unwrap()[cell.local_index()];
                for (d, (types, indices)) in
                    izip!(&sub_entity_types, &sub_entity_indices).enumerate()
                {
                    for (t, i) in izip!(types, indices) {
                        let reference_entity_dofs = element.entity_dofs(d, *i).unwrap();
                        if !reference_entity_dofs.is_empty() && !reference_entity_dofs.is_empty() {
                            let ed = &mut entity_dofs.get_mut(t).unwrap()
                                [cell.topology().sub_entity(*t, *i)];
                            while ed.len() < reference_entity_dofs.len() {
                                ed.push(ndofs);
                                ndofs += 1;
                            }
                            for dof in ed {
                                cell_dofs.push(*dof);
                            }
                        }
                        entity_closure_dofs.get_mut(t).unwrap()
                            [cell.topology().sub_entity(*t, *i)] = element
                            .entity_closure_dofs(d, *i)
                            .unwrap()
                            .iter()
                            .map(|e| cell_dofs[*e])
                            .collect::<Vec<_>>();
                    }
                }
            }
            entities_by_element.insert(*cell_type, cell_indices);
        }

        Self {
            mesh,
            elements,
            entity_dofs,
            entity_closure_dofs,
            ndofs,
            entities_by_element,
        }
    }
}

impl<
    'a,
    E: Debug + PartialEq + Eq + Clone + Copy + Hash,
    G: Mesh<EntityDescriptor = E>,
    F: MappedFiniteElement<CellType = E>,
> FunctionSpace for FunctionSpaceImpl<'a, E, G, F>
{
    type EntityDescriptor = E;
    type Mesh = G;
    type FiniteElement = F;

    fn mesh(&self) -> &G {
        self.mesh
    }

    fn elements(&self) -> &[F] {
        &self.elements
    }

    fn entities_by_element(&self, element_index: usize) -> Option<&[usize]> {
        self.entities_by_element
            .get(&self.elements[element_index].cell_type())
            .map(|v| &**v)
    }

    fn entity_dofs(&self, entity_type: E, entity_number: usize) -> Option<&[usize]> {
        if let Some(i) = self.entity_dofs.get(&entity_type) {
            if let Some(j) = i.get(entity_number) {
                Some(j)
            } else {
                None
            }
        } else {
            None
        }
    }

    fn entity_closure_dofs(&self, entity_type: E, entity_number: usize) -> Option<&[usize]> {
        if let Some(i) = self.entity_closure_dofs.get(&entity_type) {
            if let Some(j) = i.get(entity_number) {
                Some(j)
            } else {
                None
            }
        } else {
            None
        }
    }

    fn local_size(&self) -> usize {
        self.ndofs
    }

    fn global_size(&self) -> usize {
        self.ndofs
    }

    fn global_dof_index(&self, local_dof_index: usize) -> usize {
        local_dof_index
    }

    fn ownership(&self, _local_dof_index: usize) -> Ownership {
        Ownership::Owned
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ndelement::{
        ciarlet::LagrangeElementFamily,
        types::{Continuity, ReferenceCellType},
    };
    use ndmesh::shapes::unit_cube_boundary;

    #[test]
    fn test_dp0() {
        let mesh = unit_cube_boundary::<f64>(2, 2, 2, ReferenceCellType::Triangle);
        let family = LagrangeElementFamily::<f64>::new(0, Continuity::Discontinuous);

        let space = FunctionSpaceImpl::new(&mesh, &family);

        assert_eq!(
            space.local_size(),
            mesh.entity_count(ReferenceCellType::Triangle)
        );

        for cell in mesh.entity_iter(ReferenceCellType::Point) {
            assert_eq!(
                space
                    .entity_dofs(ReferenceCellType::Point, cell.local_index())
                    .unwrap()
                    .len(),
                0
            );
            assert_eq!(
                space
                    .entity_closure_dofs(ReferenceCellType::Point, cell.local_index())
                    .unwrap()
                    .len(),
                0
            );
        }
        for cell in mesh.entity_iter(ReferenceCellType::Interval) {
            assert_eq!(
                space
                    .entity_dofs(ReferenceCellType::Interval, cell.local_index())
                    .unwrap()
                    .len(),
                0
            );
            assert_eq!(
                space
                    .entity_closure_dofs(ReferenceCellType::Interval, cell.local_index())
                    .unwrap()
                    .len(),
                0
            );
        }
        for cell in mesh.entity_iter(ReferenceCellType::Triangle) {
            assert_eq!(
                space
                    .entity_dofs(ReferenceCellType::Triangle, cell.local_index())
                    .unwrap()
                    .len(),
                1
            );
            assert_eq!(
                space
                    .entity_closure_dofs(ReferenceCellType::Triangle, cell.local_index())
                    .unwrap()
                    .len(),
                1
            );
        }
    }

    #[test]
    fn test_p1() {
        let mesh = unit_cube_boundary::<f64>(2, 2, 2, ReferenceCellType::Triangle);
        let family = LagrangeElementFamily::<f64>::new(1, Continuity::Standard);

        let space = FunctionSpaceImpl::new(&mesh, &family);

        assert_eq!(
            space.local_size(),
            mesh.entity_count(ReferenceCellType::Point)
        );

        for cell in mesh.entity_iter(ReferenceCellType::Point) {
            assert_eq!(
                space
                    .entity_dofs(ReferenceCellType::Point, cell.local_index())
                    .unwrap()
                    .len(),
                1
            );
            assert_eq!(
                space
                    .entity_closure_dofs(ReferenceCellType::Point, cell.local_index())
                    .unwrap()
                    .len(),
                1
            );
        }
        for cell in mesh.entity_iter(ReferenceCellType::Interval) {
            assert_eq!(
                space
                    .entity_dofs(ReferenceCellType::Interval, cell.local_index())
                    .unwrap()
                    .len(),
                0
            );
            assert_eq!(
                space
                    .entity_closure_dofs(ReferenceCellType::Interval, cell.local_index())
                    .unwrap()
                    .len(),
                2
            );
        }
        for cell in mesh.entity_iter(ReferenceCellType::Triangle) {
            assert_eq!(
                space
                    .entity_dofs(ReferenceCellType::Triangle, cell.local_index())
                    .unwrap()
                    .len(),
                0
            );
            assert_eq!(
                space
                    .entity_closure_dofs(ReferenceCellType::Triangle, cell.local_index())
                    .unwrap()
                    .len(),
                3
            );
        }
    }

    #[test]
    fn test_dp1() {
        let mesh = unit_cube_boundary::<f64>(2, 2, 2, ReferenceCellType::Triangle);
        let family = LagrangeElementFamily::<f64>::new(1, Continuity::Discontinuous);

        let space = FunctionSpaceImpl::new(&mesh, &family);

        assert_eq!(
            space.local_size(),
            3 * mesh.entity_count(ReferenceCellType::Triangle)
        );

        for cell in mesh.entity_iter(ReferenceCellType::Point) {
            assert_eq!(
                space
                    .entity_dofs(ReferenceCellType::Point, cell.local_index())
                    .unwrap()
                    .len(),
                0
            );
            assert_eq!(
                space
                    .entity_closure_dofs(ReferenceCellType::Point, cell.local_index())
                    .unwrap()
                    .len(),
                0
            );
        }
        for cell in mesh.entity_iter(ReferenceCellType::Interval) {
            assert_eq!(
                space
                    .entity_dofs(ReferenceCellType::Interval, cell.local_index())
                    .unwrap()
                    .len(),
                0
            );
            assert_eq!(
                space
                    .entity_closure_dofs(ReferenceCellType::Interval, cell.local_index())
                    .unwrap()
                    .len(),
                0
            );
        }
        for cell in mesh.entity_iter(ReferenceCellType::Triangle) {
            assert_eq!(
                space
                    .entity_dofs(ReferenceCellType::Triangle, cell.local_index())
                    .unwrap()
                    .len(),
                3
            );
            assert_eq!(
                space
                    .entity_closure_dofs(ReferenceCellType::Triangle, cell.local_index())
                    .unwrap()
                    .len(),
                3
            );
        }
    }

    #[test]
    fn test_p2() {
        let mesh = unit_cube_boundary::<f64>(2, 2, 2, ReferenceCellType::Triangle);
        let family = LagrangeElementFamily::<f64>::new(2, Continuity::Standard);

        let space = FunctionSpaceImpl::new(&mesh, &family);

        assert_eq!(
            space.local_size(),
            mesh.entity_count(ReferenceCellType::Point)
                + mesh.entity_count(ReferenceCellType::Interval)
        );

        for cell in mesh.entity_iter(ReferenceCellType::Point) {
            assert_eq!(
                space
                    .entity_dofs(ReferenceCellType::Point, cell.local_index())
                    .unwrap()
                    .len(),
                1
            );
            assert_eq!(
                space
                    .entity_closure_dofs(ReferenceCellType::Point, cell.local_index())
                    .unwrap()
                    .len(),
                1
            );
        }
        for cell in mesh.entity_iter(ReferenceCellType::Interval) {
            assert_eq!(
                space
                    .entity_dofs(ReferenceCellType::Interval, cell.local_index())
                    .unwrap()
                    .len(),
                1
            );
            assert_eq!(
                space
                    .entity_closure_dofs(ReferenceCellType::Interval, cell.local_index())
                    .unwrap()
                    .len(),
                3
            );
        }
        for cell in mesh.entity_iter(ReferenceCellType::Triangle) {
            assert_eq!(
                space
                    .entity_dofs(ReferenceCellType::Triangle, cell.local_index())
                    .unwrap()
                    .len(),
                0
            );
            assert_eq!(
                space
                    .entity_closure_dofs(ReferenceCellType::Triangle, cell.local_index())
                    .unwrap()
                    .len(),
                6
            );
        }
    }

    #[test]
    fn test_p3() {
        let mesh = unit_cube_boundary::<f64>(2, 2, 2, ReferenceCellType::Triangle);
        let family = LagrangeElementFamily::<f64>::new(3, Continuity::Standard);

        let space = FunctionSpaceImpl::new(&mesh, &family);

        assert_eq!(
            space.local_size(),
            mesh.entity_count(ReferenceCellType::Point)
                + 2 * mesh.entity_count(ReferenceCellType::Interval)
                + mesh.entity_count(ReferenceCellType::Triangle)
        );

        for cell in mesh.entity_iter(ReferenceCellType::Point) {
            assert_eq!(
                space
                    .entity_dofs(ReferenceCellType::Point, cell.local_index())
                    .unwrap()
                    .len(),
                1
            );
            assert_eq!(
                space
                    .entity_closure_dofs(ReferenceCellType::Point, cell.local_index())
                    .unwrap()
                    .len(),
                1
            );
        }
        for cell in mesh.entity_iter(ReferenceCellType::Interval) {
            assert_eq!(
                space
                    .entity_dofs(ReferenceCellType::Interval, cell.local_index())
                    .unwrap()
                    .len(),
                2
            );
            assert_eq!(
                space
                    .entity_closure_dofs(ReferenceCellType::Interval, cell.local_index())
                    .unwrap()
                    .len(),
                4
            );
        }
        for cell in mesh.entity_iter(ReferenceCellType::Triangle) {
            assert_eq!(
                space
                    .entity_dofs(ReferenceCellType::Triangle, cell.local_index())
                    .unwrap()
                    .len(),
                1
            );
            assert_eq!(
                space
                    .entity_closure_dofs(ReferenceCellType::Triangle, cell.local_index())
                    .unwrap()
                    .len(),
                10
            );
        }
    }

    #[test]
    fn test_p1_quad() {
        let mesh = unit_cube_boundary::<f64>(2, 2, 2, ReferenceCellType::Quadrilateral);
        let family = LagrangeElementFamily::<f64>::new(1, Continuity::Standard);

        let space = FunctionSpaceImpl::new(&mesh, &family);

        assert_eq!(
            space.local_size(),
            mesh.entity_count(ReferenceCellType::Point)
        );

        for cell in mesh.entity_iter(ReferenceCellType::Point) {
            assert_eq!(
                space
                    .entity_dofs(ReferenceCellType::Point, cell.local_index())
                    .unwrap()
                    .len(),
                1
            );
            assert_eq!(
                space
                    .entity_closure_dofs(ReferenceCellType::Point, cell.local_index())
                    .unwrap()
                    .len(),
                1
            );
        }
        for cell in mesh.entity_iter(ReferenceCellType::Interval) {
            assert_eq!(
                space
                    .entity_dofs(ReferenceCellType::Interval, cell.local_index())
                    .unwrap()
                    .len(),
                0
            );
            assert_eq!(
                space
                    .entity_closure_dofs(ReferenceCellType::Interval, cell.local_index())
                    .unwrap()
                    .len(),
                2
            );
        }
        for cell in mesh.entity_iter(ReferenceCellType::Quadrilateral) {
            assert_eq!(
                space
                    .entity_dofs(ReferenceCellType::Quadrilateral, cell.local_index())
                    .unwrap()
                    .len(),
                0
            );
            assert_eq!(
                space
                    .entity_closure_dofs(ReferenceCellType::Quadrilateral, cell.local_index())
                    .unwrap()
                    .len(),
                4
            );
        }
    }

    #[test]
    fn test_p2_quad() {
        let mesh = unit_cube_boundary::<f64>(2, 2, 2, ReferenceCellType::Quadrilateral);
        let family = LagrangeElementFamily::<f64>::new(2, Continuity::Standard);

        let space = FunctionSpaceImpl::new(&mesh, &family);

        assert_eq!(
            space.local_size(),
            mesh.entity_count(ReferenceCellType::Point)
                + mesh.entity_count(ReferenceCellType::Interval)
                + mesh.entity_count(ReferenceCellType::Quadrilateral)
        );

        for cell in mesh.entity_iter(ReferenceCellType::Point) {
            assert_eq!(
                space
                    .entity_dofs(ReferenceCellType::Point, cell.local_index())
                    .unwrap()
                    .len(),
                1
            );
            assert_eq!(
                space
                    .entity_closure_dofs(ReferenceCellType::Point, cell.local_index())
                    .unwrap()
                    .len(),
                1
            );
        }
        for cell in mesh.entity_iter(ReferenceCellType::Interval) {
            assert_eq!(
                space
                    .entity_dofs(ReferenceCellType::Interval, cell.local_index())
                    .unwrap()
                    .len(),
                1
            );
            assert_eq!(
                space
                    .entity_closure_dofs(ReferenceCellType::Interval, cell.local_index())
                    .unwrap()
                    .len(),
                3
            );
        }
        for cell in mesh.entity_iter(ReferenceCellType::Quadrilateral) {
            assert_eq!(
                space
                    .entity_dofs(ReferenceCellType::Quadrilateral, cell.local_index())
                    .unwrap()
                    .len(),
                1
            );
            assert_eq!(
                space
                    .entity_closure_dofs(ReferenceCellType::Quadrilateral, cell.local_index())
                    .unwrap()
                    .len(),
                9
            );
        }
    }
}
