//! RON I/O
use crate::traits::Mesh;
#[cfg(feature = "mpi")]
use crate::traits::ParallelMesh;
#[cfg(feature = "mpi")]
use mpi::traits::Communicator;
use std::fs;

pub trait ConvertToSerializable {
    //! Convert to/from a RON string
    type SerializableType: serde::Serialize;
    /// Convert to ron
    fn to_serializable(&self) -> Self::SerializableType;
    /// Convert from ron
    fn from_serializable(ron: Self::SerializableType) -> Self;
}

pub trait RONExport: Mesh {
    //! Mesh export for RON

    /// Generate the RON string for a mesh
    fn to_ron_string(&self) -> String;

    /// Export as RON
    fn export_as_ron(&self, filename: &str) {
        let ron_s = self.to_ron_string();
        fs::write(filename, ron_s).expect("Unable to write file");
    }
}

pub trait RONImport: Sized + Mesh {
    //! Mesh import for RON

    /// Generate mesh from a RON string
    fn from_ron_string(s: String) -> Self;

    /// Import from RON
    fn import_from_ron(filename: &str) -> Self {
        let content = fs::read_to_string(filename).expect("Unable to read file");
        Self::from_ron_string(content)
    }
}

#[cfg(feature = "mpi")]
#[derive(Debug, serde::Serialize, serde::Deserialize)]
/// Summary I/O data for a parallel mesh
pub struct ParallelMeshSummaryData {
    mpi_ranks: i32,
}

#[cfg(feature = "mpi")]
pub trait RONExportParallel<'a, C: Communicator + 'a>: ParallelMesh<C = C>
where
    Self::LocalMesh: RONExport,
    Self: 'a,
{
    //! Parallel mesh export for RON

    /// Export as RON
    fn export_as_ron(&'a self, filename: &str) {
        let parts = filename.split('.').collect::<Vec<_>>();
        assert!(parts.len() > 1);
        let sub_filename = format!(
            "{}.{}.{}",
            parts[0..parts.len() - 1].join("."),
            self.comm().rank(),
            parts[parts.len() - 1]
        );

        self.local_mesh().export_as_ron(&sub_filename);
        if self.comm().rank() == 0 {
            let mesh_data = ParallelMeshSummaryData {
                mpi_ranks: self.comm().size(),
            };
            fs::write(filename, ron::to_string(&mesh_data).unwrap()).expect("Unable to write file");
        }
    }
}

#[cfg(feature = "mpi")]
pub trait RONImportParallel<'a, C: Communicator + 'a>: Sized + ParallelMesh<C = C>
where
    Self::LocalMesh: RONImport,
    Self: 'a,
{
    //! Parallel mesh import for RON

    /// Create parallel mesh from comm and local_mesh
    fn create_from_ron_info(comm: &'a C, local_mesh: Self::LocalMesh) -> Self;

    /// Export as RON
    fn import_from_ron(comm: &'a C, filename: &str) -> Self {
        let parts = filename.split('.').collect::<Vec<_>>();
        assert!(parts.len() > 1);
        let sub_filename = format!(
            "{}.{}.{}",
            parts[0..parts.len() - 1].join("."),
            comm.rank(),
            parts[parts.len() - 1]
        );

        let content = fs::read_to_string(filename).expect("Unable to read file");
        let summary: ParallelMeshSummaryData = ron::from_str(&content).unwrap();

        if summary.mpi_ranks != comm.size() {
            panic!("Incorrect number of MPI ranks");
        }

        let local_mesh = Self::LocalMesh::import_from_ron(&sub_filename);
        Self::create_from_ron_info(comm, local_mesh)
    }
}
