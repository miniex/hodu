pub mod capture;

use crate::{
    ops::{Op, OpParams},
    types::{DType, Layout, Shape, SymbolicLayout},
};

// Re-exports
pub use capture::{CaptureBoard, CaptureBoardId};

/// Snapshot-local tensor ID (normalized from runtime TensorId)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SnapshotTensorId(pub usize);

/// Snapshot input specification
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SnapshotInput {
    pub name: String,
    pub id: SnapshotTensorId,
    pub shape: Shape,
    pub dtype: DType,
}

/// Snapshot target (output) specification
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SnapshotTarget {
    pub name: String,
    pub id: SnapshotTensorId,
}

/// Snapshot constant tensor (weights, biases, etc.)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SnapshotConstant {
    pub id: SnapshotTensorId,
    pub name: Option<String>,
    pub shape: Shape,
    pub dtype: DType,
    /// Raw tensor data in bytes
    pub data: Vec<u8>,
}

/// Snapshot node (operation)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SnapshotNode {
    pub op: Op,
    pub params: Option<OpParams>,
    pub input_ids: Vec<SnapshotTensorId>,
    pub output_id: SnapshotTensorId,
    pub input_layouts: Vec<Layout>,
    pub output_layout: Layout,
    pub output_dtype: DType,
    /// Symbolic output layout for operations with data-dependent output shapes
    pub symbolic_output_layout: Option<SymbolicLayout>,
}

/// Hodu Snapshot - serializable IR representation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Snapshot {
    pub name: Option<String>,
    pub inputs: Vec<SnapshotInput>,
    pub constants: Vec<SnapshotConstant>,
    pub targets: Vec<SnapshotTarget>,
    pub nodes: Vec<SnapshotNode>,
}

impl Snapshot {
    pub fn new() -> Self {
        Self {
            name: None,
            inputs: Vec::new(),
            constants: Vec::new(),
            targets: Vec::new(),
            nodes: Vec::new(),
        }
    }

    pub fn with_name(name: impl Into<String>) -> Self {
        Self {
            name: Some(name.into()),
            inputs: Vec::new(),
            constants: Vec::new(),
            targets: Vec::new(),
            nodes: Vec::new(),
        }
    }

    #[cfg(feature = "serde")]
    pub fn to_bytes(&self) -> crate::error::HoduResult<Vec<u8>> {
        postcard::to_allocvec(self).map_err(|e| crate::error::HoduError::SerializationFailed(e.to_string()))
    }

    #[cfg(feature = "serde")]
    pub fn from_bytes(data: &[u8]) -> crate::error::HoduResult<Self> {
        postcard::from_bytes(data).map_err(|e| crate::error::HoduError::DeserializationFailed(e.to_string()))
    }

    #[cfg(feature = "serde")]
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> crate::error::HoduResult<()> {
        let bytes = self.to_bytes()?;
        std::fs::write(path, bytes).map_err(|e| crate::error::HoduError::IoError(e.to_string()))
    }

    #[cfg(feature = "serde")]
    pub fn load(path: impl AsRef<std::path::Path>) -> crate::error::HoduResult<Self> {
        let bytes = std::fs::read(path).map_err(|e| crate::error::HoduError::IoError(e.to_string()))?;
        Self::from_bytes(&bytes)
    }
}

impl Default for Snapshot {
    fn default() -> Self {
        Self::new()
    }
}
