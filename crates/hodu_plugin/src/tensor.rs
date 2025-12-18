//! Tensor data types for cross-plugin communication

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// Plugin data type enum (independent from hodu_core::DType for ABI stability)
///
/// This enum has fixed discriminant values to ensure ABI stability across
/// plugin versions. New types can be added with new discriminant values without
/// breaking existing plugins.
///
/// Note: This enum is `#[non_exhaustive]` - new types may be added in future versions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
#[non_exhaustive]
pub enum PluginDType {
    /// Boolean
    BOOL = 0,
    /// 8-bit float (E4M3)
    F8E4M3 = 1,
    /// 8-bit float (E5M2)
    F8E5M2 = 2,
    /// 16-bit bfloat
    BF16 = 3,
    /// 16-bit float
    F16 = 4,
    /// 32-bit float
    F32 = 5,
    /// 64-bit float
    F64 = 6,
    /// 8-bit unsigned integer
    U8 = 7,
    /// 16-bit unsigned integer
    U16 = 8,
    /// 32-bit unsigned integer
    U32 = 9,
    /// 64-bit unsigned integer
    U64 = 10,
    /// 8-bit signed integer
    I8 = 11,
    /// 16-bit signed integer
    I16 = 12,
    /// 32-bit signed integer
    I32 = 13,
    /// 64-bit signed integer
    I64 = 14,
}

impl PluginDType {
    /// Get size of this data type in bytes
    pub const fn size_in_bytes(&self) -> usize {
        match self {
            Self::BOOL | Self::F8E4M3 | Self::F8E5M2 | Self::U8 | Self::I8 => 1,
            Self::BF16 | Self::F16 | Self::U16 | Self::I16 => 2,
            Self::F32 | Self::U32 | Self::I32 => 4,
            Self::F64 | Self::U64 | Self::I64 => 8,
        }
    }

    /// Get name of this data type
    pub const fn name(&self) -> &'static str {
        match self {
            Self::BOOL => "bool",
            Self::F8E4M3 => "f8e4m3",
            Self::F8E5M2 => "f8e5m2",
            Self::BF16 => "bf16",
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
        }
    }

    /// Check if this is a floating point type
    pub const fn is_float(&self) -> bool {
        matches!(
            self,
            Self::F8E4M3 | Self::F8E5M2 | Self::BF16 | Self::F16 | Self::F32 | Self::F64
        )
    }

    /// Check if this is an integer type
    pub const fn is_integer(&self) -> bool {
        matches!(
            self,
            Self::U8 | Self::U16 | Self::U32 | Self::U64 | Self::I8 | Self::I16 | Self::I32 | Self::I64
        )
    }

    /// Check if this is a signed type
    pub const fn is_signed(&self) -> bool {
        matches!(
            self,
            Self::F8E4M3
                | Self::F8E5M2
                | Self::BF16
                | Self::F16
                | Self::F32
                | Self::F64
                | Self::I8
                | Self::I16
                | Self::I32
                | Self::I64
        )
    }
}

/// Error type for PluginDType parsing
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseDTypeError {
    dtype: String,
}

impl ParseDTypeError {
    fn new(dtype: impl Into<String>) -> Self {
        Self { dtype: dtype.into() }
    }
}

impl fmt::Display for ParseDTypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Unknown dtype: {}", self.dtype)
    }
}

impl std::error::Error for ParseDTypeError {}

impl FromStr for PluginDType {
    type Err = ParseDTypeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "bool" => Ok(Self::BOOL),
            "f8e4m3" => Ok(Self::F8E4M3),
            "f8e5m2" => Ok(Self::F8E5M2),
            "bf16" => Ok(Self::BF16),
            "f16" => Ok(Self::F16),
            "f32" => Ok(Self::F32),
            "f64" => Ok(Self::F64),
            "u8" => Ok(Self::U8),
            "u16" => Ok(Self::U16),
            "u32" => Ok(Self::U32),
            "u64" => Ok(Self::U64),
            "i8" => Ok(Self::I8),
            "i16" => Ok(Self::I16),
            "i32" => Ok(Self::I32),
            "i64" => Ok(Self::I64),
            _ => Err(ParseDTypeError::new(s)),
        }
    }
}

impl std::fmt::Display for PluginDType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Raw tensor data for cross-plugin communication
///
/// This struct is used to pass tensor data between the CLI and plugins
/// without depending on the full Tensor type and registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorData {
    /// Raw bytes of tensor data
    pub data: Vec<u8>,
    /// Shape dimensions
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: PluginDType,
}

/// Error type for tensor data validation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorDataError {
    /// Data size doesn't match expected size from shape and dtype
    SizeMismatch { expected: usize, actual: usize },
    /// Shape contains zero dimension (invalid tensor)
    ZeroDimension,
    /// Shape product overflows usize
    ShapeOverflow,
}

impl std::fmt::Display for TensorDataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SizeMismatch { expected, actual } => {
                write!(f, "Data size mismatch: expected {} bytes, got {}", expected, actual)
            },
            Self::ZeroDimension => {
                write!(f, "Shape contains zero dimension")
            },
            Self::ShapeOverflow => {
                write!(f, "Shape product overflows usize")
            },
        }
    }
}

impl std::error::Error for TensorDataError {}

impl TensorData {
    /// Create new tensor data without validation
    pub fn new(data: Vec<u8>, shape: Vec<usize>, dtype: PluginDType) -> Self {
        Self { data, shape, dtype }
    }

    /// Create new tensor data with validation
    ///
    /// Returns an error if:
    /// - Shape contains a zero dimension
    /// - Shape product overflows usize
    /// - Data size doesn't match the expected size from shape and dtype
    pub fn new_checked(data: Vec<u8>, shape: Vec<usize>, dtype: PluginDType) -> Result<Self, TensorDataError> {
        // Check for zero dimensions (unless scalar)
        if !shape.is_empty() && shape.contains(&0) {
            return Err(TensorDataError::ZeroDimension);
        }

        // Use checked multiplication to prevent overflow
        let numel = Self::checked_numel(&shape).ok_or(TensorDataError::ShapeOverflow)?;
        let expected_size = numel
            .checked_mul(dtype.size_in_bytes())
            .ok_or(TensorDataError::ShapeOverflow)?;

        if data.len() != expected_size {
            return Err(TensorDataError::SizeMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }
        Ok(Self { data, shape, dtype })
    }

    /// Compute number of elements with overflow checking
    fn checked_numel(shape: &[usize]) -> Option<usize> {
        shape.iter().try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
    }

    /// Number of elements in the tensor
    ///
    /// Note: This may overflow for very large shapes. Use `checked_numel()` for validation.
    pub fn numel(&self) -> usize {
        Self::checked_numel(&self.shape).unwrap_or(usize::MAX)
    }

    /// Size of data in bytes
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }

    /// Check if tensor data is valid (size matches shape * dtype)
    ///
    /// Returns false if:
    /// - Shape contains a zero dimension (unless scalar)
    /// - Shape product overflows
    /// - Data size doesn't match expected size
    pub fn is_valid(&self) -> bool {
        // Check for zero dimensions (unless scalar)
        if !self.shape.is_empty() && self.shape.contains(&0) {
            return false;
        }

        // Use checked arithmetic
        let Some(numel) = Self::checked_numel(&self.shape) else {
            return false;
        };
        let Some(expected_size) = numel.checked_mul(self.dtype.size_in_bytes()) else {
            return false;
        };
        self.data.len() == expected_size
    }

    /// Get tensor rank (number of dimensions)
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Check if tensor is scalar (rank 0)
    pub fn is_scalar(&self) -> bool {
        self.shape.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plugin_dtype_from_str() {
        assert_eq!("f32".parse::<PluginDType>().unwrap(), PluginDType::F32);
        assert_eq!("F32".parse::<PluginDType>().unwrap(), PluginDType::F32);
        assert_eq!("bool".parse::<PluginDType>().unwrap(), PluginDType::BOOL);
        assert_eq!("i64".parse::<PluginDType>().unwrap(), PluginDType::I64);
        assert_eq!("bf16".parse::<PluginDType>().unwrap(), PluginDType::BF16);
        assert!("invalid".parse::<PluginDType>().is_err());
    }

    #[test]
    fn test_parse_dtype_error() {
        let err = "invalid".parse::<PluginDType>().unwrap_err();
        assert_eq!(err.to_string(), "Unknown dtype: invalid");
    }

    #[test]
    fn test_tensor_data_new_checked_valid() {
        // 2x3 F32 tensor = 6 elements * 4 bytes = 24 bytes
        let data = vec![0u8; 24];
        let result = TensorData::new_checked(data, vec![2, 3], PluginDType::F32);
        assert!(result.is_ok());
    }

    #[test]
    fn test_tensor_data_new_checked_invalid() {
        // Wrong size: 2x3 F32 needs 24 bytes, providing 12
        let data = vec![0u8; 12];
        let result = TensorData::new_checked(data, vec![2, 3], PluginDType::F32);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(
            err,
            TensorDataError::SizeMismatch {
                expected: 24,
                actual: 12
            }
        );
    }

    #[test]
    fn test_tensor_data_is_valid() {
        let valid = TensorData::new(vec![0u8; 8], vec![2], PluginDType::F32);
        assert!(valid.is_valid());

        let invalid = TensorData::new(vec![0u8; 4], vec![2], PluginDType::F32);
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_tensor_data_is_scalar() {
        let scalar = TensorData::new(vec![0u8; 4], vec![], PluginDType::F32);
        assert!(scalar.is_scalar());
        assert_eq!(scalar.rank(), 0);

        let non_scalar = TensorData::new(vec![0u8; 4], vec![1], PluginDType::F32);
        assert!(!non_scalar.is_scalar());
        assert_eq!(non_scalar.rank(), 1);
    }

    #[test]
    fn test_tensor_data_numel() {
        let tensor = TensorData::new(vec![], vec![2, 3, 4], PluginDType::F32);
        assert_eq!(tensor.numel(), 24);

        let scalar = TensorData::new(vec![], vec![], PluginDType::F32);
        assert_eq!(scalar.numel(), 1);
    }

    #[test]
    fn test_plugin_dtype_size() {
        assert_eq!(PluginDType::BOOL.size_in_bytes(), 1);
        assert_eq!(PluginDType::F16.size_in_bytes(), 2);
        assert_eq!(PluginDType::F32.size_in_bytes(), 4);
        assert_eq!(PluginDType::F64.size_in_bytes(), 8);
        assert_eq!(PluginDType::I8.size_in_bytes(), 1);
        assert_eq!(PluginDType::I64.size_in_bytes(), 8);
    }

    #[test]
    fn test_plugin_dtype_display() {
        assert_eq!(PluginDType::F32.to_string(), "f32");
        assert_eq!(PluginDType::BOOL.to_string(), "bool");
        assert_eq!(PluginDType::BF16.to_string(), "bf16");
    }

    #[test]
    fn test_plugin_dtype_properties() {
        assert!(PluginDType::F32.is_float());
        assert!(!PluginDType::F32.is_integer());
        assert!(PluginDType::F32.is_signed());

        assert!(!PluginDType::I32.is_float());
        assert!(PluginDType::I32.is_integer());
        assert!(PluginDType::I32.is_signed());

        assert!(!PluginDType::U32.is_float());
        assert!(PluginDType::U32.is_integer());
        assert!(!PluginDType::U32.is_signed());
    }

    #[test]
    fn test_tensor_data_zero_dimension() {
        // Zero dimension should fail validation
        let result = TensorData::new_checked(vec![], vec![0, 5], PluginDType::F32);
        assert_eq!(result.unwrap_err(), TensorDataError::ZeroDimension);

        let result = TensorData::new_checked(vec![], vec![2, 0, 3], PluginDType::F32);
        assert_eq!(result.unwrap_err(), TensorDataError::ZeroDimension);

        // is_valid should also reject zero dimensions
        let tensor = TensorData::new(vec![], vec![0, 5], PluginDType::F32);
        assert!(!tensor.is_valid());
    }

    #[test]
    fn test_tensor_data_overflow() {
        // Very large shape that would overflow
        let result = TensorData::new_checked(vec![], vec![usize::MAX, 2], PluginDType::F32);
        assert_eq!(result.unwrap_err(), TensorDataError::ShapeOverflow);

        // numel should return MAX on overflow instead of panicking
        let tensor = TensorData::new(vec![], vec![usize::MAX, 2], PluginDType::F32);
        assert_eq!(tensor.numel(), usize::MAX);
        assert!(!tensor.is_valid());
    }
}
