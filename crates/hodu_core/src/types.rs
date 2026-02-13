mod compiler;
mod device;
mod dim;
mod dtype;
mod dynamic_registry;
#[cfg(feature = "serde")]
mod format;
mod layout;
mod shape;
mod symbolic_layout;
mod symbolic_shape;

pub use compiler::Compiler;
pub use device::Device;
pub use dim::{Dim, DynamicDimId};
pub use dtype::DType;
pub use dtypes::*;
pub use dynamic_registry::{clear_resolved_dimensions, get_resolved_dimension, resolve_dimension};
#[cfg(feature = "serde")]
pub use format::Format;
pub use layout::Layout;
pub use shape::Shape;
pub use symbolic_layout::SymbolicLayout;
pub use symbolic_shape::SymbolicShape;

mod dtypes {
    #![allow(non_upper_case_globals)]
    use super::DType;

    pub const bool: DType = DType::BOOL;
    pub const float8e4m3: DType = DType::F8E4M3;
    pub const f8e4m3: DType = DType::F8E4M3;
    #[cfg(feature = "f8e5m2")]
    pub const float8e5m2: DType = DType::F8E5M2;
    #[cfg(feature = "f8e5m2")]
    pub const f8e5m2: DType = DType::F8E5M2;
    pub const bfloat16: DType = DType::BF16;
    pub const bf16: DType = DType::BF16;
    pub const float16: DType = DType::F16;
    pub const f16: DType = DType::F16;
    pub const half: DType = DType::F16;
    pub const float32: DType = DType::F32;
    pub const f32: DType = DType::F32;
    #[cfg(feature = "f64")]
    pub const float64: DType = DType::F64;
    #[cfg(feature = "f64")]
    pub const f64: DType = DType::F64;
    pub const uint8: DType = DType::U8;
    pub const u8: DType = DType::U8;
    #[cfg(feature = "u16")]
    pub const uint16: DType = DType::U16;
    #[cfg(feature = "u16")]
    pub const u16: DType = DType::U16;
    pub const uint32: DType = DType::U32;
    pub const u32: DType = DType::U32;
    #[cfg(feature = "u64")]
    pub const uint64: DType = DType::U64;
    #[cfg(feature = "u64")]
    pub const u64: DType = DType::U64;
    pub const int8: DType = DType::I8;
    pub const i8: DType = DType::I8;
    #[cfg(feature = "i16")]
    pub const int16: DType = DType::I16;
    #[cfg(feature = "i16")]
    pub const i16: DType = DType::I16;
    pub const int32: DType = DType::I32;
    pub const i32: DType = DType::I32;
    #[cfg(feature = "i64")]
    pub const int64: DType = DType::I64;
    #[cfg(feature = "i64")]
    pub const i64: DType = DType::I64;
}
