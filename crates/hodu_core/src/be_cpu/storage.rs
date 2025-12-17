#![allow(clippy::upper_case_acronyms)]

mod ops_binary;
mod ops_bitwise;
mod ops_concat_split;
mod ops_conv;
mod ops_einsum;
mod ops_indexing;
mod ops_linalg;
mod ops_matrix;
mod ops_padding;
mod ops_reduce;
mod ops_resize;
mod ops_scan;
mod ops_shape_memory;
mod ops_sort;
mod ops_unary;
mod ops_windowing;

use crate::{
    be::{device::BackendDeviceT, storage::BackendStorageT},
    be_cpu::device::CpuDevice,
    error::{HoduError, HoduResult},
    op_metadatas,
    ops::Op,
    scalar::Scalar,
    types::{DType, Device, Layout},
};
use core::ffi::c_void;
use float8::F8E4M3;
#[cfg(feature = "f8e5m2")]
use float8::F8E5M2;
use half::{bf16, f16};

#[derive(Debug, Clone)]
pub enum CpuStorage {
    BOOL(Vec<bool>),
    F8E4M3(Vec<F8E4M3>),
    #[cfg(feature = "f8e5m2")]
    F8E5M2(Vec<F8E5M2>),
    BF16(Vec<bf16>),
    F16(Vec<f16>),
    F32(Vec<f32>),
    #[cfg(feature = "f64")]
    F64(Vec<f64>),
    U8(Vec<u8>),
    #[cfg(feature = "u16")]
    U16(Vec<u16>),
    U32(Vec<u32>),
    #[cfg(feature = "u64")]
    U64(Vec<u64>),
    I8(Vec<i8>),
    #[cfg(feature = "i16")]
    I16(Vec<i16>),
    I32(Vec<i32>),
    #[cfg(feature = "i64")]
    I64(Vec<i64>),
}

// #[derive(Debug, Clone)]
// pub enum CpuStorageRef<'a> {
//     BOOL(&'a [bool]),
//     F8E4M3(&'a [F8E4M3]),
//     #[cfg(feature = "f8e5m2")]
//     F8E5M2(&'a [F8E5M2]),
//     BF16(&'a [bf16]),
//     F16(&'a [f16]),
//     F32(&'a [f32]),
//     #[cfg(feature = "f64")]
//     F64(&'a [f64]),
//     U8(&'a [u8]),
//     #[cfg(feature = "u16")]
//     U16(&'a [u16]),
//     U32(&'a [u32]),
//     #[cfg(feature = "u64")]
//     U64(&'a [u64]),
//     I8(&'a [i8]),
//     #[cfg(feature = "i16")]
//     I16(&'a [i16]),
//     I32(&'a [i32]),
//     #[cfg(feature = "i64")]
//     I64(&'a [i64]),
// }

impl CpuStorage {
    /// Get raw pointer to the underlying data
    pub fn as_ptr(&self) -> *const u8 {
        match self {
            Self::BOOL(v) => v.as_ptr() as *const u8,
            Self::F8E4M3(v) => v.as_ptr() as *const u8,
            #[cfg(feature = "f8e5m2")]
            Self::F8E5M2(v) => v.as_ptr() as *const u8,
            Self::BF16(v) => v.as_ptr() as *const u8,
            Self::F16(v) => v.as_ptr() as *const u8,
            Self::F32(v) => v.as_ptr() as *const u8,
            #[cfg(feature = "f64")]
            Self::F64(v) => v.as_ptr() as *const u8,
            Self::U8(v) => v.as_ptr(),
            #[cfg(feature = "u16")]
            Self::U16(v) => v.as_ptr() as *const u8,
            Self::U32(v) => v.as_ptr() as *const u8,
            #[cfg(feature = "u64")]
            Self::U64(v) => v.as_ptr() as *const u8,
            Self::I8(v) => v.as_ptr() as *const u8,
            #[cfg(feature = "i16")]
            Self::I16(v) => v.as_ptr() as *const u8,
            Self::I32(v) => v.as_ptr() as *const u8,
            #[cfg(feature = "i64")]
            Self::I64(v) => v.as_ptr() as *const u8,
        }
    }

    /// Get mutable raw pointer to the underlying data
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        match self {
            Self::BOOL(v) => v.as_mut_ptr() as *mut u8,
            Self::F8E4M3(v) => v.as_mut_ptr() as *mut u8,
            #[cfg(feature = "f8e5m2")]
            Self::F8E5M2(v) => v.as_mut_ptr() as *mut u8,
            Self::BF16(v) => v.as_mut_ptr() as *mut u8,
            Self::F16(v) => v.as_mut_ptr() as *mut u8,
            Self::F32(v) => v.as_mut_ptr() as *mut u8,
            #[cfg(feature = "f64")]
            Self::F64(v) => v.as_mut_ptr() as *mut u8,
            Self::U8(v) => v.as_mut_ptr(),
            #[cfg(feature = "u16")]
            Self::U16(v) => v.as_mut_ptr() as *mut u8,
            Self::U32(v) => v.as_mut_ptr() as *mut u8,
            #[cfg(feature = "u64")]
            Self::U64(v) => v.as_mut_ptr() as *mut u8,
            Self::I8(v) => v.as_mut_ptr() as *mut u8,
            #[cfg(feature = "i16")]
            Self::I16(v) => v.as_mut_ptr() as *mut u8,
            Self::I32(v) => v.as_mut_ptr() as *mut u8,
            #[cfg(feature = "i64")]
            Self::I64(v) => v.as_mut_ptr() as *mut u8,
        }
    }

    pub fn from_vec<T: 'static>(vec: Vec<T>) -> Self {
        let any_vec = &vec as &dyn core::any::Any;

        macro_rules! try_downcast {
            ($type:ty, $variant:ident) => {
                if let Some(v) = any_vec.downcast_ref::<Vec<$type>>() {
                    return Self::$variant(v.clone());
                }
            };
            ($type:ty, $variant:ident, $cfg:meta) => {
                #[cfg($cfg)]
                if let Some(v) = any_vec.downcast_ref::<Vec<$type>>() {
                    return Self::$variant(v.clone());
                }
            };
        }

        try_downcast!(bool, BOOL);
        try_downcast!(F8E4M3, F8E4M3);
        try_downcast!(F8E5M2, F8E5M2, feature = "f8e5m2");
        try_downcast!(bf16, BF16);
        try_downcast!(f16, F16);
        try_downcast!(f32, F32);
        try_downcast!(f64, F64, feature = "f64");
        try_downcast!(u8, U8);
        try_downcast!(u16, U16, feature = "u16");
        try_downcast!(u32, U32);
        try_downcast!(u64, U64, feature = "u64");
        try_downcast!(i8, I8);
        try_downcast!(i16, I16, feature = "i16");
        try_downcast!(i32, I32);
        try_downcast!(i64, I64, feature = "i64");

        panic!("Unsupported vector type for CpuStorage");
    }

    pub fn from_bytes(bytes: &[u8], dtype: DType) -> HoduResult<Self> {
        use half::{bf16, f16};

        macro_rules! from_bytes_float_convert {
            ($bytes:expr, $elem_size:expr, $float_ty:ty) => {{
                if $bytes.len() % $elem_size != 0 {
                    return Err(HoduError::InternalError(format!(
                        "Invalid byte length for dtype: expected multiple of {}, got {}",
                        $elem_size,
                        $bytes.len()
                    )));
                }
                let mut data = Vec::with_capacity($bytes.len() / $elem_size);
                for chunk in $bytes.chunks_exact($elem_size) {
                    let f = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    data.push(<$float_ty>::from_f32(f));
                }
                data
            }};
        }

        macro_rules! from_bytes_direct {
            ($bytes:expr, $elem_size:expr, $ty:ty) => {{
                if $bytes.len() % $elem_size != 0 {
                    return Err(HoduError::InternalError(format!(
                        "Invalid byte length for dtype: expected multiple of {}, got {}",
                        $elem_size,
                        $bytes.len()
                    )));
                }
                let mut data = Vec::with_capacity($bytes.len() / $elem_size);
                for chunk in $bytes.chunks_exact($elem_size) {
                    let mut arr = [0u8; 16];
                    arr[..$elem_size].copy_from_slice(chunk);
                    data.push(<$ty>::from_le_bytes(arr[..$elem_size].try_into().unwrap()));
                }
                data
            }};
        }

        Ok(match dtype {
            DType::BOOL => Self::BOOL(bytes.iter().map(|&b| b != 0).collect()),
            DType::F8E4M3 => Self::F8E4M3(from_bytes_float_convert!(bytes, 4, F8E4M3)),
            #[cfg(feature = "f8e5m2")]
            DType::F8E5M2 => Self::F8E5M2(from_bytes_float_convert!(bytes, 4, F8E5M2)),
            DType::BF16 => Self::BF16(from_bytes_float_convert!(bytes, 4, bf16)),
            DType::F16 => Self::F16(from_bytes_float_convert!(bytes, 4, f16)),
            DType::F32 => Self::F32(from_bytes_direct!(bytes, 4, f32)),
            #[cfg(feature = "f64")]
            DType::F64 => Self::F64(from_bytes_direct!(bytes, 8, f64)),
            DType::U8 => Self::U8(bytes.to_vec()),
            #[cfg(feature = "u16")]
            DType::U16 => Self::U16(from_bytes_direct!(bytes, 2, u16)),
            DType::U32 => Self::U32(from_bytes_direct!(bytes, 4, u32)),
            #[cfg(feature = "u64")]
            DType::U64 => Self::U64(from_bytes_direct!(bytes, 8, u64)),
            DType::I8 => Self::I8(bytes.iter().map(|&n| n as i8).collect()),
            #[cfg(feature = "i16")]
            DType::I16 => Self::I16(from_bytes_direct!(bytes, 2, i16)),
            DType::I32 => Self::I32(from_bytes_direct!(bytes, 4, i32)),
            #[cfg(feature = "i64")]
            DType::I64 => Self::I64(from_bytes_direct!(bytes, 8, i64)),
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        macro_rules! to_bytes_float_convert {
            ($data:expr, $elem_size:expr) => {{
                let mut bytes = Vec::with_capacity($data.len() * $elem_size);
                for &f in $data {
                    bytes.extend_from_slice(&f32::from(f).to_le_bytes());
                }
                bytes
            }};
        }

        macro_rules! to_bytes_direct {
            ($data:expr, $elem_size:expr) => {{
                let mut bytes = Vec::with_capacity($data.len() * $elem_size);
                for &n in $data {
                    bytes.extend_from_slice(&n.to_le_bytes());
                }
                bytes
            }};
        }

        match self {
            Self::BOOL(data) => data.iter().map(|&b| if b { 1u8 } else { 0u8 }).collect(),
            Self::F8E4M3(data) => to_bytes_float_convert!(data, 4),
            #[cfg(feature = "f8e5m2")]
            Self::F8E5M2(data) => to_bytes_float_convert!(data, 4),
            Self::BF16(data) => to_bytes_float_convert!(data, 4),
            Self::F16(data) => to_bytes_float_convert!(data, 4),
            Self::F32(data) => to_bytes_direct!(data, 4),
            #[cfg(feature = "f64")]
            Self::F64(data) => to_bytes_direct!(data, 8),
            Self::U8(data) => data.clone(),
            #[cfg(feature = "u16")]
            Self::U16(data) => to_bytes_direct!(data, 2),
            Self::U32(data) => to_bytes_direct!(data, 4),
            #[cfg(feature = "u64")]
            Self::U64(data) => to_bytes_direct!(data, 8),
            Self::I8(data) => data.iter().map(|&n| n as u8).collect(),
            #[cfg(feature = "i16")]
            Self::I16(data) => to_bytes_direct!(data, 2),
            Self::I32(data) => to_bytes_direct!(data, 4),
            #[cfg(feature = "i64")]
            Self::I64(data) => to_bytes_direct!(data, 8),
        }
    }
}

impl BackendStorageT for CpuStorage {
    type BackendDevice = CpuDevice;

    fn dtype(&self) -> DType {
        match self {
            Self::BOOL(_) => DType::BOOL,
            Self::F8E4M3(_) => DType::F8E4M3,
            #[cfg(feature = "f8e5m2")]
            Self::F8E5M2(_) => DType::F8E5M2,
            Self::BF16(_) => DType::BF16,
            Self::F16(_) => DType::F16,
            Self::F32(_) => DType::F32,
            #[cfg(feature = "f64")]
            Self::F64(_) => DType::F64,
            Self::U8(_) => DType::U8,
            #[cfg(feature = "u16")]
            Self::U16(_) => DType::U16,
            Self::U32(_) => DType::U32,
            #[cfg(feature = "u64")]
            Self::U64(_) => DType::U64,
            Self::I8(_) => DType::I8,
            #[cfg(feature = "i16")]
            Self::I16(_) => DType::I16,
            Self::I32(_) => DType::I32,
            #[cfg(feature = "i64")]
            Self::I64(_) => DType::I64,
        }
    }

    fn device(&self) -> Device {
        Device::CPU
    }

    fn backend_device(&self) -> &CpuDevice {
        &CpuDevice
    }

    fn to_cpu_storage(&self) -> HoduResult<CpuStorage> {
        Ok(self.clone())
    }

    fn const_set(&mut self, scalar: Scalar, layout: &Layout) -> HoduResult<()> {
        let shape = layout.shape();
        let strides = layout.strides();
        let offset = layout.offset();
        let num_els = layout.size();
        let expected_dtype = self.dtype();
        let got_dtype = scalar.dtype();

        // Build metadata: [num_els, num_dims, shape..., strides..., offset]
        let mut metadata = Vec::with_capacity(2 + shape.ndim() * 2 + 1);
        metadata.push(num_els);
        metadata.push(shape.ndim());
        for i in 0..shape.ndim() {
            metadata.push(shape[i]);
        }
        for &stride in strides.iter().take(shape.ndim()) {
            metadata.push(stride);
        }
        metadata.push(offset);

        macro_rules! call_kernel {
            ($data:expr, $kernel_variant:ident, $val:expr) => {{
                let out_ptr = $data.as_mut_ptr() as *mut c_void;
                hodu_cpu_kernels::call_ops_const_set(
                    hodu_cpu_kernels::const_set::$kernel_variant,
                    out_ptr,
                    &metadata,
                    $val,
                )?;
            }};
        }

        match (self, scalar) {
            (Self::BOOL(data), Scalar::BOOL(v)) => call_kernel!(data, BOOL, v as u8),
            (Self::F8E4M3(data), Scalar::F8E4M3(v)) => call_kernel!(data, F8E4M3, v),
            #[cfg(feature = "f8e5m2")]
            (Self::F8E5M2(data), Scalar::F8E5M2(v)) => call_kernel!(data, F8E5M2, v),
            (Self::BF16(data), Scalar::BF16(v)) => call_kernel!(data, BF16, v),
            (Self::F16(data), Scalar::F16(v)) => call_kernel!(data, F16, v),
            (Self::F32(data), Scalar::F32(v)) => call_kernel!(data, F32, v),
            #[cfg(feature = "f64")]
            (Self::F64(data), Scalar::F64(v)) => call_kernel!(data, F64, v),
            (Self::U8(data), Scalar::U8(v)) => call_kernel!(data, U8, v),
            #[cfg(feature = "u16")]
            (Self::U16(data), Scalar::U16(v)) => call_kernel!(data, U16, v),
            (Self::U32(data), Scalar::U32(v)) => call_kernel!(data, U32, v),
            #[cfg(feature = "u64")]
            (Self::U64(data), Scalar::U64(v)) => call_kernel!(data, U64, v),
            (Self::I8(data), Scalar::I8(v)) => call_kernel!(data, I8, v),
            #[cfg(feature = "i16")]
            (Self::I16(data), Scalar::I16(v)) => call_kernel!(data, I16, v),
            (Self::I32(data), Scalar::I32(v)) => call_kernel!(data, I32, v),
            #[cfg(feature = "i64")]
            (Self::I64(data), Scalar::I64(v)) => call_kernel!(data, I64, v),
            _ => {
                return Err(HoduError::DTypeMismatch {
                    expected: expected_dtype,
                    got: got_dtype,
                })
            },
        }

        Ok(())
    }

    fn call_ops_binary(
        &self,
        rhs_storage: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
        op: Op,
    ) -> HoduResult<Self> {
        ops_binary::call_ops_binary(self, rhs_storage, lhs_layout, rhs_layout, op)
    }

    fn call_ops_binary_logical(
        &self,
        rhs_storage: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
        op: Op,
    ) -> HoduResult<Self> {
        ops_binary::call_ops_binary_logical(self, rhs_storage, lhs_layout, rhs_layout, op)
    }

    fn call_ops_bitwise_binary(
        &self,
        rhs_storage: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
        op: Op,
    ) -> HoduResult<Self> {
        ops_bitwise::call_ops_bitwise_binary(self, rhs_storage, lhs_layout, rhs_layout, op)
    }

    fn call_ops_bitwise_unary(&self, layout: &Layout, op: Op) -> HoduResult<Self> {
        ops_bitwise::call_ops_bitwise_unary(self, layout, op)
    }

    fn call_ops_bitwise_unary_scalar(&self, layout: &Layout, shift: u32, op: Op) -> HoduResult<Self> {
        ops_bitwise::call_ops_bitwise_unary_scalar(self, layout, shift, op)
    }

    fn call_ops_cmp(&self, rhs_storage: &Self, lhs_layout: &Layout, rhs_layout: &Layout, op: Op) -> HoduResult<Self> {
        ops_binary::call_ops_cmp(self, rhs_storage, lhs_layout, rhs_layout, op)
    }

    fn call_ops_cmp_scalar(&self, layout: &Layout, scalar: Scalar, op: Op) -> HoduResult<Self> {
        ops_unary::call_ops_cmp_scalar(self, layout, scalar, op)
    }

    fn call_ops_unary(&self, layout: &Layout, op: Op) -> HoduResult<Self> {
        ops_unary::call_ops_unary(self, layout, op)
    }

    fn call_ops_unary_logical(&self, layout: &Layout, op: Op) -> HoduResult<Self> {
        ops_unary::call_ops_unary_logical(self, layout, op)
    }

    fn call_ops_unary_scalar(&self, layout: &Layout, scalar: Scalar, op: Op) -> HoduResult<Self> {
        ops_unary::call_ops_unary_scalar(self, layout, scalar, op)
    }

    fn call_ops_matmul(
        &self,
        rhs_storage: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
        op: Op,
    ) -> HoduResult<Self> {
        ops_matrix::call_ops_matmul(self, rhs_storage, lhs_layout, rhs_layout, op)
    }

    fn call_ops_dot(&self, rhs_storage: &Self, lhs_layout: &Layout, rhs_layout: &Layout, op: Op) -> HoduResult<Self> {
        ops_matrix::call_ops_dot(self, rhs_storage, lhs_layout, rhs_layout, op)
    }

    fn call_ops_det(&self, layout: &Layout) -> HoduResult<Self> {
        ops_linalg::call_ops_det(self, layout)
    }

    fn call_ops_inv(&self, layout: &Layout) -> HoduResult<Self> {
        ops_linalg::call_ops_inv(self, layout)
    }

    fn call_ops_trace(&self, layout: &Layout) -> HoduResult<Self> {
        ops_linalg::call_ops_trace(self, layout)
    }

    fn call_ops_reduce(&self, layout: &Layout, dims: &[usize], keep_dim: bool, op: Op) -> HoduResult<Self> {
        ops_reduce::call_ops_reduce(self, layout, dims, keep_dim, op)
    }

    fn call_ops_concat(&self, others: &[&Self], layouts: &[&Layout], dim: usize, op: Op) -> HoduResult<Self> {
        ops_concat_split::call_ops_concat(self, others, layouts, dim, op)
    }

    fn call_ops_split(&self, layout: &Layout, dim: usize, start: usize, size: usize, op: Op) -> HoduResult<Self> {
        ops_concat_split::call_ops_split(self, layout, dim, start, size, op)
    }

    fn call_ops_index_select(
        &self,
        layout: &Layout,
        indices_storage: &Self,
        indices_layout: &Layout,
        dim: usize,
        op: Op,
    ) -> HoduResult<Self> {
        ops_indexing::call_ops_index_select(self, layout, indices_storage, indices_layout, dim, op)
    }

    fn call_ops_index_put(
        &self,
        layout: &Layout,
        indices_storage: &Self,
        indices_layout: &Layout,
        values_storage: &Self,
        values_layout: &Layout,
        dim: usize,
        op: Op,
    ) -> HoduResult<Self> {
        ops_indexing::call_ops_index_put(
            self,
            layout,
            indices_storage,
            indices_layout,
            values_storage,
            values_layout,
            dim,
            op,
        )
    }

    fn call_ops_gather(
        &self,
        layout: &Layout,
        indices_storage: &Self,
        indices_layout: &Layout,
        dim: usize,
        op: Op,
    ) -> HoduResult<Self> {
        ops_indexing::call_ops_gather(self, layout, indices_storage, indices_layout, dim, op)
    }

    fn call_ops_scatter(
        &self,
        layout: &Layout,
        indices_storage: &Self,
        indices_layout: &Layout,
        src_storage: &Self,
        src_layout: &Layout,
        dim: usize,
        op: Op,
    ) -> HoduResult<Self> {
        ops_indexing::call_ops_scatter(
            self,
            layout,
            indices_storage,
            indices_layout,
            src_storage,
            src_layout,
            dim,
            op,
        )
    }

    fn call_ops_onehot(
        &self,
        layout: &Layout,
        num_classes: usize,
        axis: usize,
        output_dtype: crate::types::DType,
        op: Op,
    ) -> HoduResult<Self> {
        ops_indexing::call_ops_onehot(self, layout, num_classes, axis, output_dtype, op)
    }

    fn call_ops_conv(
        &self,
        layout: &Layout,
        weight_storage: &Self,
        weight_layout: &Layout,
        stride: &[usize],
        padding: &[usize],
        dilation: &[usize],
        op: Op,
    ) -> HoduResult<Self> {
        ops_conv::call_ops_conv(
            self,
            layout,
            weight_storage,
            weight_layout,
            stride,
            padding,
            dilation,
            op,
        )
    }

    fn call_ops_conv_grad_weight(
        &self,
        layout: &Layout,
        grad_output_storage: &Self,
        grad_output_layout: &Layout,
        weight_shape: &crate::types::Shape,
        stride: &[usize],
        padding: &[usize],
        dilation: &[usize],
        op: Op,
    ) -> HoduResult<Self> {
        ops_conv::call_ops_conv_grad_weight(
            self,
            layout,
            grad_output_storage,
            grad_output_layout,
            weight_shape,
            stride,
            padding,
            dilation,
            op,
        )
    }

    fn call_ops_reduce_window(
        &self,
        layout: &Layout,
        window_shape: &[usize],
        strides: &[usize],
        padding: &[usize],
        op: Op,
    ) -> HoduResult<Self> {
        ops_windowing::call_ops_reduce_window(self, layout, window_shape, strides, padding, op)
    }

    fn call_ops_pad(
        &self,
        layout: &Layout,
        pad_before: &[usize],
        pad_after: &[usize],
        pad_value: Scalar,
        op: Op,
    ) -> HoduResult<Self> {
        ops_padding::call_ops_pad(self, layout, pad_before, pad_after, pad_value, op)
    }

    fn call_ops_resize(
        &self,
        layout: &Layout,
        output_shape: &[usize],
        mode: crate::op_params::ResizeMode,
        coord_transform: crate::op_params::ResizeCoordTransform,
        nearest_mode: crate::op_params::ResizeNearestMode,
    ) -> HoduResult<Self> {
        ops_resize::call_ops_resize(self, layout, output_shape, mode, coord_transform, nearest_mode)
    }

    fn call_ops_cumsum(&self, layout: &Layout, dim: usize) -> HoduResult<Self> {
        ops_scan::call_ops_cumsum(self, layout, dim)
    }

    fn call_ops_cumprod(&self, layout: &Layout, dim: usize) -> HoduResult<Self> {
        ops_scan::call_ops_cumprod(self, layout, dim)
    }

    fn call_ops_einsum(
        &self,
        inputs: &[&Self],
        input_layouts: &[&Layout],
        parsed: &crate::einsum::ParsedEinsum,
    ) -> HoduResult<Self> {
        ops_einsum::call_ops_einsum(self, inputs, input_layouts, parsed)
    }

    fn call_ops_flip(&self, layout: &Layout, dims: &[usize]) -> HoduResult<Self> {
        ops_shape_memory::call_ops_flip(self, layout, dims)
    }

    fn call_topk(
        &self,
        layout: &Layout,
        k: usize,
        last_dim_size: usize,
        outer_size: usize,
        largest: bool,
        sorted: bool,
    ) -> HoduResult<(Self, Self)> {
        ops_sort::call_topk(self, layout, k, last_dim_size, outer_size, largest, sorted)
    }

    fn call_nonzero(&self, layout: &Layout) -> HoduResult<(Self, usize)> {
        ops_indexing::call_nonzero(self, layout)
    }

    fn call_unique(&self, layout: &Layout) -> HoduResult<(Self, Self, Self, usize)> {
        ops_indexing::call_unique(self, layout)
    }

    fn call_compress(
        &self,
        layout: &Layout,
        condition: &Self,
        condition_layout: &Layout,
        axis: Option<usize>,
    ) -> HoduResult<(Self, usize)> {
        ops_indexing::call_compress(self, layout, condition, condition_layout, axis)
    }

    fn to_dtype(&self, layout: &Layout, target_dtype: DType) -> HoduResult<Self> {
        if self.dtype() == target_dtype {
            return self.contiguous(layout);
        }

        let metadata = op_metadatas::cast_metadata(layout);
        let num_els = layout.size();

        // Create output storage with target dtype
        let mut output = CpuDevice::allocate(num_els, target_dtype)?;

        // Build kernel name: cast_<src>_to_<dst>
        let kernel_name = format!("hodu_cpu_cast_{}_to_{}", self.dtype(), target_dtype);
        let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
        let kernel = hodu_cpu_kernels::CastKernel(kernel_name_static);

        let in_ptr = self.as_ptr() as *const c_void;
        let out_ptr = output.as_mut_ptr() as *mut c_void;

        hodu_cpu_kernels::call_ops_cast(kernel, in_ptr, out_ptr, &metadata)?;

        Ok(output)
    }

    fn contiguous(&self, layout: &Layout) -> HoduResult<Self> {
        // If already contiguous with zero offset, just clone
        if layout.is_contiguous() && layout.offset() == 0 {
            return Ok(self.clone());
        }

        let metadata = op_metadatas::contiguous_metadata(layout);
        let num_els = layout.size();

        // Create output storage
        let mut output = CpuDevice::allocate(num_els, self.dtype())?;

        macro_rules! call_kernel {
            ($in_data:expr, $out_data:expr, $kernel_variant:ident) => {{
                let in_ptr = $in_data.as_ptr() as *const c_void;
                let out_ptr = $out_data.as_mut_ptr() as *mut c_void;
                hodu_cpu_kernels::call_ops_contiguous(
                    hodu_cpu_kernels::contiguous::$kernel_variant,
                    in_ptr,
                    out_ptr,
                    &metadata,
                )?;
            }};
        }

        match (self, &mut output) {
            (Self::BOOL(inp), Self::BOOL(out)) => call_kernel!(inp, out, BOOL),
            (Self::F8E4M3(inp), Self::F8E4M3(out)) => call_kernel!(inp, out, F8E4M3),
            #[cfg(feature = "f8e5m2")]
            (Self::F8E5M2(inp), Self::F8E5M2(out)) => call_kernel!(inp, out, F8E5M2),
            (Self::BF16(inp), Self::BF16(out)) => call_kernel!(inp, out, BF16),
            (Self::F16(inp), Self::F16(out)) => call_kernel!(inp, out, F16),
            (Self::F32(inp), Self::F32(out)) => call_kernel!(inp, out, F32),
            #[cfg(feature = "f64")]
            (Self::F64(inp), Self::F64(out)) => call_kernel!(inp, out, F64),
            (Self::U8(inp), Self::U8(out)) => call_kernel!(inp, out, U8),
            #[cfg(feature = "u16")]
            (Self::U16(inp), Self::U16(out)) => call_kernel!(inp, out, U16),
            (Self::U32(inp), Self::U32(out)) => call_kernel!(inp, out, U32),
            #[cfg(feature = "u64")]
            (Self::U64(inp), Self::U64(out)) => call_kernel!(inp, out, U64),
            (Self::I8(inp), Self::I8(out)) => call_kernel!(inp, out, I8),
            #[cfg(feature = "i16")]
            (Self::I16(inp), Self::I16(out)) => call_kernel!(inp, out, I16),
            (Self::I32(inp), Self::I32(out)) => call_kernel!(inp, out, I32),
            #[cfg(feature = "i64")]
            (Self::I64(inp), Self::I64(out)) => call_kernel!(inp, out, I64),
            _ => unreachable!(),
        }

        Ok(output)
    }
}
