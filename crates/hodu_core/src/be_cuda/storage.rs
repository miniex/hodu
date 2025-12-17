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
    be::storage::BackendStorageT,
    be_cpu::storage::CpuStorage,
    be_cuda::device::CudaDevice,
    error::{HoduError, HoduResult},
    op_metadatas,
    ops::Op,
    scalar::Scalar,
    types::{DType, Device, Layout, Shape},
};
use float8::F8E4M3;
#[cfg(feature = "f8e5m2")]
use float8::F8E5M2;
use half::{bf16, f16};
use hodu_cuda_kernels::cuda::{CudaSlice, DevicePtr, DeviceRepr};

#[derive(Debug, Clone)]
pub(crate) enum CudaStorageData {
    BOOL(CudaSlice<bool>),
    F8E4M3(CudaSlice<F8E4M3>),
    #[cfg(feature = "f8e5m2")]
    F8E5M2(CudaSlice<F8E5M2>),
    BF16(CudaSlice<bf16>),
    F16(CudaSlice<f16>),
    F32(CudaSlice<f32>),
    #[cfg(feature = "f64")]
    F64(CudaSlice<f64>),
    U8(CudaSlice<u8>),
    #[cfg(feature = "u16")]
    U16(CudaSlice<u16>),
    U32(CudaSlice<u32>),
    #[cfg(feature = "u64")]
    U64(CudaSlice<u64>),
    I8(CudaSlice<i8>),
    #[cfg(feature = "i16")]
    I16(CudaSlice<i16>),
    I32(CudaSlice<i32>),
    #[cfg(feature = "i64")]
    I64(CudaSlice<i64>),
}

#[derive(Debug, Clone)]
pub struct CudaStorage {
    device_id: usize,
    device: Arc<CudaDevice>,
    data: CudaStorageData,
}

impl CudaStorage {
    pub(crate) fn new(device_id: usize, device: Arc<CudaDevice>, data: CudaStorageData) -> Self {
        Self {
            device_id,
            device,
            data,
        }
    }

    pub fn device_id(&self) -> usize {
        self.device_id
    }

    fn get_device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Get device pointer to the underlying CUDA data
    pub fn as_ptr(&self) -> *const u8 {
        match &self.data {
            CudaStorageData::BOOL(s) => s.device_ptr().as_ptr() as *const u8,
            CudaStorageData::F8E4M3(s) => s.device_ptr().as_ptr() as *const u8,
            #[cfg(feature = "f8e5m2")]
            CudaStorageData::F8E5M2(s) => s.device_ptr().as_ptr() as *const u8,
            CudaStorageData::BF16(s) => s.device_ptr().as_ptr() as *const u8,
            CudaStorageData::F16(s) => s.device_ptr().as_ptr() as *const u8,
            CudaStorageData::F32(s) => s.device_ptr().as_ptr() as *const u8,
            #[cfg(feature = "f64")]
            CudaStorageData::F64(s) => s.device_ptr().as_ptr() as *const u8,
            CudaStorageData::U8(s) => s.device_ptr().as_ptr() as *const u8,
            #[cfg(feature = "u16")]
            CudaStorageData::U16(s) => s.device_ptr().as_ptr() as *const u8,
            CudaStorageData::U32(s) => s.device_ptr().as_ptr() as *const u8,
            #[cfg(feature = "u64")]
            CudaStorageData::U64(s) => s.device_ptr().as_ptr() as *const u8,
            CudaStorageData::I8(s) => s.device_ptr().as_ptr() as *const u8,
            #[cfg(feature = "i16")]
            CudaStorageData::I16(s) => s.device_ptr().as_ptr() as *const u8,
            CudaStorageData::I32(s) => s.device_ptr().as_ptr() as *const u8,
            #[cfg(feature = "i64")]
            CudaStorageData::I64(s) => s.device_ptr().as_ptr() as *const u8,
        }
    }

    /// Get mutable device pointer to the underlying CUDA data
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        match &mut self.data {
            CudaStorageData::BOOL(s) => s.device_ptr().as_ptr() as *mut u8,
            CudaStorageData::F8E4M3(s) => s.device_ptr().as_ptr() as *mut u8,
            #[cfg(feature = "f8e5m2")]
            CudaStorageData::F8E5M2(s) => s.device_ptr().as_ptr() as *mut u8,
            CudaStorageData::BF16(s) => s.device_ptr().as_ptr() as *mut u8,
            CudaStorageData::F16(s) => s.device_ptr().as_ptr() as *mut u8,
            CudaStorageData::F32(s) => s.device_ptr().as_ptr() as *mut u8,
            #[cfg(feature = "f64")]
            CudaStorageData::F64(s) => s.device_ptr().as_ptr() as *mut u8,
            CudaStorageData::U8(s) => s.device_ptr().as_ptr() as *mut u8,
            #[cfg(feature = "u16")]
            CudaStorageData::U16(s) => s.device_ptr().as_ptr() as *mut u8,
            CudaStorageData::U32(s) => s.device_ptr().as_ptr() as *mut u8,
            #[cfg(feature = "u64")]
            CudaStorageData::U64(s) => s.device_ptr().as_ptr() as *mut u8,
            CudaStorageData::I8(s) => s.device_ptr().as_ptr() as *mut u8,
            #[cfg(feature = "i16")]
            CudaStorageData::I16(s) => s.device_ptr().as_ptr() as *mut u8,
            CudaStorageData::I32(s) => s.device_ptr().as_ptr() as *mut u8,
            #[cfg(feature = "i64")]
            CudaStorageData::I64(s) => s.device_ptr().as_ptr() as *mut u8,
        }
    }

    pub fn len(&self) -> usize {
        match &self.data {
            CudaStorageData::BOOL(s) => s.len(),
            CudaStorageData::F8E4M3(s) => s.len(),
            #[cfg(feature = "f8e5m2")]
            CudaStorageData::F8E5M2(s) => s.len(),
            CudaStorageData::BF16(s) => s.len(),
            CudaStorageData::F16(s) => s.len(),
            CudaStorageData::F32(s) => s.len(),
            #[cfg(feature = "f64")]
            CudaStorageData::F64(s) => s.len(),
            CudaStorageData::U8(s) => s.len(),
            #[cfg(feature = "u16")]
            CudaStorageData::U16(s) => s.len(),
            CudaStorageData::U32(s) => s.len(),
            #[cfg(feature = "u64")]
            CudaStorageData::U64(s) => s.len(),
            CudaStorageData::I8(s) => s.len(),
            #[cfg(feature = "i16")]
            CudaStorageData::I16(s) => s.len(),
            CudaStorageData::I32(s) => s.len(),
            #[cfg(feature = "i64")]
            CudaStorageData::I64(s) => s.len(),
        }
    }

    pub fn to_cpu<T: DeviceRepr + Clone>(&self, slice: &CudaSlice<T>) -> HoduResult<Vec<T>> {
        let device = self.get_device();
        let stream = device.context().default_stream();
        let mut result = vec![unsafe { core::mem::zeroed() }; slice.len()];
        stream
            .memcpy_dtoh(slice, &mut result)
            .map_err(|e| HoduError::BackendError(format!("CUDA memcpy_dtoh failed: {:?}", e)))?;
        Ok(result)
    }

    pub fn from_cpu_storage(cpu_storage: &CpuStorage, device_id: usize) -> HoduResult<Self> {
        let device = CudaDevice::get(device_id)?;
        let data = Self::cpu_to_cuda_storage_data(&device, cpu_storage)?;
        Ok(Self::new(device_id, device, data))
    }

    fn cpu_to_cuda_storage_data(device: &CudaDevice, cpu_storage: &CpuStorage) -> HoduResult<CudaStorageData> {
        match cpu_storage {
            CpuStorage::BOOL(data) => Ok(CudaStorageData::BOOL(device.new_buffer_with_data(data)?)),
            CpuStorage::F8E4M3(data) => Ok(CudaStorageData::F8E4M3(device.new_buffer_with_data(data)?)),
            #[cfg(feature = "f8e5m2")]
            CpuStorage::F8E5M2(data) => Ok(CudaStorageData::F8E5M2(device.new_buffer_with_data(data)?)),
            CpuStorage::BF16(data) => Ok(CudaStorageData::BF16(device.new_buffer_with_data(data)?)),
            CpuStorage::F16(data) => Ok(CudaStorageData::F16(device.new_buffer_with_data(data)?)),
            CpuStorage::F32(data) => Ok(CudaStorageData::F32(device.new_buffer_with_data(data)?)),
            #[cfg(feature = "f64")]
            CpuStorage::F64(data) => Ok(CudaStorageData::F64(device.new_buffer_with_data(data)?)),
            CpuStorage::U8(data) => Ok(CudaStorageData::U8(device.new_buffer_with_data(data)?)),
            #[cfg(feature = "u16")]
            CpuStorage::U16(data) => Ok(CudaStorageData::U16(device.new_buffer_with_data(data)?)),
            CpuStorage::U32(data) => Ok(CudaStorageData::U32(device.new_buffer_with_data(data)?)),
            #[cfg(feature = "u64")]
            CpuStorage::U64(data) => Ok(CudaStorageData::U64(device.new_buffer_with_data(data)?)),
            CpuStorage::I8(data) => Ok(CudaStorageData::I8(device.new_buffer_with_data(data)?)),
            #[cfg(feature = "i16")]
            CpuStorage::I16(data) => Ok(CudaStorageData::I16(device.new_buffer_with_data(data)?)),
            CpuStorage::I32(data) => Ok(CudaStorageData::I32(device.new_buffer_with_data(data)?)),
            #[cfg(feature = "i64")]
            CpuStorage::I64(data) => Ok(CudaStorageData::I64(device.new_buffer_with_data(data)?)),
        }
    }
}

impl BackendStorageT for CudaStorage {
    type BackendDevice = CudaDevice;

    fn dtype(&self) -> DType {
        match &self.data {
            CudaStorageData::BOOL(_) => DType::BOOL,
            CudaStorageData::F8E4M3(_) => DType::F8E4M3,
            #[cfg(feature = "f8e5m2")]
            CudaStorageData::F8E5M2(_) => DType::F8E5M2,
            CudaStorageData::BF16(_) => DType::BF16,
            CudaStorageData::F16(_) => DType::F16,
            CudaStorageData::F32(_) => DType::F32,
            #[cfg(feature = "f64")]
            CudaStorageData::F64(_) => DType::F64,
            CudaStorageData::U8(_) => DType::U8,
            #[cfg(feature = "u16")]
            CudaStorageData::U16(_) => DType::U16,
            CudaStorageData::U32(_) => DType::U32,
            #[cfg(feature = "u64")]
            CudaStorageData::U64(_) => DType::U64,
            CudaStorageData::I8(_) => DType::I8,
            #[cfg(feature = "i16")]
            CudaStorageData::I16(_) => DType::I16,
            CudaStorageData::I32(_) => DType::I32,
            #[cfg(feature = "i64")]
            CudaStorageData::I64(_) => DType::I64,
        }
    }

    fn device(&self) -> Device {
        Device::CUDA(self.device_id)
    }

    fn backend_device(&self) -> &CudaDevice {
        &self.device
    }

    fn to_cpu_storage(&self) -> HoduResult<CpuStorage> {
        match &self.data {
            CudaStorageData::BOOL(s) => Ok(CpuStorage::BOOL(self.to_cpu(s)?)),
            CudaStorageData::F8E4M3(s) => Ok(CpuStorage::F8E4M3(self.to_cpu(s)?)),
            #[cfg(feature = "f8e5m2")]
            CudaStorageData::F8E5M2(s) => Ok(CpuStorage::F8E5M2(self.to_cpu(s)?)),
            CudaStorageData::BF16(s) => Ok(CpuStorage::BF16(self.to_cpu(s)?)),
            CudaStorageData::F16(s) => Ok(CpuStorage::F16(self.to_cpu(s)?)),
            CudaStorageData::F32(s) => Ok(CpuStorage::F32(self.to_cpu(s)?)),
            #[cfg(feature = "f64")]
            CudaStorageData::F64(s) => Ok(CpuStorage::F64(self.to_cpu(s)?)),
            CudaStorageData::U8(s) => Ok(CpuStorage::U8(self.to_cpu(s)?)),
            #[cfg(feature = "u16")]
            CudaStorageData::U16(s) => Ok(CpuStorage::U16(self.to_cpu(s)?)),
            CudaStorageData::U32(s) => Ok(CpuStorage::U32(self.to_cpu(s)?)),
            #[cfg(feature = "u64")]
            CudaStorageData::U64(s) => Ok(CpuStorage::U64(self.to_cpu(s)?)),
            CudaStorageData::I8(s) => Ok(CpuStorage::I8(self.to_cpu(s)?)),
            #[cfg(feature = "i16")]
            CudaStorageData::I16(s) => Ok(CpuStorage::I16(self.to_cpu(s)?)),
            CudaStorageData::I32(s) => Ok(CpuStorage::I32(self.to_cpu(s)?)),
            #[cfg(feature = "i64")]
            CudaStorageData::I64(s) => Ok(CpuStorage::I64(self.to_cpu(s)?)),
        }
    }

    fn const_set(&mut self, scalar: Scalar, layout: &Layout) -> HoduResult<()> {
        let shape = layout.shape();
        let strides = layout.strides();
        let offset = layout.offset();
        let num_els = layout.size();

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

        let dtype = self.dtype();

        // Get references before mutable borrow to avoid borrow checker issues
        let kernels = &self.device.kernels;
        let context = &self.device.context;

        let kernel_name = format!("hodu_cuda_const_set_{}", dtype);
        let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
        let kernel = hodu_cuda_kernels::kernels::Kernel(kernel_name_static);

        macro_rules! call_const_set {
            ($slice:expr, $val:expr, $ty:ty) => {{
                hodu_cuda_kernels::kernels::call_const_set(kernel, kernels, context, $slice, &metadata, $val)?
            }};
        }

        match (&mut self.data, scalar) {
            (CudaStorageData::BOOL(slice), Scalar::BOOL(v)) => call_const_set!(slice, v, bool),
            (CudaStorageData::F8E4M3(slice), Scalar::F8E4M3(v)) => call_const_set!(slice, v, float8::F8E4M3),
            #[cfg(feature = "f8e5m2")]
            (CudaStorageData::F8E5M2(slice), Scalar::F8E5M2(v)) => call_const_set!(slice, v, float8::F8E5M2),
            (CudaStorageData::BF16(slice), Scalar::BF16(v)) => call_const_set!(slice, v, half::bf16),
            (CudaStorageData::F16(slice), Scalar::F16(v)) => call_const_set!(slice, v, half::f16),
            (CudaStorageData::F32(slice), Scalar::F32(v)) => call_const_set!(slice, v, f32),
            #[cfg(feature = "f64")]
            (CudaStorageData::F64(slice), Scalar::F64(v)) => call_const_set!(slice, v, f64),
            (CudaStorageData::U8(slice), Scalar::U8(v)) => call_const_set!(slice, v, u8),
            #[cfg(feature = "u16")]
            (CudaStorageData::U16(slice), Scalar::U16(v)) => call_const_set!(slice, v, u16),
            (CudaStorageData::U32(slice), Scalar::U32(v)) => call_const_set!(slice, v, u32),
            #[cfg(feature = "u64")]
            (CudaStorageData::U64(slice), Scalar::U64(v)) => call_const_set!(slice, v, u64),
            (CudaStorageData::I8(slice), Scalar::I8(v)) => call_const_set!(slice, v, i8),
            #[cfg(feature = "i16")]
            (CudaStorageData::I16(slice), Scalar::I16(v)) => call_const_set!(slice, v, i16),
            (CudaStorageData::I32(slice), Scalar::I32(v)) => call_const_set!(slice, v, i32),
            #[cfg(feature = "i64")]
            (CudaStorageData::I64(slice), Scalar::I64(v)) => call_const_set!(slice, v, i64),
            _ => {
                return Err(HoduError::DTypeMismatch {
                    expected: dtype,
                    got: scalar.dtype(),
                })
            },
        }

        Ok(())
    }

    fn call_ops_binary(&self, rhs: &Self, lhs_layout: &Layout, rhs_layout: &Layout, op: Op) -> HoduResult<Self> {
        ops_binary::call_ops_binary(self, rhs, lhs_layout, rhs_layout, op)
    }

    fn call_ops_binary_logical(
        &self,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
        op: Op,
    ) -> HoduResult<Self> {
        ops_binary::call_ops_binary_logical(self, rhs, lhs_layout, rhs_layout, op)
    }

    fn call_ops_bitwise_binary(
        &self,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
        op: Op,
    ) -> HoduResult<Self> {
        ops_bitwise::call_ops_bitwise_binary(self, rhs, lhs_layout, rhs_layout, op)
    }

    fn call_ops_bitwise_unary(&self, layout: &Layout, op: Op) -> HoduResult<Self> {
        ops_bitwise::call_ops_bitwise_unary(self, layout, op)
    }

    fn call_ops_bitwise_unary_scalar(&self, layout: &Layout, shift: u32, op: Op) -> HoduResult<Self> {
        ops_bitwise::call_ops_bitwise_unary_scalar(self, layout, shift, op)
    }

    fn call_ops_cmp(&self, rhs: &Self, lhs_layout: &Layout, rhs_layout: &Layout, op: Op) -> HoduResult<Self> {
        ops_binary::call_ops_cmp(self, rhs, lhs_layout, rhs_layout, op)
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

    fn call_ops_matmul(&self, rhs: &Self, lhs_layout: &Layout, rhs_layout: &Layout, op: Op) -> HoduResult<Self> {
        ops_matrix::call_ops_matmul(self, rhs, lhs_layout, rhs_layout, op)
    }

    fn call_ops_dot(&self, rhs: &Self, lhs_layout: &Layout, rhs_layout: &Layout, op: Op) -> HoduResult<Self> {
        ops_matrix::call_ops_dot(self, rhs, lhs_layout, rhs_layout, op)
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
        indices: &Self,
        indices_layout: &Layout,
        dim: usize,
        op: Op,
    ) -> HoduResult<Self> {
        ops_indexing::call_ops_index_select(self, layout, indices, indices_layout, dim, op)
    }

    fn call_ops_index_put(
        &self,
        layout: &Layout,
        indices: &Self,
        indices_layout: &Layout,
        values: &Self,
        values_layout: &Layout,
        dim: usize,
        op: Op,
    ) -> HoduResult<Self> {
        ops_indexing::call_ops_index_put(self, layout, indices, indices_layout, values, values_layout, dim, op)
    }

    fn call_ops_gather(
        &self,
        layout: &Layout,
        indices: &Self,
        indices_layout: &Layout,
        dim: usize,
        op: Op,
    ) -> HoduResult<Self> {
        ops_indexing::call_ops_gather(self, layout, indices, indices_layout, dim, op)
    }

    fn call_ops_scatter(
        &self,
        layout: &Layout,
        indices: &Self,
        indices_layout: &Layout,
        src: &Self,
        src_layout: &Layout,
        dim: usize,
        op: Op,
    ) -> HoduResult<Self> {
        ops_indexing::call_ops_scatter(self, layout, indices, indices_layout, src, src_layout, dim, op)
    }

    fn call_ops_onehot(
        &self,
        layout: &Layout,
        num_classes: usize,
        axis: usize,
        output_dtype: DType,
        op: Op,
    ) -> HoduResult<Self> {
        ops_indexing::call_ops_onehot(self, layout, num_classes, axis, output_dtype, op)
    }

    fn call_ops_conv(
        &self,
        layout: &Layout,
        weight: &Self,
        weight_layout: &Layout,
        stride: &[usize],
        padding: &[usize],
        dilation: &[usize],
        op: Op,
    ) -> HoduResult<Self> {
        ops_conv::call_ops_conv(self, layout, weight, weight_layout, stride, padding, dilation, op)
    }

    fn call_ops_conv_grad_weight(
        &self,
        layout: &Layout,
        grad_output: &Self,
        grad_output_layout: &Layout,
        weight_shape: &Shape,
        stride: &[usize],
        padding: &[usize],
        dilation: &[usize],
        op: Op,
    ) -> HoduResult<Self> {
        ops_conv::call_ops_conv_grad_weight(
            self,
            layout,
            grad_output,
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

        let src_dtype = self.dtype();
        let device = self.get_device();

        let kernel_name = format!("hodu_cuda_cast_{}_to_{}", src_dtype, target_dtype);
        let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
        let kernel = hodu_cuda_kernels::kernels::Kernel(kernel_name_static);

        macro_rules! call_cast {
            ($input:expr, $input_ty:ty, $output_ty:ty) => {{
                let mut output: CudaSlice<$output_ty> = device.new_buffer(num_els as usize)?;
                hodu_cuda_kernels::kernels::call_ops_cast(
                    kernel,
                    device.kernels(),
                    device.context(),
                    $input,
                    &mut output,
                    &metadata,
                )?;
                output
            }};
        }

        match (&self.data, target_dtype) {
            (CudaStorageData::F32(input), DType::F16) => Ok(CudaStorage::new(
                self.device_id(),
                self.device.clone(),
                CudaStorageData::F16(call_cast!(input, f32, half::f16)),
            )),
            (CudaStorageData::F32(input), DType::BF16) => Ok(CudaStorage::new(
                self.device_id(),
                self.device.clone(),
                CudaStorageData::BF16(call_cast!(input, f32, half::bf16)),
            )),
            (CudaStorageData::F16(input), DType::F32) => Ok(CudaStorage::new(
                self.device_id(),
                self.device.clone(),
                CudaStorageData::F32(call_cast!(input, half::f16, f32)),
            )),
            (CudaStorageData::BF16(input), DType::F32) => Ok(CudaStorage::new(
                self.device_id(),
                self.device.clone(),
                CudaStorageData::F32(call_cast!(input, half::bf16, f32)),
            )),
            #[cfg(feature = "f64")]
            (CudaStorageData::F32(input), DType::F64) => Ok(CudaStorage::new(
                self.device_id(),
                self.device.clone(),
                CudaStorageData::F64(call_cast!(input, f32, f64)),
            )),
            #[cfg(feature = "f64")]
            (CudaStorageData::F64(input), DType::F32) => Ok(CudaStorage::new(
                self.device_id(),
                self.device.clone(),
                CudaStorageData::F32(call_cast!(input, f64, f32)),
            )),
            (CudaStorageData::I32(input), DType::F32) => Ok(CudaStorage::new(
                self.device_id(),
                self.device.clone(),
                CudaStorageData::F32(call_cast!(input, i32, f32)),
            )),
            (CudaStorageData::F32(input), DType::I32) => Ok(CudaStorage::new(
                self.device_id(),
                self.device.clone(),
                CudaStorageData::I32(call_cast!(input, f32, i32)),
            )),
            _ => Err(HoduError::NotImplemented(format!(
                "cast from {} to {}",
                src_dtype, target_dtype
            ))),
        }
    }

    fn contiguous(&self, layout: &Layout) -> HoduResult<Self> {
        if layout.is_contiguous() {
            return Ok(self.clone());
        }

        let metadata = op_metadatas::contiguous_metadata(layout);
        let num_els = layout.size();

        let dtype = self.dtype();
        let device = self.get_device();

        let kernel_name = format!("hodu_cuda_contiguous_{}", dtype);
        let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
        let kernel = hodu_cuda_kernels::kernels::Kernel(kernel_name_static);

        macro_rules! call_contiguous {
            ($input:expr, $ty:ty) => {{
                let mut output: CudaSlice<$ty> = device.new_buffer(num_els as usize)?;
                hodu_cuda_kernels::kernels::call_ops_contiguous(
                    kernel,
                    device.kernels(),
                    device.context(),
                    $input,
                    &mut output,
                    &metadata,
                )?;
                output
            }};
        }

        match &self.data {
            CudaStorageData::BOOL(input) => Ok(CudaStorage::new(
                self.device_id(),
                self.device.clone(),
                CudaStorageData::BOOL(call_contiguous!(input, bool)),
            )),
            CudaStorageData::F8E4M3(input) => Ok(CudaStorage::new(
                self.device_id(),
                self.device.clone(),
                CudaStorageData::F8E4M3(call_contiguous!(input, float8::F8E4M3)),
            )),
            #[cfg(feature = "f8e5m2")]
            CudaStorageData::F8E5M2(input) => Ok(CudaStorage::new(
                self.device_id(),
                self.device.clone(),
                CudaStorageData::F8E5M2(call_contiguous!(input, float8::F8E5M2)),
            )),
            CudaStorageData::BF16(input) => Ok(CudaStorage::new(
                self.device_id(),
                self.device.clone(),
                CudaStorageData::BF16(call_contiguous!(input, half::bf16)),
            )),
            CudaStorageData::F16(input) => Ok(CudaStorage::new(
                self.device_id(),
                self.device.clone(),
                CudaStorageData::F16(call_contiguous!(input, half::f16)),
            )),
            CudaStorageData::F32(input) => Ok(CudaStorage::new(
                self.device_id(),
                self.device.clone(),
                CudaStorageData::F32(call_contiguous!(input, f32)),
            )),
            #[cfg(feature = "f64")]
            CudaStorageData::F64(input) => Ok(CudaStorage::new(
                self.device_id(),
                self.device.clone(),
                CudaStorageData::F64(call_contiguous!(input, f64)),
            )),
            CudaStorageData::U8(input) => Ok(CudaStorage::new(
                self.device_id(),
                self.device.clone(),
                CudaStorageData::U8(call_contiguous!(input, u8)),
            )),
            #[cfg(feature = "u16")]
            CudaStorageData::U16(input) => Ok(CudaStorage::new(
                self.device_id(),
                self.device.clone(),
                CudaStorageData::U16(call_contiguous!(input, u16)),
            )),
            CudaStorageData::U32(input) => Ok(CudaStorage::new(
                self.device_id(),
                self.device.clone(),
                CudaStorageData::U32(call_contiguous!(input, u32)),
            )),
            #[cfg(feature = "u64")]
            CudaStorageData::U64(input) => Ok(CudaStorage::new(
                self.device_id(),
                self.device.clone(),
                CudaStorageData::U64(call_contiguous!(input, u64)),
            )),
            CudaStorageData::I8(input) => Ok(CudaStorage::new(
                self.device_id(),
                self.device.clone(),
                CudaStorageData::I8(call_contiguous!(input, i8)),
            )),
            #[cfg(feature = "i16")]
            CudaStorageData::I16(input) => Ok(CudaStorage::new(
                self.device_id(),
                self.device.clone(),
                CudaStorageData::I16(call_contiguous!(input, i16)),
            )),
            CudaStorageData::I32(input) => Ok(CudaStorage::new(
                self.device_id(),
                self.device.clone(),
                CudaStorageData::I32(call_contiguous!(input, i32)),
            )),
            #[cfg(feature = "i64")]
            CudaStorageData::I64(input) => Ok(CudaStorage::new(
                self.device_id(),
                self.device.clone(),
                CudaStorageData::I64(call_contiguous!(input, i64)),
            )),
        }
    }
}
