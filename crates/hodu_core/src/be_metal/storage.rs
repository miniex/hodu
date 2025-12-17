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
    be_metal::device::MetalDevice,
    error::{HoduError, HoduResult},
    op_metadatas,
    ops::Op,
    scalar::Scalar,
    types::{DType, Device, Layout, Shape},
};
use hodu_metal_kernels::{
    kernels::{call_const_set, call_ops_cast, call_ops_contiguous, const_set, contiguous, Kernel},
    metal::Buffer,
    utils::BufferOffset,
};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct MetalStorage {
    buffer: Arc<Buffer>,
    device: MetalDevice,
    count: usize,
    dtype: DType,
}

fn read_to_vec<T: Clone>(buffer: &Buffer, n: usize) -> Vec<T> {
    let ptr = buffer.contents() as *const T;
    assert!(!ptr.is_null());
    let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
    slice.to_vec()
}

impl MetalStorage {
    pub fn new(buffer: Arc<Buffer>, device: MetalDevice, count: usize, dtype: DType) -> Self {
        Self {
            buffer,
            device,
            count,
            dtype,
        }
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Get device pointer to the underlying Metal buffer data
    #[allow(dead_code)]
    pub fn as_ptr(&self) -> *const u8 {
        self.buffer.contents() as *const u8
    }

    /// Get mutable device pointer to the underlying Metal buffer data
    #[allow(dead_code)]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.buffer.contents()
    }

    pub fn to_cpu<T: Clone>(&self) -> HoduResult<Vec<T>> {
        let size = self.count * self.dtype.size_in_bytes();
        let buffer = self.device.allocate_buffer(size)?;
        {
            let command_buffer = self.device.command_buffer()?;
            command_buffer.set_label("to_cpu");
            let blit = command_buffer.blit_command_encoder();
            blit.set_label("blit_to_cpu");
            blit.copy_from_buffer(&self.buffer, 0, &buffer, 0, size);
            blit.end_encoding();
        }
        self.device.wait_until_completed()?;
        Ok(read_to_vec(&buffer, self.count))
    }

    pub fn from_cpu_storage(cpu_storage: &CpuStorage) -> HoduResult<Self> {
        let device = MetalDevice::global().clone();
        let dtype = cpu_storage.dtype();
        let count = match cpu_storage {
            CpuStorage::BOOL(v) => v.len(),
            CpuStorage::BF16(v) => v.len(),
            CpuStorage::F16(v) => v.len(),
            CpuStorage::F32(v) => v.len(),
            CpuStorage::U8(v) => v.len(),
            #[cfg(feature = "u16")]
            CpuStorage::U16(v) => v.len(),
            CpuStorage::U32(v) => v.len(),
            #[cfg(feature = "u64")]
            CpuStorage::U64(v) => v.len(),
            CpuStorage::I8(v) => v.len(),
            #[cfg(feature = "i16")]
            CpuStorage::I16(v) => v.len(),
            CpuStorage::I32(v) => v.len(),
            #[cfg(feature = "i64")]
            CpuStorage::I64(v) => v.len(),
            _ => {
                return Err(HoduError::InternalError(format!(
                    "Unsupported dtype for Metal: {:?}",
                    dtype
                )))
            },
        };
        let buffer = device.new_buffer_with_cpu_storage(cpu_storage)?;
        Ok(Self::new(buffer, device, count, dtype))
    }
}

impl BackendStorageT for MetalStorage {
    type BackendDevice = MetalDevice;

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> Device {
        Device::Metal
    }

    fn backend_device(&self) -> &MetalDevice {
        &self.device
    }

    fn to_cpu_storage(&self) -> HoduResult<CpuStorage> {
        match self.dtype {
            DType::BOOL => Ok(CpuStorage::BOOL(self.to_cpu()?)),
            DType::BF16 => Ok(CpuStorage::BF16(self.to_cpu()?)),
            DType::F16 => Ok(CpuStorage::F16(self.to_cpu()?)),
            DType::F32 => Ok(CpuStorage::F32(self.to_cpu()?)),
            DType::I8 => Ok(CpuStorage::I8(self.to_cpu()?)),
            #[cfg(feature = "i16")]
            DType::I16 => Ok(CpuStorage::I16(self.to_cpu()?)),
            DType::I32 => Ok(CpuStorage::I32(self.to_cpu()?)),
            #[cfg(feature = "i64")]
            DType::I64 => Ok(CpuStorage::I64(self.to_cpu()?)),
            DType::U8 => Ok(CpuStorage::U8(self.to_cpu()?)),
            #[cfg(feature = "u16")]
            DType::U16 => Ok(CpuStorage::U16(self.to_cpu()?)),
            DType::U32 => Ok(CpuStorage::U32(self.to_cpu()?)),
            #[cfg(feature = "u64")]
            DType::U64 => Ok(CpuStorage::U64(self.to_cpu()?)),

            // not supported
            DType::F8E4M3 => Ok(CpuStorage::F8E4M3(self.to_cpu()?)),
            #[cfg(feature = "f8e5m2")]
            DType::F8E5M2 => Ok(CpuStorage::F8E5M2(self.to_cpu()?)),
            #[cfg(feature = "f64")]
            DType::F64 => Ok(CpuStorage::F64(self.to_cpu()?)),
        }
    }

    fn const_set(&mut self, scalar: Scalar, layout: &Layout) -> HoduResult<()> {
        let shape = layout.shape();
        let strides = layout.strides();
        let offset = layout.offset();
        let num_els = layout.size();

        let command_buffer = self.device.command_buffer()?;

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

        // Match scalar type and call appropriate kernel
        macro_rules! call_kernel {
            ($scalar_variant:ident, $kernel_variant:ident, $val:expr) => {
                call_const_set(
                    const_set::$kernel_variant,
                    self.device.kernels(),
                    self.device.device(),
                    &command_buffer,
                    &self.buffer,
                    &metadata,
                    $val,
                )?
            };
        }

        match (self.dtype, scalar) {
            (DType::BOOL, Scalar::BOOL(v)) => call_kernel!(BOOL, BOOL, v),
            (DType::BF16, Scalar::BF16(v)) => call_kernel!(BF16, BF16, v),
            (DType::F16, Scalar::F16(v)) => call_kernel!(F16, F16, v),
            (DType::F32, Scalar::F32(v)) => call_kernel!(F32, F32, v),
            (DType::I8, Scalar::I8(v)) => call_kernel!(I8, I8, v),
            #[cfg(feature = "i16")]
            (DType::I16, Scalar::I16(v)) => call_kernel!(I16, I16, v),
            (DType::I32, Scalar::I32(v)) => call_kernel!(I32, I32, v),
            #[cfg(feature = "i64")]
            (DType::I64, Scalar::I64(v)) => call_kernel!(I64, I64, v),
            (DType::U8, Scalar::U8(v)) => call_kernel!(U8, U8, v),
            #[cfg(feature = "u16")]
            (DType::U16, Scalar::U16(v)) => call_kernel!(U16, U16, v),
            (DType::U32, Scalar::U32(v)) => call_kernel!(U32, U32, v),
            #[cfg(feature = "u64")]
            (DType::U64, Scalar::U64(v)) => call_kernel!(U64, U64, v),
            _ => {
                return Err(HoduError::DTypeMismatch {
                    expected: self.dtype,
                    got: scalar.dtype(),
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
        output_dtype: DType,
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
        weight_shape: &Shape,
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
        if self.dtype == target_dtype {
            // Still need to make it contiguous according to layout
            return self.contiguous(layout);
        }

        let metadata = op_metadatas::cast_metadata(layout);
        let num_els = layout.size();

        // Create output buffer with target dtype
        let output_buffer = self.device.new_buffer(num_els, target_dtype, "to_dtype")?;

        let command_buffer = self.device.command_buffer()?;

        let input = BufferOffset {
            buffer: &self.buffer,
            offset_in_bytes: 0,
        };

        // Build kernel name: cast_<src>_to_<dst>
        let kernel_name = format!("hodu_metal_cast_{}_to_{}", self.dtype, target_dtype);
        let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
        let kernel = Kernel(kernel_name_static);

        call_ops_cast(
            kernel,
            self.device.kernels(),
            self.device.device(),
            &command_buffer,
            input,
            &output_buffer,
            &metadata,
        )?;

        Ok(Self::new(output_buffer, self.device.clone(), num_els, target_dtype))
    }

    fn contiguous(&self, layout: &Layout) -> HoduResult<Self> {
        // If already contiguous, return clone
        if layout.is_contiguous() {
            return Ok(self.clone());
        }

        let metadata = op_metadatas::contiguous_metadata(layout);
        let num_els = layout.size();

        // Create output buffer
        let output_buffer = self.device.new_buffer(num_els, self.dtype, "contiguous")?;

        let command_buffer = self.device.command_buffer()?;

        let input = BufferOffset {
            buffer: &self.buffer,
            offset_in_bytes: 0,
        };

        // Call appropriate contiguous kernel based on dtype
        macro_rules! call_kernel {
            ($kernel_variant:ident) => {
                call_ops_contiguous(
                    contiguous::$kernel_variant,
                    self.device.kernels(),
                    self.device.device(),
                    &command_buffer,
                    input,
                    &output_buffer,
                    &metadata,
                )?
            };
        }

        match self.dtype {
            DType::BOOL => call_kernel!(BOOL),
            DType::BF16 => call_kernel!(BF16),
            DType::F16 => call_kernel!(F16),
            DType::F32 => call_kernel!(F32),
            DType::I8 => call_kernel!(I8),
            #[cfg(feature = "i16")]
            DType::I16 => call_kernel!(I16),
            DType::I32 => call_kernel!(I32),
            #[cfg(feature = "i64")]
            DType::I64 => call_kernel!(I64),
            DType::U8 => call_kernel!(U8),
            #[cfg(feature = "u16")]
            DType::U16 => call_kernel!(U16),
            DType::U32 => call_kernel!(U32),
            #[cfg(feature = "u64")]
            DType::U64 => call_kernel!(U64),
            _ => {
                return Err(HoduError::UnsupportedDTypeForDevice {
                    dtype: self.dtype,
                    device: Device::Metal,
                })
            },
        }

        Ok(Self::new(output_buffer, self.device.clone(), num_els, self.dtype))
    }
}
