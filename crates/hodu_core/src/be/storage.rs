#![allow(clippy::too_many_arguments)]
#![allow(clippy::upper_case_acronyms)]

use crate::{
    be::device::{BackendDevice, BackendDeviceT},
    be_cpu::storage::CpuStorage,
    error::{HoduError, HoduResult},
    ops::Op,
    scalar::Scalar,
    types::{DType, Device, Layout, Shape},
};

pub trait BackendStorageT: Sized {
    type BackendDevice: BackendDeviceT;

    fn dtype(&self) -> DType;

    fn device(&self) -> Device;

    fn backend_device(&self) -> &Self::BackendDevice;

    #[allow(dead_code)]
    fn to_cpu_storage(&self) -> HoduResult<CpuStorage>;

    fn const_set(&mut self, _: Scalar, _: &Layout) -> HoduResult<()>;

    fn call_ops_binary(&self, _: &Self, _: &Layout, _: &Layout, _: Op) -> HoduResult<Self>;

    fn call_ops_binary_logical(&self, _: &Self, _: &Layout, _: &Layout, _: Op) -> HoduResult<Self>;

    fn call_ops_bitwise_binary(&self, _: &Self, _: &Layout, _: &Layout, _: Op) -> HoduResult<Self>;

    fn call_ops_bitwise_unary(&self, _: &Layout, _: Op) -> HoduResult<Self>;

    fn call_ops_bitwise_unary_scalar(&self, _: &Layout, _: u32, _: Op) -> HoduResult<Self>;

    fn call_ops_cmp(&self, _: &Self, _: &Layout, _: &Layout, _: Op) -> HoduResult<Self>;

    fn call_ops_cmp_scalar(&self, _: &Layout, _: Scalar, _: Op) -> HoduResult<Self>;

    fn call_ops_unary(&self, _: &Layout, _: Op) -> HoduResult<Self>;

    fn call_ops_unary_logical(&self, _: &Layout, _: Op) -> HoduResult<Self>;

    fn call_ops_unary_scalar(&self, _: &Layout, _: Scalar, _: Op) -> HoduResult<Self>;

    fn call_ops_matmul(&self, _: &Self, _: &Layout, _: &Layout, _: Op) -> HoduResult<Self>;

    fn call_ops_dot(&self, _: &Self, _: &Layout, _: &Layout, _: Op) -> HoduResult<Self>;

    fn call_ops_det(&self, _: &Layout) -> HoduResult<Self>;

    fn call_ops_inv(&self, _: &Layout) -> HoduResult<Self>;

    fn call_ops_trace(&self, _: &Layout) -> HoduResult<Self>;

    fn call_ops_reduce(&self, _: &Layout, _: &[usize], _: bool, _: Op) -> HoduResult<Self>;

    fn call_ops_concat(&self, _: &[&Self], _: &[&Layout], _: usize, _: Op) -> HoduResult<Self>;

    fn call_ops_split(&self, _: &Layout, _: usize, _: usize, _: usize, _: Op) -> HoduResult<Self>;

    fn call_ops_index_select(&self, _: &Layout, _: &Self, _: &Layout, _: usize, _: Op) -> HoduResult<Self>;

    fn call_ops_index_put(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
        _: Op,
    ) -> HoduResult<Self>;

    fn call_ops_gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize, _: Op) -> HoduResult<Self>;

    fn call_ops_scatter(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
        _: Op,
    ) -> HoduResult<Self>;

    fn call_ops_onehot(&self, _: &Layout, _: usize, _: usize, _: DType, _: Op) -> HoduResult<Self>;

    /// Returns indices of non-zero elements.
    /// Output shape is [N, ndim] where N is the count of non-zero elements.
    fn call_nonzero(&self, _: &Layout) -> HoduResult<(Self, usize)>; // (indices, count)

    /// Returns unique elements, inverse indices, and counts.
    /// - values: sorted unique values [unique_count], same dtype as input
    /// - inverse: indices into values for each input element [num_els], i32
    /// - counts: count of each unique value [unique_count], i32
    fn call_unique(&self, _: &Layout) -> HoduResult<(Self, Self, Self, usize)>; // (values, inverse, counts, unique_count)

    /// Select elements based on boolean condition array
    /// Returns (selected_elements, true_count)
    fn call_compress(
        &self,
        _: &Layout,
        _condition: &Self,
        _condition_layout: &Layout,
        _axis: Option<usize>,
    ) -> HoduResult<(Self, usize)>;

    fn call_ops_conv(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &[usize],
        _: &[usize],
        _: &[usize],
        _: Op,
    ) -> HoduResult<Self>;

    fn call_ops_conv_grad_weight(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Shape,
        _: &[usize],
        _: &[usize],
        _: &[usize],
        _: Op,
    ) -> HoduResult<Self>;

    fn call_ops_reduce_window(&self, _: &Layout, _: &[usize], _: &[usize], _: &[usize], _: Op) -> HoduResult<Self>;

    fn call_ops_resize(
        &self,
        _: &Layout,
        _: &[usize],
        _: crate::op_params::ResizeMode,
        _: crate::op_params::ResizeCoordTransform,
        _: crate::op_params::ResizeNearestMode,
    ) -> HoduResult<Self>;

    fn call_ops_pad(&self, _: &Layout, _: &[usize], _: &[usize], _: Scalar, _: Op) -> HoduResult<Self>;

    fn call_ops_cumsum(&self, _: &Layout, _: usize) -> HoduResult<Self>;

    fn call_ops_cumprod(&self, _: &Layout, _: usize) -> HoduResult<Self>;

    fn call_topk(
        &self,
        _: &Layout,
        _: usize, // k
        _: usize, // last_dim_size
        _: usize, // outer_size
        _: bool,  // largest
        _: bool,  // sorted
    ) -> HoduResult<(Self, Self)>; // (values, indices)

    fn call_ops_einsum(
        &self,
        inputs: &[&Self],
        input_layouts: &[&Layout],
        parsed: &crate::einsum::ParsedEinsum,
    ) -> HoduResult<Self>;

    fn call_ops_flip(&self, _: &Layout, _: &[usize]) -> HoduResult<Self>;

    fn to_dtype(&self, _: &Layout, _: DType) -> HoduResult<Self>;

    // only implemented in BackendStorage
    // fn to_device(&self, _: &Layout, _: Device) -> HoduResult<Self>;

    fn contiguous(&self, _: &Layout) -> HoduResult<Self>;
}

#[derive(Debug, Clone)]
pub enum BackendStorage {
    CPU(CpuStorage),
    #[cfg(feature = "cuda")]
    CUDA(crate::be_cuda::storage::CudaStorage),
    #[cfg(feature = "metal")]
    Metal(crate::be_metal::storage::MetalStorage),
}

impl BackendStorage {
    pub fn dtype(&self) -> DType {
        match self {
            Self::CPU(storage) => storage.dtype(),
            #[cfg(feature = "cuda")]
            Self::CUDA(storage) => storage.dtype(),
            #[cfg(feature = "metal")]
            Self::Metal(storage) => storage.dtype(),
        }
    }

    pub fn device(&self) -> Device {
        match self {
            Self::CPU(storage) => storage.device(),
            #[cfg(feature = "cuda")]
            Self::CUDA(storage) => storage.device(),
            #[cfg(feature = "metal")]
            Self::Metal(storage) => storage.device(),
        }
    }

    #[allow(dead_code)]
    pub fn backend_device(&self) -> BackendDevice {
        match self {
            Self::CPU(storage) => BackendDevice::CPU(storage.backend_device().clone()),
            #[cfg(feature = "cuda")]
            Self::CUDA(storage) => BackendDevice::CUDA(storage.backend_device().clone()),
            #[cfg(feature = "metal")]
            Self::Metal(storage) => BackendDevice::Metal(storage.backend_device().clone()),
        }
    }

    #[allow(dead_code)]
    pub fn to_cpu_storage(&self) -> HoduResult<CpuStorage> {
        match self {
            Self::CPU(storage) => Ok(storage.clone()),
            #[cfg(feature = "cuda")]
            Self::CUDA(storage) => storage.to_cpu_storage(),
            #[cfg(feature = "metal")]
            Self::Metal(storage) => storage.to_cpu_storage(),
        }
    }

    pub(crate) fn const_set(&mut self, scalar: Scalar, layout: &Layout) -> HoduResult<()> {
        match self {
            Self::CPU(storage) => storage.const_set(scalar, layout),
            #[cfg(feature = "cuda")]
            Self::CUDA(storage) => storage.const_set(scalar, layout),
            #[cfg(feature = "metal")]
            Self::Metal(storage) => storage.const_set(scalar, layout),
        }
    }

    pub(crate) fn call_ops_binary(
        &self,
        rhs_storage: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
        op: Op,
    ) -> HoduResult<Self> {
        let lhs_device = self.device();
        let rhs_device = rhs_storage.device();
        if lhs_device != rhs_device {
            return Err(HoduError::DeviceMismatch {
                expected: lhs_device,
                got: rhs_device,
            });
        }

        match (self, rhs_storage) {
            (Self::CPU(lhs_storage), Self::CPU(rhs_storage)) => Ok(Self::CPU(lhs_storage.call_ops_binary(
                rhs_storage,
                lhs_layout,
                rhs_layout,
                op,
            )?)),
            #[cfg(feature = "cuda")]
            (Self::CUDA(lhs_storage), Self::CUDA(rhs_storage)) => Ok(Self::CUDA(lhs_storage.call_ops_binary(
                rhs_storage,
                lhs_layout,
                rhs_layout,
                op,
            )?)),
            #[cfg(feature = "metal")]
            (Self::Metal(lhs_storage), Self::Metal(rhs_storage)) => Ok(Self::Metal(lhs_storage.call_ops_binary(
                rhs_storage,
                lhs_layout,
                rhs_layout,
                op,
            )?)),
            #[cfg(any(feature = "cuda", feature = "metal"))]
            _ => Err(HoduError::DeviceMismatch {
                expected: lhs_device,
                got: rhs_device,
            }),
        }
    }

    pub(crate) fn call_ops_binary_logical(
        &self,
        rhs_storage: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
        op: Op,
    ) -> HoduResult<Self> {
        let lhs_device = self.device();
        let rhs_device = rhs_storage.device();
        if lhs_device != rhs_device {
            return Err(HoduError::DeviceMismatch {
                expected: lhs_device,
                got: rhs_device,
            });
        }

        match (self, rhs_storage) {
            (Self::CPU(lhs_storage), Self::CPU(rhs_storage)) => Ok(Self::CPU(lhs_storage.call_ops_binary_logical(
                rhs_storage,
                lhs_layout,
                rhs_layout,
                op,
            )?)),
            #[cfg(feature = "cuda")]
            (Self::CUDA(lhs_storage), Self::CUDA(rhs_storage)) => Ok(Self::CUDA(lhs_storage.call_ops_binary_logical(
                rhs_storage,
                lhs_layout,
                rhs_layout,
                op,
            )?)),
            #[cfg(feature = "metal")]
            (Self::Metal(lhs_storage), Self::Metal(rhs_storage)) => Ok(Self::Metal(
                lhs_storage.call_ops_binary_logical(rhs_storage, lhs_layout, rhs_layout, op)?,
            )),
            #[cfg(any(feature = "cuda", feature = "metal"))]
            _ => Err(HoduError::DeviceMismatch {
                expected: lhs_device,
                got: rhs_device,
            }),
        }
    }

    pub(crate) fn call_ops_bitwise_binary(
        &self,
        rhs_storage: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
        op: Op,
    ) -> HoduResult<Self> {
        let lhs_device = self.device();
        let rhs_device = rhs_storage.device();
        if lhs_device != rhs_device {
            return Err(HoduError::DeviceMismatch {
                expected: lhs_device,
                got: rhs_device,
            });
        }

        match (self, rhs_storage) {
            (Self::CPU(lhs_storage), Self::CPU(rhs_storage)) => Ok(Self::CPU(lhs_storage.call_ops_bitwise_binary(
                rhs_storage,
                lhs_layout,
                rhs_layout,
                op,
            )?)),
            #[cfg(feature = "cuda")]
            (Self::CUDA(lhs_storage), Self::CUDA(rhs_storage)) => Ok(Self::CUDA(lhs_storage.call_ops_bitwise_binary(
                rhs_storage,
                lhs_layout,
                rhs_layout,
                op,
            )?)),
            #[cfg(feature = "metal")]
            (Self::Metal(lhs_storage), Self::Metal(rhs_storage)) => Ok(Self::Metal(
                lhs_storage.call_ops_bitwise_binary(rhs_storage, lhs_layout, rhs_layout, op)?,
            )),
            #[cfg(any(feature = "cuda", feature = "metal"))]
            _ => Err(HoduError::DeviceMismatch {
                expected: lhs_device,
                got: rhs_device,
            }),
        }
    }

    pub(crate) fn call_ops_bitwise_unary(&self, layout: &Layout, op: Op) -> HoduResult<Self> {
        match self {
            Self::CPU(storage) => Ok(Self::CPU(storage.call_ops_bitwise_unary(layout, op)?)),
            #[cfg(feature = "cuda")]
            Self::CUDA(storage) => Ok(Self::CUDA(storage.call_ops_bitwise_unary(layout, op)?)),
            #[cfg(feature = "metal")]
            Self::Metal(storage) => Ok(Self::Metal(storage.call_ops_bitwise_unary(layout, op)?)),
        }
    }

    pub(crate) fn call_ops_bitwise_unary_scalar(&self, layout: &Layout, shift: u32, op: Op) -> HoduResult<Self> {
        match self {
            Self::CPU(storage) => Ok(Self::CPU(storage.call_ops_bitwise_unary_scalar(layout, shift, op)?)),
            #[cfg(feature = "cuda")]
            Self::CUDA(storage) => Ok(Self::CUDA(storage.call_ops_bitwise_unary_scalar(layout, shift, op)?)),
            #[cfg(feature = "metal")]
            Self::Metal(storage) => Ok(Self::Metal(storage.call_ops_bitwise_unary_scalar(layout, shift, op)?)),
        }
    }

    pub(crate) fn call_ops_cmp(
        &self,
        rhs_storage: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
        op: Op,
    ) -> HoduResult<Self> {
        let lhs_device = self.device();
        let rhs_device = rhs_storage.device();
        if lhs_device != rhs_device {
            return Err(HoduError::DeviceMismatch {
                expected: lhs_device,
                got: rhs_device,
            });
        }

        match (self, rhs_storage) {
            (Self::CPU(lhs_storage), Self::CPU(rhs_storage)) => Ok(Self::CPU(lhs_storage.call_ops_cmp(
                rhs_storage,
                lhs_layout,
                rhs_layout,
                op,
            )?)),
            #[cfg(feature = "cuda")]
            (Self::CUDA(lhs_storage), Self::CUDA(rhs_storage)) => Ok(Self::CUDA(lhs_storage.call_ops_cmp(
                rhs_storage,
                lhs_layout,
                rhs_layout,
                op,
            )?)),
            #[cfg(feature = "metal")]
            (Self::Metal(lhs_storage), Self::Metal(rhs_storage)) => Ok(Self::Metal(lhs_storage.call_ops_cmp(
                rhs_storage,
                lhs_layout,
                rhs_layout,
                op,
            )?)),
            #[cfg(any(feature = "cuda", feature = "metal"))]
            _ => Err(HoduError::DeviceMismatch {
                expected: lhs_device,
                got: rhs_device,
            }),
        }
    }

    pub(crate) fn call_ops_cmp_scalar(&self, layout: &Layout, scalar: Scalar, op: Op) -> HoduResult<Self> {
        match self {
            Self::CPU(storage) => Ok(Self::CPU(storage.call_ops_cmp_scalar(layout, scalar, op)?)),
            #[cfg(feature = "cuda")]
            Self::CUDA(storage) => Ok(Self::CUDA(storage.call_ops_cmp_scalar(layout, scalar, op)?)),
            #[cfg(feature = "metal")]
            Self::Metal(storage) => Ok(Self::Metal(storage.call_ops_cmp_scalar(layout, scalar, op)?)),
        }
    }

    pub(crate) fn call_ops_unary(&self, layout: &Layout, op: Op) -> HoduResult<Self> {
        match self {
            Self::CPU(storage) => Ok(Self::CPU(storage.call_ops_unary(layout, op)?)),
            #[cfg(feature = "cuda")]
            Self::CUDA(storage) => Ok(Self::CUDA(storage.call_ops_unary(layout, op)?)),
            #[cfg(feature = "metal")]
            Self::Metal(storage) => Ok(Self::Metal(storage.call_ops_unary(layout, op)?)),
        }
    }

    pub(crate) fn call_ops_unary_logical(&self, layout: &Layout, op: Op) -> HoduResult<Self> {
        match self {
            Self::CPU(storage) => Ok(Self::CPU(storage.call_ops_unary_logical(layout, op)?)),
            #[cfg(feature = "cuda")]
            Self::CUDA(storage) => Ok(Self::CUDA(storage.call_ops_unary_logical(layout, op)?)),
            #[cfg(feature = "metal")]
            Self::Metal(storage) => Ok(Self::Metal(storage.call_ops_unary_logical(layout, op)?)),
        }
    }

    pub(crate) fn call_ops_unary_scalar(&self, layout: &Layout, scalar: Scalar, op: Op) -> HoduResult<Self> {
        match self {
            Self::CPU(storage) => Ok(Self::CPU(storage.call_ops_unary_scalar(layout, scalar, op)?)),
            #[cfg(feature = "cuda")]
            Self::CUDA(storage) => Ok(Self::CUDA(storage.call_ops_unary_scalar(layout, scalar, op)?)),
            #[cfg(feature = "metal")]
            Self::Metal(storage) => Ok(Self::Metal(storage.call_ops_unary_scalar(layout, scalar, op)?)),
        }
    }

    pub(crate) fn call_ops_matmul(
        &self,
        rhs_storage: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
        op: Op,
    ) -> HoduResult<Self> {
        let lhs_device = self.device();
        let rhs_device = rhs_storage.device();
        if lhs_device != rhs_device {
            return Err(HoduError::DeviceMismatch {
                expected: lhs_device,
                got: rhs_device,
            });
        }

        match (self, rhs_storage) {
            (Self::CPU(lhs_storage), Self::CPU(rhs_storage)) => Ok(Self::CPU(lhs_storage.call_ops_matmul(
                rhs_storage,
                lhs_layout,
                rhs_layout,
                op,
            )?)),
            #[cfg(feature = "cuda")]
            (Self::CUDA(lhs_storage), Self::CUDA(rhs_storage)) => Ok(Self::CUDA(lhs_storage.call_ops_matmul(
                rhs_storage,
                lhs_layout,
                rhs_layout,
                op,
            )?)),
            #[cfg(feature = "metal")]
            (Self::Metal(lhs_storage), Self::Metal(rhs_storage)) => Ok(Self::Metal(lhs_storage.call_ops_matmul(
                rhs_storage,
                lhs_layout,
                rhs_layout,
                op,
            )?)),
            #[cfg(any(feature = "cuda", feature = "metal"))]
            _ => Err(HoduError::DeviceMismatch {
                expected: lhs_device,
                got: rhs_device,
            }),
        }
    }

    pub(crate) fn call_ops_dot(
        &self,
        rhs_storage: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
        op: Op,
    ) -> HoduResult<Self> {
        let lhs_device = self.device();
        let rhs_device = rhs_storage.device();
        if lhs_device != rhs_device {
            return Err(HoduError::DeviceMismatch {
                expected: lhs_device,
                got: rhs_device,
            });
        }

        match (self, rhs_storage) {
            (Self::CPU(lhs_storage), Self::CPU(rhs_storage)) => Ok(Self::CPU(lhs_storage.call_ops_dot(
                rhs_storage,
                lhs_layout,
                rhs_layout,
                op,
            )?)),
            #[cfg(feature = "cuda")]
            (Self::CUDA(lhs_storage), Self::CUDA(rhs_storage)) => Ok(Self::CUDA(lhs_storage.call_ops_dot(
                rhs_storage,
                lhs_layout,
                rhs_layout,
                op,
            )?)),
            #[cfg(feature = "metal")]
            (Self::Metal(lhs_storage), Self::Metal(rhs_storage)) => Ok(Self::Metal(lhs_storage.call_ops_dot(
                rhs_storage,
                lhs_layout,
                rhs_layout,
                op,
            )?)),
            #[cfg(any(feature = "cuda", feature = "metal"))]
            _ => Err(HoduError::DeviceMismatch {
                expected: lhs_device,
                got: rhs_device,
            }),
        }
    }

    pub(crate) fn call_ops_det(&self, layout: &Layout) -> HoduResult<Self> {
        match self {
            Self::CPU(storage) => Ok(Self::CPU(storage.call_ops_det(layout)?)),
            #[cfg(feature = "cuda")]
            Self::CUDA(storage) => Ok(Self::CUDA(storage.call_ops_det(layout)?)),
            #[cfg(feature = "metal")]
            Self::Metal(storage) => Ok(Self::Metal(storage.call_ops_det(layout)?)),
        }
    }

    pub(crate) fn call_ops_inv(&self, layout: &Layout) -> HoduResult<Self> {
        match self {
            Self::CPU(storage) => Ok(Self::CPU(storage.call_ops_inv(layout)?)),
            #[cfg(feature = "cuda")]
            Self::CUDA(storage) => Ok(Self::CUDA(storage.call_ops_inv(layout)?)),
            #[cfg(feature = "metal")]
            Self::Metal(storage) => Ok(Self::Metal(storage.call_ops_inv(layout)?)),
        }
    }

    pub(crate) fn call_ops_trace(&self, layout: &Layout) -> HoduResult<Self> {
        match self {
            Self::CPU(storage) => Ok(Self::CPU(storage.call_ops_trace(layout)?)),
            #[cfg(feature = "cuda")]
            Self::CUDA(storage) => Ok(Self::CUDA(storage.call_ops_trace(layout)?)),
            #[cfg(feature = "metal")]
            Self::Metal(storage) => Ok(Self::Metal(storage.call_ops_trace(layout)?)),
        }
    }

    pub(crate) fn call_ops_reduce(&self, layout: &Layout, dims: &[usize], keep_dim: bool, op: Op) -> HoduResult<Self> {
        match self {
            Self::CPU(storage) => Ok(Self::CPU(storage.call_ops_reduce(layout, dims, keep_dim, op)?)),
            #[cfg(feature = "cuda")]
            Self::CUDA(storage) => Ok(Self::CUDA(storage.call_ops_reduce(layout, dims, keep_dim, op)?)),
            #[cfg(feature = "metal")]
            Self::Metal(storage) => Ok(Self::Metal(storage.call_ops_reduce(layout, dims, keep_dim, op)?)),
        }
    }

    pub(crate) fn call_ops_concat(
        &self,
        others: &[&Self],
        layouts: &[&Layout],
        dim: usize,
        op: Op,
    ) -> HoduResult<Self> {
        // Check all storages are on the same device
        let device = self.device();
        for other in others {
            let other_device = other.device();
            if device != other_device {
                return Err(HoduError::DeviceMismatch {
                    expected: device,
                    got: other_device,
                });
            }
        }

        match self {
            Self::CPU(storage) => {
                let others_cpu: Vec<&CpuStorage> = others
                    .iter()
                    .map(|s| match s {
                        Self::CPU(cpu) => cpu,
                        #[cfg(any(feature = "cuda", feature = "metal"))]
                        _ => unreachable!("Device mismatch already checked"),
                    })
                    .collect();
                Ok(Self::CPU(storage.call_ops_concat(&others_cpu, layouts, dim, op)?))
            },
            #[cfg(feature = "cuda")]
            Self::CUDA(storage) => {
                let others_cuda: Vec<&crate::be_cuda::storage::CudaStorage> = others
                    .iter()
                    .map(|s| match s {
                        Self::CUDA(cuda) => cuda,
                        _ => unreachable!("Device mismatch already checked"),
                    })
                    .collect();
                Ok(Self::CUDA(storage.call_ops_concat(&others_cuda, layouts, dim, op)?))
            },
            #[cfg(feature = "metal")]
            Self::Metal(storage) => {
                let others_metal: Vec<&crate::be_metal::storage::MetalStorage> = others
                    .iter()
                    .map(|s| match s {
                        Self::Metal(metal) => metal,
                        _ => unreachable!("Device mismatch already checked"),
                    })
                    .collect();
                Ok(Self::Metal(storage.call_ops_concat(&others_metal, layouts, dim, op)?))
            },
        }
    }

    pub(crate) fn call_ops_split(
        &self,
        layout: &Layout,
        dim: usize,
        start: usize,
        size: usize,
        op: Op,
    ) -> HoduResult<Self> {
        match self {
            Self::CPU(storage) => Ok(Self::CPU(storage.call_ops_split(layout, dim, start, size, op)?)),
            #[cfg(feature = "cuda")]
            Self::CUDA(storage) => Ok(Self::CUDA(storage.call_ops_split(layout, dim, start, size, op)?)),
            #[cfg(feature = "metal")]
            Self::Metal(storage) => Ok(Self::Metal(storage.call_ops_split(layout, dim, start, size, op)?)),
        }
    }

    pub(crate) fn call_ops_index_select(
        &self,
        layout: &Layout,
        indices_storage: &Self,
        indices_layout: &Layout,
        dim: usize,
        op: Op,
    ) -> HoduResult<Self> {
        // Check devices match
        let device = self.device();
        let indices_device = indices_storage.device();
        if device != indices_device {
            return Err(HoduError::DeviceMismatch {
                expected: device,
                got: indices_device,
            });
        }

        match (self, indices_storage) {
            (Self::CPU(storage), Self::CPU(indices)) => Ok(Self::CPU(storage.call_ops_index_select(
                layout,
                indices,
                indices_layout,
                dim,
                op,
            )?)),
            #[cfg(feature = "cuda")]
            (Self::CUDA(storage), Self::CUDA(indices)) => Ok(Self::CUDA(storage.call_ops_index_select(
                layout,
                indices,
                indices_layout,
                dim,
                op,
            )?)),
            #[cfg(feature = "metal")]
            (Self::Metal(storage), Self::Metal(indices)) => Ok(Self::Metal(storage.call_ops_index_select(
                layout,
                indices,
                indices_layout,
                dim,
                op,
            )?)),
            #[cfg(any(feature = "cuda", feature = "metal"))]
            _ => Err(HoduError::DeviceMismatch {
                expected: device,
                got: indices_device,
            }),
        }
    }

    pub(crate) fn call_ops_index_put(
        &self,
        layout: &Layout,
        indices_storage: &Self,
        indices_layout: &Layout,
        values_storage: &Self,
        values_layout: &Layout,
        dim: usize,
        op: Op,
    ) -> HoduResult<Self> {
        // Check all devices match
        let device = self.device();
        let indices_device = indices_storage.device();
        let values_device = values_storage.device();

        if device != indices_device {
            return Err(HoduError::DeviceMismatch {
                expected: device,
                got: indices_device,
            });
        }
        if device != values_device {
            return Err(HoduError::DeviceMismatch {
                expected: device,
                got: values_device,
            });
        }

        match (self, indices_storage, values_storage) {
            (Self::CPU(storage), Self::CPU(indices), Self::CPU(values)) => Ok(Self::CPU(storage.call_ops_index_put(
                layout,
                indices,
                indices_layout,
                values,
                values_layout,
                dim,
                op,
            )?)),
            #[cfg(feature = "cuda")]
            (Self::CUDA(storage), Self::CUDA(indices), Self::CUDA(values)) => Ok(Self::CUDA(
                storage.call_ops_index_put(layout, indices, indices_layout, values, values_layout, dim, op)?,
            )),
            #[cfg(feature = "metal")]
            (Self::Metal(storage), Self::Metal(indices), Self::Metal(values)) => Ok(Self::Metal(
                storage.call_ops_index_put(layout, indices, indices_layout, values, values_layout, dim, op)?,
            )),
            #[cfg(any(feature = "cuda", feature = "metal"))]
            _ => {
                if device != indices_device {
                    Err(HoduError::DeviceMismatch {
                        expected: device,
                        got: indices_device,
                    })
                } else {
                    Err(HoduError::DeviceMismatch {
                        expected: device,
                        got: values_device,
                    })
                }
            },
        }
    }

    pub(crate) fn call_ops_gather(
        &self,
        layout: &Layout,
        indices_storage: &Self,
        indices_layout: &Layout,
        dim: usize,
        op: Op,
    ) -> HoduResult<Self> {
        // Check devices match
        let device = self.device();
        let indices_device = indices_storage.device();
        if device != indices_device {
            return Err(HoduError::DeviceMismatch {
                expected: device,
                got: indices_device,
            });
        }

        match (self, indices_storage) {
            (Self::CPU(storage), Self::CPU(indices)) => Ok(Self::CPU(storage.call_ops_gather(
                layout,
                indices,
                indices_layout,
                dim,
                op,
            )?)),
            #[cfg(feature = "cuda")]
            (Self::CUDA(storage), Self::CUDA(indices)) => Ok(Self::CUDA(storage.call_ops_gather(
                layout,
                indices,
                indices_layout,
                dim,
                op,
            )?)),
            #[cfg(feature = "metal")]
            (Self::Metal(storage), Self::Metal(indices)) => Ok(Self::Metal(storage.call_ops_gather(
                layout,
                indices,
                indices_layout,
                dim,
                op,
            )?)),
            #[cfg(any(feature = "cuda", feature = "metal"))]
            _ => Err(HoduError::DeviceMismatch {
                expected: device,
                got: indices_device,
            }),
        }
    }

    pub(crate) fn call_ops_scatter(
        &self,
        layout: &Layout,
        indices_storage: &Self,
        indices_layout: &Layout,
        src_storage: &Self,
        src_layout: &Layout,
        dim: usize,
        op: Op,
    ) -> HoduResult<Self> {
        // Check all devices match
        let device = self.device();
        let indices_device = indices_storage.device();
        let src_device = src_storage.device();

        if device != indices_device {
            return Err(HoduError::DeviceMismatch {
                expected: device,
                got: indices_device,
            });
        }
        if device != src_device {
            return Err(HoduError::DeviceMismatch {
                expected: device,
                got: src_device,
            });
        }

        match (self, indices_storage, src_storage) {
            (Self::CPU(storage), Self::CPU(indices), Self::CPU(src)) => Ok(Self::CPU(storage.call_ops_scatter(
                layout,
                indices,
                indices_layout,
                src,
                src_layout,
                dim,
                op,
            )?)),
            #[cfg(feature = "cuda")]
            (Self::CUDA(storage), Self::CUDA(indices), Self::CUDA(src)) => Ok(Self::CUDA(storage.call_ops_scatter(
                layout,
                indices,
                indices_layout,
                src,
                src_layout,
                dim,
                op,
            )?)),
            #[cfg(feature = "metal")]
            (Self::Metal(storage), Self::Metal(indices), Self::Metal(src)) => Ok(Self::Metal(
                storage.call_ops_scatter(layout, indices, indices_layout, src, src_layout, dim, op)?,
            )),
            #[cfg(any(feature = "cuda", feature = "metal"))]
            _ => {
                if device != indices_device {
                    Err(HoduError::DeviceMismatch {
                        expected: device,
                        got: indices_device,
                    })
                } else {
                    Err(HoduError::DeviceMismatch {
                        expected: device,
                        got: src_device,
                    })
                }
            },
        }
    }

    pub(crate) fn call_ops_onehot(
        &self,
        layout: &Layout,
        num_classes: usize,
        axis: usize,
        output_dtype: DType,
        op: Op,
    ) -> HoduResult<Self> {
        match self {
            Self::CPU(storage) => Ok(Self::CPU(storage.call_ops_onehot(
                layout,
                num_classes,
                axis,
                output_dtype,
                op,
            )?)),
            #[cfg(feature = "cuda")]
            Self::CUDA(storage) => Ok(Self::CUDA(storage.call_ops_onehot(
                layout,
                num_classes,
                axis,
                output_dtype,
                op,
            )?)),
            #[cfg(feature = "metal")]
            Self::Metal(storage) => Ok(Self::Metal(storage.call_ops_onehot(
                layout,
                num_classes,
                axis,
                output_dtype,
                op,
            )?)),
        }
    }

    pub(crate) fn call_nonzero(&self, layout: &Layout) -> HoduResult<(Self, usize)> {
        match self {
            Self::CPU(storage) => {
                let (indices, count) = storage.call_nonzero(layout)?;
                Ok((Self::CPU(indices), count))
            },
            #[cfg(feature = "cuda")]
            Self::CUDA(storage) => {
                let (indices, count) = storage.call_nonzero(layout)?;
                Ok((Self::CUDA(indices), count))
            },
            #[cfg(feature = "metal")]
            Self::Metal(storage) => {
                let (indices, count) = storage.call_nonzero(layout)?;
                Ok((Self::Metal(indices), count))
            },
        }
    }

    pub(crate) fn call_unique(&self, layout: &Layout) -> HoduResult<(Self, Self, Self, usize)> {
        match self {
            Self::CPU(storage) => {
                let (values, inverse, counts, unique_count) = storage.call_unique(layout)?;
                Ok((Self::CPU(values), Self::CPU(inverse), Self::CPU(counts), unique_count))
            },
            #[cfg(feature = "cuda")]
            Self::CUDA(storage) => {
                let (values, inverse, counts, unique_count) = storage.call_unique(layout)?;
                Ok((
                    Self::CUDA(values),
                    Self::CUDA(inverse),
                    Self::CUDA(counts),
                    unique_count,
                ))
            },
            #[cfg(feature = "metal")]
            Self::Metal(storage) => {
                let (values, inverse, counts, unique_count) = storage.call_unique(layout)?;
                Ok((
                    Self::Metal(values),
                    Self::Metal(inverse),
                    Self::Metal(counts),
                    unique_count,
                ))
            },
        }
    }

    pub(crate) fn call_compress(
        &self,
        layout: &Layout,
        condition_storage: &Self,
        condition_layout: &Layout,
        axis: Option<usize>,
    ) -> HoduResult<(Self, usize)> {
        // Check devices match
        let device = self.device();
        let condition_device = condition_storage.device();
        if device != condition_device {
            return Err(HoduError::DeviceMismatch {
                expected: device,
                got: condition_device,
            });
        }

        match (self, condition_storage) {
            (Self::CPU(storage), Self::CPU(condition)) => {
                let (result, count) = storage.call_compress(layout, condition, condition_layout, axis)?;
                Ok((Self::CPU(result), count))
            },
            #[cfg(feature = "cuda")]
            (Self::CUDA(storage), Self::CUDA(condition)) => {
                let (result, count) = storage.call_compress(layout, condition, condition_layout, axis)?;
                Ok((Self::CUDA(result), count))
            },
            #[cfg(feature = "metal")]
            (Self::Metal(storage), Self::Metal(condition)) => {
                let (result, count) = storage.call_compress(layout, condition, condition_layout, axis)?;
                Ok((Self::Metal(result), count))
            },
            #[cfg(any(feature = "cuda", feature = "metal"))]
            _ => Err(HoduError::DeviceMismatch {
                expected: device,
                got: condition_device,
            }),
        }
    }

    pub(crate) fn call_ops_conv(
        &self,
        layout: &Layout,
        weight_storage: &Self,
        weight_layout: &Layout,
        stride: &[usize],
        padding: &[usize],
        dilation: &[usize],
        op: Op,
    ) -> HoduResult<Self> {
        // Check devices match
        let device = self.device();
        let weight_device = weight_storage.device();
        if device != weight_device {
            return Err(HoduError::DeviceMismatch {
                expected: device,
                got: weight_device,
            });
        }

        match (self, weight_storage) {
            (Self::CPU(storage), Self::CPU(weight)) => Ok(Self::CPU(storage.call_ops_conv(
                layout,
                weight,
                weight_layout,
                stride,
                padding,
                dilation,
                op,
            )?)),
            #[cfg(feature = "cuda")]
            (Self::CUDA(storage), Self::CUDA(weight)) => Ok(Self::CUDA(storage.call_ops_conv(
                layout,
                weight,
                weight_layout,
                stride,
                padding,
                dilation,
                op,
            )?)),
            #[cfg(feature = "metal")]
            (Self::Metal(storage), Self::Metal(weight)) => Ok(Self::Metal(storage.call_ops_conv(
                layout,
                weight,
                weight_layout,
                stride,
                padding,
                dilation,
                op,
            )?)),
            #[cfg(any(feature = "cuda", feature = "metal"))]
            _ => Err(HoduError::DeviceMismatch {
                expected: device,
                got: weight_device,
            }),
        }
    }

    pub(crate) fn call_ops_conv_grad_weight(
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
        // Check devices match
        let device = self.device();
        let grad_output_device = grad_output_storage.device();
        if device != grad_output_device {
            return Err(HoduError::DeviceMismatch {
                expected: device,
                got: grad_output_device,
            });
        }

        match (self, grad_output_storage) {
            (Self::CPU(storage), Self::CPU(grad_output)) => Ok(Self::CPU(storage.call_ops_conv_grad_weight(
                layout,
                grad_output,
                grad_output_layout,
                weight_shape,
                stride,
                padding,
                dilation,
                op,
            )?)),
            #[cfg(feature = "cuda")]
            (Self::CUDA(storage), Self::CUDA(grad_output)) => Ok(Self::CUDA(storage.call_ops_conv_grad_weight(
                layout,
                grad_output,
                grad_output_layout,
                weight_shape,
                stride,
                padding,
                dilation,
                op,
            )?)),
            #[cfg(feature = "metal")]
            (Self::Metal(storage), Self::Metal(grad_output)) => Ok(Self::Metal(storage.call_ops_conv_grad_weight(
                layout,
                grad_output,
                grad_output_layout,
                weight_shape,
                stride,
                padding,
                dilation,
                op,
            )?)),
            #[cfg(any(feature = "cuda", feature = "metal"))]
            _ => Err(HoduError::DeviceMismatch {
                expected: device,
                got: grad_output_device,
            }),
        }
    }

    pub(crate) fn call_ops_reduce_window(
        &self,
        layout: &Layout,
        window_shape: &[usize],
        strides: &[usize],
        padding: &[usize],
        op: Op,
    ) -> HoduResult<Self> {
        match self {
            Self::CPU(storage) => Ok(Self::CPU(storage.call_ops_reduce_window(
                layout,
                window_shape,
                strides,
                padding,
                op,
            )?)),
            #[cfg(feature = "cuda")]
            Self::CUDA(storage) => Ok(Self::CUDA(storage.call_ops_reduce_window(
                layout,
                window_shape,
                strides,
                padding,
                op,
            )?)),
            #[cfg(feature = "metal")]
            Self::Metal(storage) => Ok(Self::Metal(storage.call_ops_reduce_window(
                layout,
                window_shape,
                strides,
                padding,
                op,
            )?)),
        }
    }

    pub(crate) fn call_ops_pad(
        &self,
        layout: &Layout,
        pad_before: &[usize],
        pad_after: &[usize],
        pad_value: Scalar,
        op: Op,
    ) -> HoduResult<Self> {
        match self {
            Self::CPU(storage) => Ok(Self::CPU(
                storage.call_ops_pad(layout, pad_before, pad_after, pad_value, op)?,
            )),
            #[cfg(feature = "cuda")]
            Self::CUDA(storage) => Ok(Self::CUDA(
                storage.call_ops_pad(layout, pad_before, pad_after, pad_value, op)?,
            )),
            #[cfg(feature = "metal")]
            Self::Metal(storage) => Ok(Self::Metal(
                storage.call_ops_pad(layout, pad_before, pad_after, pad_value, op)?,
            )),
        }
    }

    pub(crate) fn call_ops_resize(
        &self,
        layout: &Layout,
        output_shape: &[usize],
        mode: crate::op_params::ResizeMode,
        coord_transform: crate::op_params::ResizeCoordTransform,
        nearest_mode: crate::op_params::ResizeNearestMode,
    ) -> HoduResult<Self> {
        match self {
            Self::CPU(storage) => Ok(Self::CPU(storage.call_ops_resize(
                layout,
                output_shape,
                mode,
                coord_transform,
                nearest_mode,
            )?)),
            #[cfg(feature = "cuda")]
            Self::CUDA(storage) => Ok(Self::CUDA(storage.call_ops_resize(
                layout,
                output_shape,
                mode,
                coord_transform,
                nearest_mode,
            )?)),
            #[cfg(feature = "metal")]
            Self::Metal(storage) => Ok(Self::Metal(storage.call_ops_resize(
                layout,
                output_shape,
                mode,
                coord_transform,
                nearest_mode,
            )?)),
        }
    }

    pub(crate) fn call_ops_cumsum(&self, layout: &Layout, dim: usize) -> HoduResult<Self> {
        match self {
            Self::CPU(storage) => Ok(Self::CPU(storage.call_ops_cumsum(layout, dim)?)),
            #[cfg(feature = "cuda")]
            Self::CUDA(storage) => Ok(Self::CUDA(storage.call_ops_cumsum(layout, dim)?)),
            #[cfg(feature = "metal")]
            Self::Metal(storage) => Ok(Self::Metal(storage.call_ops_cumsum(layout, dim)?)),
        }
    }

    pub(crate) fn call_ops_cumprod(&self, layout: &Layout, dim: usize) -> HoduResult<Self> {
        match self {
            Self::CPU(storage) => Ok(Self::CPU(storage.call_ops_cumprod(layout, dim)?)),
            #[cfg(feature = "cuda")]
            Self::CUDA(storage) => Ok(Self::CUDA(storage.call_ops_cumprod(layout, dim)?)),
            #[cfg(feature = "metal")]
            Self::Metal(storage) => Ok(Self::Metal(storage.call_ops_cumprod(layout, dim)?)),
        }
    }

    pub(crate) fn call_ops_einsum(
        &self,
        inputs: &[&Self],
        input_layouts: &[&Layout],
        parsed: &crate::einsum::ParsedEinsum,
    ) -> HoduResult<Self> {
        match self {
            Self::CPU(storage) => {
                let cpu_inputs: Vec<&CpuStorage> = inputs
                    .iter()
                    .map(|s| match s {
                        Self::CPU(cpu_storage) => Ok(cpu_storage),
                        #[cfg(any(feature = "cuda", feature = "metal"))]
                        _ => Err(HoduError::DeviceMismatch {
                            expected: Device::CPU,
                            got: s.device(),
                        }),
                    })
                    .collect::<HoduResult<Vec<_>>>()?;
                Ok(Self::CPU(storage.call_ops_einsum(
                    &cpu_inputs,
                    input_layouts,
                    parsed,
                )?))
            },
            #[cfg(feature = "cuda")]
            Self::CUDA(storage) => {
                let cuda_inputs: Vec<&crate::be_cuda::storage::CudaStorage> = inputs
                    .iter()
                    .map(|s| match s {
                        Self::CUDA(cuda_storage) => Ok(cuda_storage),
                        _ => Err(HoduError::DeviceMismatch {
                            expected: Device::CUDA(0),
                            got: s.device(),
                        }),
                    })
                    .collect::<HoduResult<Vec<_>>>()?;
                Ok(Self::CUDA(storage.call_ops_einsum(
                    &cuda_inputs,
                    input_layouts,
                    parsed,
                )?))
            },
            #[cfg(feature = "metal")]
            Self::Metal(storage) => {
                let metal_inputs: Vec<&crate::be_metal::storage::MetalStorage> = inputs
                    .iter()
                    .map(|s| match s {
                        Self::Metal(metal_storage) => Ok(metal_storage),
                        _ => Err(HoduError::DeviceMismatch {
                            expected: Device::Metal,
                            got: s.device(),
                        }),
                    })
                    .collect::<HoduResult<Vec<_>>>()?;
                Ok(Self::Metal(storage.call_ops_einsum(
                    &metal_inputs,
                    input_layouts,
                    parsed,
                )?))
            },
        }
    }

    pub(crate) fn call_ops_flip(&self, layout: &Layout, dims: &[usize]) -> HoduResult<Self> {
        match self {
            Self::CPU(storage) => Ok(Self::CPU(storage.call_ops_flip(layout, dims)?)),
            #[cfg(feature = "cuda")]
            Self::CUDA(storage) => Ok(Self::CUDA(storage.call_ops_flip(layout, dims)?)),
            #[cfg(feature = "metal")]
            Self::Metal(storage) => Ok(Self::Metal(storage.call_ops_flip(layout, dims)?)),
        }
    }

    pub(crate) fn call_topk(
        &self,
        layout: &Layout,
        k: usize,
        last_dim_size: usize,
        outer_size: usize,
        largest: bool,
        sorted: bool,
    ) -> HoduResult<(Self, Self)> {
        match self {
            Self::CPU(storage) => {
                let (values, indices) = storage.call_topk(layout, k, last_dim_size, outer_size, largest, sorted)?;
                Ok((Self::CPU(values), Self::CPU(indices)))
            },
            #[cfg(feature = "cuda")]
            Self::CUDA(storage) => {
                let (values, indices) = storage.call_topk(layout, k, last_dim_size, outer_size, largest, sorted)?;
                Ok((Self::CUDA(values), Self::CUDA(indices)))
            },
            #[cfg(feature = "metal")]
            Self::Metal(storage) => {
                let (values, indices) = storage.call_topk(layout, k, last_dim_size, outer_size, largest, sorted)?;
                Ok((Self::Metal(values), Self::Metal(indices)))
            },
        }
    }

    pub(crate) fn to_dtype(&self, layout: &Layout, target_dtype: DType) -> HoduResult<Self> {
        match self {
            Self::CPU(storage) => {
                let converted_storage = storage.to_dtype(layout, target_dtype)?;
                Ok(Self::CPU(converted_storage))
            },
            #[cfg(feature = "cuda")]
            Self::CUDA(storage) => {
                let converted_storage = storage.to_dtype(layout, target_dtype)?;
                Ok(Self::CUDA(converted_storage))
            },
            #[cfg(feature = "metal")]
            Self::Metal(storage) => {
                let converted_storage = storage.to_dtype(layout, target_dtype)?;
                Ok(Self::Metal(converted_storage))
            },
        }
    }

    pub(crate) fn to_device(&self, layout: &Layout, target_device: Device) -> HoduResult<Self> {
        match self {
            Self::CPU(storage) => match target_device {
                Device::CPU => {
                    let contiguous_storage = storage.contiguous(layout)?;
                    Ok(Self::CPU(contiguous_storage))
                },
                #[cfg(feature = "cuda")]
                Device::CUDA(device_id) => {
                    use crate::be_cuda::storage::CudaStorage;
                    let contiguous_storage = storage.contiguous(layout)?;
                    let converted_storage = CudaStorage::from_cpu_storage(&contiguous_storage, device_id)?;
                    Ok(Self::CUDA(converted_storage))
                },
                #[cfg(feature = "metal")]
                Device::Metal => {
                    use crate::be_metal::storage::MetalStorage;
                    let contiguous_storage = storage.contiguous(layout)?;
                    let converted_storage = MetalStorage::from_cpu_storage(&contiguous_storage)?;
                    Ok(Self::Metal(converted_storage))
                },
                #[cfg(all(feature = "metal-device", not(feature = "metal")))]
                Device::Metal => Err(HoduError::UnsupportedDevice(Device::Metal)),
            },
            #[cfg(feature = "cuda")]
            Self::CUDA(storage) => match target_device {
                Device::CPU => {
                    let contiguous_storage = storage.contiguous(layout)?;
                    let converted_storage = contiguous_storage.to_cpu_storage()?;
                    Ok(Self::CPU(converted_storage))
                },
                #[cfg(feature = "cuda")]
                Device::CUDA(device_id) => {
                    if storage.device_id() == device_id {
                        // Same device - just make contiguous
                        let contiguous_storage = storage.contiguous(layout)?;
                        Ok(Self::CUDA(contiguous_storage))
                    } else {
                        // Different device - transfer via CPU
                        let contiguous_storage = storage.contiguous(layout)?;
                        let cpu_storage = contiguous_storage.to_cpu_storage()?;
                        let converted_storage =
                            crate::be_cuda::storage::CudaStorage::from_cpu_storage(&cpu_storage, device_id)?;
                        Ok(Self::CUDA(converted_storage))
                    }
                },
                #[cfg(feature = "metal")]
                Device::Metal => {
                    use crate::be_metal::storage::MetalStorage;
                    let contiguous_storage = storage.contiguous(layout)?;
                    let cpu_storage = contiguous_storage.to_cpu_storage()?;
                    let converted_storage = MetalStorage::from_cpu_storage(&cpu_storage)?;
                    Ok(Self::Metal(converted_storage))
                },
            },
            #[cfg(feature = "metal")]
            Self::Metal(storage) => match target_device {
                Device::CPU => {
                    let contiguoused_storage = storage.contiguous(layout)?;
                    let converted_storage = contiguoused_storage.to_cpu_storage()?;
                    Ok(Self::CPU(converted_storage))
                },
                #[cfg(feature = "cuda")]
                Device::CUDA(device_id) => {
                    use crate::be_cuda::storage::CudaStorage;
                    let contiguous_storage = storage.contiguous(layout)?;
                    let cpu_storage = contiguous_storage.to_cpu_storage()?;
                    let converted_storage = CudaStorage::from_cpu_storage(&cpu_storage, device_id)?;
                    Ok(Self::CUDA(converted_storage))
                },
                #[cfg(feature = "metal")]
                Device::Metal => {
                    let contiguous_storage = storage.contiguous(layout)?;
                    Ok(Self::Metal(contiguous_storage))
                },
            },
        }
    }

    pub(crate) fn contiguous(&self, layout: &Layout) -> HoduResult<Self> {
        match self {
            Self::CPU(storage) => {
                let contiguous_storage = storage.contiguous(layout)?;
                Ok(Self::CPU(contiguous_storage))
            },
            #[cfg(feature = "cuda")]
            Self::CUDA(storage) => {
                let contiguous_storage = storage.contiguous(layout)?;
                Ok(Self::CUDA(contiguous_storage))
            },
            #[cfg(feature = "metal")]
            Self::Metal(storage) => {
                let contiguous_storage = storage.contiguous(layout)?;
                Ok(Self::Metal(contiguous_storage))
            },
        }
    }
}
