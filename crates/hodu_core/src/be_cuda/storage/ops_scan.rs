use crate::{
    be::storage::BackendStorageT,
    be_cuda::storage::{CudaStorage, CudaStorageData},
    error::HoduResult,
    types::Layout,
};
use hodu_cuda_kernels::{cuda::CudaSlice, kernels};
use std::sync::Arc;

pub fn call_ops_cumsum(input_storage: &CudaStorage, input_layout: &Layout, dim: usize) -> HoduResult<CudaStorage> {
    let dtype = input_storage.dtype();
    let device = input_storage.get_device();
    let shape = input_layout.shape();
    let num_els = shape.size();

    let metadata = crate::op_metadatas::scan_metadata(input_layout, dim);

    let kernel_name = format!("hodu_cuda_cumsum_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    let device_id = input_storage.device_id;
    let device_arc = Arc::clone(&input_storage.device);

    macro_rules! call_cumsum {
        ($input:expr, $ty:ty) => {{
            let mut output: CudaSlice<$ty> = device.new_buffer(num_els)?;
            kernels::call_ops_cumsum(
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

    match &input_storage.data {
        CudaStorageData::F8E4M3(input) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::F8E4M3(call_cumsum!(input, float8::F8E4M3)),
        )),
        #[cfg(feature = "f8e5m2")]
        CudaStorageData::F8E5M2(input) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::F8E5M2(call_cumsum!(input, float8::F8E5M2)),
        )),
        CudaStorageData::BF16(input) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::BF16(call_cumsum!(input, half::bf16)),
        )),
        CudaStorageData::F16(input) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::F16(call_cumsum!(input, half::f16)),
        )),
        CudaStorageData::F32(input) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::F32(call_cumsum!(input, f32)),
        )),
        #[cfg(feature = "f64")]
        CudaStorageData::F64(input) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::F64(call_cumsum!(input, f64)),
        )),
        CudaStorageData::U8(input) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::U8(call_cumsum!(input, u8)),
        )),
        #[cfg(feature = "u16")]
        CudaStorageData::U16(input) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::U16(call_cumsum!(input, u16)),
        )),
        CudaStorageData::U32(input) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::U32(call_cumsum!(input, u32)),
        )),
        #[cfg(feature = "u64")]
        CudaStorageData::U64(input) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::U64(call_cumsum!(input, u64)),
        )),
        CudaStorageData::I8(input) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::I8(call_cumsum!(input, i8)),
        )),
        #[cfg(feature = "i16")]
        CudaStorageData::I16(input) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::I16(call_cumsum!(input, i16)),
        )),
        CudaStorageData::I32(input) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::I32(call_cumsum!(input, i32)),
        )),
        #[cfg(feature = "i64")]
        CudaStorageData::I64(input) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::I64(call_cumsum!(input, i64)),
        )),
        _ => unreachable!("cumsum does not support bool"),
    }
}

pub fn call_ops_cumprod(input_storage: &CudaStorage, input_layout: &Layout, dim: usize) -> HoduResult<CudaStorage> {
    let dtype = input_storage.dtype();
    let device = input_storage.get_device();
    let shape = input_layout.shape();
    let num_els = shape.size();

    let metadata = crate::op_metadatas::scan_metadata(input_layout, dim);

    let kernel_name = format!("hodu_cuda_cumprod_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    let device_id = input_storage.device_id;
    let device_arc = Arc::clone(&input_storage.device);

    macro_rules! call_cumprod {
        ($input:expr, $ty:ty) => {{
            let mut output: CudaSlice<$ty> = device.new_buffer(num_els)?;
            kernels::call_ops_cumprod(
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

    match &input_storage.data {
        CudaStorageData::F8E4M3(input) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::F8E4M3(call_cumprod!(input, float8::F8E4M3)),
        )),
        #[cfg(feature = "f8e5m2")]
        CudaStorageData::F8E5M2(input) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::F8E5M2(call_cumprod!(input, float8::F8E5M2)),
        )),
        CudaStorageData::BF16(input) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::BF16(call_cumprod!(input, half::bf16)),
        )),
        CudaStorageData::F16(input) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::F16(call_cumprod!(input, half::f16)),
        )),
        CudaStorageData::F32(input) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::F32(call_cumprod!(input, f32)),
        )),
        #[cfg(feature = "f64")]
        CudaStorageData::F64(input) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::F64(call_cumprod!(input, f64)),
        )),
        CudaStorageData::U8(input) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::U8(call_cumprod!(input, u8)),
        )),
        #[cfg(feature = "u16")]
        CudaStorageData::U16(input) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::U16(call_cumprod!(input, u16)),
        )),
        CudaStorageData::U32(input) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::U32(call_cumprod!(input, u32)),
        )),
        #[cfg(feature = "u64")]
        CudaStorageData::U64(input) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::U64(call_cumprod!(input, u64)),
        )),
        CudaStorageData::I8(input) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::I8(call_cumprod!(input, i8)),
        )),
        #[cfg(feature = "i16")]
        CudaStorageData::I16(input) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::I16(call_cumprod!(input, i16)),
        )),
        CudaStorageData::I32(input) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::I32(call_cumprod!(input, i32)),
        )),
        #[cfg(feature = "i64")]
        CudaStorageData::I64(input) => Ok(CudaStorage::new(
            device_id,
            device_arc,
            CudaStorageData::I64(call_cumprod!(input, i64)),
        )),
        _ => unreachable!("cumprod does not support bool"),
    }
}
