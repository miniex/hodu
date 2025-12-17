use crate::{
    be::{device::BackendDeviceT, storage::BackendStorageT},
    be_cpu::{device::CpuDevice, storage::CpuStorage},
    error::HoduResult,
    types::Layout,
};
use core::ffi::c_void;

pub fn call_ops_cumsum(storage: &CpuStorage, layout: &Layout, dim: usize) -> HoduResult<CpuStorage> {
    let dtype = storage.dtype();
    let shape = layout.shape();
    let num_els = shape.size();

    let kernel_name = format!("hodu_cpu_cumsum_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = hodu_cpu_kernels::macros::Kernel(kernel_name_static);

    let metadata = crate::op_metadatas::scan_metadata(layout, dim);

    let mut output = CpuDevice::allocate(num_els, dtype)?;

    macro_rules! call_cumsum {
        ($input_data:expr, $out_data:expr) => {{
            let input_ptr = $input_data.as_ptr() as *const c_void;
            let out_ptr = $out_data.as_mut_ptr() as *mut c_void;
            hodu_cpu_kernels::call_ops_cumsum(kernel, input_ptr, out_ptr, &metadata)?;
        }};
    }

    match (storage, &mut output) {
        (CpuStorage::F8E4M3(input), CpuStorage::F8E4M3(out)) => call_cumsum!(input, out),
        #[cfg(feature = "f8e5m2")]
        (CpuStorage::F8E5M2(input), CpuStorage::F8E5M2(out)) => call_cumsum!(input, out),
        (CpuStorage::BF16(input), CpuStorage::BF16(out)) => call_cumsum!(input, out),
        (CpuStorage::F16(input), CpuStorage::F16(out)) => call_cumsum!(input, out),
        (CpuStorage::F32(input), CpuStorage::F32(out)) => call_cumsum!(input, out),
        #[cfg(feature = "f64")]
        (CpuStorage::F64(input), CpuStorage::F64(out)) => call_cumsum!(input, out),
        (CpuStorage::U8(input), CpuStorage::U8(out)) => call_cumsum!(input, out),
        #[cfg(feature = "u16")]
        (CpuStorage::U16(input), CpuStorage::U16(out)) => call_cumsum!(input, out),
        (CpuStorage::U32(input), CpuStorage::U32(out)) => call_cumsum!(input, out),
        #[cfg(feature = "u64")]
        (CpuStorage::U64(input), CpuStorage::U64(out)) => call_cumsum!(input, out),
        (CpuStorage::I8(input), CpuStorage::I8(out)) => call_cumsum!(input, out),
        #[cfg(feature = "i16")]
        (CpuStorage::I16(input), CpuStorage::I16(out)) => call_cumsum!(input, out),
        (CpuStorage::I32(input), CpuStorage::I32(out)) => call_cumsum!(input, out),
        #[cfg(feature = "i64")]
        (CpuStorage::I64(input), CpuStorage::I64(out)) => call_cumsum!(input, out),
        _ => unreachable!("dtype mismatch or unsupported dtype for cumsum"),
    }

    Ok(output)
}

pub fn call_ops_cumprod(storage: &CpuStorage, layout: &Layout, dim: usize) -> HoduResult<CpuStorage> {
    let dtype = storage.dtype();
    let shape = layout.shape();
    let num_els = shape.size();

    let kernel_name = format!("hodu_cpu_cumprod_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = hodu_cpu_kernels::macros::Kernel(kernel_name_static);

    let metadata = crate::op_metadatas::scan_metadata(layout, dim);

    let mut output = CpuDevice::allocate(num_els, dtype)?;

    macro_rules! call_cumprod {
        ($input_data:expr, $out_data:expr) => {{
            let input_ptr = $input_data.as_ptr() as *const c_void;
            let out_ptr = $out_data.as_mut_ptr() as *mut c_void;
            hodu_cpu_kernels::call_ops_cumprod(kernel, input_ptr, out_ptr, &metadata)?;
        }};
    }

    match (storage, &mut output) {
        (CpuStorage::F8E4M3(input), CpuStorage::F8E4M3(out)) => call_cumprod!(input, out),
        #[cfg(feature = "f8e5m2")]
        (CpuStorage::F8E5M2(input), CpuStorage::F8E5M2(out)) => call_cumprod!(input, out),
        (CpuStorage::BF16(input), CpuStorage::BF16(out)) => call_cumprod!(input, out),
        (CpuStorage::F16(input), CpuStorage::F16(out)) => call_cumprod!(input, out),
        (CpuStorage::F32(input), CpuStorage::F32(out)) => call_cumprod!(input, out),
        #[cfg(feature = "f64")]
        (CpuStorage::F64(input), CpuStorage::F64(out)) => call_cumprod!(input, out),
        (CpuStorage::U8(input), CpuStorage::U8(out)) => call_cumprod!(input, out),
        #[cfg(feature = "u16")]
        (CpuStorage::U16(input), CpuStorage::U16(out)) => call_cumprod!(input, out),
        (CpuStorage::U32(input), CpuStorage::U32(out)) => call_cumprod!(input, out),
        #[cfg(feature = "u64")]
        (CpuStorage::U64(input), CpuStorage::U64(out)) => call_cumprod!(input, out),
        (CpuStorage::I8(input), CpuStorage::I8(out)) => call_cumprod!(input, out),
        #[cfg(feature = "i16")]
        (CpuStorage::I16(input), CpuStorage::I16(out)) => call_cumprod!(input, out),
        (CpuStorage::I32(input), CpuStorage::I32(out)) => call_cumprod!(input, out),
        #[cfg(feature = "i64")]
        (CpuStorage::I64(input), CpuStorage::I64(out)) => call_cumprod!(input, out),
        _ => unreachable!("dtype mismatch or unsupported dtype for cumprod"),
    }

    Ok(output)
}
