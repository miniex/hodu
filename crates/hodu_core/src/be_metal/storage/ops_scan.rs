use crate::{be::storage::BackendStorageT, be_metal::storage::MetalStorage, error::HoduResult, types::Layout};
use hodu_metal_kernels::{kernels, utils::BufferOffset};

pub fn call_ops_cumsum(input_storage: &MetalStorage, input_layout: &Layout, dim: usize) -> HoduResult<MetalStorage> {
    let dtype = input_storage.dtype();
    let device = input_storage.backend_device();

    let shape = input_layout.shape();
    let num_els = shape.size();

    let output_buffer = device.new_buffer(num_els, dtype, "cumsum_output")?;
    let metadata = crate::op_metadatas::scan_metadata(input_layout, dim);

    let kernel_name = format!("hodu_metal_cumsum_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    let input_offset = BufferOffset::zero_offset(input_storage.buffer());
    let command_buffer = device.command_buffer()?;

    kernels::call_ops_cumsum(
        kernel,
        device.kernels(),
        device.device(),
        &command_buffer,
        input_offset,
        &output_buffer,
        &metadata,
    )?;

    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(MetalStorage::new(output_buffer, device.clone(), num_els, dtype))
}

pub fn call_ops_cumprod(input_storage: &MetalStorage, input_layout: &Layout, dim: usize) -> HoduResult<MetalStorage> {
    let dtype = input_storage.dtype();
    let device = input_storage.backend_device();

    let shape = input_layout.shape();
    let num_els = shape.size();

    let output_buffer = device.new_buffer(num_els, dtype, "cumprod_output")?;
    let metadata = crate::op_metadatas::scan_metadata(input_layout, dim);

    let kernel_name = format!("hodu_metal_cumprod_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    let input_offset = BufferOffset::zero_offset(input_storage.buffer());
    let command_buffer = device.command_buffer()?;

    kernels::call_ops_cumprod(
        kernel,
        device.kernels(),
        device.device(),
        &command_buffer,
        input_offset,
        &output_buffer,
        &metadata,
    )?;

    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(MetalStorage::new(output_buffer, device.clone(), num_els, dtype))
}
