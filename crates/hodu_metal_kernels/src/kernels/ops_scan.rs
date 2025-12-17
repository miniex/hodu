//! Scan operations
//!
//! This module provides scan operations that compute prefix operations:
//! - cumsum: Cumulative sum along a dimension
//! - cumprod: Cumulative product along a dimension

use crate::{
    error::MetalKernelError,
    kernel::Kernels,
    kernels::macros::ops,
    metal::{Buffer, ComputeCommandEncoder, Device},
    set_params,
    source::Source,
    utils::{linear_split, BufferOffset, EncoderProvider},
};
use objc2_metal::MTLResourceUsage;

ops!(cumsum, cumprod);

/// Executes a cumsum operation.
///
/// Computes cumulative sum along the specified dimension.
///
/// # Arguments
/// * `kernel` - Cumsum kernel (e.g., cumsum::F32)
/// * `kernels` - Kernel cache
/// * `device` - Metal device to execute on
/// * `ep` - Encoder provider (command buffer)
/// * `input` - Input tensor buffer
/// * `output` - Output buffer (cumsum result)
/// * `metadata` - Metadata describing tensor shape and scan dimension
///
/// # Metadata Layout
/// - `metadata[0]`: num_els (total number of elements)
/// - `metadata[1]`: num_dims (number of dimensions)
/// - `metadata[2..2+num_dims]`: shape
/// - `metadata[2+num_dims..2+2*num_dims]`: strides
/// - `metadata[2+2*num_dims]`: offset
/// - `metadata[3+2*num_dims]`: dim (dimension to scan along)
pub fn call_ops_cumsum(
    kernel: Kernel,
    kernels: &Kernels,
    device: &Device,
    ep: impl EncoderProvider,
    input: BufferOffset,
    output: &Buffer,
    metadata: &[usize],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Scan, kernel.0)?;

    let num_els = metadata[0];

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (&input, output, metadata));

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}

/// Executes a cumprod operation.
///
/// Computes cumulative product along the specified dimension.
///
/// # Arguments
/// * `kernel` - Cumprod kernel (e.g., cumprod::F32)
/// * `kernels` - Kernel cache
/// * `device` - Metal device to execute on
/// * `ep` - Encoder provider (command buffer)
/// * `input` - Input tensor buffer
/// * `output` - Output buffer (cumprod result)
/// * `metadata` - Metadata describing tensor shape and scan dimension
///
/// # Metadata Layout
/// - `metadata[0]`: num_els (total number of elements)
/// - `metadata[1]`: num_dims (number of dimensions)
/// - `metadata[2..2+num_dims]`: shape
/// - `metadata[2+num_dims..2+2*num_dims]`: strides
/// - `metadata[2+2*num_dims]`: offset
/// - `metadata[3+2*num_dims]`: dim (dimension to scan along)
pub fn call_ops_cumprod(
    kernel: Kernel,
    kernels: &Kernels,
    device: &Device,
    ep: impl EncoderProvider,
    input: BufferOffset,
    output: &Buffer,
    metadata: &[usize],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Scan, kernel.0)?;

    let num_els = metadata[0];

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (&input, output, metadata));

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}
