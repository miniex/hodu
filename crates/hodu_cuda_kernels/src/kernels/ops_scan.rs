//! Scan operations
//!
//! This module provides scan operations that compute prefix operations:
//! - cumsum: Cumulative sum along a dimension
//! - cumprod: Cumulative product along a dimension

use crate::{
    cuda::*,
    error::{CudaKernelError, Result},
    kernel::Kernels,
    kernels::macros::ops,
    source::Source,
};

ops!(cumsum, cumprod);

/// Execute a cumsum operation
///
/// Computes cumulative sum along the specified dimension.
///
/// # Arguments
/// * `kernel` - The cumsum kernel (e.g., cumsum::F32)
/// * `kernels` - Kernel cache
/// * `context` - CUDA context
/// * `input` - Input tensor device slice
/// * `output` - Output tensor device slice (cumsum result)
/// * `metadata` - Metadata describing tensor shape and scan dimension
///
/// # Metadata layout
/// - metadata[0]: num_els (total number of elements)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: shape
/// - metadata[2+num_dims..2+2*num_dims]: strides
/// - metadata[2+2*num_dims]: offset
/// - metadata[3+2*num_dims]: dim (dimension to scan along)
pub fn call_ops_cumsum<T>(
    kernel: crate::kernels::macros::Kernel,
    kernels: &Kernels,
    context: &Arc<CudaContext>,
    input: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
    metadata: &[usize],
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr,
{
    let func = kernels.load_function(context, Source::OpsScan, kernel.0)?;

    let num_dims = metadata[1];
    let dim = metadata[3 + 2 * num_dims];
    let shape = &metadata[2..2 + num_dims];

    let mut outer_size = 1usize;
    for d in 0..dim {
        outer_size *= shape[d];
    }
    let mut inner_size = 1usize;
    for d in (dim + 1)..num_dims {
        inner_size *= shape[d];
    }
    let num_scans = outer_size * inner_size;

    let block_size = 256u32;
    let grid_size = (num_scans as u32).div_ceil(block_size).max(1);

    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let stream = context.default_stream();
    let metadata_dev = stream
        .memcpy_stod(metadata)
        .map_err(|e| CudaKernelError::MemoryError(format!("Failed to copy metadata: {:?}", e)))?;

    unsafe {
        func.launch(&stream, cfg, |args| {
            args.arg(input).arg(output).arg(&metadata_dev);
        })
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to launch kernel: {:?}", e)))?;
    }

    Ok(())
}

/// Execute a cumprod operation
///
/// Computes cumulative product along the specified dimension.
///
/// # Arguments
/// * `kernel` - The cumprod kernel (e.g., cumprod::F32)
/// * `kernels` - Kernel cache
/// * `context` - CUDA context
/// * `input` - Input tensor device slice
/// * `output` - Output tensor device slice (cumprod result)
/// * `metadata` - Metadata describing tensor shape and scan dimension
///
/// # Metadata layout
/// - metadata[0]: num_els (total number of elements)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: shape
/// - metadata[2+num_dims..2+2*num_dims]: strides
/// - metadata[2+2*num_dims]: offset
/// - metadata[3+2*num_dims]: dim (dimension to scan along)
pub fn call_ops_cumprod<T>(
    kernel: crate::kernels::macros::Kernel,
    kernels: &Kernels,
    context: &Arc<CudaContext>,
    input: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
    metadata: &[usize],
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr,
{
    let func = kernels.load_function(context, Source::OpsScan, kernel.0)?;

    let num_dims = metadata[1];
    let dim = metadata[3 + 2 * num_dims];
    let shape = &metadata[2..2 + num_dims];

    let mut outer_size = 1usize;
    for d in 0..dim {
        outer_size *= shape[d];
    }
    let mut inner_size = 1usize;
    for d in (dim + 1)..num_dims {
        inner_size *= shape[d];
    }
    let num_scans = outer_size * inner_size;

    let block_size = 256u32;
    let grid_size = (num_scans as u32).div_ceil(block_size).max(1);

    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let stream = context.default_stream();
    let metadata_dev = stream
        .memcpy_stod(metadata)
        .map_err(|e| CudaKernelError::MemoryError(format!("Failed to copy metadata: {:?}", e)))?;

    unsafe {
        func.launch(&stream, cfg, |args| {
            args.arg(input).arg(output).arg(&metadata_dev);
        })
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to launch kernel: {:?}", e)))?;
    }

    Ok(())
}
