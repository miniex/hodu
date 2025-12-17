//! Scan operations
//!
//! This module provides scan operations that compute prefix operations:
//! - cumsum: Cumulative sum along a dimension
//! - cumprod: Cumulative product along a dimension
//!
//! All operations support multiple data types.

use crate::{error::Result, kernels::macros::ops};
use core::ffi::c_void;

ops!(cumsum, cumprod);

/// Call cumsum operation by kernel name
///
/// Computes cumulative sum along the specified dimension.
///
/// # Metadata layout
/// - metadata[0]: num_els (total number of elements)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: shape
/// - metadata[2+num_dims..2+2*num_dims]: strides
/// - metadata[2+2*num_dims]: offset
/// - metadata[3+2*num_dims]: dim (dimension to scan along)
///
/// # Safety
/// - `input` must point to valid tensor data of the appropriate type
/// - `output` must point to a valid output buffer with sufficient capacity
/// - Metadata must accurately describe the tensor layout
pub fn call_ops_cumsum(
    kernel_name: crate::kernels::macros::Kernel,
    input: *const c_void,
    output: *mut c_void,
    metadata: &[usize],
) -> Result<()> {
    unsafe {
        dispatch_cumsum(kernel_name.0, input, output, metadata.as_ptr());
    }
    Ok(())
}

macro_rules! declare_and_dispatch_cumsum {
    ($($dtype:ident),* $(,)?) => {
        paste::paste! {
            extern "C" {
                $(
                    fn [<hodu_cpu_cumsum_ $dtype>](
                        input: *const c_void,
                        output: *mut c_void,
                        metadata: *const usize,
                    );
                )*
            }

            unsafe fn dispatch_cumsum(
                kernel_name: &str,
                input: *const c_void,
                output: *mut c_void,
                metadata: *const usize,
            ) {
                match kernel_name {
                    $(
                        concat!("hodu_cpu_cumsum_", stringify!($dtype)) => {
                            [<hodu_cpu_cumsum_ $dtype>](input, output, metadata)
                        }
                    )*
                    _ => panic!("Unknown kernel: {}", kernel_name),
                }
            }
        }
    };
}

declare_and_dispatch_cumsum!(f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64);

/// Call cumprod operation by kernel name
///
/// Computes cumulative product along the specified dimension.
///
/// # Metadata layout
/// - metadata[0]: num_els (total number of elements)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: shape
/// - metadata[2+num_dims..2+2*num_dims]: strides
/// - metadata[2+2*num_dims]: offset
/// - metadata[3+2*num_dims]: dim (dimension to scan along)
///
/// # Safety
/// - `input` must point to valid tensor data of the appropriate type
/// - `output` must point to a valid output buffer with sufficient capacity
/// - Metadata must accurately describe the tensor layout
pub fn call_ops_cumprod(
    kernel_name: crate::kernels::macros::Kernel,
    input: *const c_void,
    output: *mut c_void,
    metadata: &[usize],
) -> Result<()> {
    unsafe {
        dispatch_cumprod(kernel_name.0, input, output, metadata.as_ptr());
    }
    Ok(())
}

macro_rules! declare_and_dispatch_cumprod {
    ($($dtype:ident),* $(,)?) => {
        paste::paste! {
            extern "C" {
                $(
                    fn [<hodu_cpu_cumprod_ $dtype>](
                        input: *const c_void,
                        output: *mut c_void,
                        metadata: *const usize,
                    );
                )*
            }

            unsafe fn dispatch_cumprod(
                kernel_name: &str,
                input: *const c_void,
                output: *mut c_void,
                metadata: *const usize,
            ) {
                match kernel_name {
                    $(
                        concat!("hodu_cpu_cumprod_", stringify!($dtype)) => {
                            [<hodu_cpu_cumprod_ $dtype>](input, output, metadata)
                        }
                    )*
                    _ => panic!("Unknown kernel: {}", kernel_name),
                }
            }
        }
    };
}

declare_and_dispatch_cumprod!(f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64);
