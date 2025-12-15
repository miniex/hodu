use crate::{error::HoduResult, scalar::Scalar, tensor::Tensor};

impl Tensor {
    pub fn softmax<T: Into<Scalar>>(&self, dim: T) -> HoduResult<Self> {
        let dim_scalar = dim.into();
        let dim_i32 = dim_scalar.to_i32();
        let ndim = self.ndim() as i32;
        let dim_usize = if dim_i32 < 0 {
            (ndim + dim_i32) as u32
        } else {
            dim_i32 as u32
        } as usize;

        // Numerical stability: subtract max
        // softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
        let max_val = self.max(&[dim_usize], true)?;
        let shifted = self.sub(&max_val)?;
        let exp_vals = shifted.exp()?;
        let sum_exp = exp_vals.sum(&[dim_usize], true)?;
        exp_vals.div(&sum_exp)
    }

    pub fn log_softmax<D: Into<Scalar>>(&self, dim: D) -> HoduResult<Self> {
        let dim_scalar = dim.into();
        let dim_i32 = dim_scalar.to_i32();
        let ndim = self.ndim() as i32;
        let dim_usize = if dim_i32 < 0 {
            (ndim + dim_i32) as u32
        } else {
            dim_i32 as u32
        } as usize;

        // Numerical stability: log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
        let max_val = self.max(&[dim_usize], true)?;
        let shifted = self.sub(&max_val)?;
        let exp_vals = shifted.exp()?;
        let sum_exp = exp_vals.sum(&[dim_usize], true)?;
        let log_sum_exp = sum_exp.ln()?;
        shifted.sub(&log_sum_exp)
    }

    pub fn lrn(&self, size: usize, alpha: f64, beta: f64, k: f64) -> HoduResult<Self> {
        let ndim = self.ndim();
        if ndim < 3 {
            return Err(crate::error::HoduError::InvalidArgument(format!(
                "LRN requires at least 3D input, got {}D",
                ndim
            )));
        }
        if size == 0 || size.is_multiple_of(2) {
            return Err(crate::error::HoduError::InvalidArgument(format!(
                "LRN size must be odd and > 0, got {}",
                size
            )));
        }

        let shape = self.shape();
        let shape_dims = shape.dims();
        let dtype = self.dtype();
        let channels = shape_dims[1]; // Channel dimension is always dim 1

        // Square the input
        let x_squared = self.square()?;

        // Reshape to [N, C, -1] for easier processing
        let batch_size = shape_dims[0];
        let spatial_size: usize = shape_dims[2..].iter().product();
        let x_squared_3d = x_squared.reshape([batch_size, channels, spatial_size])?;

        // Transpose to [N, spatial, C] for conv1d along channels
        let x_transposed = x_squared_3d.permute(&[0, 2, 1])?; // [N, spatial, C]
        let x_for_conv = x_transposed.reshape([batch_size * spatial_size, 1, channels])?; // [N*spatial, 1, C]

        // Create a box filter kernel [1, 1, size] filled with 1.0
        let kernel = Tensor::ones([1, 1, size], dtype)?.to_device(self.device())?;

        // Pad channels so output has same size
        let pad = size / 2;

        // Apply conv1d with padding to compute local sum of squares
        // conv1d signature: (weight, stride, padding, dilation)
        let local_sum = x_for_conv.conv1d(&kernel, 1, pad, 1)?; // [N*spatial, 1, C]

        // Reshape back to original spatial dimensions
        let local_sum_3d = local_sum.reshape([batch_size, spatial_size, channels])?; // [N, spatial, C]
        let local_sum_transposed = local_sum_3d.permute(&[0, 2, 1])?; // [N, C, spatial]
        let local_sum_reshaped = local_sum_transposed.reshape(shape_dims)?; // Original shape

        // Apply LRN formula: x / (k + alpha * local_sum)^beta
        let scaled = local_sum_reshaped.mul_scalar(alpha)?;
        let biased = scaled.add_scalar(k)?;
        let denominator = biased.pow_scalar(beta)?;

        self.div(&denominator)
    }
}
