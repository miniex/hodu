use super::VjpCompute;
use crate::{
    error::{HoduError, HoduResult},
    op_params::{OpParams, ScanParams},
    ops::ScanOp,
    tensor::{tensor_from_id, TensorId},
};

impl VjpCompute for ScanOp {
    fn compute_vjp(
        &self,
        _inputs: &[TensorId],
        _output: TensorId,
        grad_output: TensorId,
        op_params: &OpParams,
    ) -> HoduResult<Vec<TensorId>> {
        match self {
            ScanOp::CumSum => {
                let OpParams::Scan(ScanParams { dim }) = op_params else {
                    return Err(HoduError::VjpFunctionNotFound("CumSum requires ScanParams".to_string()));
                };

                let grad_tensor = tensor_from_id(grad_output);

                // Gradient of cumsum(x, dim) is flip(cumsum(flip(grad, dim), dim), dim)
                // This is equivalent to reverse cumsum
                let flipped = grad_tensor.flip(&[*dim])?;
                let cumsum_flipped = flipped.cumsum(*dim)?;
                let grad_input = cumsum_flipped.flip(&[*dim])?;

                Ok(vec![grad_input.id()])
            },
            ScanOp::CumProd => {
                let OpParams::Scan(ScanParams { dim }) = op_params else {
                    return Err(HoduError::VjpFunctionNotFound(
                        "CumProd requires ScanParams".to_string(),
                    ));
                };

                let grad_tensor = tensor_from_id(grad_output);
                let output_tensor = tensor_from_id(_output);
                let input_tensor = tensor_from_id(_inputs[0]);

                // Gradient of cumprod(x, dim):
                // grad_x = flip(cumsum(flip(grad_y * y, dim), dim), dim) / x
                let grad_times_output = grad_tensor.mul(&output_tensor)?;
                let flipped = grad_times_output.flip(&[*dim])?;
                let cumsum_flipped = flipped.cumsum(*dim)?;
                let reverse_cumsum = cumsum_flipped.flip(&[*dim])?;
                let grad_input = reverse_cumsum.div(&input_tensor)?;

                Ok(vec![grad_input.id()])
            },
        }
    }
}
