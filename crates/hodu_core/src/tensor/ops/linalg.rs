use crate::{
    error::HoduResult,
    ops::{DetParams, InvParams, LinalgOp, Op, OpParams, TraceParams},
    scalar::Scalar,
    tensor::{create_builder_tensor, from_storage_with_context, gradient, Tensor},
    types::{Layout, Shape},
    utils::valid::{validate_dtype_for_device, validate_dtype_for_op, validate_requires_grad_for_op},
};

impl Tensor {
    /// Compute the determinant of a square matrix.
    ///
    /// # Input
    /// - Square matrix `[..., N, N]` (supports batched input)
    ///
    /// # Output
    /// - Scalar `[...]` (batch dimensions preserved)
    ///
    /// # Example
    /// ```ignore
    /// let matrix = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    /// let det = matrix.det()?; // Returns -2.0
    /// ```
    pub fn det(&self) -> HoduResult<Self> {
        let op = Op::Linalg(LinalgOp::Det);

        // Validate dtype for device and operation
        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), op.clone())?;

        let shape = self.shape();
        let ndim = shape.ndim();

        // Validate shape - need at least 2D and square matrix
        if ndim < 2 {
            return Err(crate::error::HoduError::InvalidArgument(
                "det requires at least 2D tensor".to_string(),
            ));
        }

        let n = shape.dims()[ndim - 1];
        let m = shape.dims()[ndim - 2];

        if n != m {
            return Err(crate::error::HoduError::InvalidArgument(format!(
                "det requires square matrix, got {}×{}",
                m, n
            )));
        }

        // Compute output shape (batch dimensions only)
        let output_shape = if ndim == 2 {
            Shape::new(&[1])
        } else {
            Shape::new(&shape.dims()[..ndim - 2])
        };

        let result_layout = Layout::from_shape(&output_shape);
        let self_layout = self.layout();
        let validate_requires_grad = validate_requires_grad_for_op(op.clone());

        if crate::snapshot::capture::is_active() {
            let requires_grad = self.is_requires_grad() && validate_requires_grad;
            let (result_id, result_tensor) = create_builder_tensor(result_layout.clone(), self.dtype(), requires_grad);

            crate::snapshot::capture::capture_operation(
                op.clone(),
                Some(OpParams::Det(DetParams)),
                vec![self.id()],
                result_id,
                vec![self_layout],
                result_layout,
            )?;

            if requires_grad {
                gradient::record_operation(vec![self.id()], result_id, op, OpParams::Det(DetParams))?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| storage.call_ops_det(&self_layout))?;

            let requires_grad = self.is_requires_grad() && validate_requires_grad;
            let result = from_storage_with_context(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                gradient::record_operation(vec![self.id()], result.id(), op, OpParams::Det(DetParams))?;
            }

            Ok(result)
        }
    }

    /// Compute the inverse of a square matrix.
    ///
    /// # Input
    /// - Square matrix `[..., N, N]` (supports batched input)
    ///
    /// # Output
    /// - Inverse matrix `[..., N, N]` (same shape as input)
    ///
    /// # Example
    /// ```ignore
    /// let matrix = Tensor::from_slice(&[4.0, 7.0, 2.0, 6.0], &[2, 2])?;
    /// let inv = matrix.inv()?; // Returns [[0.6, -0.7], [-0.2, 0.4]]
    /// ```
    ///
    /// # Notes
    /// - For singular matrices, the result will contain inf/nan values.
    /// - Uses Gauss-Jordan elimination with partial pivoting for numerical stability.
    pub fn inv(&self) -> HoduResult<Self> {
        let op = Op::Linalg(LinalgOp::Inv);

        // Validate dtype for device and operation
        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), op.clone())?;

        let shape = self.shape();
        let ndim = shape.ndim();

        // Validate shape - need at least 2D and square matrix
        if ndim < 2 {
            return Err(crate::error::HoduError::InvalidArgument(
                "inv requires at least 2D tensor".to_string(),
            ));
        }

        let n = shape.dims()[ndim - 1];
        let m = shape.dims()[ndim - 2];

        if n != m {
            return Err(crate::error::HoduError::InvalidArgument(format!(
                "inv requires square matrix, got {}×{}",
                m, n
            )));
        }

        // Output shape is same as input
        let output_shape = shape.clone();
        let result_layout = Layout::from_shape(&output_shape);
        let self_layout = self.layout();
        let validate_requires_grad = validate_requires_grad_for_op(op.clone());

        if crate::snapshot::capture::is_active() {
            let requires_grad = self.is_requires_grad() && validate_requires_grad;
            let (result_id, result_tensor) = create_builder_tensor(result_layout.clone(), self.dtype(), requires_grad);

            crate::snapshot::capture::capture_operation(
                op.clone(),
                Some(OpParams::Inv(InvParams)),
                vec![self.id()],
                result_id,
                vec![self_layout],
                result_layout,
            )?;

            if requires_grad {
                gradient::record_operation(vec![self.id()], result_id, op, OpParams::Inv(InvParams))?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| storage.call_ops_inv(&self_layout))?;

            let requires_grad = self.is_requires_grad() && validate_requires_grad;
            let result = from_storage_with_context(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                gradient::record_operation(vec![self.id()], result.id(), op, OpParams::Inv(InvParams))?;
            }

            Ok(result)
        }
    }

    /// Compute the trace of a square matrix (sum of diagonal elements).
    ///
    /// # Input
    /// - Square matrix `[..., N, N]` (supports batched input)
    ///
    /// # Output
    /// - Scalar `[...]` (batch dimensions preserved)
    ///
    /// # Example
    /// ```ignore
    /// let matrix = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    /// let tr = matrix.trace()?; // Returns 5.0 (1.0 + 4.0)
    /// ```
    pub fn trace(&self) -> HoduResult<Self> {
        let op = Op::Linalg(LinalgOp::Trace);

        // Validate dtype for device and operation
        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), op.clone())?;

        let shape = self.shape();
        let ndim = shape.ndim();

        // Validate shape - need at least 2D and square matrix
        if ndim < 2 {
            return Err(crate::error::HoduError::InvalidArgument(
                "trace requires at least 2D tensor".to_string(),
            ));
        }

        let n = shape.dims()[ndim - 1];
        let m = shape.dims()[ndim - 2];

        if n != m {
            return Err(crate::error::HoduError::InvalidArgument(format!(
                "trace requires square matrix, got {}×{}",
                m, n
            )));
        }

        // Compute output shape (batch dimensions only)
        let output_shape = if ndim == 2 {
            Shape::new(&[1])
        } else {
            Shape::new(&shape.dims()[..ndim - 2])
        };

        let result_layout = Layout::from_shape(&output_shape);
        let self_layout = self.layout();
        let validate_requires_grad = validate_requires_grad_for_op(op.clone());

        if crate::snapshot::capture::is_active() {
            let requires_grad = self.is_requires_grad() && validate_requires_grad;
            let (result_id, result_tensor) = create_builder_tensor(result_layout.clone(), self.dtype(), requires_grad);

            crate::snapshot::capture::capture_operation(
                op.clone(),
                Some(OpParams::Trace(TraceParams)),
                vec![self.id()],
                result_id,
                vec![self_layout],
                result_layout,
            )?;

            if requires_grad {
                gradient::record_operation(vec![self.id()], result_id, op, OpParams::Trace(TraceParams))?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| storage.call_ops_trace(&self_layout))?;

            let requires_grad = self.is_requires_grad() && validate_requires_grad;
            let result = from_storage_with_context(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                gradient::record_operation(vec![self.id()], result.id(), op, OpParams::Trace(TraceParams))?;
            }

            Ok(result)
        }
    }

    /// Solve a linear system Ax = b.
    ///
    /// # Input
    /// - `self` (A): Square matrix `[..., N, N]` (coefficient matrix)
    /// - `b`: Vector or matrix `[..., N]` or `[..., N, K]` (right-hand side)
    ///
    /// # Output
    /// - Solution `x` with shape `[..., N]` or `[..., N, K]`
    ///
    /// # Example
    /// ```ignore
    /// let a = Tensor::from_slice(&[2.0, 1.0, 1.0, 3.0], &[2, 2])?;
    /// let b = Tensor::from_slice(&[4.0, 7.0], &[2])?;
    /// let x = a.solve(&b)?; // Solves Ax = b
    /// ```
    ///
    /// # Notes
    /// - Internally uses `inv(A) @ b` for simplicity.
    /// - For singular matrices, the result will contain inf/nan values.
    pub fn solve(&self, b: &Tensor) -> HoduResult<Self> {
        // solve(A, b) = inv(A) @ b
        let inv_a = self.inv()?;

        // For 1D b, unsqueeze -> matmul -> squeeze
        if b.ndim() == 1 {
            let b_col = b.unsqueeze(-1)?;
            let x_col = inv_a.matmul(&b_col)?;
            x_col.squeeze(&[-1])
        } else {
            inv_a.matmul(b)
        }
    }

    /// Returns the lower triangular part of the matrix, zeroing out elements above the k-th diagonal.
    ///
    /// # Input
    /// - Matrix `[..., N, M]` (supports batched input)
    /// - `diagonal`: The diagonal above which to zero elements.
    ///   - `k = 0` (default): main diagonal
    ///   - `k > 0`: above main diagonal
    ///   - `k < 0`: below main diagonal
    ///
    /// # Output
    /// - Lower triangular matrix `[..., N, M]` (same shape as input)
    ///
    /// # Example
    /// ```ignore
    /// let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3])?;
    /// let lower = x.tril(0)?;
    /// // [[1, 0, 0],
    /// //  [4, 5, 0],
    /// //  [7, 8, 9]]
    /// ```
    pub fn tril<D: Into<Scalar>>(&self, diagonal: D) -> HoduResult<Self> {
        let diagonal = diagonal.into().to_i32();
        let shape = self.shape();
        let ndim = shape.ndim();

        if ndim < 2 {
            return Err(crate::error::HoduError::InvalidArgument(
                "tril requires at least 2D tensor".to_string(),
            ));
        }

        let n = shape.dims()[ndim - 2];
        let m = shape.dims()[ndim - 1];

        // Create row indices [0, 1, 2, ...] with shape [N, 1]
        let row_indices = Self::arange(0i32, n as i32, 1i32)?
            .reshape([n, 1])?
            .to_device(self.device())?;

        // Create col indices [0, 1, 2, ...] with shape [1, M]
        let col_indices = Self::arange(0i32, m as i32, 1i32)?
            .reshape([1, m])?
            .to_device(self.device())?;

        // tril mask: row >= col - diagonal → row + diagonal >= col
        let row_plus_diag = row_indices.add_scalar(diagonal)?;
        let mask = row_plus_diag.ge(&col_indices)?;

        // Convert mask to input dtype and multiply
        let mask_typed = mask.to_dtype(self.dtype())?;
        self.mul(&mask_typed)
    }

    /// Returns the upper triangular part of the matrix, zeroing out elements below the k-th diagonal.
    ///
    /// # Input
    /// - Matrix `[..., N, M]` (supports batched input)
    /// - `diagonal`: The diagonal below which to zero elements.
    ///   - `k = 0` (default): main diagonal
    ///   - `k > 0`: above main diagonal
    ///   - `k < 0`: below main diagonal
    ///
    /// # Output
    /// - Upper triangular matrix `[..., N, M]` (same shape as input)
    ///
    /// # Example
    /// ```ignore
    /// let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3])?;
    /// let upper = x.triu(0)?;
    /// // [[1, 2, 3],
    /// //  [0, 5, 6],
    /// //  [0, 0, 9]]
    /// ```
    pub fn triu<D: Into<Scalar>>(&self, diagonal: D) -> HoduResult<Self> {
        let diagonal = diagonal.into().to_i32();
        let shape = self.shape();
        let ndim = shape.ndim();

        if ndim < 2 {
            return Err(crate::error::HoduError::InvalidArgument(
                "triu requires at least 2D tensor".to_string(),
            ));
        }

        let n = shape.dims()[ndim - 2];
        let m = shape.dims()[ndim - 1];

        // Create row indices [0, 1, 2, ...] with shape [N, 1]
        let row_indices = Self::arange(0i32, n as i32, 1i32)?
            .reshape([n, 1])?
            .to_device(self.device())?;

        // Create col indices [0, 1, 2, ...] with shape [1, M]
        let col_indices = Self::arange(0i32, m as i32, 1i32)?
            .reshape([1, m])?
            .to_device(self.device())?;

        // triu mask: col >= row + diagonal
        let row_plus_diag = row_indices.add_scalar(diagonal)?;
        let mask = col_indices.ge(&row_plus_diag)?;

        // Convert mask to input dtype and multiply
        let mask_typed = mask.to_dtype(self.dtype())?;
        self.mul(&mask_typed)
    }
}
