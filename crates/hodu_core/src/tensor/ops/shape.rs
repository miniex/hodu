use crate::{
    error::{HoduError, HoduResult},
    ops::{
        BroadcastParams, FlattenParams, Op, OpParams, PermuteParams, ReshapeParams, ShapeOp, ShapeScalarsOp,
        SliceParams, SqueezeParams, TransposeParams, UnsqueezeParams,
    },
    scalar::Scalar,
    tensor::{create_builder_tensor, from_shared_storage_with, gradient, Tensor},
    types::Shape,
};

impl Tensor {
    pub fn reshape(&self, shape: impl Into<Shape>) -> HoduResult<Self> {
        let shape = shape.into();
        let current_size = self.size();
        let new_size = shape.size();

        if current_size != new_size {
            return Err(HoduError::incompatible_shapes(
                self.shape(),
                shape,
                Op::Shape(ShapeOp::Reshape),
            ));
        }

        if !self.is_contiguous() {
            let contiguous = self.contiguous()?;
            return contiguous.reshape(shape);
        }

        let current_layout = self.layout();
        let new_layout = current_layout.reshape(&shape)?;
        let requires_grad = self.is_requires_grad();

        if crate::snapshot::capture::is_active() {
            let (result_id, result_tensor) = create_builder_tensor(new_layout.clone(), self.dtype(), requires_grad);

            crate::snapshot::capture::capture_operation(
                Op::Shape(ShapeOp::Reshape),
                None,
                vec![self.id()],
                result_id,
                vec![current_layout],
                new_layout,
            )?;

            if requires_grad {
                gradient::record_operation(
                    vec![self.id()],
                    result_id,
                    Op::Shape(ShapeOp::Reshape),
                    OpParams::Reshape(ReshapeParams),
                )?;
            }

            Ok(result_tensor)
        } else {
            let result = from_shared_storage_with(self, new_layout, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                gradient::record_operation(
                    vec![self.id()],
                    result.id(),
                    Op::Shape(ShapeOp::Reshape),
                    OpParams::Reshape(ReshapeParams),
                )?;
            }

            Ok(result)
        }
    }

    pub fn view(&self, shape: impl Into<Shape>) -> HoduResult<Self> {
        self.reshape(shape)
    }

    pub fn flatten(&self) -> HoduResult<Self> {
        if !self.is_contiguous() {
            let contiguous = self.contiguous()?;
            return contiguous.flatten();
        }

        let current_layout = self.layout();
        let new_layout = current_layout.flatten()?;
        let requires_grad = self.is_requires_grad();

        if crate::snapshot::capture::is_active() {
            let (result_id, result_tensor) = create_builder_tensor(new_layout.clone(), self.dtype(), requires_grad);

            crate::snapshot::capture::capture_operation(
                Op::Shape(ShapeOp::Flatten),
                None,
                vec![self.id()],
                result_id,
                vec![current_layout],
                new_layout,
            )?;

            if requires_grad {
                gradient::record_operation(
                    vec![self.id()],
                    result_id,
                    Op::Shape(ShapeOp::Flatten),
                    OpParams::Flatten(FlattenParams),
                )?;
            }

            Ok(result_tensor)
        } else {
            let result = from_shared_storage_with(self, new_layout, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                gradient::record_operation(
                    vec![self.id()],
                    result.id(),
                    Op::Shape(ShapeOp::Flatten),
                    OpParams::Flatten(FlattenParams),
                )?;
            }

            Ok(result)
        }
    }

    pub fn squeeze<D: Into<Scalar> + Copy>(&self, dims: &[D]) -> HoduResult<Self> {
        if !self.is_contiguous() {
            let contiguous = self.contiguous()?;
            return contiguous.squeeze(dims);
        }

        let dims_i32: Vec<i32> = dims
            .iter()
            .map(|&d| {
                let scalar = d.into();
                scalar.to_i32()
            })
            .collect();

        let current_layout = self.layout();
        let new_layout = current_layout.squeeze(&dims_i32)?;
        let requires_grad = self.is_requires_grad();

        if crate::snapshot::capture::is_active() {
            let (result_id, result_tensor) = create_builder_tensor(new_layout.clone(), self.dtype(), requires_grad);

            crate::snapshot::capture::capture_operation(
                Op::Shape(ShapeOp::Squeeze),
                None,
                vec![self.id()],
                result_id,
                vec![current_layout],
                new_layout,
            )?;

            if requires_grad {
                gradient::record_operation(
                    vec![self.id()],
                    result_id,
                    Op::Shape(ShapeOp::Squeeze),
                    OpParams::Squeeze(SqueezeParams),
                )?;
            }

            Ok(result_tensor)
        } else {
            let result = from_shared_storage_with(self, new_layout, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                gradient::record_operation(
                    vec![self.id()],
                    result.id(),
                    Op::Shape(ShapeOp::Squeeze),
                    OpParams::Squeeze(SqueezeParams),
                )?;
            }

            Ok(result)
        }
    }

    pub fn unsqueeze<D: Into<Scalar>>(&self, dim: D) -> HoduResult<Self> {
        if !self.is_contiguous() {
            let contiguous = self.contiguous()?;
            return contiguous.unsqueeze(dim);
        }

        let dim_scalar = dim.into();
        let dim_i32 = dim_scalar.to_i32();

        let current_layout = self.layout();
        let new_layout = current_layout.unsqueeze(dim_i32)?;
        let requires_grad = self.is_requires_grad();

        if crate::snapshot::capture::is_active() {
            let (result_id, result_tensor) = create_builder_tensor(new_layout.clone(), self.dtype(), requires_grad);

            crate::snapshot::capture::capture_operation(
                Op::Shape(ShapeOp::Unsqueeze),
                None,
                vec![self.id()],
                result_id,
                vec![current_layout],
                new_layout,
            )?;

            if requires_grad {
                gradient::record_operation(
                    vec![self.id()],
                    result_id,
                    Op::Shape(ShapeOp::Unsqueeze),
                    OpParams::Unsqueeze(UnsqueezeParams),
                )?;
            }

            Ok(result_tensor)
        } else {
            let result = from_shared_storage_with(self, new_layout, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                gradient::record_operation(
                    vec![self.id()],
                    result.id(),
                    Op::Shape(ShapeOp::Unsqueeze),
                    OpParams::Unsqueeze(UnsqueezeParams),
                )?;
            }

            Ok(result)
        }
    }

    pub fn broadcast(&self, target_shape: impl Into<Shape>) -> HoduResult<Self> {
        let target_shape = target_shape.into();
        if !self.is_contiguous() {
            let contiguous = self.contiguous()?;
            return contiguous.broadcast(&target_shape);
        }

        let current_layout = self.layout();
        let new_layout = current_layout.broadcast_to(&target_shape)?;
        let requires_grad = self.is_requires_grad();

        if crate::snapshot::capture::is_active() {
            let (result_id, result_tensor) = create_builder_tensor(new_layout.clone(), self.dtype(), requires_grad);

            crate::snapshot::capture::capture_operation(
                Op::Shape(ShapeOp::Broadcast),
                None,
                vec![self.id()],
                result_id,
                vec![current_layout],
                new_layout,
            )?;

            if requires_grad {
                gradient::record_operation(
                    vec![self.id()],
                    result_id,
                    Op::Shape(ShapeOp::Broadcast),
                    OpParams::Broadcast(BroadcastParams),
                )?;
            }

            Ok(result_tensor)
        } else {
            let result = from_shared_storage_with(self, new_layout, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                gradient::record_operation(
                    vec![self.id()],
                    result.id(),
                    Op::Shape(ShapeOp::Broadcast),
                    OpParams::Broadcast(BroadcastParams),
                )?;
            }

            Ok(result)
        }
    }

    pub fn broadcast_like(&self, other: &Self) -> HoduResult<Self> {
        self.broadcast(other.shape())
    }

    pub fn broadcast_left(&self, added_dims: &[usize]) -> HoduResult<Self> {
        let current_shape = self.shape();
        let current_dims = current_shape.dims();

        let mut new_dims = added_dims.to_vec();
        new_dims.extend_from_slice(current_dims);

        self.broadcast(Shape::from(new_dims))
    }

    pub fn transpose<D1: Into<Scalar>, D2: Into<Scalar>>(&self, dim1: D1, dim2: D2) -> HoduResult<Self> {
        if !self.is_contiguous() {
            let contiguous = self.contiguous()?;
            return contiguous.transpose(dim1, dim2);
        }

        let dim1_scalar = dim1.into();
        let dim2_scalar = dim2.into();
        let dim1_i32 = dim1_scalar.to_i32();
        let dim2_i32 = dim2_scalar.to_i32();

        let current_layout = self.layout();
        let new_layout = current_layout.transpose(dim1_i32, dim2_i32)?;
        let requires_grad = self.is_requires_grad();

        if crate::snapshot::capture::is_active() {
            let (result_id, result_tensor) = create_builder_tensor(new_layout.clone(), self.dtype(), requires_grad);

            crate::snapshot::capture::capture_operation(
                Op::Shape(ShapeOp::Transpose),
                None,
                vec![self.id()],
                result_id,
                vec![current_layout],
                new_layout,
            )?;

            if requires_grad {
                gradient::record_operation(
                    vec![self.id()],
                    result_id,
                    Op::Shape(ShapeOp::Transpose),
                    OpParams::Transpose(TransposeParams),
                )?;
            }

            Ok(result_tensor)
        } else {
            let result = from_shared_storage_with(self, new_layout, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                gradient::record_operation(
                    vec![self.id()],
                    result.id(),
                    Op::Shape(ShapeOp::Transpose),
                    OpParams::Transpose(TransposeParams),
                )?;
            }

            Ok(result)
        }
    }

    pub fn t(&self) -> HoduResult<Self> {
        self.transpose(-2, -1)
    }

    pub fn permute<A: Into<Scalar> + Copy>(&self, axes: &[A]) -> HoduResult<Self> {
        if !self.is_contiguous() {
            let contiguous = self.contiguous()?;
            return contiguous.permute(axes);
        }

        let axes_i32: Vec<i32> = axes
            .iter()
            .map(|&axis| {
                let axis_scalar = axis.into();
                axis_scalar.to_i32()
            })
            .collect();

        let current_layout = self.layout();
        let new_layout = current_layout.permute(&axes_i32)?;
        let requires_grad = self.is_requires_grad();

        if crate::snapshot::capture::is_active() {
            let (result_id, result_tensor) = create_builder_tensor(new_layout.clone(), self.dtype(), requires_grad);

            crate::snapshot::capture::capture_operation(
                Op::Shape(ShapeOp::Permute),
                None,
                vec![self.id()],
                result_id,
                vec![current_layout],
                new_layout,
            )?;

            if requires_grad {
                gradient::record_operation(
                    vec![self.id()],
                    result_id,
                    Op::Shape(ShapeOp::Permute),
                    OpParams::Permute(PermuteParams),
                )?;
            }

            Ok(result_tensor)
        } else {
            let result = from_shared_storage_with(self, new_layout, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                gradient::record_operation(
                    vec![self.id()],
                    result.id(),
                    Op::Shape(ShapeOp::Permute),
                    OpParams::Permute(PermuteParams),
                )?;
            }

            Ok(result)
        }
    }

    pub fn repeat(&self, repeats: &[usize]) -> HoduResult<Self> {
        let input_shape = self.shape();
        let ndim = input_shape.ndim();

        if repeats.len() != ndim {
            return Err(HoduError::InvalidLayout {
                reason: format!("repeats length {} must match tensor rank {}", repeats.len(), ndim),
            });
        }

        // Use tile operation: repeat each element before moving to next
        // For each dimension, we expand then reshape
        let mut result = self.clone();

        for (dim_idx, &repeat_count) in repeats.iter().enumerate() {
            if repeat_count == 1 {
                continue; // No need to repeat
            }

            let current_shape = result.shape();
            let current_dims = current_shape.dims();

            // Add a new dimension after current dim and broadcast
            let mut expanded_shape = current_dims.to_vec();
            expanded_shape.insert(dim_idx + 1, 1);

            // Reshape to add singleton dimension
            result = result.reshape(&expanded_shape)?;

            // Broadcast the singleton dimension
            let mut broadcast_shape = expanded_shape.clone();
            broadcast_shape[dim_idx + 1] = repeat_count;
            result = result.broadcast(&broadcast_shape)?;

            // Reshape to merge the repeated dimension
            let mut final_shape = current_dims.to_vec();
            final_shape[dim_idx] *= repeat_count;
            result = result.reshape(&final_shape)?;
        }

        Ok(result)
    }

    pub fn slice<D: Into<Scalar>, S: Into<Scalar> + Copy>(
        &self,
        dim: D,
        start: S,
        end: Option<S>,
        step: S,
    ) -> HoduResult<Self> {
        if !self.is_contiguous() {
            let contiguous = self.contiguous()?;
            return contiguous.slice(dim, start, end, step);
        }

        let dim_scalar = dim.into();
        let dim_i32 = dim_scalar.to_i32();

        let start_scalar = start.into();
        let start_i32 = start_scalar.to_i32();

        let end_i32 = end.map(|e| {
            let end_scalar = e.into();
            end_scalar.to_i32()
        });

        let step_scalar = step.into();
        let step_i32 = step_scalar.to_i32();

        let current_layout = self.layout();
        let new_layout = current_layout.slice(dim_i32, start_i32, end_i32, step_i32)?;
        let requires_grad = self.is_requires_grad();

        let end_scalar = Scalar::from(end_i32.unwrap_or(i32::MAX));
        let op_params = OpParams::Slice(SliceParams {
            dim: dim_scalar,
            start: start_scalar,
            end: end_scalar,
            step: step_scalar,
        });

        if crate::snapshot::capture::is_active() {
            let (result_id, result_tensor) = create_builder_tensor(new_layout.clone(), self.dtype(), requires_grad);

            crate::snapshot::capture::capture_operation(
                Op::ShapeScalars(ShapeScalarsOp::Slice),
                Some(op_params.clone()),
                vec![self.id()],
                result_id,
                vec![current_layout],
                new_layout,
            )?;

            if requires_grad {
                gradient::record_operation(
                    vec![self.id()],
                    result_id,
                    Op::ShapeScalars(ShapeScalarsOp::Slice),
                    op_params,
                )?;
            }

            Ok(result_tensor)
        } else {
            let result = from_shared_storage_with(self, new_layout, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                gradient::record_operation(
                    vec![self.id()],
                    result.id(),
                    Op::ShapeScalars(ShapeScalarsOp::Slice),
                    op_params,
                )?;
            }

            Ok(result)
        }
    }

    /// Rearranges data from depth (channel) dimension into spatial blocks.
    /// Also known as PixelShuffle in PyTorch.
    ///
    /// Input shape: [N, C*block_size*block_size, H, W]
    /// Output shape: [N, C, H*block_size, W*block_size]
    ///
    /// # Arguments
    /// * `block_size` - The size of spatial blocks (upscale factor)
    /// * `mode` - Rearrangement mode: "DCR" (default) or "CRD"
    ///   - DCR: depth-column-row order (ONNX default, PyTorch PixelShuffle)
    ///   - CRD: column-row-depth order
    pub fn depth_to_space(&self, block_size: usize, mode: &str) -> HoduResult<Self> {
        let shape = self.shape();
        let dims = shape.dims();

        if dims.len() != 4 {
            return Err(HoduError::InvalidLayout {
                reason: format!("depth_to_space requires 4D input [N,C,H,W], got {}D", dims.len()),
            });
        }

        let (n, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        let r = block_size;

        if c % (r * r) != 0 {
            return Err(HoduError::InvalidLayout {
                reason: format!("channels {} must be divisible by block_size^2 ({})", c, r * r),
            });
        }

        let c_out = c / (r * r);

        match mode {
            "DCR" => {
                // [N, C*r*r, H, W] -> [N, C, r, r, H, W] -> [N, C, H, r, W, r] -> [N, C, H*r, W*r]
                let reshaped = self.reshape([n, c_out, r, r, h, w])?;
                let permuted = reshaped.permute(&[0, 1, 4, 2, 5, 3])?;
                permuted.contiguous()?.reshape([n, c_out, h * r, w * r])
            },
            "CRD" => {
                // [N, r, r, C, H, W] -> [N, C, H, r, W, r] -> [N, C, H*r, W*r]
                let reshaped = self.reshape([n, r, r, c_out, h, w])?;
                let permuted = reshaped.permute(&[0, 3, 4, 1, 5, 2])?;
                permuted.contiguous()?.reshape([n, c_out, h * r, w * r])
            },
            _ => Err(HoduError::InvalidLayout {
                reason: format!("depth_to_space mode must be 'DCR' or 'CRD', got '{}'", mode),
            }),
        }
    }

    /// Rearranges data from spatial blocks into the depth (channel) dimension.
    /// Also known as PixelUnshuffle in PyTorch.
    ///
    /// Input shape: [N, C, H*block_size, W*block_size]
    /// Output shape: [N, C*block_size*block_size, H, W]
    ///
    /// # Arguments
    /// * `block_size` - The size of spatial blocks (downscale factor)
    /// * `mode` - Rearrangement mode: "DCR" (default) or "CRD"
    ///   - DCR: depth-column-row order (ONNX default)
    ///   - CRD: column-row-depth order
    pub fn space_to_depth(&self, block_size: usize, mode: &str) -> HoduResult<Self> {
        let shape = self.shape();
        let dims = shape.dims();

        if dims.len() != 4 {
            return Err(HoduError::InvalidLayout {
                reason: format!("space_to_depth requires 4D input [N,C,H,W], got {}D", dims.len()),
            });
        }

        let (n, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        let r = block_size;

        if h % r != 0 || w % r != 0 {
            return Err(HoduError::InvalidLayout {
                reason: format!("height {} and width {} must be divisible by block_size {}", h, w, r),
            });
        }

        let h_out = h / r;
        let w_out = w / r;
        let c_out = c * r * r;

        match mode {
            "DCR" => {
                // [N, C, H*r, W*r] -> [N, C, H, r, W, r] -> [N, C, r, r, H, W] -> [N, C*r*r, H, W]
                let reshaped = self.reshape([n, c, h_out, r, w_out, r])?;
                let permuted = reshaped.permute(&[0, 1, 3, 5, 2, 4])?;
                permuted.contiguous()?.reshape([n, c_out, h_out, w_out])
            },
            "CRD" => {
                // [N, C, H, r, W, r] -> [N, r, r, C, H, W] -> [N, C*r*r, H, W]
                let reshaped = self.reshape([n, c, h_out, r, w_out, r])?;
                let permuted = reshaped.permute(&[0, 3, 5, 1, 2, 4])?;
                permuted.contiguous()?.reshape([n, c_out, h_out, w_out])
            },
            _ => Err(HoduError::InvalidLayout {
                reason: format!("space_to_depth mode must be 'DCR' or 'CRD', got '{}'", mode),
            }),
        }
    }
}
