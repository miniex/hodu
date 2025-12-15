use crate::{
    error::HoduResult,
    ops::{
        CompressParams, GatherParams, IndexPutParams, IndexSelectParams, IndexingOp, NonzeroParams, OnehotoParams, Op,
        OpParams, ScatterAddParams, ScatterMaxParams, ScatterMinParams, ScatterParams, UniqueParams,
    },
    scalar::Scalar,
    tensor::{create_builder_tensor, from_storage_with_context, gradient, Tensor},
    types::{DType, Dim, DynamicDimId, Layout, Shape, SymbolicLayout, SymbolicShape},
    utils::valid::{
        validate_dtype_for_device, validate_dtype_for_op, validate_indices_dtype, validate_requires_grad_for_op,
        validate_same_device, validate_same_dtype,
    },
};

impl Tensor {
    pub fn index_select<D: Into<Scalar>>(&self, dim: D, indices: &Self) -> HoduResult<Self> {
        let dim_scalar = dim.into();
        let dim_i32 = dim_scalar.to_i32();
        let ndim = self.ndim() as i32;
        let dim_usize = if dim_i32 < 0 {
            (ndim + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, indices], Op::Indexing(IndexingOp::IndexSelect))?;
        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), Op::Indexing(IndexingOp::IndexSelect))?;
        validate_indices_dtype(indices, Op::Indexing(IndexingOp::IndexSelect))?;
        let validate_requires_grad = validate_requires_grad_for_op(Op::Indexing(IndexingOp::IndexSelect));

        let shape = self.shape();
        let shape_dims = shape.dims();
        let indices_size = indices.size();
        let mut output_dims = shape_dims.to_vec();
        output_dims[dim_usize] = indices_size;

        let result_layout = Layout::from_shape(&crate::types::Shape::from(output_dims));
        let self_layout = self.layout();
        let indices_layout = indices.layout();

        if crate::snapshot::capture::is_active() {
            let requires_grad = self.is_requires_grad() && validate_requires_grad;
            let (result_id, result_tensor) = create_builder_tensor(result_layout.clone(), self.dtype(), requires_grad);

            let op_params = OpParams::IndexSelect(IndexSelectParams { dim: dim_scalar });

            crate::snapshot::capture::capture_operation(
                Op::Indexing(IndexingOp::IndexSelect),
                Some(op_params.clone()),
                vec![self.id(), indices.id()],
                result_id,
                vec![self_layout, indices_layout],
                result_layout,
            )?;

            if requires_grad {
                gradient::record_operation(
                    vec![self.id(), indices.id()],
                    result_id,
                    Op::Indexing(IndexingOp::IndexSelect),
                    op_params,
                )?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| {
                indices.with_storage(|indices_storage| {
                    storage.call_ops_index_select(
                        &self_layout,
                        indices_storage,
                        &indices_layout,
                        dim_usize,
                        Op::Indexing(IndexingOp::IndexSelect),
                    )
                })
            })?;

            let requires_grad = self.is_requires_grad() && validate_requires_grad;
            let result = from_storage_with_context(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                gradient::record_operation(
                    vec![self.id(), indices.id()],
                    result.id(),
                    Op::Indexing(IndexingOp::IndexSelect),
                    OpParams::IndexSelect(IndexSelectParams { dim: dim_scalar }),
                )?;
            }

            Ok(result)
        }
    }

    pub fn index_put<D: Into<Scalar>>(&self, dim: D, indices: &Self, values: &Self) -> HoduResult<Self> {
        let dim_scalar = dim.into();
        let dim_i32 = dim_scalar.to_i32();
        let ndim = self.ndim() as i32;
        let dim_usize = if dim_i32 < 0 {
            (ndim + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, indices, values], Op::Indexing(IndexingOp::IndexPut))?;
        validate_same_dtype(&[self, values], Op::Indexing(IndexingOp::IndexPut))?;
        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), Op::Indexing(IndexingOp::IndexPut))?;
        validate_indices_dtype(indices, Op::Indexing(IndexingOp::IndexPut))?;
        let validate_requires_grad = validate_requires_grad_for_op(Op::Indexing(IndexingOp::IndexPut));

        let result_layout = self.layout();
        let indices_layout = indices.layout();
        let values_layout = values.layout();

        if crate::snapshot::capture::is_active() {
            let requires_grad = (self.is_requires_grad() || values.is_requires_grad()) && validate_requires_grad;
            let (result_id, result_tensor) = create_builder_tensor(result_layout.clone(), self.dtype(), requires_grad);

            let op_params = OpParams::IndexPut(IndexPutParams { dim: dim_scalar });

            crate::snapshot::capture::capture_operation(
                Op::Indexing(IndexingOp::IndexPut),
                Some(op_params.clone()),
                vec![self.id(), values.id(), indices.id()],
                result_id,
                vec![result_layout.clone(), values_layout, indices_layout],
                result_layout.clone(),
            )?;

            if requires_grad {
                gradient::record_operation(
                    vec![self.id(), values.id(), indices.id()],
                    result_id,
                    Op::Indexing(IndexingOp::IndexPut),
                    op_params,
                )?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| {
                indices.with_storage(|indices_storage| {
                    values.with_storage(|values_storage| {
                        storage.call_ops_index_put(
                            &result_layout,
                            indices_storage,
                            &indices_layout,
                            values_storage,
                            &values_layout,
                            dim_usize,
                            Op::Indexing(IndexingOp::IndexPut),
                        )
                    })
                })
            })?;

            let requires_grad = (self.is_requires_grad() || values.is_requires_grad()) && validate_requires_grad;
            let result = from_storage_with_context(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                gradient::record_operation(
                    vec![self.id(), values.id(), indices.id()],
                    result.id(),
                    Op::Indexing(IndexingOp::IndexPut),
                    OpParams::IndexPut(IndexPutParams { dim: dim_scalar }),
                )?;
            }

            Ok(result)
        }
    }

    pub fn gather<D: Into<Scalar>>(&self, dim: D, indices: &Self) -> HoduResult<Self> {
        let dim_scalar = dim.into();
        let dim_i32 = dim_scalar.to_i32();
        let ndim = self.ndim() as i32;
        let dim_usize = if dim_i32 < 0 {
            (ndim + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, indices], Op::Indexing(IndexingOp::Gather))?;
        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), Op::Indexing(IndexingOp::Gather))?;
        validate_indices_dtype(indices, Op::Indexing(IndexingOp::Gather))?;
        let validate_requires_grad = validate_requires_grad_for_op(Op::Indexing(IndexingOp::Gather));

        // Output has same shape as indices
        let self_layout = self.layout();
        let result_layout = indices.layout();

        if crate::snapshot::capture::is_active() {
            let requires_grad = self.is_requires_grad() && validate_requires_grad;
            let (result_id, result_tensor) = create_builder_tensor(result_layout.clone(), self.dtype(), requires_grad);

            let op_params = OpParams::Gather(GatherParams { dim: dim_scalar });

            crate::snapshot::capture::capture_operation(
                Op::Indexing(IndexingOp::Gather),
                Some(op_params.clone()),
                vec![self.id(), indices.id()],
                result_id,
                vec![self_layout.clone(), result_layout.clone()],
                result_layout,
            )?;

            if requires_grad {
                gradient::record_operation(
                    vec![self.id(), indices.id()],
                    result_id,
                    Op::Indexing(IndexingOp::Gather),
                    op_params,
                )?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| {
                indices.with_storage(|indices_storage| {
                    storage.call_ops_gather(
                        &self_layout,
                        indices_storage,
                        &result_layout,
                        dim_usize,
                        Op::Indexing(IndexingOp::Gather),
                    )
                })
            })?;

            let requires_grad = self.is_requires_grad() && validate_requires_grad;
            let result = from_storage_with_context(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                gradient::record_operation(
                    vec![self.id(), indices.id()],
                    result.id(),
                    Op::Indexing(IndexingOp::Gather),
                    OpParams::Gather(GatherParams { dim: dim_scalar }),
                )?;
            }

            Ok(result)
        }
    }

    pub fn gather_nd(&self, indices: &Self) -> HoduResult<Self> {
        let data_shape = self.shape();
        let data_dims = data_shape.dims();
        let data_ndim = data_dims.len();

        let indices_shape = indices.shape();
        let indices_dims = indices_shape.dims();
        let indices_ndim = indices_dims.len();

        if indices_ndim < 1 {
            return Err(crate::error::HoduError::InvalidArgument(
                "gather_nd: indices must have at least 1 dimension".to_string(),
            ));
        }

        let k = indices_dims[indices_ndim - 1];
        if k > data_ndim {
            return Err(crate::error::HoduError::InvalidArgument(format!(
                "gather_nd: indices last dim ({}) cannot exceed data ndim ({})",
                k, data_ndim
            )));
        }

        let batch_dims: Vec<usize> = indices_dims[..indices_ndim - 1].to_vec();
        let batch_size: usize = batch_dims.iter().product::<usize>().max(1);

        let slice_dims: Vec<usize> = data_dims[k..].to_vec();
        let slice_size: usize = slice_dims.iter().product::<usize>().max(1);

        // Compute strides for first k dimensions
        let mut strides = vec![1usize; k];
        for i in (0..k).rev() {
            if i < k - 1 {
                strides[i] = strides[i + 1] * data_dims[i + 1];
            } else {
                strides[i] = slice_size;
            }
        }

        let indices_flat = indices.reshape([batch_size, k])?;

        // Convert k-dimensional indices to linear indices
        let strides_tensor =
            Self::from_slice(strides.iter().map(|&s| s as i32).collect::<Vec<_>>(), [k])?.to_device(self.device())?;
        let indices_i32 = indices_flat.to_dtype(DType::I32)?;
        let weighted = indices_i32.mul(&strides_tensor)?;
        let linear_indices = weighted.sum(&[1], false)?;

        let indexed_size: usize = data_dims[..k].iter().product::<usize>().max(1);
        let data_flat = self.reshape([indexed_size, slice_size])?;

        let gathered = data_flat.index_select(0, &linear_indices)?;

        let mut output_shape = batch_dims;
        output_shape.extend(slice_dims);
        if output_shape.is_empty() {
            output_shape.push(1);
        }
        gathered.reshape(output_shape)
    }

    pub fn scatter<D: Into<Scalar>>(&self, dim: D, indices: &Self, src: &Self) -> HoduResult<Self> {
        let dim_scalar = dim.into();
        let dim_i32 = dim_scalar.to_i32();
        let ndim = self.ndim() as i32;
        let dim_usize = if dim_i32 < 0 {
            (ndim + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, indices, src], Op::Indexing(IndexingOp::Scatter))?;
        validate_same_dtype(&[self, src], Op::Indexing(IndexingOp::Scatter))?;
        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), Op::Indexing(IndexingOp::Scatter))?;
        validate_indices_dtype(indices, Op::Indexing(IndexingOp::Scatter))?;
        let validate_requires_grad = validate_requires_grad_for_op(Op::Indexing(IndexingOp::Scatter));

        // Output has same shape as self
        let result_layout = self.layout();
        let indices_layout = indices.layout();
        let src_layout = src.layout();

        if crate::snapshot::capture::is_active() {
            let requires_grad = (self.is_requires_grad() || src.is_requires_grad()) && validate_requires_grad;
            let (result_id, result_tensor) = create_builder_tensor(result_layout.clone(), self.dtype(), requires_grad);

            let op_params = OpParams::Scatter(ScatterParams { dim: dim_scalar });

            crate::snapshot::capture::capture_operation(
                Op::Indexing(IndexingOp::Scatter),
                Some(op_params.clone()),
                vec![self.id(), src.id(), indices.id()],
                result_id,
                vec![result_layout.clone(), src_layout.clone(), indices_layout.clone()],
                result_layout,
            )?;

            if requires_grad {
                gradient::record_operation(
                    vec![self.id(), src.id(), indices.id()],
                    result_id,
                    Op::Indexing(IndexingOp::Scatter),
                    op_params,
                )?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| {
                indices.with_storage(|indices_storage| {
                    src.with_storage(|src_storage| {
                        storage.call_ops_scatter(
                            &result_layout,
                            indices_storage,
                            &indices_layout,
                            src_storage,
                            &src_layout,
                            dim_usize,
                            Op::Indexing(IndexingOp::Scatter),
                        )
                    })
                })
            })?;

            let requires_grad = (self.is_requires_grad() || src.is_requires_grad()) && validate_requires_grad;
            let result = from_storage_with_context(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                gradient::record_operation(
                    vec![self.id(), src.id(), indices.id()],
                    result.id(),
                    Op::Indexing(IndexingOp::Scatter),
                    OpParams::Scatter(ScatterParams { dim: dim_scalar }),
                )?;
            }

            Ok(result)
        }
    }

    pub fn scatter_add<D: Into<Scalar>>(&self, dim: D, indices: &Self, src: &Self) -> HoduResult<Self> {
        let dim_scalar = dim.into();
        let dim_i32 = dim_scalar.to_i32();
        let ndim = self.ndim() as i32;
        let dim_usize = if dim_i32 < 0 {
            (ndim + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, indices, src], Op::Indexing(IndexingOp::ScatterAdd))?;
        validate_same_dtype(&[self, src], Op::Indexing(IndexingOp::ScatterAdd))?;
        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), Op::Indexing(IndexingOp::ScatterAdd))?;
        validate_indices_dtype(indices, Op::Indexing(IndexingOp::ScatterAdd))?;
        let validate_requires_grad = validate_requires_grad_for_op(Op::Indexing(IndexingOp::ScatterAdd));

        let result_layout = self.layout();

        if crate::snapshot::capture::is_active() {
            let requires_grad = (self.is_requires_grad() || src.is_requires_grad()) && validate_requires_grad;
            let (result_id, result_tensor) = create_builder_tensor(result_layout.clone(), self.dtype(), requires_grad);

            let op_params = OpParams::ScatterAdd(ScatterAddParams { dim: dim_scalar });

            crate::snapshot::capture::capture_operation(
                Op::Indexing(IndexingOp::ScatterAdd),
                Some(op_params.clone()),
                vec![self.id(), src.id(), indices.id()],
                result_id,
                vec![self.layout(), src.layout(), indices.layout()],
                result_layout,
            )?;

            if requires_grad {
                gradient::record_operation(
                    vec![self.id(), src.id(), indices.id()],
                    result_id,
                    Op::Indexing(IndexingOp::ScatterAdd),
                    op_params,
                )?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| {
                indices.with_storage(|indices_storage| {
                    src.with_storage(|src_storage| {
                        storage.call_ops_scatter(
                            &self.layout(),
                            indices_storage,
                            &indices.layout(),
                            src_storage,
                            &src.layout(),
                            dim_usize,
                            Op::Indexing(IndexingOp::ScatterAdd),
                        )
                    })
                })
            })?;

            let requires_grad = (self.is_requires_grad() || src.is_requires_grad()) && validate_requires_grad;
            let result = from_storage_with_context(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                gradient::record_operation(
                    vec![self.id(), src.id(), indices.id()],
                    result.id(),
                    Op::Indexing(IndexingOp::ScatterAdd),
                    OpParams::ScatterAdd(ScatterAddParams { dim: dim_scalar }),
                )?;
            }

            Ok(result)
        }
    }

    pub fn scatter_max<D: Into<Scalar>>(&self, dim: D, indices: &Self, src: &Self) -> HoduResult<Self> {
        let dim_scalar = dim.into();
        let dim_i32 = dim_scalar.to_i32();
        let ndim = self.ndim() as i32;
        let dim_usize = if dim_i32 < 0 {
            (ndim + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, indices, src], Op::Indexing(IndexingOp::ScatterMax))?;
        validate_same_dtype(&[self, src], Op::Indexing(IndexingOp::ScatterMax))?;
        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), Op::Indexing(IndexingOp::ScatterMax))?;
        validate_indices_dtype(indices, Op::Indexing(IndexingOp::ScatterMax))?;
        let validate_requires_grad = validate_requires_grad_for_op(Op::Indexing(IndexingOp::ScatterMax));

        let result_layout = self.layout();

        if crate::snapshot::capture::is_active() {
            let requires_grad = (self.is_requires_grad() || src.is_requires_grad()) && validate_requires_grad;
            let (result_id, result_tensor) = create_builder_tensor(result_layout.clone(), self.dtype(), requires_grad);

            let op_params = OpParams::ScatterMax(ScatterMaxParams { dim: dim_scalar });

            crate::snapshot::capture::capture_operation(
                Op::Indexing(IndexingOp::ScatterMax),
                Some(op_params.clone()),
                vec![self.id(), src.id(), indices.id()],
                result_id,
                vec![self.layout(), src.layout(), indices.layout()],
                result_layout,
            )?;

            if requires_grad {
                gradient::record_operation(
                    vec![self.id(), src.id(), indices.id()],
                    result_id,
                    Op::Indexing(IndexingOp::ScatterMax),
                    op_params,
                )?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| {
                indices.with_storage(|indices_storage| {
                    src.with_storage(|src_storage| {
                        storage.call_ops_scatter(
                            &self.layout(),
                            indices_storage,
                            &indices.layout(),
                            src_storage,
                            &src.layout(),
                            dim_usize,
                            Op::Indexing(IndexingOp::ScatterMax),
                        )
                    })
                })
            })?;

            let requires_grad = (self.is_requires_grad() || src.is_requires_grad()) && validate_requires_grad;
            let result = from_storage_with_context(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                gradient::record_operation(
                    vec![self.id(), src.id(), indices.id()],
                    result.id(),
                    Op::Indexing(IndexingOp::ScatterMax),
                    OpParams::ScatterMax(ScatterMaxParams { dim: dim_scalar }),
                )?;
            }

            Ok(result)
        }
    }

    pub fn scatter_min<D: Into<Scalar>>(&self, dim: D, indices: &Self, src: &Self) -> HoduResult<Self> {
        let dim_scalar = dim.into();
        let dim_i32 = dim_scalar.to_i32();
        let ndim = self.ndim() as i32;
        let dim_usize = if dim_i32 < 0 {
            (ndim + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, indices, src], Op::Indexing(IndexingOp::ScatterMin))?;
        validate_same_dtype(&[self, src], Op::Indexing(IndexingOp::ScatterMin))?;
        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), Op::Indexing(IndexingOp::ScatterMin))?;
        validate_indices_dtype(indices, Op::Indexing(IndexingOp::ScatterMin))?;
        let validate_requires_grad = validate_requires_grad_for_op(Op::Indexing(IndexingOp::ScatterMin));

        let result_layout = self.layout();

        if crate::snapshot::capture::is_active() {
            let requires_grad = (self.is_requires_grad() || src.is_requires_grad()) && validate_requires_grad;
            let (result_id, result_tensor) = create_builder_tensor(result_layout.clone(), self.dtype(), requires_grad);

            let op_params = OpParams::ScatterMin(ScatterMinParams { dim: dim_scalar });

            crate::snapshot::capture::capture_operation(
                Op::Indexing(IndexingOp::ScatterMin),
                Some(op_params.clone()),
                vec![self.id(), src.id(), indices.id()],
                result_id,
                vec![self.layout(), src.layout(), indices.layout()],
                result_layout,
            )?;

            if requires_grad {
                gradient::record_operation(
                    vec![self.id(), src.id(), indices.id()],
                    result_id,
                    Op::Indexing(IndexingOp::ScatterMin),
                    op_params,
                )?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| {
                indices.with_storage(|indices_storage| {
                    src.with_storage(|src_storage| {
                        storage.call_ops_scatter(
                            &self.layout(),
                            indices_storage,
                            &indices.layout(),
                            src_storage,
                            &src.layout(),
                            dim_usize,
                            Op::Indexing(IndexingOp::ScatterMin),
                        )
                    })
                })
            })?;

            let requires_grad = (self.is_requires_grad() || src.is_requires_grad()) && validate_requires_grad;
            let result = from_storage_with_context(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                gradient::record_operation(
                    vec![self.id(), src.id(), indices.id()],
                    result.id(),
                    Op::Indexing(IndexingOp::ScatterMin),
                    OpParams::ScatterMin(ScatterMinParams { dim: dim_scalar }),
                )?;
            }

            Ok(result)
        }
    }

    pub fn scatter_nd(&self, indices: &Self, updates: &Self) -> HoduResult<Self> {
        let data_shape = self.shape();
        let data_dims = data_shape.dims();
        let data_ndim = data_dims.len();

        let indices_shape = indices.shape();
        let indices_dims = indices_shape.dims();
        let indices_ndim = indices_dims.len();

        if indices_ndim < 1 {
            return Err(crate::error::HoduError::InvalidArgument(
                "scatter_nd: indices must have at least 1 dimension".to_string(),
            ));
        }

        let k = indices_dims[indices_ndim - 1];
        if k > data_ndim {
            return Err(crate::error::HoduError::InvalidArgument(format!(
                "scatter_nd: indices last dim ({}) cannot exceed data ndim ({})",
                k, data_ndim
            )));
        }

        let batch_dims: Vec<usize> = indices_dims[..indices_ndim - 1].to_vec();
        let batch_size: usize = batch_dims.iter().product::<usize>().max(1);

        let slice_dims: Vec<usize> = data_dims[k..].to_vec();
        let slice_size: usize = slice_dims.iter().product::<usize>().max(1);

        // Compute strides for first k dimensions
        let mut strides = vec![1usize; k];
        for i in (0..k).rev() {
            if i < k - 1 {
                strides[i] = strides[i + 1] * data_dims[i + 1];
            } else {
                strides[i] = slice_size;
            }
        }

        let indices_flat = indices.reshape([batch_size, k])?;

        // Convert to linear indices
        let strides_tensor =
            Self::from_slice(strides.iter().map(|&s| s as i32).collect::<Vec<_>>(), [k])?.to_device(self.device())?;
        let indices_i32 = indices_flat.to_dtype(DType::I32)?;
        let weighted = indices_i32.mul(&strides_tensor)?;
        let linear_indices = weighted.sum(&[1], false)?;

        let indexed_size: usize = data_dims[..k].iter().product::<usize>().max(1);
        let data_flat = self.reshape([indexed_size, slice_size])?;

        let updates_flat = updates.reshape([batch_size, slice_size])?;

        // Expand linear_indices for scatter
        let linear_indices_expanded = linear_indices.unsqueeze(-1)?.broadcast([batch_size, slice_size])?;

        let result_flat = data_flat.scatter(0, &linear_indices_expanded, &updates_flat)?;

        result_flat.reshape(data_dims.to_vec())
    }

    /// Convert integer indices to one-hot encoded vectors.
    ///
    /// # Arguments
    /// * `num_classes` - Number of classes (depth of one-hot dimension)
    /// * `axis` - Dimension to insert the one-hot encoding (default: -1, last dimension)
    /// * `dtype` - Output data type (default: F32)
    ///
    /// # Returns
    /// A new tensor with one-hot encoded vectors. The output shape is the input shape
    /// with `num_classes` inserted at the specified axis.
    ///
    /// # Example
    /// ```ignore
    /// let indices = Tensor::from_slice(&[0i32, 1, 2, 0], &[4])?;
    /// let onehot = indices.onehot(3, -1, DType::F32)?;
    /// // onehot shape: [4, 3]
    /// // onehot values: [[1,0,0], [0,1,0], [0,0,1], [1,0,0]]
    /// ```
    pub fn onehot<N: Into<Scalar>, A: Into<Scalar>>(&self, num_classes: N, axis: A, dtype: DType) -> HoduResult<Self> {
        let num_classes_scalar = num_classes.into();
        let axis_scalar = axis.into();
        let num_classes_usize = num_classes_scalar.to_usize();
        let axis_i32 = axis_scalar.to_i32();

        // Normalize axis
        let ndim = self.ndim() as i32;
        let output_ndim = ndim + 1;
        let axis_usize = if axis_i32 < 0 {
            (output_ndim + axis_i32) as usize
        } else {
            axis_i32 as usize
        };

        // Validate
        validate_dtype_for_device(dtype, self.device())?;
        validate_dtype_for_op(self.dtype(), Op::Indexing(IndexingOp::Onehot))?;
        validate_indices_dtype(self, Op::Indexing(IndexingOp::Onehot))?;

        // Compute output shape: insert num_classes at axis position
        let input_shape = self.shape();
        let mut output_dims = Vec::with_capacity(output_ndim as usize);
        for (i, &dim) in input_shape.dims().iter().enumerate() {
            if i == axis_usize {
                output_dims.push(num_classes_usize);
            }
            output_dims.push(dim);
        }
        // If axis == output_ndim - 1 (last position)
        if axis_usize == output_ndim as usize - 1 {
            output_dims.push(num_classes_usize);
        }

        let result_layout = Layout::from_shape(&Shape::from(output_dims));
        let self_layout = self.layout();

        if crate::snapshot::capture::is_active() {
            // Onehot is non-differentiable
            let requires_grad = false;
            let (result_id, result_tensor) = create_builder_tensor(result_layout.clone(), dtype, requires_grad);

            let op_params = OpParams::Onehoto(OnehotoParams {
                num_classes: num_classes_scalar,
                axis: axis_scalar,
                dtype,
            });

            crate::snapshot::capture::capture_operation(
                Op::Indexing(IndexingOp::Onehot),
                Some(op_params),
                vec![self.id()],
                result_id,
                vec![self_layout],
                result_layout,
            )?;

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| {
                storage.call_ops_onehot(
                    &self_layout,
                    num_classes_usize,
                    axis_usize,
                    dtype,
                    Op::Indexing(IndexingOp::Onehot),
                )
            })?;

            // Onehot is non-differentiable
            let result = from_storage_with_context(storage, result_layout, true, false);

            Ok(result)
        }
    }

    /// Returns the indices of non-zero elements in the tensor.
    ///
    /// # Returns
    /// A tensor of shape `[N, ndim]` where N is the number of non-zero elements
    /// and ndim is the number of dimensions in the input tensor. Each row contains
    /// the multi-dimensional index of a non-zero element.
    ///
    /// Note: The output size is data-dependent. In capture mode, the maximum possible
    /// size is allocated (num_elements * ndim).
    /// This operation does not support gradients.
    ///
    /// # Example
    /// ```ignore
    /// let x = Tensor::from_slice(&[0.0f32, 1.0, 0.0, 2.0, 0.0, 3.0], &[2, 3])?;
    /// let indices = x.nonzero()?;
    /// // indices shape: [3, 2] (3 non-zero elements, 2 dimensions)
    /// // indices values: [[0, 1], [1, 0], [1, 2]]
    /// ```
    pub fn nonzero(&self) -> HoduResult<Self> {
        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), Op::Indexing(IndexingOp::Nonzero))?;

        let self_layout = self.layout();
        let ndim = self.ndim();
        let num_els = self_layout.size();

        if crate::snapshot::capture::is_active() {
            // In capture mode, use symbolic shape for data-dependent output
            // Output shape: [N, ndim] where N is dynamic (0 to num_els)
            let dynamic_dim_id = DynamicDimId::new();
            let symbolic_shape = SymbolicShape::new(vec![
                Dim::dynamic_with_id(dynamic_dim_id, Some(num_els)),
                Dim::Concrete(ndim),
            ]);
            let symbolic_layout = SymbolicLayout::from_shape(&symbolic_shape)
                .expect("Symbolic shape with max_bound should create valid layout");

            // Allocate using max bounds
            let max_result_layout = symbolic_layout
                .to_max_layout()
                .expect("Symbolic layout with max_bound should produce max layout");

            // Nonzero is non-differentiable
            let requires_grad = false;
            let (result_id, result_tensor) =
                create_builder_tensor(max_result_layout.clone(), DType::I32, requires_grad);

            let op_params = OpParams::Nonzero(NonzeroParams {
                dynamic_count_dim: Some(dynamic_dim_id),
            });

            crate::snapshot::capture::capture_operation_with_symbolic(
                Op::Indexing(IndexingOp::Nonzero),
                Some(op_params),
                vec![self.id()],
                result_id,
                vec![self_layout],
                max_result_layout,
                Some(symbolic_layout),
            )?;

            Ok(result_tensor)
        } else {
            let (storage, count) = self.with_storage(|storage| storage.call_nonzero(&self_layout))?;

            let result_layout = Layout::from_shape(&Shape::from(vec![count, ndim]));

            // Nonzero is non-differentiable
            let result = from_storage_with_context(storage, result_layout, true, false);

            Ok(result)
        }
    }

    /// Returns unique elements, inverse indices, and counts.
    ///
    /// The input tensor is flattened before finding unique elements.
    /// Returns a tuple of (values, inverse, counts):
    /// - values: 1D tensor of sorted unique values, same dtype as input
    /// - inverse: 1D tensor where input.flatten() = values[inverse]
    /// - counts: 1D tensor with count of each unique value
    ///
    /// Note: The output size is data-dependent. In capture mode, the maximum possible
    /// size is allocated (num_elements for values and counts).
    /// This operation does not support gradients.
    ///
    /// # Example
    /// ```ignore
    /// let x = Tensor::from_slice(&[1, 2, 2, 3, 1, 4, 3], &[7])?;
    /// let (values, inverse, counts) = x.unique()?;
    /// // values: [1, 2, 3, 4]
    /// // inverse: [0, 1, 1, 2, 0, 3, 2]
    /// // counts: [2, 2, 2, 1]
    /// ```
    pub fn unique(&self) -> HoduResult<(Self, Self, Self)> {
        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), Op::Indexing(IndexingOp::Unique))?;

        let self_layout = self.layout();
        let num_els = self_layout.size();

        if crate::snapshot::capture::is_active() {
            // In capture mode, use symbolic shape for data-dependent outputs
            // values shape: [M] where M is dynamic (0 to num_els)
            // inverse shape: [num_els] - concrete
            // counts shape: [M] where M is dynamic (same as values)
            let dynamic_dim_id = DynamicDimId::new();

            let values_symbolic_shape = SymbolicShape::new(vec![Dim::dynamic_with_id(dynamic_dim_id, Some(num_els))]);
            let values_symbolic_layout = SymbolicLayout::from_shape(&values_symbolic_shape)
                .expect("Symbolic shape with max_bound should create valid layout");

            // Allocate using max bounds
            let max_values_layout = values_symbolic_layout
                .to_max_layout()
                .expect("Symbolic layout with max_bound should produce max layout");
            let inverse_layout = Layout::from_shape(&Shape::from(vec![num_els]));
            let max_counts_layout = max_values_layout.clone();

            // Unique is non-differentiable
            let requires_grad = false;

            // Create builder tensors for all three outputs
            let (values_id, values_tensor) =
                create_builder_tensor(max_values_layout.clone(), self.dtype(), requires_grad);
            let (inverse_id, inverse_tensor) = create_builder_tensor(inverse_layout.clone(), DType::I32, requires_grad);
            let (counts_id, counts_tensor) =
                create_builder_tensor(max_counts_layout.clone(), DType::I32, requires_grad);

            let op_params = OpParams::Unique(UniqueParams {
                inverse_id,
                counts_id,
                dynamic_count_dim: Some(dynamic_dim_id),
            });

            crate::snapshot::capture::capture_operation_with_symbolic(
                Op::Indexing(IndexingOp::Unique),
                Some(op_params),
                vec![self.id()],
                values_id,
                vec![self_layout],
                max_values_layout,
                Some(values_symbolic_layout),
            )?;

            Ok((values_tensor, inverse_tensor, counts_tensor))
        } else {
            let (values_storage, inverse_storage, counts_storage, unique_count) =
                self.with_storage(|storage| storage.call_unique(&self_layout))?;

            let values_layout = Layout::from_shape(&Shape::from(vec![unique_count]));
            let inverse_layout = Layout::from_shape(&Shape::from(vec![num_els]));
            let counts_layout = Layout::from_shape(&Shape::from(vec![unique_count]));

            // Unique is non-differentiable
            let values = from_storage_with_context(values_storage, values_layout, true, false);
            let inverse = from_storage_with_context(inverse_storage, inverse_layout, true, false);
            let counts = from_storage_with_context(counts_storage, counts_layout, true, false);

            Ok((values, inverse, counts))
        }
    }

    /// Selects elements from the input tensor based on a boolean condition.
    ///
    /// Like NumPy's `np.compress(condition, a, axis)`:
    /// - condition: 1-D boolean tensor specifying which elements to select
    /// - axis: Optional axis along which to compress. If None, input is flattened first.
    ///
    /// Note: The output size is data-dependent (count of True values in condition).
    /// In capture mode, the maximum possible size is allocated.
    /// This operation does not support gradients.
    ///
    /// # Arguments
    /// * `condition` - 1-D boolean tensor
    /// * `axis` - Optional axis along which to compress. If None, input is flattened.
    ///
    /// # Returns
    /// A tensor containing elements selected where condition is True.
    ///
    /// # Example
    /// ```ignore
    /// let a = Tensor::from_slice(&[1, 2, 3, 4, 5, 6], &[3, 2])?;  // [[1, 2], [3, 4], [5, 6]]
    /// let condition = Tensor::from_slice(&[false, true, true], &[3])?;
    /// let result = a.compress(&condition, Some(0))?;  // [[3, 4], [5, 6]]
    /// ```
    pub fn compress<A: Into<Option<i32>>>(&self, condition: &Self, axis: A) -> HoduResult<Self> {
        use crate::error::HoduError;

        let axis_opt: Option<i32> = axis.into();

        // Validate devices match
        validate_same_device(&[self, condition], Op::Indexing(IndexingOp::Compress))?;
        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), Op::Indexing(IndexingOp::Compress))?;

        // Validate condition is BOOL
        if condition.dtype() != DType::BOOL {
            return Err(HoduError::DTypeMismatch {
                expected: DType::BOOL,
                got: condition.dtype(),
            });
        }

        // Validate condition is 1-D
        if condition.ndim() != 1 {
            return Err(HoduError::BackendError(format!(
                "compress condition must be 1-D, got {}D",
                condition.ndim()
            )));
        }

        let self_layout = self.layout();
        let condition_layout = condition.layout();
        let condition_size = condition_layout.size();

        // Normalize and validate axis
        let axis_usize: Option<usize> = match axis_opt {
            Some(ax) => {
                let ndim = self.ndim() as i32;
                let normalized = if ax < 0 { ndim + ax } else { ax };
                if normalized < 0 || normalized >= ndim {
                    return Err(HoduError::InvalidAxis {
                        axis: ax,
                        ndim: self.ndim(),
                    });
                }
                let ax_usize = normalized as usize;
                // Validate condition size matches axis dimension
                if condition_size != self.shape()[ax_usize] {
                    return Err(HoduError::BackendError(format!(
                        "compress: condition size {} != axis {} size {}",
                        condition_size,
                        ax_usize,
                        self.shape()[ax_usize]
                    )));
                }
                Some(ax_usize)
            },
            None => {
                // Flatten mode: condition size must match total elements
                let num_els = self_layout.size();
                if condition_size != num_els {
                    return Err(HoduError::BackendError(format!(
                        "compress: condition size {} != flattened input size {}",
                        condition_size, num_els
                    )));
                }
                None
            },
        };

        // Convert axis to Scalar for params
        let axis_scalar: Option<Scalar> = axis_opt.map(Scalar::from);

        if crate::snapshot::capture::is_active() {
            // In capture mode, use symbolic shape for data-dependent output
            let dynamic_dim_id = DynamicDimId::new();

            // Compute max output shape
            let (max_shape, symbolic_shape) = match axis_usize {
                Some(ax) => {
                    // Output shape: input shape with axis dimension replaced by condition_size (max)
                    let input_shape = self.shape();
                    let mut max_dims = Vec::with_capacity(input_shape.ndim());
                    let mut symbolic_dims = Vec::with_capacity(input_shape.ndim());

                    for (i, &dim) in input_shape.dims().iter().enumerate() {
                        if i == ax {
                            max_dims.push(condition_size); // Max is all True
                            symbolic_dims.push(Dim::dynamic_with_id(dynamic_dim_id, Some(condition_size)));
                        } else {
                            max_dims.push(dim);
                            symbolic_dims.push(Dim::Concrete(dim));
                        }
                    }
                    (Shape::from(max_dims), SymbolicShape::new(symbolic_dims))
                },
                None => {
                    // Flatten mode: output is 1-D with max size = condition_size
                    let max_dims = vec![condition_size];
                    let symbolic_dims = vec![Dim::dynamic_with_id(dynamic_dim_id, Some(condition_size))];
                    (Shape::from(max_dims), SymbolicShape::new(symbolic_dims))
                },
            };

            let symbolic_layout = SymbolicLayout::from_shape(&symbolic_shape)
                .expect("Symbolic shape with max_bound should create valid layout");
            let max_result_layout = Layout::from_shape(&max_shape);

            // Compress is non-differentiable
            let requires_grad = false;
            let (result_id, result_tensor) =
                create_builder_tensor(max_result_layout.clone(), self.dtype(), requires_grad);

            let op_params = OpParams::Compress(CompressParams {
                axis: axis_scalar,
                dynamic_count_dim: Some(dynamic_dim_id),
            });

            crate::snapshot::capture::capture_operation_with_symbolic(
                Op::Indexing(IndexingOp::Compress),
                Some(op_params),
                vec![self.id(), condition.id()],
                result_id,
                vec![self_layout, condition_layout],
                max_result_layout,
                Some(symbolic_layout),
            )?;

            Ok(result_tensor)
        } else {
            let (storage, true_count) = self.with_storage(|storage| {
                condition.with_storage(|condition_storage| {
                    storage.call_compress(&self_layout, condition_storage, &condition_layout, axis_usize)
                })
            })?;

            // Compute actual output shape based on true_count
            let result_shape = match axis_usize {
                Some(ax) => {
                    let input_shape = self.shape();
                    let mut output_dims = input_shape.dims().to_vec();
                    output_dims[ax] = true_count;
                    Shape::from(output_dims)
                },
                None => Shape::from(vec![true_count]),
            };
            let result_layout = Layout::from_shape(&result_shape);

            // Compress is non-differentiable
            let result = from_storage_with_context(storage, result_layout, true, false);

            Ok(result)
        }
    }
}
