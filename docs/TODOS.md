# TODOS.md

**Serialization:** (🟡 Important)
- To be determined

**Backend:** (🔴 Critical)
- [x] CPU SIMD support
- [x] CPU parallelization support
- [x] CUDA support
- [x] Metal support
- [ ] OS-provided BLAS support
  - [x] aarch64-apple-darwin (Accelerate framework)
  - [ ] x86_64-apple-darwin (Accelerate framework)
  - [ ] x86_64-unknown-linux-gnu (system BLAS)
  - [ ] aarch64-unknown-linux-gnu (system BLAS)

**Tensor Creation & Initialization:** (🔴 Critical)
- [ ] Implement initialization functions (xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal)
- [x] Implement basic creation ops (eye, arange, linspace, uniform, normal)
- [x] Implement tril/triu (ONNX: Trilu) - extract triangular matrix
- [ ] Implement diag/diagonal - create/extract diagonal matrix

**ONNX Compatibility - Tensor Operations:** (🔴 Critical)
- [x] Implement padding operations (pad_constant, pad_reflect, pad_replicate, pad_circular)
- [x] Implement flip operation
- [x] Implement repeat operation (ONNX: Tile)
- [x] Implement expand operation (broadcast)
- [x] Implement ceil, floor, round operations (ONNX: Ceil, Floor, Round)
- [x] Implement cumsum operation (ONNX: CumSum)

**ONNX Compatibility - Unary Operations:** (🟡 Important)
- [x] Implement erf (ONNX: Erf) - used in accurate GELU
- [x] Implement inverse trigonometric (asin, acos, atan)
- [x] Implement hyperbolic (sinh, cosh, tanh)
- [x] Implement inverse hyperbolic (asinh, acosh, atanh)
- [x] Implement hardswish(hardsilu), hardsigmoid (ONNX: HardSwish, HardSigmoid)
- [x] Implement selu (ONNX: Selu) - Scaled ELU activation
- [x] Implement celu (ONNX: Celu) - Continuous ELU activation
- [x] Implement softsign (ONNX: Softsign) - x/(1+|x|)

**ONNX Compatibility - Other Operations:** (🟡 Important)
- [x] Implement einsum (ONNX: Einsum)
- [x] Implement resize/upsample (ONNX: Resize)
- [x] Implement topk (ONNX: TopK)
- [x] Implement nonzero (ONNX: NonZero)
- [x] Implement onehot (ONNX: OneHot)
- [x] Implement rem (ONNX: Mod)
- [x] Implement isnan (ONNX: IsNaN) - check for NaN values
- [x] Implement isinf (ONNX: IsInf) - check for Inf values
- [x] Implement unique (ONNX: Unique) - find unique elements
- [x] Implement reduce_logsumexp (ONNX: ReduceLogSumExp) - numerically stable log(sum(exp(x)))
- [ ] Implement scatter_nd (ONNX: ScatterND) - N-dimensional scatter
- [ ] Implement gather_nd (ONNX: GatherND) - N-dimensional gather
- [ ] Implement depth_to_space (ONNX: DepthToSpace) - pixel shuffle for super-resolution
- [ ] Implement space_to_depth (ONNX: SpaceToDepth) - inverse pixel shuffle
- [ ] Implement grid_sample (ONNX: GridSample) - 2D/3D spatial sampling

**ONNX Compatibility - Matrix Operations:** (🟢 Nice-to-have)
- [x] Implement det (ONNX: Det) - matrix determinant

**Linear Algebra Operations:** (🟢 Nice-to-have)
- [x] Implement inv - matrix inverse
- [x] Implement solve - linear system solver (Ax = b)
- [ ] Implement svd - singular value decomposition
- [ ] Implement eig/eigvals - eigenvalue decomposition
- [ ] Implement cholesky - Cholesky decomposition
- [ ] Implement qr - QR decomposition
- [ ] Implement lu - LU decomposition (factorization)
- [x] Implement trace - matrix trace (sum of diagonal)
- [ ] Implement slogdet - sign and log of determinant (numerically stable)
- [ ] Implement matrix_rank - rank of matrix
- [ ] Implement pinv - Moore-Penrose pseudo-inverse

**ONNX Compatibility - Low Priority:** (🟢 Nice-to-have)
- [x] Implement compress (ONNX: Compress) - select elements based on condition
- [x] Implement reduce_logsum (ONNX: ReduceLogSum) - log(sum(x))
- [x] Implement bitwise operations (ONNX: BitShift, BitwiseAnd, BitwiseOr, BitwiseXor, BitwiseNot) - shl, shr, bitwise_and, bitwise_or, bitwise_xor, bitwise_not (integer only)
- [x] Implement lrn (ONNX: LRN) - Local Response Normalization (deprecated, rarely used)

**Recurrent Layers:** (🔴 Critical)
- [x] Implement RNN
- [x] Implement LSTM
- [x] Implement GRU

**Attention Layers:** (🔴 Critical)
- [x] Implement MultiheadAttention
- [x] Implement ScaledDotProductAttention

**Pooling Layers:**
- [x] Implement pooling layers
- [x] Implement GlobalAvgPool, GlobalMaxPool (🟡 Important)
- [x] Implement FractionalMaxPool (🟢 Nice-to-have)

**Normalization Layers:**
- [x] Implement normalization layers
- [x] Implement GroupNorm, InstanceNorm (🟡 Important)
- [x] Implement RMSNorm (🟡 Important)

**Activation Functions:** (🟡 Important)
- [x] Implement Swish/SiLU, Mish
- [x] Implement PReLU, RReLU

**Loss Functions:** (🟡 Important)
- [x] Implement SmoothL1Loss
- [x] Implement KLDivLoss
- [x] Implement CosineEmbeddingLoss

**Optimizers:** (🟡 Important)
- [x] Implement RMSprop
- [x] Implement Adagrad

**DataSet**
- [x] Implement Dataset

**Tensor-level Normalization Ops:** (🟢 Nice-to-have - nn modules use basic ops)
- [ ] Implement batch_norm tensor op - fused `(x - mean) / sqrt(var + eps) * γ + β`
- [ ] Implement layer_norm tensor op - fused normalize last N dimensions
- [ ] Implement group_norm tensor op - fused per-group normalization
- [ ] Implement instance_norm tensor op - fused per-instance normalization
- [ ] Implement rms_norm tensor op - fused `x / sqrt(mean(x²) + eps) * γ`

**Tensor Manipulation Ops:** (🟡 Important)
- [ ] Implement tile/repeat_interleave - repeat tensor elements
- [ ] Implement roll - circular shift along axes (PyTorch)
- [ ] Implement cumprod - cumulative product (like cumsum)

**Detection Ops:** (🟢 Nice-to-have)
- [ ] Implement roi_align/roi_pool - Region of Interest pooling (Object Detection)
- [ ] Implement nms - Non-Maximum Suppression (Object Detection)
- [ ] Implement deformable_conv - Deformable Convolution
