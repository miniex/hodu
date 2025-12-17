#include <metal_stdlib>
using namespace metal;

// Cumulative sum operation: computes prefix sum along a dimension
//
// Metadata layout:
// - metadata[0]: num_els (total number of elements)
// - metadata[1]: num_dims (number of dimensions)
// - metadata[2..2+num_dims]: shape
// - metadata[2+num_dims..2+2*num_dims]: strides
// - metadata[2+2*num_dims]: offset
// - metadata[3+2*num_dims]: dim (dimension to scan along)

#define CUMSUM_OP(TYPE, TYPE_SUFFIX)                                                               \
    kernel void hodu_metal_cumsum_##TYPE_SUFFIX(                                                   \
        device const TYPE *input [[buffer(0)]], device TYPE *output [[buffer(1)]],                 \
        constant size_t *metadata [[buffer(2)]], uint tid [[thread_position_in_grid]]) {           \
        const size_t num_dims = metadata[1];                                                       \
        constant size_t *shape = metadata + 2;                                                     \
        constant size_t *strides = metadata + 2 + num_dims;                                        \
        const size_t offset = metadata[2 + 2 * num_dims];                                          \
        const size_t dim = metadata[3 + 2 * num_dims];                                             \
                                                                                                   \
        if (num_dims == 0) {                                                                       \
            if (tid == 0) {                                                                        \
                output[0] = input[offset];                                                         \
            }                                                                                      \
            return;                                                                                \
        }                                                                                          \
                                                                                                   \
        /* Compute output strides (contiguous row-major) */                                        \
        size_t out_strides[16];                                                                    \
        out_strides[num_dims - 1] = 1;                                                             \
        for (size_t d = num_dims - 1; d > 0; d--) {                                                \
            out_strides[d - 1] = out_strides[d] * shape[d];                                        \
        }                                                                                          \
                                                                                                   \
        size_t outer_size = 1;                                                                     \
        for (size_t d = 0; d < dim; d++) {                                                         \
            outer_size *= shape[d];                                                                \
        }                                                                                          \
        size_t inner_size = 1;                                                                     \
        for (size_t d = dim + 1; d < num_dims; d++) {                                              \
            inner_size *= shape[d];                                                                \
        }                                                                                          \
        const size_t scan_size = shape[dim];                                                       \
        const size_t scan_stride = strides[dim];                                                   \
        const size_t num_scans = outer_size * inner_size;                                          \
                                                                                                   \
        if (tid >= num_scans)                                                                      \
            return;                                                                                \
                                                                                                   \
        const size_t outer = tid / inner_size;                                                     \
        const size_t inner = tid % inner_size;                                                     \
                                                                                                   \
        TYPE acc = static_cast<TYPE>(0);                                                           \
        for (size_t s = 0; s < scan_size; s++) {                                                   \
            size_t in_idx = offset;                                                                \
            size_t out_idx = 0;                                                                    \
            size_t tmp_outer = outer;                                                              \
            for (size_t d = 0; d < dim; d++) {                                                     \
                size_t coord = tmp_outer % shape[d];                                               \
                tmp_outer /= shape[d];                                                             \
                in_idx += coord * strides[d];                                                      \
                out_idx += coord * out_strides[d];                                                 \
            }                                                                                      \
            in_idx += s * scan_stride;                                                             \
            out_idx += s * out_strides[dim];                                                       \
            size_t tmp_inner = inner;                                                              \
            for (size_t d = num_dims - 1; d > dim; d--) {                                          \
                size_t coord = tmp_inner % shape[d];                                               \
                tmp_inner /= shape[d];                                                             \
                in_idx += coord * strides[d];                                                      \
                out_idx += coord * out_strides[d];                                                 \
            }                                                                                      \
            acc += input[in_idx];                                                                  \
            output[out_idx] = acc;                                                                 \
        }                                                                                          \
    }

CUMSUM_OP(bfloat, bf16)
CUMSUM_OP(half, f16)
CUMSUM_OP(float, f32)
CUMSUM_OP(uint8_t, u8)
CUMSUM_OP(uint16_t, u16)
CUMSUM_OP(uint32_t, u32)
CUMSUM_OP(uint64_t, u64)
CUMSUM_OP(int8_t, i8)
CUMSUM_OP(int16_t, i16)
CUMSUM_OP(int32_t, i32)
CUMSUM_OP(int64_t, i64)

// Cumulative product operation: computes prefix product along a dimension
//
// Metadata layout:
// - metadata[0]: num_els (total number of elements)
// - metadata[1]: num_dims (number of dimensions)
// - metadata[2..2+num_dims]: shape
// - metadata[2+num_dims..2+2*num_dims]: strides
// - metadata[2+2*num_dims]: offset
// - metadata[3+2*num_dims]: dim (dimension to scan along)

#define CUMPROD_OP(TYPE, TYPE_SUFFIX)                                                              \
    kernel void hodu_metal_cumprod_##TYPE_SUFFIX(                                                  \
        device const TYPE *input [[buffer(0)]], device TYPE *output [[buffer(1)]],                 \
        constant size_t *metadata [[buffer(2)]], uint tid [[thread_position_in_grid]]) {           \
        const size_t num_dims = metadata[1];                                                       \
        constant size_t *shape = metadata + 2;                                                     \
        constant size_t *strides = metadata + 2 + num_dims;                                        \
        const size_t offset = metadata[2 + 2 * num_dims];                                          \
        const size_t dim = metadata[3 + 2 * num_dims];                                             \
                                                                                                   \
        if (num_dims == 0) {                                                                       \
            if (tid == 0) {                                                                        \
                output[0] = input[offset];                                                         \
            }                                                                                      \
            return;                                                                                \
        }                                                                                          \
                                                                                                   \
        /* Compute output strides (contiguous row-major) */                                        \
        size_t out_strides[16];                                                                    \
        out_strides[num_dims - 1] = 1;                                                             \
        for (size_t d = num_dims - 1; d > 0; d--) {                                                \
            out_strides[d - 1] = out_strides[d] * shape[d];                                        \
        }                                                                                          \
                                                                                                   \
        size_t outer_size = 1;                                                                     \
        for (size_t d = 0; d < dim; d++) {                                                         \
            outer_size *= shape[d];                                                                \
        }                                                                                          \
        size_t inner_size = 1;                                                                     \
        for (size_t d = dim + 1; d < num_dims; d++) {                                              \
            inner_size *= shape[d];                                                                \
        }                                                                                          \
        const size_t scan_size = shape[dim];                                                       \
        const size_t scan_stride = strides[dim];                                                   \
        const size_t num_scans = outer_size * inner_size;                                          \
                                                                                                   \
        if (tid >= num_scans)                                                                      \
            return;                                                                                \
                                                                                                   \
        const size_t outer = tid / inner_size;                                                     \
        const size_t inner = tid % inner_size;                                                     \
                                                                                                   \
        TYPE acc = static_cast<TYPE>(1);                                                           \
        for (size_t s = 0; s < scan_size; s++) {                                                   \
            size_t in_idx = offset;                                                                \
            size_t out_idx = 0;                                                                    \
            size_t tmp_outer = outer;                                                              \
            for (size_t d = 0; d < dim; d++) {                                                     \
                size_t coord = tmp_outer % shape[d];                                               \
                tmp_outer /= shape[d];                                                             \
                in_idx += coord * strides[d];                                                      \
                out_idx += coord * out_strides[d];                                                 \
            }                                                                                      \
            in_idx += s * scan_stride;                                                             \
            out_idx += s * out_strides[dim];                                                       \
            size_t tmp_inner = inner;                                                              \
            for (size_t d = num_dims - 1; d > dim; d--) {                                          \
                size_t coord = tmp_inner % shape[d];                                               \
                tmp_inner /= shape[d];                                                             \
                in_idx += coord * strides[d];                                                      \
                out_idx += coord * out_strides[d];                                                 \
            }                                                                                      \
            acc *= input[in_idx];                                                                  \
            output[out_idx] = acc;                                                                 \
        }                                                                                          \
    }

CUMPROD_OP(bfloat, bf16)
CUMPROD_OP(half, f16)
CUMPROD_OP(float, f32)
CUMPROD_OP(uint8_t, u8)
CUMPROD_OP(uint16_t, u16)
CUMPROD_OP(uint32_t, u32)
CUMPROD_OP(uint64_t, u64)
CUMPROD_OP(int8_t, i8)
CUMPROD_OP(int16_t, i16)
CUMPROD_OP(int32_t, i32)
CUMPROD_OP(int64_t, i64)
