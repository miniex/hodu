#include "ops_scan.h"
#include "types.h"

#define IMPL_CUMSUM(TYPE, TYPE_SUFFIX)                                                             \
    void hodu_cpu_cumsum_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata) {  \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *shape = &metadata[2];                                                        \
        const size_t *strides = &metadata[2 + num_dims];                                           \
        const size_t offset = metadata[2 + 2 * num_dims];                                          \
        const size_t dim = metadata[3 + 2 * num_dims];                                             \
                                                                                                   \
        const TYPE *in = (const TYPE *)input + offset;                                             \
        TYPE *out = (TYPE *)output;                                                                \
                                                                                                   \
        if (num_dims == 0) {                                                                       \
            out[0] = in[0];                                                                        \
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
                                                                                                   \
        for (size_t outer = 0; outer < outer_size; outer++) {                                      \
            for (size_t inner = 0; inner < inner_size; inner++) {                                  \
                TYPE acc = (TYPE)0;                                                                \
                for (size_t s = 0; s < scan_size; s++) {                                           \
                    size_t in_idx = 0;                                                             \
                    size_t out_idx = 0;                                                            \
                    size_t tmp_outer = outer;                                                      \
                    for (size_t d = 0; d < dim; d++) {                                             \
                        size_t coord = tmp_outer % shape[d];                                       \
                        tmp_outer /= shape[d];                                                     \
                        in_idx += coord * strides[d];                                              \
                        out_idx += coord * out_strides[d];                                         \
                    }                                                                              \
                    in_idx += s * scan_stride;                                                     \
                    out_idx += s * out_strides[dim];                                               \
                    size_t tmp_inner = inner;                                                      \
                    for (size_t d = num_dims - 1; d > dim; d--) {                                  \
                        size_t coord = tmp_inner % shape[d];                                       \
                        tmp_inner /= shape[d];                                                     \
                        in_idx += coord * strides[d];                                              \
                        out_idx += coord * out_strides[d];                                         \
                    }                                                                              \
                    acc += in[in_idx];                                                             \
                    out[out_idx] = acc;                                                            \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }

IMPL_CUMSUM(f32_t, f32)
IMPL_CUMSUM(f64_t, f64)
IMPL_CUMSUM(u8_t, u8)
IMPL_CUMSUM(u16_t, u16)
IMPL_CUMSUM(u32_t, u32)
IMPL_CUMSUM(u64_t, u64)
IMPL_CUMSUM(i8_t, i8)
IMPL_CUMSUM(i16_t, i16)
IMPL_CUMSUM(i32_t, i32)
IMPL_CUMSUM(i64_t, i64)

#define IMPL_CUMSUM_CONVERT(TYPE, TYPE_SUFFIX, TO_FLOAT, FROM_FLOAT)                               \
    void hodu_cpu_cumsum_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata) {  \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *shape = &metadata[2];                                                        \
        const size_t *strides = &metadata[2 + num_dims];                                           \
        const size_t offset = metadata[2 + 2 * num_dims];                                          \
        const size_t dim = metadata[3 + 2 * num_dims];                                             \
                                                                                                   \
        const TYPE *in = (const TYPE *)input + offset;                                             \
        TYPE *out = (TYPE *)output;                                                                \
                                                                                                   \
        if (num_dims == 0) {                                                                       \
            out[0] = in[0];                                                                        \
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
                                                                                                   \
        for (size_t outer = 0; outer < outer_size; outer++) {                                      \
            for (size_t inner = 0; inner < inner_size; inner++) {                                  \
                float acc = 0.0f;                                                                  \
                for (size_t s = 0; s < scan_size; s++) {                                           \
                    size_t in_idx = 0;                                                             \
                    size_t out_idx = 0;                                                            \
                    size_t tmp_outer = outer;                                                      \
                    for (size_t d = 0; d < dim; d++) {                                             \
                        size_t coord = tmp_outer % shape[d];                                       \
                        tmp_outer /= shape[d];                                                     \
                        in_idx += coord * strides[d];                                              \
                        out_idx += coord * out_strides[d];                                         \
                    }                                                                              \
                    in_idx += s * scan_stride;                                                     \
                    out_idx += s * out_strides[dim];                                               \
                    size_t tmp_inner = inner;                                                      \
                    for (size_t d = num_dims - 1; d > dim; d--) {                                  \
                        size_t coord = tmp_inner % shape[d];                                       \
                        tmp_inner /= shape[d];                                                     \
                        in_idx += coord * strides[d];                                              \
                        out_idx += coord * out_strides[d];                                         \
                    }                                                                              \
                    acc += TO_FLOAT(in[in_idx]);                                                   \
                    out[out_idx] = FROM_FLOAT(acc);                                                \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }

IMPL_CUMSUM_CONVERT(f8e4m3_t, f8e4m3, f8e4m3_to_float, float_to_f8e4m3)
IMPL_CUMSUM_CONVERT(f8e5m2_t, f8e5m2, f8e5m2_to_float, float_to_f8e5m2)
IMPL_CUMSUM_CONVERT(bf16_t, bf16, bf16_to_float, float_to_bf16)
IMPL_CUMSUM_CONVERT(f16_t, f16, f16_to_float, float_to_f16)

#define IMPL_CUMPROD(TYPE, TYPE_SUFFIX)                                                            \
    void hodu_cpu_cumprod_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata) { \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *shape = &metadata[2];                                                        \
        const size_t *strides = &metadata[2 + num_dims];                                           \
        const size_t offset = metadata[2 + 2 * num_dims];                                          \
        const size_t dim = metadata[3 + 2 * num_dims];                                             \
                                                                                                   \
        const TYPE *in = (const TYPE *)input + offset;                                             \
        TYPE *out = (TYPE *)output;                                                                \
                                                                                                   \
        if (num_dims == 0) {                                                                       \
            out[0] = in[0];                                                                        \
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
                                                                                                   \
        for (size_t outer = 0; outer < outer_size; outer++) {                                      \
            for (size_t inner = 0; inner < inner_size; inner++) {                                  \
                TYPE acc = (TYPE)1;                                                                \
                for (size_t s = 0; s < scan_size; s++) {                                           \
                    size_t in_idx = 0;                                                             \
                    size_t out_idx = 0;                                                            \
                    size_t tmp_outer = outer;                                                      \
                    for (size_t d = 0; d < dim; d++) {                                             \
                        size_t coord = tmp_outer % shape[d];                                       \
                        tmp_outer /= shape[d];                                                     \
                        in_idx += coord * strides[d];                                              \
                        out_idx += coord * out_strides[d];                                         \
                    }                                                                              \
                    in_idx += s * scan_stride;                                                     \
                    out_idx += s * out_strides[dim];                                               \
                    size_t tmp_inner = inner;                                                      \
                    for (size_t d = num_dims - 1; d > dim; d--) {                                  \
                        size_t coord = tmp_inner % shape[d];                                       \
                        tmp_inner /= shape[d];                                                     \
                        in_idx += coord * strides[d];                                              \
                        out_idx += coord * out_strides[d];                                         \
                    }                                                                              \
                    acc *= in[in_idx];                                                             \
                    out[out_idx] = acc;                                                            \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }

IMPL_CUMPROD(f32_t, f32)
IMPL_CUMPROD(f64_t, f64)
IMPL_CUMPROD(u8_t, u8)
IMPL_CUMPROD(u16_t, u16)
IMPL_CUMPROD(u32_t, u32)
IMPL_CUMPROD(u64_t, u64)
IMPL_CUMPROD(i8_t, i8)
IMPL_CUMPROD(i16_t, i16)
IMPL_CUMPROD(i32_t, i32)
IMPL_CUMPROD(i64_t, i64)

#define IMPL_CUMPROD_CONVERT(TYPE, TYPE_SUFFIX, TO_FLOAT, FROM_FLOAT)                              \
    void hodu_cpu_cumprod_##TYPE_SUFFIX(const void *input, void *output, const size_t *metadata) { \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *shape = &metadata[2];                                                        \
        const size_t *strides = &metadata[2 + num_dims];                                           \
        const size_t offset = metadata[2 + 2 * num_dims];                                          \
        const size_t dim = metadata[3 + 2 * num_dims];                                             \
                                                                                                   \
        const TYPE *in = (const TYPE *)input + offset;                                             \
        TYPE *out = (TYPE *)output;                                                                \
                                                                                                   \
        if (num_dims == 0) {                                                                       \
            out[0] = in[0];                                                                        \
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
                                                                                                   \
        for (size_t outer = 0; outer < outer_size; outer++) {                                      \
            for (size_t inner = 0; inner < inner_size; inner++) {                                  \
                float acc = 1.0f;                                                                  \
                for (size_t s = 0; s < scan_size; s++) {                                           \
                    size_t in_idx = 0;                                                             \
                    size_t out_idx = 0;                                                            \
                    size_t tmp_outer = outer;                                                      \
                    for (size_t d = 0; d < dim; d++) {                                             \
                        size_t coord = tmp_outer % shape[d];                                       \
                        tmp_outer /= shape[d];                                                     \
                        in_idx += coord * strides[d];                                              \
                        out_idx += coord * out_strides[d];                                         \
                    }                                                                              \
                    in_idx += s * scan_stride;                                                     \
                    out_idx += s * out_strides[dim];                                               \
                    size_t tmp_inner = inner;                                                      \
                    for (size_t d = num_dims - 1; d > dim; d--) {                                  \
                        size_t coord = tmp_inner % shape[d];                                       \
                        tmp_inner /= shape[d];                                                     \
                        in_idx += coord * strides[d];                                              \
                        out_idx += coord * out_strides[d];                                         \
                    }                                                                              \
                    acc *= TO_FLOAT(in[in_idx]);                                                   \
                    out[out_idx] = FROM_FLOAT(acc);                                                \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }

IMPL_CUMPROD_CONVERT(f8e4m3_t, f8e4m3, f8e4m3_to_float, float_to_f8e4m3)
IMPL_CUMPROD_CONVERT(f8e5m2_t, f8e5m2, f8e5m2_to_float, float_to_f8e5m2)
IMPL_CUMPROD_CONVERT(bf16_t, bf16, bf16_to_float, float_to_bf16)
IMPL_CUMPROD_CONVERT(f16_t, f16, f16_to_float, float_to_f16)
