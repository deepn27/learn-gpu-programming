#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include <cuda_runtime.h>

namespace fused_mlp {

constexpr int kMatrixDim = 4096;

template <typename T>
struct AccumulatorType {
    using type = T;
};

template <>
struct AccumulatorType<c10::Half> {
    using type = float;
};

template <>
struct AccumulatorType<c10::BFloat16> {
    using type = float;
};

template <typename T>
struct ScalarConversion {
    using scalar_t = T;

    __device__ static inline float to_float(T x) {
        return static_cast<float>(x);
    }

    __device__ static inline T from_float(float x) {
        return static_cast<T>(x);
    }
};

template <>
struct ScalarConversion<c10::Half> {
    using scalar_t = c10::Half;

    __device__ static inline float to_float(c10::Half x) {
        return static_cast<float>(x);
    }

    __device__ static inline c10::Half from_float(float x) {
        return static_cast<c10::Half>(x);
    }
};

template <>
struct ScalarConversion<c10::BFloat16> {
    using scalar_t = c10::BFloat16;

    __device__ static inline float to_float(c10::BFloat16 x) {
        return static_cast<float>(x);
    }

    __device__ static inline c10::BFloat16 from_float(float x) {
        return static_cast<c10::BFloat16>(x);
    }
};

__device__ inline float silu_float(float x) {
    return x / (1.0f + expf(-x));
}

template <typename scalar_t>
__device__ inline scalar_t silu(scalar_t x) {
    float xf = ScalarConversion<scalar_t>::to_float(x);
    float yf = silu_float(xf);
    return ScalarConversion<scalar_t>::from_float(yf);
}

inline void validate_inputs(const at::Tensor& x, const at::Tensor& w1, const at::Tensor& w2) {
    TORCH_CHECK(x.dim() == 2, "Input must be rank-2 tensor");
    TORCH_CHECK(w1.dim() == 2 && w2.dim() == 2, "Weights must be rank-2 tensors");

    TORCH_CHECK(x.size(0) == kMatrixDim && x.size(1) == kMatrixDim, "Input must be 4096x4096");
    TORCH_CHECK(w1.size(0) == kMatrixDim && w1.size(1) == kMatrixDim, "W1 must be 4096x4096");
    TORCH_CHECK(w2.size(0) == kMatrixDim && w2.size(1) == kMatrixDim, "W2 must be 4096x4096");

    TORCH_CHECK(x.device().is_cuda(), "Input tensor must reside on CUDA device");
    TORCH_CHECK(w1.device() == x.device() && w2.device() == x.device(), "All tensors must be on the same device");

    TORCH_CHECK(x.dtype() == w1.dtype() && x.dtype() == w2.dtype(), "All tensors must share dtype");
}

}  // namespace fused_mlp
