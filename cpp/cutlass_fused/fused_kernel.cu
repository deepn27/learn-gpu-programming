#include "fused_common.cuh"

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>

namespace fused_mlp {
namespace cutlass_impl {

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int TILE_K = 16;
constexpr int INPUT_CHUNK = 32;
constexpr int THREADS_X = 16;
constexpr int THREADS_Y = 16;
constexpr int THREAD_M = BLOCK_M / THREADS_Y;
constexpr int THREAD_N = BLOCK_N / THREADS_X;
constexpr int Z_FRAG_SIZE = THREAD_M * THREAD_N;
constexpr int Y_FRAG_SIZE = THREAD_M * TILE_K;

static_assert(BLOCK_M % THREADS_Y == 0, "BLOCK_M must be divisible by THREADS_Y");
static_assert(BLOCK_N % THREADS_X == 0, "BLOCK_N must be divisible by THREADS_X");

template <typename scalar_t>
__global__ void fused_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ w1,
    const scalar_t* __restrict__ w2,
    scalar_t* __restrict__ out,
    int64_t m,
    int64_t k,
    int64_t hidden,
    int64_t n) {
    using acc_t = typename AccumulatorType<scalar_t>::type;
    using ToAccum = cutlass::NumericConverter<acc_t, scalar_t>;
    using FromAccum = cutlass::NumericConverter<scalar_t, acc_t>;

    __shared__ scalar_t shared_x[BLOCK_M * INPUT_CHUNK];
    __shared__ scalar_t shared_w1[TILE_K * INPUT_CHUNK];
    __shared__ scalar_t shared_w2[TILE_K * BLOCK_N];

    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;

    const int row_start = block_row * BLOCK_M;
    const int col_start = block_col * BLOCK_N;

    const int thread_linear = threadIdx.y * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * blockDim.y;

    cutlass::Array<acc_t, Z_FRAG_SIZE> z_frag;
    for (int idx = 0; idx < Z_FRAG_SIZE; ++idx) {
        z_frag[idx] = acc_t(0);
    }

    ToAccum to_accum;
    FromAccum from_accum;

    for (int hidden_start = 0; hidden_start < hidden; hidden_start += TILE_K) {
        cutlass::Array<acc_t, Y_FRAG_SIZE> y_frag;
        for (int idx = 0; idx < Y_FRAG_SIZE; ++idx) {
            y_frag[idx] = acc_t(0);
        }

        for (int input_start = 0; input_start < k; input_start += INPUT_CHUNK) {
            for (int linear_idx = thread_linear; linear_idx < BLOCK_M * INPUT_CHUNK; linear_idx += total_threads) {
                const int local_row = linear_idx / INPUT_CHUNK;
                const int local_col = linear_idx % INPUT_CHUNK;
                const int global_row = row_start + local_row;
                const int global_col = input_start + local_col;

                scalar_t value = scalar_t(0);
                if (global_row < m && global_col < k) {
                    value = x[global_row * k + global_col];
                }
                shared_x[local_row * INPUT_CHUNK + local_col] = value;
            }

            for (int linear_idx = thread_linear; linear_idx < TILE_K * INPUT_CHUNK; linear_idx += total_threads) {
                const int local_hidden = linear_idx / INPUT_CHUNK;
                const int local_col = linear_idx % INPUT_CHUNK;
                const int global_hidden = hidden_start + local_hidden;
                const int global_col = input_start + local_col;

                scalar_t value = scalar_t(0);
                if (global_hidden < hidden && global_col < k) {
                    value = w1[global_hidden * k + global_col];
                }
                shared_w1[local_hidden * INPUT_CHUNK + local_col] = value;
            }

            __syncthreads();

            for (int inner = 0; inner < INPUT_CHUNK; ++inner) {
                acc_t a_vals[THREAD_M];
                #pragma unroll
                for (int i = 0; i < THREAD_M; ++i) {
                    const int local_row = threadIdx.y * THREAD_M + i;
                    const int global_row = row_start + local_row;
                    if (global_row < m) {
                        const scalar_t a = shared_x[local_row * INPUT_CHUNK + inner];
                        a_vals[i] = to_accum(a);
                    } else {
                        a_vals[i] = acc_t(0);
                    }
                }

                #pragma unroll
                for (int kk = 0; kk < TILE_K; ++kk) {
                    const scalar_t b_val = shared_w1[kk * INPUT_CHUNK + inner];
                    const acc_t b = to_accum(b_val);
                    #pragma unroll
                    for (int i = 0; i < THREAD_M; ++i) {
                        y_frag[i * TILE_K + kk] += a_vals[i] * b;
                    }
                }
            }

            __syncthreads();
        }

        for (int idx = 0; idx < Y_FRAG_SIZE; ++idx) {
            y_frag[idx] = silu_float(static_cast<float>(y_frag[idx]));
        }

        for (int linear_idx = thread_linear; linear_idx < TILE_K * BLOCK_N; linear_idx += total_threads) {
            const int local_hidden = linear_idx / BLOCK_N;
            const int local_col = linear_idx % BLOCK_N;
            const int global_hidden = hidden_start + local_hidden;
            const int global_col = col_start + local_col;

            scalar_t value = scalar_t(0);
            if (global_hidden < hidden && global_col < n) {
                value = w2[global_col * hidden + global_hidden];
            }
            shared_w2[local_hidden * BLOCK_N + local_col] = value;
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            cutlass::Array<acc_t, THREAD_N> w_vals;
            #pragma unroll
            for (int j = 0; j < THREAD_N; ++j) {
                const int local_col = threadIdx.x * THREAD_N + j;
                const int global_col = col_start + local_col;
                if (global_col < n) {
                    const scalar_t wv = shared_w2[kk * BLOCK_N + local_col];
                    w_vals[j] = to_accum(wv);
                } else {
                    w_vals[j] = acc_t(0);
                }
            }

            #pragma unroll
            for (int i = 0; i < THREAD_M; ++i) {
                const acc_t y_val = y_frag[i * TILE_K + kk];
                #pragma unroll
                for (int j = 0; j < THREAD_N; ++j) {
                    z_frag[i * THREAD_N + j] += y_val * w_vals[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < THREAD_M; ++i) {
        const int local_row = threadIdx.y * THREAD_M + i;
        const int global_row = row_start + local_row;
        if (global_row >= m) {
            continue;
        }
        #pragma unroll
        for (int j = 0; j < THREAD_N; ++j) {
            const int local_col = threadIdx.x * THREAD_N + j;
            const int global_col = col_start + local_col;
            if (global_col >= n) {
                continue;
            }
            const acc_t value = z_frag[i * THREAD_N + j];
            out[global_row * n + global_col] = from_accum(value);
        }
    }
}


template <typename scalar_t>
void launch_kernel(
    const at::Tensor& x,
    const at::Tensor& w1,
    const at::Tensor& w2,
    at::Tensor& out,
    cudaStream_t stream) {
    const dim3 block_dim(THREADS_X, THREADS_Y);
    const dim3 grid_dim(
        (out.size(1) + BLOCK_N - 1) / BLOCK_N,
        (out.size(0) + BLOCK_M - 1) / BLOCK_M);

    fused_kernel<scalar_t><<<grid_dim, block_dim, 0, stream>>>(
        x.data_ptr<scalar_t>(),
        w1.data_ptr<scalar_t>(),
        w2.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        out.size(0),
        x.size(1),
        w1.size(0),
        out.size(1));

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cutlass_impl

void launch_cutlass_fused(
    const at::Tensor& x,
    const at::Tensor& w1,
    const at::Tensor& w2,
    at::Tensor& out,
    cudaStream_t stream) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "cutlass_fused_kernel", [&] {
        cutlass_impl::launch_kernel<scalar_t>(x, w1, w2, out, stream);
    });
}

}  // namespace fused_mlp
