#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "fused_common.cuh"

namespace fused_mlp {
void launch_cutlass_fused(
    const at::Tensor& x,
    const at::Tensor& w1,
    const at::Tensor& w2,
    at::Tensor& out,
    cudaStream_t stream);
}

namespace {

torch::Tensor fused_forward(torch::Tensor x, torch::Tensor w1, torch::Tensor w2) {
    fused_mlp::validate_inputs(x, w1, w2);

    at::cuda::CUDAGuard device_guard(x.device());

    auto x_contig = x.contiguous();
    auto w1_contig = w1.contiguous();
    auto w2_contig = w2.contiguous();

    auto options = x_contig.options();
    auto out = torch::empty({x_contig.size(0), w2_contig.size(0)}, options);

    auto stream = at::cuda::getCurrentCUDAStream();
    fused_mlp::launch_cutlass_fused(x_contig, w1_contig, w2_contig, out, stream.stream());

    return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_forward", &fused_forward, "CUTLASS fused MLP forward pass");
}
