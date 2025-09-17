"""PyTorch reference implementation for the fused MLP benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .benchmark_utils import LatencyStats, measure_latency


@dataclass
class BaselineResult:
    latency: LatencyStats
    output: torch.Tensor


def fused_mlp_reference(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """Compute the baseline fused MLP using PyTorch ops."""

    y = F.linear(x, w1, bias=None)
    y = F.silu(y)
    z = F.linear(y, w2, bias=None)
    return z


def run_baseline(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    *,
    warmup: int = 10,
    iters: int = 100,
) -> BaselineResult:
    """Benchmark the reference fused MLP for the provided tensors."""

    device = x.device

    def op() -> torch.Tensor:
        return fused_mlp_reference(x, w1, w2)

    latency = measure_latency(op, warmup=warmup, iters=iters, device=device)
    # Run one final time to return the output tensor for validation purposes.
    with torch.inference_mode():
        out = fused_mlp_reference(x, w1, w2)
    return BaselineResult(latency=latency, output=out)


__all__ = ["BaselineResult", "fused_mlp_reference", "run_baseline"]
