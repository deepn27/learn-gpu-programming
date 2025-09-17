"""Entry point for benchmarking baseline and fused CUDA kernels."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from .baseline import run_baseline
from .benchmark_utils import LatencyStats, measure_latency, summarize_results
from .extensions import load_cutlass_fused, load_native_fused


@dataclass
class BenchmarkOutcome:
    name: str
    latency: LatencyStats
    max_abs_diff: float | None = None


def _resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_arg)


def _resolve_dtype(dtype_arg: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[dtype_arg]
    except KeyError as exc:  # pragma: no cover - guard rails for CLI usage
        raise ValueError(f"Unsupported dtype '{dtype_arg}'. Choose from {list(mapping)}") from exc


def _allocate_inputs(dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (4096, 4096)
    generator = torch.Generator(device=device.type) if device.type == "cuda" else torch.Generator()
    x = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    w1 = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    w2 = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    return x, w1, w2


def _benchmark_extension(
    name: str,
    forward_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    *,
    warmup: int,
    iters: int,
    baseline_output: torch.Tensor,
) -> BenchmarkOutcome:
    def op() -> torch.Tensor:
        return forward_fn(x, w1, w2)

    latency = measure_latency(op, warmup=warmup, iters=iters, device=x.device)
    with torch.inference_mode():
        out = forward_fn(x, w1, w2)
    max_abs_diff = (out - baseline_output).abs().max().item()
    return BenchmarkOutcome(name=name, latency=latency, max_abs_diff=max_abs_diff)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default=None, help="Device to benchmark on (default: cuda if available).")
    parser.add_argument("--dtype", type=str, default="float16", help="Data type: float32|float16|bfloat16")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations before measuring.")
    parser.add_argument("--iters", type=int, default=50, help="Number of iterations for latency measurement.")
    parser.add_argument("--skip-cutlass", action="store_true", help="Skip the CUTLASS fused kernel benchmark.")
    parser.add_argument("--skip-native", action="store_true", help="Skip the native fused kernel benchmark.")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype)

    if device.type != "cuda":
        print("[WARN] CUDA device not available - fused kernels require CUDA for execution.")

    x, w1, w2 = _allocate_inputs(dtype, device)

    baseline_result = run_baseline(x, w1, w2, warmup=args.warmup, iters=args.iters)
    outcomes: list[BenchmarkOutcome] = [
        BenchmarkOutcome(name="PyTorch baseline", latency=baseline_result.latency, max_abs_diff=0.0)
    ]

    if device.type == "cuda" and not args.skip_native:
        native_module = load_native_fused()
        outcomes.append(
            _benchmark_extension(
                "Native fused kernel",
                native_module.fused_forward,
                x,
                w1,
                w2,
                warmup=args.warmup,
                iters=args.iters,
                baseline_output=baseline_result.output,
            )
        )

    if device.type == "cuda" and not args.skip_cutlass:
        cutlass_module = load_cutlass_fused()
        outcomes.append(
            _benchmark_extension(
                "CUTLASS fused kernel",
                cutlass_module.fused_forward,
                x,
                w1,
                w2,
                warmup=args.warmup,
                iters=args.iters,
                baseline_output=baseline_result.output,
            )
        )

    table = summarize_results([(outcome.name, outcome.latency) for outcome in outcomes])
    print(table)

    print("\nVerification (max |diff| vs baseline):")
    for outcome in outcomes:
        if outcome.max_abs_diff is None:
            print(f"  - {outcome.name}: n/a")
        else:
            print(f"  - {outcome.name}: {outcome.max_abs_diff:.6f}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
