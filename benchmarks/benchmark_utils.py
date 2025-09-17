"""Utility helpers for benchmarking fused MLP implementations."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable, List, Sequence

import torch


@dataclass
class LatencyStats:
    """Simple container tracking latency samples and summary statistics."""

    samples_ms: List[float]

    @property
    def p50_ms(self) -> float:
        return statistics.median(self.samples_ms)

    @property
    def p99_ms(self) -> float:
        if not self.samples_ms:
            return float("nan")
        sorted_samples = sorted(self.samples_ms)
        index = min(int(round(0.99 * (len(sorted_samples) - 1))), len(sorted_samples) - 1)
        return sorted_samples[index]


def _ensure_inference_context(func: Callable[[], torch.Tensor]) -> Callable[[], torch.Tensor]:
    def wrapper() -> torch.Tensor:
        with torch.inference_mode():
            return func()

    return wrapper


def measure_latency(
    func: Callable[[], torch.Tensor],
    *,
    warmup: int = 10,
    iters: int = 100,
    device: torch.device | str | None = None,
) -> LatencyStats:
    """Measure latency of ``func`` returning :class:`LatencyStats`.

    The returned latencies are recorded in milliseconds. If ``device`` resolves
    to a CUDA device we rely on CUDA events to collect precise timings,
    otherwise a CPU wall clock is used.
    """

    if iters <= 0:
        raise ValueError("iters must be > 0")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    resolved_device: torch.device
    if device is None:
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(device)

    wrapped = _ensure_inference_context(func)

    samples: List[float] = []

    if resolved_device.type == "cuda":
        # Warmup
        for _ in range(warmup):
            wrapped()
        torch.cuda.synchronize(resolved_device)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        for _ in range(iters):
            start_event.record()
            wrapped()
            end_event.record()
            end_event.synchronize()
            samples.append(start_event.elapsed_time(end_event))

        torch.cuda.synchronize(resolved_device)
    else:
        # CPU path
        for _ in range(warmup):
            wrapped()

        for _ in range(iters):
            start = time.perf_counter()
            wrapped()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            samples.append((end - start) * 1000.0)

    return LatencyStats(samples)


def summarize_results(name_latency_pairs: Sequence[tuple[str, LatencyStats]]) -> str:
    """Format latency statistics as a Markdown table."""

    try:
        from tabulate import tabulate
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency missing at runtime
        raise RuntimeError(
            "tabulate is required for pretty printing results. Install it via 'pip install tabulate'."
        ) from exc

    headers = ["Implementation", "P50 (ms)", "P99 (ms)"]
    rows = [
        (
            name,
            f"{stats.p50_ms:.3f}" if stats.samples_ms else "n/a",
            f"{stats.p99_ms:.3f}" if stats.samples_ms else "n/a",
        )
        for name, stats in name_latency_pairs
    ]
    return tabulate(rows, headers=headers, tablefmt="github")
