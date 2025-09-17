# Fused 4096×4096 MLP Benchmarks

This repository provides three implementations of the following fused computation targeting NVIDIA A100 (SM80) GPUs:

```
Y = Linear(X, W1)        # 4096 × 4096, no bias
Y = SiLU(Y)
Z = Linear(Y, W2)        # 4096 × 4096, no bias
```

The goal is to compare latency (P50, P99) for:

1. **PyTorch baseline** – stock operations executed sequentially.
2. **CUTLASS fused kernel** – a custom CUDA extension that keeps data in shared memory and relies on CUTLASS utilities.
3. **Native fused kernel** – a hand-written CUDA kernel that implements the same tiling strategy without CUTLASS helpers.

## Repository Layout

```
benchmarks/
  baseline.py          # PyTorch reference and benchmarking harness
  benchmark_utils.py   # Latency measurement helpers (P50/P99)
  extensions.py        # Compiles CUDA extensions via torch.utils.cpp_extension
  run_all.py           # CLI script to benchmark all implementations
cpp/
  common/fused_common.cuh   # Shared tensor validation + math helpers
  cutlass_fused/             # CUTLASS-based fused kernel + bindings
  native_fused/              # Native CUDA fused kernel + bindings
```

## Prerequisites

* CUDA-capable GPU (the kernels are tuned for A100 / SM80 but work on other recent architectures).
* Python 3.10+
* PyTorch with CUDA support.
* CUTLASS v3.5.1 checkout (or newer) available locally.
* `tabulate` Python package for pretty-printing benchmark tables.

Clone CUTLASS next to the repository or point `CUTLASS_DIR` at an existing checkout:

```bash
mkdir -p external
git clone --depth=1 --branch v3.5.1 https://github.com/NVIDIA/cutlass.git external/cutlass
```

Install Python dependencies:

```bash
pip install torch tabulate
```

## Running Benchmarks

Invoke the benchmark driver to compile extensions on-the-fly and collect latency numbers:

```bash
python -m benchmarks.run_all --device cuda --dtype float16 --warmup 20 --iters 100
```

Arguments:

* `--device` (default: `cuda` if available) – device to run benchmarks on.
* `--dtype` (default: `float16`) – choose between `float16`, `float32`, and `bfloat16`.
* `--warmup` – warmup iterations discarded from measurements.
* `--iters` – number of timed iterations.
* `--skip-native` / `--skip-cutlass` – optional flags to skip compiling either extension.

The script prints a Markdown table with P50/P99 latencies and validates that fused kernels match the baseline (max absolute difference).

## Implementation Notes

* Both fused kernels operate on fixed 4096×4096 matrices and use a 64×64 output tile per thread block with 16×16 threads.
* The CUTLASS variant uses `cutlass::Array` and `cutlass::NumericConverter` helpers to manage per-thread fragments and type conversion while keeping the fused pipeline inside a single kernel launch.
* The native variant mirrors the tiling strategy using explicit CUDA math.
* Accumulation occurs in FP32 for float16/bfloat16 inputs.

## Troubleshooting

* Ensure the CUTLASS headers are discoverable. Set `CUTLASS_DIR` or place the repository under `external/cutlass`.
* Compile time can be significant the first time the extensions build; subsequent runs reuse the cached binaries.
* To change the targeted GPU architecture, set `TORCH_CUDA_ARCH_LIST` before running `benchmarks.run_all`.
