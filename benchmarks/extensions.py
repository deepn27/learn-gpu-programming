"""Helpers for building/loading fused CUDA extensions."""

from __future__ import annotations

import functools
import os
import pathlib
from typing import Optional

from torch.utils.cpp_extension import load as load_extension


_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_CPP_ROOT = _REPO_ROOT / "cpp"


def _cuda_arch_flags() -> list[str]:
    arch_env = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if arch_env:
        arches = [arch.strip() for arch in arch_env.split() if arch.strip()]
        flags = [f"-gencode=arch=compute_{arch},code=sm_{arch}" for arch in arches]
        if flags:
            return flags
    # Default to Ampere / SM80 which matches the target environment (A100).
    return ["-gencode=arch=compute_80,code=sm_80"]


def _load_extension(name: str, sources: list[pathlib.Path], *, extra_include_paths: Optional[list[pathlib.Path]] = None) -> object:
    include_dirs = [str(_CPP_ROOT / "common")] + [str(path) for path in (extra_include_paths or [])]

    extra_cuda_cflags = ["-O3", "-lineinfo", "--use_fast_math"] + _cuda_arch_flags()
    extra_cflags = ["-O3", "-std=c++17"]

    return load_extension(
        name=name,
        sources=[str(src) for src in sources],
        extra_include_paths=include_dirs,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        with_cuda=True,
        verbose=False,
    )


@functools.lru_cache(maxsize=None)
def load_native_fused() -> object:
    """Compile and return the native fused CUDA extension."""

    src_dir = _CPP_ROOT / "native_fused"
    sources = [src_dir / "binding.cpp", src_dir / "fused_kernel.cu"]
    return _load_extension("fused_mlp_native", sources)


def _resolve_cutlass_dir() -> pathlib.Path:
    candidates = []
    env = os.environ.get("CUTLASS_DIR")
    if env:
        candidates.append(pathlib.Path(env))
    candidates.append(_REPO_ROOT / "external" / "cutlass")
    candidates.append(_REPO_ROOT / "cutlass")

    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Unable to locate CUTLASS. Set the CUTLASS_DIR environment variable or clone the library "
        "into 'external/cutlass'."
    )


@functools.lru_cache(maxsize=None)
def load_cutlass_fused() -> object:
    """Compile and return the CUTLASS-based fused CUDA extension."""

    cutlass_dir = _resolve_cutlass_dir()
    src_dir = _CPP_ROOT / "cutlass_fused"
    sources = [src_dir / "binding.cpp", src_dir / "fused_kernel.cu"]
    return _load_extension("fused_mlp_cutlass", sources, extra_include_paths=[cutlass_dir])


__all__ = ["load_native_fused", "load_cutlass_fused"]
