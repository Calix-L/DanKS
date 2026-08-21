from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Any

import torch
from torch.utils.cpp_extension import CUDA_HOME as TORCH_CUDA_HOME, load_inline


_MODULE: Any | None = None


def _cuda_home() -> Path | None:
    configured = os.environ.get("CUDA_HOME", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    if TORCH_CUDA_HOME:
        return Path(TORCH_CUDA_HOME).resolve()
    nvcc = shutil.which("nvcc")
    return Path(nvcc).resolve().parents[1] if nvcc else None


def _configure_cuda_env() -> None:
    cuda_home = _cuda_home()
    if cuda_home is None:
        raise RuntimeError("CUDA toolkit not found; set CUDA_HOME or add nvcc to PATH")
    nvcc = cuda_home / "bin" / "nvcc"
    if not nvcc.is_file():
        raise RuntimeError(f"CUDA nvcc not found: {nvcc}")
    os.environ["CUDA_HOME"] = str(cuda_home)
    os.environ["PATH"] = os.pathsep.join((str(cuda_home / "bin"), os.environ.get("PATH", "")))


def available() -> bool:
    cuda_home = _cuda_home()
    return bool(
        torch.cuda.is_available()
        and cuda_home is not None
        and (cuda_home / "bin" / "nvcc").is_file()
    )


def module() -> Any:
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    _configure_cuda_env()
    cpp_sources = r"""
#include <torch/extension.h>

torch::Tensor frontier_keep_mask(torch::Tensor features);
"""
    cuda_sources = r"""
#include <torch/extension.h>

namespace {

__device__ __forceinline__ bool dominates_row(
    const double* a,
    double b0,
    double b1,
    double b2,
    double b3,
    double b4,
    double b5,
    double b6,
    double b7,
    double b8,
    double b9,
    double b10,
    double b11,
    double b12,
    int a_idx,
    int b_idx
) {
    constexpr double eps = 1.0e-12;
    if (
        (a[0] > b0) ||
        (a[2] > b2) ||
        (a[3] > b3) ||
        (a[4] > b4 + eps)
    ) {
        return false;
    }
    if (
        (a[1] + eps < b1) ||
        (a[5] + eps < b5) ||
        (a[6] + eps < b6) ||
        (a[7] + eps < b7) ||
        (a[8] + eps < b8) ||
        (a[9] + eps < b9) ||
        (a[10] + eps < b10) ||
        (a[11] + eps < b11) ||
        (a[12] + eps < b12)
    ) {
        return false;
    }
    const bool strict =
        (a[0] < b0) ||
        (a[1] > b1 + eps) ||
        (a[2] < b2) ||
        (a[3] < b3) ||
        (a[4] + eps < b4) ||
        (a[5] > b5 + eps) ||
        (a[6] > b6 + eps) ||
        (a[7] > b7 + eps) ||
        (a[8] > b8 + eps) ||
        (a[9] > b9 + eps) ||
        (a[10] > b10 + eps) ||
        (a[11] > b11 + eps) ||
        (a[12] > b12 + eps);
    const bool previous_equal = a_idx < b_idx;
    return strict || previous_equal;
}

__global__ void frontier_keep_mask_kernel(const double* __restrict__ features, int8_t* __restrict__ keep, int n) {
    const int row = blockIdx.x;
    if (row >= n) return;
    const double* row_ptr = features + row * 13;
    const double b0 = row_ptr[0];
    const double b1 = row_ptr[1];
    const double b2 = row_ptr[2];
    const double b3 = row_ptr[3];
    const double b4 = row_ptr[4];
    const double b5 = row_ptr[5];
    const double b6 = row_ptr[6];
    const double b7 = row_ptr[7];
    const double b8 = row_ptr[8];
    const double b9 = row_ptr[9];
    const double b10 = row_ptr[10];
    const double b11 = row_ptr[11];
    const double b12 = row_ptr[12];
    int dominated = 0;
    for (int col = threadIdx.x; col < n; col += blockDim.x) {
        const double* col_ptr = features + col * 13;
        if (dominates_row(col_ptr, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, col, row)) {
            dominated = 1;
        }
    }
    const int any_dominated = __syncthreads_or(dominated);
    if (threadIdx.x == 0) keep[row] = any_dominated ? 0 : 1;
}

}  // namespace

torch::Tensor frontier_keep_mask(torch::Tensor features) {
    TORCH_CHECK(features.is_cuda(), "features must be CUDA tensor");
    TORCH_CHECK(features.scalar_type() == torch::kFloat64, "features must be float64");
    TORCH_CHECK(features.dim() == 2 && features.size(1) == 13, "features must have shape [N, 13]");
    auto contiguous = features.contiguous();
    const int n = static_cast<int>(contiguous.size(0));
    auto keep = torch::empty({n}, torch::TensorOptions().device(contiguous.device()).dtype(torch::kInt8));
    if (n == 0) return keep;
    int threads = 256;
    if (n <= 32) {
        threads = 32;
    } else if (n <= 64) {
        threads = 64;
    } else if (n <= 128) {
        threads = 128;
    }
    frontier_keep_mask_kernel<<<n, threads>>>(contiguous.data_ptr<double>(), keep.data_ptr<int8_t>(), n);
    return keep;
}
"""
    _MODULE = load_inline(
        name="danks_gpu_frontier_v9",
        cpp_sources=[cpp_sources],
        cuda_sources=[cuda_sources],
        functions=["frontier_keep_mask"],
        extra_cuda_cflags=["-O3"],
        verbose=os.environ.get("DANKS_VERBOSE_CUDA_BUILD", "").strip().lower() in {"1", "true", "yes", "on"},
    )
    return _MODULE


def frontier_keep_mask(features: torch.Tensor) -> torch.Tensor:
    return module().frontier_keep_mask(features)
