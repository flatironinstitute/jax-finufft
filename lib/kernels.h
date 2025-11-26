// GPU kernel declarations for jax-finufft using XLA typed FFI (api_version=4).
//
// This module declares FINUFFT GPU operations as XLA FFI custom calls with
// typed buffer interfaces for CUDA execution.

#ifndef _JAX_FINUFFT_KERNELS_H_
#define _JAX_FINUFFT_KERNELS_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace jax_finufft {
namespace gpu {

// =============================================================================
// Type 1 NUFFT handlers (non-uniform to uniform)
// =============================================================================

ffi::Error nufft1d1f_impl(
    cudaStream_t stream, float eps, int64_t iflag, int64_t n_tot,
    int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2, int64_t n_k_3,
    int64_t modeord, double upsampfac, int64_t gpu_method, int64_t gpu_sort,
    int64_t gpu_kerevalmeth, int64_t gpu_maxbatchsize, int64_t debug,
    ffi::AnyBuffer source, ffi::AnyBuffer points_x, ffi::AnyBuffer points_y,
    ffi::AnyBuffer points_z, ffi::Result<ffi::AnyBuffer> output);

ffi::Error nufft1d1_impl(
    cudaStream_t stream, double eps, int64_t iflag, int64_t n_tot,
    int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2, int64_t n_k_3,
    int64_t modeord, double upsampfac, int64_t gpu_method, int64_t gpu_sort,
    int64_t gpu_kerevalmeth, int64_t gpu_maxbatchsize, int64_t debug,
    ffi::AnyBuffer source, ffi::AnyBuffer points_x, ffi::AnyBuffer points_y,
    ffi::AnyBuffer points_z, ffi::Result<ffi::AnyBuffer> output);

ffi::Error nufft2d1f_impl(
    cudaStream_t stream, float eps, int64_t iflag, int64_t n_tot,
    int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2, int64_t n_k_3,
    int64_t modeord, double upsampfac, int64_t gpu_method, int64_t gpu_sort,
    int64_t gpu_kerevalmeth, int64_t gpu_maxbatchsize, int64_t debug,
    ffi::AnyBuffer source, ffi::AnyBuffer points_x, ffi::AnyBuffer points_y,
    ffi::AnyBuffer points_z, ffi::Result<ffi::AnyBuffer> output);

ffi::Error nufft2d1_impl(
    cudaStream_t stream, double eps, int64_t iflag, int64_t n_tot,
    int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2, int64_t n_k_3,
    int64_t modeord, double upsampfac, int64_t gpu_method, int64_t gpu_sort,
    int64_t gpu_kerevalmeth, int64_t gpu_maxbatchsize, int64_t debug,
    ffi::AnyBuffer source, ffi::AnyBuffer points_x, ffi::AnyBuffer points_y,
    ffi::AnyBuffer points_z, ffi::Result<ffi::AnyBuffer> output);

ffi::Error nufft3d1f_impl(
    cudaStream_t stream, float eps, int64_t iflag, int64_t n_tot,
    int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2, int64_t n_k_3,
    int64_t modeord, double upsampfac, int64_t gpu_method, int64_t gpu_sort,
    int64_t gpu_kerevalmeth, int64_t gpu_maxbatchsize, int64_t debug,
    ffi::AnyBuffer source, ffi::AnyBuffer points_x, ffi::AnyBuffer points_y,
    ffi::AnyBuffer points_z, ffi::Result<ffi::AnyBuffer> output);

ffi::Error nufft3d1_impl(
    cudaStream_t stream, double eps, int64_t iflag, int64_t n_tot,
    int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2, int64_t n_k_3,
    int64_t modeord, double upsampfac, int64_t gpu_method, int64_t gpu_sort,
    int64_t gpu_kerevalmeth, int64_t gpu_maxbatchsize, int64_t debug,
    ffi::AnyBuffer source, ffi::AnyBuffer points_x, ffi::AnyBuffer points_y,
    ffi::AnyBuffer points_z, ffi::Result<ffi::AnyBuffer> output);

// =============================================================================
// Type 2 NUFFT handlers (uniform to non-uniform)
// =============================================================================

ffi::Error nufft1d2f_impl(
    cudaStream_t stream, float eps, int64_t iflag, int64_t n_tot,
    int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2, int64_t n_k_3,
    int64_t modeord, double upsampfac, int64_t gpu_method, int64_t gpu_sort,
    int64_t gpu_kerevalmeth, int64_t gpu_maxbatchsize, int64_t debug,
    ffi::AnyBuffer source, ffi::AnyBuffer points_x, ffi::AnyBuffer points_y,
    ffi::AnyBuffer points_z, ffi::Result<ffi::AnyBuffer> output);

ffi::Error nufft1d2_impl(
    cudaStream_t stream, double eps, int64_t iflag, int64_t n_tot,
    int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2, int64_t n_k_3,
    int64_t modeord, double upsampfac, int64_t gpu_method, int64_t gpu_sort,
    int64_t gpu_kerevalmeth, int64_t gpu_maxbatchsize, int64_t debug,
    ffi::AnyBuffer source, ffi::AnyBuffer points_x, ffi::AnyBuffer points_y,
    ffi::AnyBuffer points_z, ffi::Result<ffi::AnyBuffer> output);

ffi::Error nufft2d2f_impl(
    cudaStream_t stream, float eps, int64_t iflag, int64_t n_tot,
    int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2, int64_t n_k_3,
    int64_t modeord, double upsampfac, int64_t gpu_method, int64_t gpu_sort,
    int64_t gpu_kerevalmeth, int64_t gpu_maxbatchsize, int64_t debug,
    ffi::AnyBuffer source, ffi::AnyBuffer points_x, ffi::AnyBuffer points_y,
    ffi::AnyBuffer points_z, ffi::Result<ffi::AnyBuffer> output);

ffi::Error nufft2d2_impl(
    cudaStream_t stream, double eps, int64_t iflag, int64_t n_tot,
    int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2, int64_t n_k_3,
    int64_t modeord, double upsampfac, int64_t gpu_method, int64_t gpu_sort,
    int64_t gpu_kerevalmeth, int64_t gpu_maxbatchsize, int64_t debug,
    ffi::AnyBuffer source, ffi::AnyBuffer points_x, ffi::AnyBuffer points_y,
    ffi::AnyBuffer points_z, ffi::Result<ffi::AnyBuffer> output);

ffi::Error nufft3d2f_impl(
    cudaStream_t stream, float eps, int64_t iflag, int64_t n_tot,
    int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2, int64_t n_k_3,
    int64_t modeord, double upsampfac, int64_t gpu_method, int64_t gpu_sort,
    int64_t gpu_kerevalmeth, int64_t gpu_maxbatchsize, int64_t debug,
    ffi::AnyBuffer source, ffi::AnyBuffer points_x, ffi::AnyBuffer points_y,
    ffi::AnyBuffer points_z, ffi::Result<ffi::AnyBuffer> output);

ffi::Error nufft3d2_impl(
    cudaStream_t stream, double eps, int64_t iflag, int64_t n_tot,
    int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2, int64_t n_k_3,
    int64_t modeord, double upsampfac, int64_t gpu_method, int64_t gpu_sort,
    int64_t gpu_kerevalmeth, int64_t gpu_maxbatchsize, int64_t debug,
    ffi::AnyBuffer source, ffi::AnyBuffer points_x, ffi::AnyBuffer points_y,
    ffi::AnyBuffer points_z, ffi::Result<ffi::AnyBuffer> output);

// =============================================================================
// Type 3 NUFFT handlers (non-uniform to non-uniform)
// =============================================================================

ffi::Error nufft1d3f_impl(
    cudaStream_t stream, float eps, int64_t iflag, int64_t n_tot,
    int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2, int64_t n_k_3,
    int64_t modeord, double upsampfac, int64_t gpu_method, int64_t gpu_sort,
    int64_t gpu_kerevalmeth, int64_t gpu_maxbatchsize, int64_t debug,
    ffi::AnyBuffer source, ffi::AnyBuffer points_x, ffi::AnyBuffer points_y,
    ffi::AnyBuffer points_z, ffi::AnyBuffer target_x, ffi::AnyBuffer target_y,
    ffi::AnyBuffer target_z, ffi::Result<ffi::AnyBuffer> output);

ffi::Error nufft1d3_impl(
    cudaStream_t stream, double eps, int64_t iflag, int64_t n_tot,
    int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2, int64_t n_k_3,
    int64_t modeord, double upsampfac, int64_t gpu_method, int64_t gpu_sort,
    int64_t gpu_kerevalmeth, int64_t gpu_maxbatchsize, int64_t debug,
    ffi::AnyBuffer source, ffi::AnyBuffer points_x, ffi::AnyBuffer points_y,
    ffi::AnyBuffer points_z, ffi::AnyBuffer target_x, ffi::AnyBuffer target_y,
    ffi::AnyBuffer target_z, ffi::Result<ffi::AnyBuffer> output);

ffi::Error nufft2d3f_impl(
    cudaStream_t stream, float eps, int64_t iflag, int64_t n_tot,
    int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2, int64_t n_k_3,
    int64_t modeord, double upsampfac, int64_t gpu_method, int64_t gpu_sort,
    int64_t gpu_kerevalmeth, int64_t gpu_maxbatchsize, int64_t debug,
    ffi::AnyBuffer source, ffi::AnyBuffer points_x, ffi::AnyBuffer points_y,
    ffi::AnyBuffer points_z, ffi::AnyBuffer target_x, ffi::AnyBuffer target_y,
    ffi::AnyBuffer target_z, ffi::Result<ffi::AnyBuffer> output);

ffi::Error nufft2d3_impl(
    cudaStream_t stream, double eps, int64_t iflag, int64_t n_tot,
    int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2, int64_t n_k_3,
    int64_t modeord, double upsampfac, int64_t gpu_method, int64_t gpu_sort,
    int64_t gpu_kerevalmeth, int64_t gpu_maxbatchsize, int64_t debug,
    ffi::AnyBuffer source, ffi::AnyBuffer points_x, ffi::AnyBuffer points_y,
    ffi::AnyBuffer points_z, ffi::AnyBuffer target_x, ffi::AnyBuffer target_y,
    ffi::AnyBuffer target_z, ffi::Result<ffi::AnyBuffer> output);

ffi::Error nufft3d3f_impl(
    cudaStream_t stream, float eps, int64_t iflag, int64_t n_tot,
    int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2, int64_t n_k_3,
    int64_t modeord, double upsampfac, int64_t gpu_method, int64_t gpu_sort,
    int64_t gpu_kerevalmeth, int64_t gpu_maxbatchsize, int64_t debug,
    ffi::AnyBuffer source, ffi::AnyBuffer points_x, ffi::AnyBuffer points_y,
    ffi::AnyBuffer points_z, ffi::AnyBuffer target_x, ffi::AnyBuffer target_y,
    ffi::AnyBuffer target_z, ffi::Result<ffi::AnyBuffer> output);

ffi::Error nufft3d3_impl(
    cudaStream_t stream, double eps, int64_t iflag, int64_t n_tot,
    int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2, int64_t n_k_3,
    int64_t modeord, double upsampfac, int64_t gpu_method, int64_t gpu_sort,
    int64_t gpu_kerevalmeth, int64_t gpu_maxbatchsize, int64_t debug,
    ffi::AnyBuffer source, ffi::AnyBuffer points_x, ffi::AnyBuffer points_y,
    ffi::AnyBuffer points_z, ffi::AnyBuffer target_x, ffi::AnyBuffer target_y,
    ffi::AnyBuffer target_z, ffi::Result<ffi::AnyBuffer> output);

}  // namespace gpu
}  // namespace jax_finufft

#endif
