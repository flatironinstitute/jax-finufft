// GPU kernel implementations for jax-finufft using XLA typed FFI.
//
// This module implements FINUFFT GPU operations as XLA FFI custom calls
// with typed buffer interfaces for CUDA execution.

#include <complex>
#include <cstdint>

#include "cufinufft_wrapper.h"
#include "kernels.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
using namespace jax_finufft::gpu;

namespace jax_finufft {
namespace gpu {

// =============================================================================
// Core NUFFT execution logic
// =============================================================================

/// Execute NUFFT transform for a batch of inputs on GPU.
template <int ndim, typename T, int type>
ffi::Error run_nufft(cudaStream_t stream, cufinufft_opts opts, T eps, int iflag, int64_t n_tot,
                     int n_transf, int64_t n_j, const int64_t* n_k, T* x, T* y, T* z,
                     std::complex<T>* c, T* s, T* t, T* u, std::complex<T>* F) {
  int64_t n_k_total = 1;
  if constexpr (type != 3) {
    for (int d = 0; d < ndim; ++d) {
      n_k_total *= n_k[d];
    }
  } else {
    n_k_total = n_k[0];
  }

  // Set stream and device in options
  update_opts<T>(&opts, ndim, stream);
  int device_ordinal;
  cudaError_t cuda_err = cudaGetDevice(&device_ordinal);
  if (cuda_err != cudaSuccess) {
    return ffi::Error::Internal("cudaGetDevice failed: " +
                                std::string(cudaGetErrorString(cuda_err)));
  }
  device_type device{device_ordinal};
  update_opts<T>(&opts, ndim, device);

  typename plan_type<T>::type plan;
  int64_t n_k_mutable[3] = {n_k[0], n_k[1], n_k[2]};
  int ret = makeplan<T>(type, ndim, n_k_mutable, iflag, n_transf, eps, &plan, &opts);
  // ret == 1 is FINUFFT_WARN_EPS_TOO_SMALL (warning, not error)
  if (ret > 1) {
    return ffi::Error::Internal("cuFINUFFT makeplan failed with code " + std::to_string(ret));
  }

  for (int64_t index = 0; index < n_tot; ++index) {
    int64_t i = index * n_j;
    int64_t j = index * n_j * n_transf;
    int64_t k = index * n_k_total * n_transf;

    if constexpr (type != 3) {
      ret = setpts<T>(plan, n_j, &x[i], y_index<ndim, T>(y, i), z_index<ndim, T>(z, i), 0, nullptr,
                      nullptr, nullptr);
    } else {
      int64_t i_target = index * n_k_total;
      ret = setpts<T>(plan, n_j, &x[i], y_index<ndim, T>(y, i), z_index<ndim, T>(z, i), n_k_total,
                      &s[i_target], y_index<ndim, T>(t, i_target), z_index<ndim, T>(u, i_target));
    }
    if (ret != 0) {
      destroy<T>(plan);
      return ffi::Error::Internal("cuFINUFFT setpts failed with code " + std::to_string(ret));
    }

    ret = execute<T>(plan, &c[j], &F[k]);
    if (ret != 0) {
      destroy<T>(plan);
      return ffi::Error::Internal("cuFINUFFT execute failed with code " + std::to_string(ret));
    }

    // Synchronize after each execute to ensure completion before next setpts
    // (setpts modifies plan state that execute reads asynchronously)
    cudaStreamSynchronize(stream);
  }

  destroy<T>(plan);

  cuda_err = cudaGetLastError();
  if (cuda_err != cudaSuccess) {
    return ffi::Error::Internal("CUDA error: " + std::string(cudaGetErrorString(cuda_err)));
  }

  return ffi::Error::Success();
}

// =============================================================================
// Helper to build cufinufft_opts from attributes
// =============================================================================

template <typename T>
cufinufft_opts build_opts(int64_t modeord, double upsampfac, int64_t gpu_method, int64_t gpu_sort,
                          int64_t gpu_kerevalmeth, int64_t gpu_maxbatchsize, int64_t debug) {
  cufinufft_opts opts;
  default_opts<T>(&opts);
  opts.modeord = static_cast<int>(modeord);
  opts.upsampfac = upsampfac;
  opts.gpu_method = static_cast<int>(gpu_method);
  opts.gpu_sort = static_cast<int>(gpu_sort);
  opts.gpu_kerevalmeth = static_cast<int>(gpu_kerevalmeth);
  opts.gpu_maxbatchsize = static_cast<int>(gpu_maxbatchsize);
  opts.debug = static_cast<int>(debug);
  return opts;
}

// =============================================================================
// Type 1 NUFFT implementations (non-uniform to uniform)
// =============================================================================

template <int ndim, typename T>
ffi::Error nufft1_impl(cudaStream_t stream, T eps, int64_t iflag, int64_t n_tot, int64_t n_transf,
                       int64_t n_j, int64_t n_k_1, int64_t n_k_2, int64_t n_k_3, int64_t modeord,
                       double upsampfac, int64_t gpu_method, int64_t gpu_sort,
                       int64_t gpu_kerevalmeth, int64_t gpu_maxbatchsize, int64_t debug,
                       ffi::AnyBuffer source, ffi::AnyBuffer points_x, ffi::AnyBuffer points_y,
                       ffi::AnyBuffer points_z, ffi::Result<ffi::AnyBuffer> output) {
  cufinufft_opts opts = build_opts<T>(modeord, upsampfac, gpu_method, gpu_sort, gpu_kerevalmeth,
                                      gpu_maxbatchsize, debug);

  int64_t n_k[3] = {n_k_1, n_k_2, n_k_3};

  auto* c = reinterpret_cast<std::complex<T>*>(source.untyped_data());
  auto* x = reinterpret_cast<T*>(points_x.untyped_data());
  T* y = nullptr;
  T* z = nullptr;
  if constexpr (ndim > 1) {
    y = reinterpret_cast<T*>(points_y.untyped_data());
  }
  if constexpr (ndim > 2) {
    z = reinterpret_cast<T*>(points_z.untyped_data());
  }
  auto* F = reinterpret_cast<std::complex<T>*>(output->untyped_data());

  return run_nufft<ndim, T, 1>(stream, opts, eps, static_cast<int>(iflag), n_tot,
                               static_cast<int>(n_transf), n_j, n_k, x, y, z, c, nullptr, nullptr,
                               nullptr, F);
}

// =============================================================================
// Type 2 NUFFT implementations (uniform to non-uniform)
// =============================================================================

template <int ndim, typename T>
ffi::Error nufft2_impl(cudaStream_t stream, T eps, int64_t iflag, int64_t n_tot, int64_t n_transf,
                       int64_t n_j, int64_t n_k_1, int64_t n_k_2, int64_t n_k_3, int64_t modeord,
                       double upsampfac, int64_t gpu_method, int64_t gpu_sort,
                       int64_t gpu_kerevalmeth, int64_t gpu_maxbatchsize, int64_t debug,
                       ffi::AnyBuffer source, ffi::AnyBuffer points_x, ffi::AnyBuffer points_y,
                       ffi::AnyBuffer points_z, ffi::Result<ffi::AnyBuffer> output) {
  cufinufft_opts opts = build_opts<T>(modeord, upsampfac, gpu_method, gpu_sort, gpu_kerevalmeth,
                                      gpu_maxbatchsize, debug);

  int64_t n_k[3] = {n_k_1, n_k_2, n_k_3};

  auto* F = reinterpret_cast<std::complex<T>*>(source.untyped_data());
  auto* x = reinterpret_cast<T*>(points_x.untyped_data());
  T* y = nullptr;
  T* z = nullptr;
  if constexpr (ndim > 1) {
    y = reinterpret_cast<T*>(points_y.untyped_data());
  }
  if constexpr (ndim > 2) {
    z = reinterpret_cast<T*>(points_z.untyped_data());
  }
  auto* c = reinterpret_cast<std::complex<T>*>(output->untyped_data());

  return run_nufft<ndim, T, 2>(stream, opts, eps, static_cast<int>(iflag), n_tot,
                               static_cast<int>(n_transf), n_j, n_k, x, y, z, c, nullptr, nullptr,
                               nullptr, F);
}

// =============================================================================
// Type 3 NUFFT implementations (non-uniform to non-uniform)
// =============================================================================

template <int ndim, typename T>
ffi::Error nufft3_impl(cudaStream_t stream, T eps, int64_t iflag, int64_t n_tot, int64_t n_transf,
                       int64_t n_j, int64_t n_k_1, int64_t n_k_2, int64_t n_k_3, int64_t modeord,
                       double upsampfac, int64_t gpu_method, int64_t gpu_sort,
                       int64_t gpu_kerevalmeth, int64_t gpu_maxbatchsize, int64_t debug,
                       ffi::AnyBuffer source, ffi::AnyBuffer points_x, ffi::AnyBuffer points_y,
                       ffi::AnyBuffer points_z, ffi::AnyBuffer target_x, ffi::AnyBuffer target_y,
                       ffi::AnyBuffer target_z, ffi::Result<ffi::AnyBuffer> output) {
  cufinufft_opts opts = build_opts<T>(modeord, upsampfac, gpu_method, gpu_sort, gpu_kerevalmeth,
                                      gpu_maxbatchsize, debug);

  int64_t n_k[3] = {n_k_1, n_k_2, n_k_3};

  auto* c = reinterpret_cast<std::complex<T>*>(source.untyped_data());
  auto* x = reinterpret_cast<T*>(points_x.untyped_data());
  T* y = nullptr;
  T* z = nullptr;
  auto* s = reinterpret_cast<T*>(target_x.untyped_data());
  T* t = nullptr;
  T* u = nullptr;
  if constexpr (ndim > 1) {
    y = reinterpret_cast<T*>(points_y.untyped_data());
    t = reinterpret_cast<T*>(target_y.untyped_data());
  }
  if constexpr (ndim > 2) {
    z = reinterpret_cast<T*>(points_z.untyped_data());
    u = reinterpret_cast<T*>(target_z.untyped_data());
  }
  auto* F = reinterpret_cast<std::complex<T>*>(output->untyped_data());

  return run_nufft<ndim, T, 3>(stream, opts, eps, static_cast<int>(iflag), n_tot,
                               static_cast<int>(n_transf), n_j, n_k, x, y, z, c, s, t, u, F);
}

// =============================================================================
// Explicit template instantiations - Type 1
// =============================================================================

ffi::Error nufft1d1f_impl(cudaStream_t stream, float eps, int64_t iflag, int64_t n_tot,
                          int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2,
                          int64_t n_k_3, int64_t modeord, double upsampfac, int64_t gpu_method,
                          int64_t gpu_sort, int64_t gpu_kerevalmeth, int64_t gpu_maxbatchsize,
                          int64_t debug, ffi::AnyBuffer source, ffi::AnyBuffer points_x,
                          ffi::AnyBuffer points_y, ffi::AnyBuffer points_z,
                          ffi::Result<ffi::AnyBuffer> output) {
  return nufft1_impl<1, float>(stream, eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3,
                               modeord, upsampfac, gpu_method, gpu_sort, gpu_kerevalmeth,
                               gpu_maxbatchsize, debug, source, points_x, points_y, points_z,
                               output);
}

ffi::Error nufft1d1_impl(cudaStream_t stream, double eps, int64_t iflag, int64_t n_tot,
                         int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2,
                         int64_t n_k_3, int64_t modeord, double upsampfac, int64_t gpu_method,
                         int64_t gpu_sort, int64_t gpu_kerevalmeth, int64_t gpu_maxbatchsize,
                         int64_t debug, ffi::AnyBuffer source, ffi::AnyBuffer points_x,
                         ffi::AnyBuffer points_y, ffi::AnyBuffer points_z,
                         ffi::Result<ffi::AnyBuffer> output) {
  return nufft1_impl<1, double>(stream, eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3,
                                modeord, upsampfac, gpu_method, gpu_sort, gpu_kerevalmeth,
                                gpu_maxbatchsize, debug, source, points_x, points_y, points_z,
                                output);
}

ffi::Error nufft2d1f_impl(cudaStream_t stream, float eps, int64_t iflag, int64_t n_tot,
                          int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2,
                          int64_t n_k_3, int64_t modeord, double upsampfac, int64_t gpu_method,
                          int64_t gpu_sort, int64_t gpu_kerevalmeth, int64_t gpu_maxbatchsize,
                          int64_t debug, ffi::AnyBuffer source, ffi::AnyBuffer points_x,
                          ffi::AnyBuffer points_y, ffi::AnyBuffer points_z,
                          ffi::Result<ffi::AnyBuffer> output) {
  return nufft1_impl<2, float>(stream, eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3,
                               modeord, upsampfac, gpu_method, gpu_sort, gpu_kerevalmeth,
                               gpu_maxbatchsize, debug, source, points_x, points_y, points_z,
                               output);
}

ffi::Error nufft2d1_impl(cudaStream_t stream, double eps, int64_t iflag, int64_t n_tot,
                         int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2,
                         int64_t n_k_3, int64_t modeord, double upsampfac, int64_t gpu_method,
                         int64_t gpu_sort, int64_t gpu_kerevalmeth, int64_t gpu_maxbatchsize,
                         int64_t debug, ffi::AnyBuffer source, ffi::AnyBuffer points_x,
                         ffi::AnyBuffer points_y, ffi::AnyBuffer points_z,
                         ffi::Result<ffi::AnyBuffer> output) {
  return nufft1_impl<2, double>(stream, eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3,
                                modeord, upsampfac, gpu_method, gpu_sort, gpu_kerevalmeth,
                                gpu_maxbatchsize, debug, source, points_x, points_y, points_z,
                                output);
}

ffi::Error nufft3d1f_impl(cudaStream_t stream, float eps, int64_t iflag, int64_t n_tot,
                          int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2,
                          int64_t n_k_3, int64_t modeord, double upsampfac, int64_t gpu_method,
                          int64_t gpu_sort, int64_t gpu_kerevalmeth, int64_t gpu_maxbatchsize,
                          int64_t debug, ffi::AnyBuffer source, ffi::AnyBuffer points_x,
                          ffi::AnyBuffer points_y, ffi::AnyBuffer points_z,
                          ffi::Result<ffi::AnyBuffer> output) {
  return nufft1_impl<3, float>(stream, eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3,
                               modeord, upsampfac, gpu_method, gpu_sort, gpu_kerevalmeth,
                               gpu_maxbatchsize, debug, source, points_x, points_y, points_z,
                               output);
}

ffi::Error nufft3d1_impl(cudaStream_t stream, double eps, int64_t iflag, int64_t n_tot,
                         int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2,
                         int64_t n_k_3, int64_t modeord, double upsampfac, int64_t gpu_method,
                         int64_t gpu_sort, int64_t gpu_kerevalmeth, int64_t gpu_maxbatchsize,
                         int64_t debug, ffi::AnyBuffer source, ffi::AnyBuffer points_x,
                         ffi::AnyBuffer points_y, ffi::AnyBuffer points_z,
                         ffi::Result<ffi::AnyBuffer> output) {
  return nufft1_impl<3, double>(stream, eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3,
                                modeord, upsampfac, gpu_method, gpu_sort, gpu_kerevalmeth,
                                gpu_maxbatchsize, debug, source, points_x, points_y, points_z,
                                output);
}

// =============================================================================
// Explicit template instantiations - Type 2
// =============================================================================

ffi::Error nufft1d2f_impl(cudaStream_t stream, float eps, int64_t iflag, int64_t n_tot,
                          int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2,
                          int64_t n_k_3, int64_t modeord, double upsampfac, int64_t gpu_method,
                          int64_t gpu_sort, int64_t gpu_kerevalmeth, int64_t gpu_maxbatchsize,
                          int64_t debug, ffi::AnyBuffer source, ffi::AnyBuffer points_x,
                          ffi::AnyBuffer points_y, ffi::AnyBuffer points_z,
                          ffi::Result<ffi::AnyBuffer> output) {
  return nufft2_impl<1, float>(stream, eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3,
                               modeord, upsampfac, gpu_method, gpu_sort, gpu_kerevalmeth,
                               gpu_maxbatchsize, debug, source, points_x, points_y, points_z,
                               output);
}

ffi::Error nufft1d2_impl(cudaStream_t stream, double eps, int64_t iflag, int64_t n_tot,
                         int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2,
                         int64_t n_k_3, int64_t modeord, double upsampfac, int64_t gpu_method,
                         int64_t gpu_sort, int64_t gpu_kerevalmeth, int64_t gpu_maxbatchsize,
                         int64_t debug, ffi::AnyBuffer source, ffi::AnyBuffer points_x,
                         ffi::AnyBuffer points_y, ffi::AnyBuffer points_z,
                         ffi::Result<ffi::AnyBuffer> output) {
  return nufft2_impl<1, double>(stream, eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3,
                                modeord, upsampfac, gpu_method, gpu_sort, gpu_kerevalmeth,
                                gpu_maxbatchsize, debug, source, points_x, points_y, points_z,
                                output);
}

ffi::Error nufft2d2f_impl(cudaStream_t stream, float eps, int64_t iflag, int64_t n_tot,
                          int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2,
                          int64_t n_k_3, int64_t modeord, double upsampfac, int64_t gpu_method,
                          int64_t gpu_sort, int64_t gpu_kerevalmeth, int64_t gpu_maxbatchsize,
                          int64_t debug, ffi::AnyBuffer source, ffi::AnyBuffer points_x,
                          ffi::AnyBuffer points_y, ffi::AnyBuffer points_z,
                          ffi::Result<ffi::AnyBuffer> output) {
  return nufft2_impl<2, float>(stream, eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3,
                               modeord, upsampfac, gpu_method, gpu_sort, gpu_kerevalmeth,
                               gpu_maxbatchsize, debug, source, points_x, points_y, points_z,
                               output);
}

ffi::Error nufft2d2_impl(cudaStream_t stream, double eps, int64_t iflag, int64_t n_tot,
                         int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2,
                         int64_t n_k_3, int64_t modeord, double upsampfac, int64_t gpu_method,
                         int64_t gpu_sort, int64_t gpu_kerevalmeth, int64_t gpu_maxbatchsize,
                         int64_t debug, ffi::AnyBuffer source, ffi::AnyBuffer points_x,
                         ffi::AnyBuffer points_y, ffi::AnyBuffer points_z,
                         ffi::Result<ffi::AnyBuffer> output) {
  return nufft2_impl<2, double>(stream, eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3,
                                modeord, upsampfac, gpu_method, gpu_sort, gpu_kerevalmeth,
                                gpu_maxbatchsize, debug, source, points_x, points_y, points_z,
                                output);
}

ffi::Error nufft3d2f_impl(cudaStream_t stream, float eps, int64_t iflag, int64_t n_tot,
                          int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2,
                          int64_t n_k_3, int64_t modeord, double upsampfac, int64_t gpu_method,
                          int64_t gpu_sort, int64_t gpu_kerevalmeth, int64_t gpu_maxbatchsize,
                          int64_t debug, ffi::AnyBuffer source, ffi::AnyBuffer points_x,
                          ffi::AnyBuffer points_y, ffi::AnyBuffer points_z,
                          ffi::Result<ffi::AnyBuffer> output) {
  return nufft2_impl<3, float>(stream, eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3,
                               modeord, upsampfac, gpu_method, gpu_sort, gpu_kerevalmeth,
                               gpu_maxbatchsize, debug, source, points_x, points_y, points_z,
                               output);
}

ffi::Error nufft3d2_impl(cudaStream_t stream, double eps, int64_t iflag, int64_t n_tot,
                         int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2,
                         int64_t n_k_3, int64_t modeord, double upsampfac, int64_t gpu_method,
                         int64_t gpu_sort, int64_t gpu_kerevalmeth, int64_t gpu_maxbatchsize,
                         int64_t debug, ffi::AnyBuffer source, ffi::AnyBuffer points_x,
                         ffi::AnyBuffer points_y, ffi::AnyBuffer points_z,
                         ffi::Result<ffi::AnyBuffer> output) {
  return nufft2_impl<3, double>(stream, eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3,
                                modeord, upsampfac, gpu_method, gpu_sort, gpu_kerevalmeth,
                                gpu_maxbatchsize, debug, source, points_x, points_y, points_z,
                                output);
}

// =============================================================================
// Explicit template instantiations - Type 3
// =============================================================================

ffi::Error nufft1d3f_impl(cudaStream_t stream, float eps, int64_t iflag, int64_t n_tot,
                          int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2,
                          int64_t n_k_3, int64_t modeord, double upsampfac, int64_t gpu_method,
                          int64_t gpu_sort, int64_t gpu_kerevalmeth, int64_t gpu_maxbatchsize,
                          int64_t debug, ffi::AnyBuffer source, ffi::AnyBuffer points_x,
                          ffi::AnyBuffer points_y, ffi::AnyBuffer points_z,
                          ffi::AnyBuffer target_x, ffi::AnyBuffer target_y,
                          ffi::AnyBuffer target_z, ffi::Result<ffi::AnyBuffer> output) {
  return nufft3_impl<1, float>(stream, eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3,
                               modeord, upsampfac, gpu_method, gpu_sort, gpu_kerevalmeth,
                               gpu_maxbatchsize, debug, source, points_x, points_y, points_z,
                               target_x, target_y, target_z, output);
}

ffi::Error nufft1d3_impl(cudaStream_t stream, double eps, int64_t iflag, int64_t n_tot,
                         int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2,
                         int64_t n_k_3, int64_t modeord, double upsampfac, int64_t gpu_method,
                         int64_t gpu_sort, int64_t gpu_kerevalmeth, int64_t gpu_maxbatchsize,
                         int64_t debug, ffi::AnyBuffer source, ffi::AnyBuffer points_x,
                         ffi::AnyBuffer points_y, ffi::AnyBuffer points_z, ffi::AnyBuffer target_x,
                         ffi::AnyBuffer target_y, ffi::AnyBuffer target_z,
                         ffi::Result<ffi::AnyBuffer> output) {
  return nufft3_impl<1, double>(stream, eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3,
                                modeord, upsampfac, gpu_method, gpu_sort, gpu_kerevalmeth,
                                gpu_maxbatchsize, debug, source, points_x, points_y, points_z,
                                target_x, target_y, target_z, output);
}

ffi::Error nufft2d3f_impl(cudaStream_t stream, float eps, int64_t iflag, int64_t n_tot,
                          int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2,
                          int64_t n_k_3, int64_t modeord, double upsampfac, int64_t gpu_method,
                          int64_t gpu_sort, int64_t gpu_kerevalmeth, int64_t gpu_maxbatchsize,
                          int64_t debug, ffi::AnyBuffer source, ffi::AnyBuffer points_x,
                          ffi::AnyBuffer points_y, ffi::AnyBuffer points_z,
                          ffi::AnyBuffer target_x, ffi::AnyBuffer target_y,
                          ffi::AnyBuffer target_z, ffi::Result<ffi::AnyBuffer> output) {
  return nufft3_impl<2, float>(stream, eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3,
                               modeord, upsampfac, gpu_method, gpu_sort, gpu_kerevalmeth,
                               gpu_maxbatchsize, debug, source, points_x, points_y, points_z,
                               target_x, target_y, target_z, output);
}

ffi::Error nufft2d3_impl(cudaStream_t stream, double eps, int64_t iflag, int64_t n_tot,
                         int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2,
                         int64_t n_k_3, int64_t modeord, double upsampfac, int64_t gpu_method,
                         int64_t gpu_sort, int64_t gpu_kerevalmeth, int64_t gpu_maxbatchsize,
                         int64_t debug, ffi::AnyBuffer source, ffi::AnyBuffer points_x,
                         ffi::AnyBuffer points_y, ffi::AnyBuffer points_z, ffi::AnyBuffer target_x,
                         ffi::AnyBuffer target_y, ffi::AnyBuffer target_z,
                         ffi::Result<ffi::AnyBuffer> output) {
  return nufft3_impl<2, double>(stream, eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3,
                                modeord, upsampfac, gpu_method, gpu_sort, gpu_kerevalmeth,
                                gpu_maxbatchsize, debug, source, points_x, points_y, points_z,
                                target_x, target_y, target_z, output);
}

ffi::Error nufft3d3f_impl(cudaStream_t stream, float eps, int64_t iflag, int64_t n_tot,
                          int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2,
                          int64_t n_k_3, int64_t modeord, double upsampfac, int64_t gpu_method,
                          int64_t gpu_sort, int64_t gpu_kerevalmeth, int64_t gpu_maxbatchsize,
                          int64_t debug, ffi::AnyBuffer source, ffi::AnyBuffer points_x,
                          ffi::AnyBuffer points_y, ffi::AnyBuffer points_z,
                          ffi::AnyBuffer target_x, ffi::AnyBuffer target_y,
                          ffi::AnyBuffer target_z, ffi::Result<ffi::AnyBuffer> output) {
  return nufft3_impl<3, float>(stream, eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3,
                               modeord, upsampfac, gpu_method, gpu_sort, gpu_kerevalmeth,
                               gpu_maxbatchsize, debug, source, points_x, points_y, points_z,
                               target_x, target_y, target_z, output);
}

ffi::Error nufft3d3_impl(cudaStream_t stream, double eps, int64_t iflag, int64_t n_tot,
                         int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2,
                         int64_t n_k_3, int64_t modeord, double upsampfac, int64_t gpu_method,
                         int64_t gpu_sort, int64_t gpu_kerevalmeth, int64_t gpu_maxbatchsize,
                         int64_t debug, ffi::AnyBuffer source, ffi::AnyBuffer points_x,
                         ffi::AnyBuffer points_y, ffi::AnyBuffer points_z, ffi::AnyBuffer target_x,
                         ffi::AnyBuffer target_y, ffi::AnyBuffer target_z,
                         ffi::Result<ffi::AnyBuffer> output) {
  return nufft3_impl<3, double>(stream, eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3,
                                modeord, upsampfac, gpu_method, gpu_sort, gpu_kerevalmeth,
                                gpu_maxbatchsize, debug, source, points_x, points_y, points_z,
                                target_x, target_y, target_z, output);
}

}  // namespace gpu
}  // namespace jax_finufft
