// CPU FFI bindings for jax-finufft using XLA typed FFI (api_version=4).
//
// This module exposes FINUFFT operations as XLA FFI custom calls with typed
// buffer interfaces, enabling seamless integration with JAX's compilation
// pipeline.

#include "jax_finufft_cpu.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>

#include <complex>
#include <cstdint>
#include <type_traits>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
namespace nb = nanobind;

namespace jax_finufft {
namespace cpu {

// =============================================================================
// Core NUFFT execution logic
// =============================================================================

/// Execute NUFFT transform for a batch of inputs.
template <int ndim, typename T, int type>
ffi::Error run_nufft(finufft_opts opts, T eps, int iflag, int64_t n_tot, int n_transf, int64_t n_j,
                     const int64_t* n_k, T* x, T* y, T* z, std::complex<T>* c, T* s, T* t, T* u,
                     std::complex<T>* F) {
  int64_t n_k_total = 1;
  if constexpr (type != 3) {
    for (int d = 0; d < ndim; ++d) {
      n_k_total *= n_k[d];
    }
  } else {
    n_k_total = n_k[0];
  }

  typename plan_type<T>::type plan;
  int64_t n_k_mutable[3] = {n_k[0], n_k[1], n_k[2]};
  int ret = makeplan<T>(type, ndim, n_k_mutable, iflag, n_transf, eps, &plan, &opts);
  // ret == 1 is FINUFFT_WARN_EPS_TOO_SMALL (warning, not error)
  if (ret > 1) {
    return ffi::Error::Internal("FINUFFT makeplan failed with code " + std::to_string(ret));
  }

  for (int64_t index = 0; index < n_tot; ++index) {
    int64_t i = index * n_j;
    int64_t j = i * n_transf;
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
      return ffi::Error::Internal("FINUFFT setpts failed with code " + std::to_string(ret));
    }

    ret = execute<T>(plan, &c[j], &F[k]);
    if (ret != 0) {
      destroy<T>(plan);
      return ffi::Error::Internal("FINUFFT execute failed with code " + std::to_string(ret));
    }
  }

  destroy<T>(plan);
  return ffi::Error::Success();
}

// =============================================================================
// Type 1 NUFFT: Non-uniform to uniform (adjoint)
// =============================================================================

template <int ndim, typename T>
ffi::Error nufft1_impl(T eps, int64_t iflag, int64_t n_tot, int64_t n_transf, int64_t n_j,
                       int64_t n_k_1, int64_t n_k_2, int64_t n_k_3, int64_t modeord, int64_t debug,
                       int64_t spread_debug, int64_t showwarn, int64_t nthreads, int64_t fftw,
                       int64_t spread_sort, int64_t spread_kerevalmeth, int64_t spread_kerpad,
                       double upsampfac, int64_t spread_thread, int64_t maxbatchsize,
                       int64_t spread_nthr_atomic, int64_t spread_max_sp_size,
                       ffi::AnyBuffer source, ffi::AnyBuffer points_x, ffi::AnyBuffer points_y,
                       ffi::AnyBuffer points_z, ffi::Result<ffi::AnyBuffer> output) {
  finufft_opts opts;
  default_opts<T>(&opts);
  opts.modeord = static_cast<int>(modeord);
  opts.debug = static_cast<int>(debug);
  opts.spread_debug = static_cast<int>(spread_debug);
  opts.showwarn = static_cast<int>(showwarn);
  opts.nthreads = static_cast<int>(nthreads);
  opts.fftw = static_cast<int>(fftw);
  opts.spread_sort = static_cast<int>(spread_sort);
  opts.spread_kerevalmeth = static_cast<int>(spread_kerevalmeth);
  opts.spread_kerpad = static_cast<int>(spread_kerpad);
  opts.upsampfac = upsampfac;
  opts.spread_thread = static_cast<int>(spread_thread);
  opts.maxbatchsize = static_cast<int>(maxbatchsize);
  opts.spread_nthr_atomic = static_cast<int>(spread_nthr_atomic);
  opts.spread_max_sp_size = static_cast<int>(spread_max_sp_size);

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

  return run_nufft<ndim, T, 1>(opts, eps, static_cast<int>(iflag), n_tot,
                               static_cast<int>(n_transf), n_j, n_k, x, y, z, c, nullptr, nullptr,
                               nullptr, F);
}

// =============================================================================
// Type 2 NUFFT: Uniform to non-uniform (forward)
// =============================================================================

template <int ndim, typename T>
ffi::Error nufft2_impl(T eps, int64_t iflag, int64_t n_tot, int64_t n_transf, int64_t n_j,
                       int64_t n_k_1, int64_t n_k_2, int64_t n_k_3, int64_t modeord, int64_t debug,
                       int64_t spread_debug, int64_t showwarn, int64_t nthreads, int64_t fftw,
                       int64_t spread_sort, int64_t spread_kerevalmeth, int64_t spread_kerpad,
                       double upsampfac, int64_t spread_thread, int64_t maxbatchsize,
                       int64_t spread_nthr_atomic, int64_t spread_max_sp_size,
                       ffi::AnyBuffer source, ffi::AnyBuffer points_x, ffi::AnyBuffer points_y,
                       ffi::AnyBuffer points_z, ffi::Result<ffi::AnyBuffer> output) {
  finufft_opts opts;
  default_opts<T>(&opts);
  opts.modeord = static_cast<int>(modeord);
  opts.debug = static_cast<int>(debug);
  opts.spread_debug = static_cast<int>(spread_debug);
  opts.showwarn = static_cast<int>(showwarn);
  opts.nthreads = static_cast<int>(nthreads);
  opts.fftw = static_cast<int>(fftw);
  opts.spread_sort = static_cast<int>(spread_sort);
  opts.spread_kerevalmeth = static_cast<int>(spread_kerevalmeth);
  opts.spread_kerpad = static_cast<int>(spread_kerpad);
  opts.upsampfac = upsampfac;
  opts.spread_thread = static_cast<int>(spread_thread);
  opts.maxbatchsize = static_cast<int>(maxbatchsize);
  opts.spread_nthr_atomic = static_cast<int>(spread_nthr_atomic);
  opts.spread_max_sp_size = static_cast<int>(spread_max_sp_size);

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

  return run_nufft<ndim, T, 2>(opts, eps, static_cast<int>(iflag), n_tot,
                               static_cast<int>(n_transf), n_j, n_k, x, y, z, c, nullptr, nullptr,
                               nullptr, F);
}

// =============================================================================
// Type 3 NUFFT: Non-uniform to non-uniform
// =============================================================================

template <int ndim, typename T>
ffi::Error nufft3_impl(T eps, int64_t iflag, int64_t n_tot, int64_t n_transf, int64_t n_j,
                       int64_t n_k_1, int64_t n_k_2, int64_t n_k_3, int64_t modeord, int64_t debug,
                       int64_t spread_debug, int64_t showwarn, int64_t nthreads, int64_t fftw,
                       int64_t spread_sort, int64_t spread_kerevalmeth, int64_t spread_kerpad,
                       double upsampfac, int64_t spread_thread, int64_t maxbatchsize,
                       int64_t spread_nthr_atomic, int64_t spread_max_sp_size,
                       ffi::AnyBuffer source, ffi::AnyBuffer points_x, ffi::AnyBuffer points_y,
                       ffi::AnyBuffer points_z, ffi::AnyBuffer targets_s, ffi::AnyBuffer targets_t,
                       ffi::AnyBuffer targets_u, ffi::Result<ffi::AnyBuffer> output) {
  finufft_opts opts;
  default_opts<T>(&opts);
  opts.modeord = static_cast<int>(modeord);
  opts.debug = static_cast<int>(debug);
  opts.spread_debug = static_cast<int>(spread_debug);
  opts.showwarn = static_cast<int>(showwarn);
  opts.nthreads = static_cast<int>(nthreads);
  opts.fftw = static_cast<int>(fftw);
  opts.spread_sort = static_cast<int>(spread_sort);
  opts.spread_kerevalmeth = static_cast<int>(spread_kerevalmeth);
  opts.spread_kerpad = static_cast<int>(spread_kerpad);
  opts.upsampfac = upsampfac;
  opts.spread_thread = static_cast<int>(spread_thread);
  opts.maxbatchsize = static_cast<int>(maxbatchsize);
  opts.spread_nthr_atomic = static_cast<int>(spread_nthr_atomic);
  opts.spread_max_sp_size = static_cast<int>(spread_max_sp_size);

  int64_t n_k[3] = {n_k_1, n_k_2, n_k_3};

  auto* c = reinterpret_cast<std::complex<T>*>(source.untyped_data());
  auto* x = reinterpret_cast<T*>(points_x.untyped_data());
  auto* s = reinterpret_cast<T*>(targets_s.untyped_data());
  T* y = nullptr;
  T* z = nullptr;
  T* t = nullptr;
  T* u = nullptr;
  if constexpr (ndim > 1) {
    y = reinterpret_cast<T*>(points_y.untyped_data());
    t = reinterpret_cast<T*>(targets_t.untyped_data());
  }
  if constexpr (ndim > 2) {
    z = reinterpret_cast<T*>(points_z.untyped_data());
    u = reinterpret_cast<T*>(targets_u.untyped_data());
  }
  auto* F = reinterpret_cast<std::complex<T>*>(output->untyped_data());

  return run_nufft<ndim, T, 3>(opts, eps, static_cast<int>(iflag), n_tot,
                               static_cast<int>(n_transf), n_j, n_k, x, y, z, c, s, t, u, F);
}

// =============================================================================
// Dimension-specific wrapper functions for Type 1/2
// These call the core run_nufft with the correct number of point arrays
// =============================================================================

// Helper to setup finufft_opts from FFI attributes
template <typename T>
finufft_opts setup_opts(int64_t modeord, int64_t debug, int64_t spread_debug, int64_t showwarn,
                        int64_t nthreads, int64_t fftw, int64_t spread_sort,
                        int64_t spread_kerevalmeth, int64_t spread_kerpad, double upsampfac,
                        int64_t spread_thread, int64_t maxbatchsize, int64_t spread_nthr_atomic,
                        int64_t spread_max_sp_size) {
  finufft_opts opts;
  default_opts<T>(&opts);
  opts.modeord = static_cast<int>(modeord);
  opts.debug = static_cast<int>(debug);
  opts.spread_debug = static_cast<int>(spread_debug);
  opts.showwarn = static_cast<int>(showwarn);
  opts.nthreads = static_cast<int>(nthreads);
  opts.fftw = static_cast<int>(fftw);
  opts.spread_sort = static_cast<int>(spread_sort);
  opts.spread_kerevalmeth = static_cast<int>(spread_kerevalmeth);
  opts.spread_kerpad = static_cast<int>(spread_kerpad);
  opts.upsampfac = upsampfac;
  opts.spread_thread = static_cast<int>(spread_thread);
  opts.maxbatchsize = static_cast<int>(maxbatchsize);
  opts.spread_nthr_atomic = static_cast<int>(spread_nthr_atomic);
  opts.spread_max_sp_size = static_cast<int>(spread_max_sp_size);
  return opts;
}

// 1D Type 1 wrapper (only points_x)
template <typename T>
ffi::Error nufft1d1_wrapper(T eps, int64_t iflag, int64_t n_tot, int64_t n_transf, int64_t n_j,
                            int64_t n_k_1, int64_t n_k_2, int64_t n_k_3, int64_t modeord,
                            int64_t debug, int64_t spread_debug, int64_t showwarn,
                            int64_t nthreads, int64_t fftw, int64_t spread_sort,
                            int64_t spread_kerevalmeth, int64_t spread_kerpad, double upsampfac,
                            int64_t spread_thread, int64_t maxbatchsize,
                            int64_t spread_nthr_atomic, int64_t spread_max_sp_size,
                            ffi::AnyBuffer source, ffi::AnyBuffer points_x,
                            ffi::Result<ffi::AnyBuffer> output) {
  auto opts = setup_opts<T>(modeord, debug, spread_debug, showwarn, nthreads, fftw, spread_sort,
                            spread_kerevalmeth, spread_kerpad, upsampfac, spread_thread,
                            maxbatchsize, spread_nthr_atomic, spread_max_sp_size);
  int64_t n_k[3] = {n_k_1, n_k_2, n_k_3};
  auto* c = reinterpret_cast<std::complex<T>*>(source.untyped_data());
  auto* x = reinterpret_cast<T*>(points_x.untyped_data());
  auto* F = reinterpret_cast<std::complex<T>*>(output->untyped_data());
  return run_nufft<1, T, 1>(opts, eps, static_cast<int>(iflag), n_tot, static_cast<int>(n_transf),
                            n_j, n_k, x, nullptr, nullptr, c, nullptr, nullptr, nullptr, F);
}

// 2D Type 1 wrapper (points_x, points_y)
template <typename T>
ffi::Error nufft2d1_wrapper(T eps, int64_t iflag, int64_t n_tot, int64_t n_transf, int64_t n_j,
                            int64_t n_k_1, int64_t n_k_2, int64_t n_k_3, int64_t modeord,
                            int64_t debug, int64_t spread_debug, int64_t showwarn,
                            int64_t nthreads, int64_t fftw, int64_t spread_sort,
                            int64_t spread_kerevalmeth, int64_t spread_kerpad, double upsampfac,
                            int64_t spread_thread, int64_t maxbatchsize,
                            int64_t spread_nthr_atomic, int64_t spread_max_sp_size,
                            ffi::AnyBuffer source, ffi::AnyBuffer points_x,
                            ffi::AnyBuffer points_y, ffi::Result<ffi::AnyBuffer> output) {
  auto opts = setup_opts<T>(modeord, debug, spread_debug, showwarn, nthreads, fftw, spread_sort,
                            spread_kerevalmeth, spread_kerpad, upsampfac, spread_thread,
                            maxbatchsize, spread_nthr_atomic, spread_max_sp_size);
  int64_t n_k[3] = {n_k_1, n_k_2, n_k_3};
  auto* c = reinterpret_cast<std::complex<T>*>(source.untyped_data());
  auto* x = reinterpret_cast<T*>(points_x.untyped_data());
  auto* y = reinterpret_cast<T*>(points_y.untyped_data());
  auto* F = reinterpret_cast<std::complex<T>*>(output->untyped_data());
  return run_nufft<2, T, 1>(opts, eps, static_cast<int>(iflag), n_tot, static_cast<int>(n_transf),
                            n_j, n_k, x, y, nullptr, c, nullptr, nullptr, nullptr, F);
}

// 3D Type 1 wrapper (points_x, points_y, points_z) - same as original nufft1_impl<3, T>
template <typename T>
ffi::Error nufft3d1_wrapper(T eps, int64_t iflag, int64_t n_tot, int64_t n_transf, int64_t n_j,
                            int64_t n_k_1, int64_t n_k_2, int64_t n_k_3, int64_t modeord,
                            int64_t debug, int64_t spread_debug, int64_t showwarn,
                            int64_t nthreads, int64_t fftw, int64_t spread_sort,
                            int64_t spread_kerevalmeth, int64_t spread_kerpad, double upsampfac,
                            int64_t spread_thread, int64_t maxbatchsize,
                            int64_t spread_nthr_atomic, int64_t spread_max_sp_size,
                            ffi::AnyBuffer source, ffi::AnyBuffer points_x,
                            ffi::AnyBuffer points_y, ffi::AnyBuffer points_z,
                            ffi::Result<ffi::AnyBuffer> output) {
  auto opts = setup_opts<T>(modeord, debug, spread_debug, showwarn, nthreads, fftw, spread_sort,
                            spread_kerevalmeth, spread_kerpad, upsampfac, spread_thread,
                            maxbatchsize, spread_nthr_atomic, spread_max_sp_size);
  int64_t n_k[3] = {n_k_1, n_k_2, n_k_3};
  auto* c = reinterpret_cast<std::complex<T>*>(source.untyped_data());
  auto* x = reinterpret_cast<T*>(points_x.untyped_data());
  auto* y = reinterpret_cast<T*>(points_y.untyped_data());
  auto* z = reinterpret_cast<T*>(points_z.untyped_data());
  auto* F = reinterpret_cast<std::complex<T>*>(output->untyped_data());
  return run_nufft<3, T, 1>(opts, eps, static_cast<int>(iflag), n_tot, static_cast<int>(n_transf),
                            n_j, n_k, x, y, z, c, nullptr, nullptr, nullptr, F);
}

// 1D Type 2 wrapper (only points_x)
template <typename T>
ffi::Error nufft1d2_wrapper(T eps, int64_t iflag, int64_t n_tot, int64_t n_transf, int64_t n_j,
                            int64_t n_k_1, int64_t n_k_2, int64_t n_k_3, int64_t modeord,
                            int64_t debug, int64_t spread_debug, int64_t showwarn,
                            int64_t nthreads, int64_t fftw, int64_t spread_sort,
                            int64_t spread_kerevalmeth, int64_t spread_kerpad, double upsampfac,
                            int64_t spread_thread, int64_t maxbatchsize,
                            int64_t spread_nthr_atomic, int64_t spread_max_sp_size,
                            ffi::AnyBuffer source, ffi::AnyBuffer points_x,
                            ffi::Result<ffi::AnyBuffer> output) {
  auto opts = setup_opts<T>(modeord, debug, spread_debug, showwarn, nthreads, fftw, spread_sort,
                            spread_kerevalmeth, spread_kerpad, upsampfac, spread_thread,
                            maxbatchsize, spread_nthr_atomic, spread_max_sp_size);
  int64_t n_k[3] = {n_k_1, n_k_2, n_k_3};
  auto* F = reinterpret_cast<std::complex<T>*>(source.untyped_data());
  auto* x = reinterpret_cast<T*>(points_x.untyped_data());
  auto* c = reinterpret_cast<std::complex<T>*>(output->untyped_data());
  return run_nufft<1, T, 2>(opts, eps, static_cast<int>(iflag), n_tot, static_cast<int>(n_transf),
                            n_j, n_k, x, nullptr, nullptr, c, nullptr, nullptr, nullptr, F);
}

// 2D Type 2 wrapper (points_x, points_y)
template <typename T>
ffi::Error nufft2d2_wrapper(T eps, int64_t iflag, int64_t n_tot, int64_t n_transf, int64_t n_j,
                            int64_t n_k_1, int64_t n_k_2, int64_t n_k_3, int64_t modeord,
                            int64_t debug, int64_t spread_debug, int64_t showwarn,
                            int64_t nthreads, int64_t fftw, int64_t spread_sort,
                            int64_t spread_kerevalmeth, int64_t spread_kerpad, double upsampfac,
                            int64_t spread_thread, int64_t maxbatchsize,
                            int64_t spread_nthr_atomic, int64_t spread_max_sp_size,
                            ffi::AnyBuffer source, ffi::AnyBuffer points_x,
                            ffi::AnyBuffer points_y, ffi::Result<ffi::AnyBuffer> output) {
  auto opts = setup_opts<T>(modeord, debug, spread_debug, showwarn, nthreads, fftw, spread_sort,
                            spread_kerevalmeth, spread_kerpad, upsampfac, spread_thread,
                            maxbatchsize, spread_nthr_atomic, spread_max_sp_size);
  int64_t n_k[3] = {n_k_1, n_k_2, n_k_3};
  auto* F = reinterpret_cast<std::complex<T>*>(source.untyped_data());
  auto* x = reinterpret_cast<T*>(points_x.untyped_data());
  auto* y = reinterpret_cast<T*>(points_y.untyped_data());
  auto* c = reinterpret_cast<std::complex<T>*>(output->untyped_data());
  return run_nufft<2, T, 2>(opts, eps, static_cast<int>(iflag), n_tot, static_cast<int>(n_transf),
                            n_j, n_k, x, y, nullptr, c, nullptr, nullptr, nullptr, F);
}

// 3D Type 2 wrapper (points_x, points_y, points_z)
template <typename T>
ffi::Error nufft3d2_wrapper(T eps, int64_t iflag, int64_t n_tot, int64_t n_transf, int64_t n_j,
                            int64_t n_k_1, int64_t n_k_2, int64_t n_k_3, int64_t modeord,
                            int64_t debug, int64_t spread_debug, int64_t showwarn,
                            int64_t nthreads, int64_t fftw, int64_t spread_sort,
                            int64_t spread_kerevalmeth, int64_t spread_kerpad, double upsampfac,
                            int64_t spread_thread, int64_t maxbatchsize,
                            int64_t spread_nthr_atomic, int64_t spread_max_sp_size,
                            ffi::AnyBuffer source, ffi::AnyBuffer points_x,
                            ffi::AnyBuffer points_y, ffi::AnyBuffer points_z,
                            ffi::Result<ffi::AnyBuffer> output) {
  auto opts = setup_opts<T>(modeord, debug, spread_debug, showwarn, nthreads, fftw, spread_sort,
                            spread_kerevalmeth, spread_kerpad, upsampfac, spread_thread,
                            maxbatchsize, spread_nthr_atomic, spread_max_sp_size);
  int64_t n_k[3] = {n_k_1, n_k_2, n_k_3};
  auto* F = reinterpret_cast<std::complex<T>*>(source.untyped_data());
  auto* x = reinterpret_cast<T*>(points_x.untyped_data());
  auto* y = reinterpret_cast<T*>(points_y.untyped_data());
  auto* z = reinterpret_cast<T*>(points_z.untyped_data());
  auto* c = reinterpret_cast<std::complex<T>*>(output->untyped_data());
  return run_nufft<3, T, 2>(opts, eps, static_cast<int>(iflag), n_tot, static_cast<int>(n_transf),
                            n_j, n_k, x, y, z, c, nullptr, nullptr, nullptr, F);
}

// =============================================================================
// Dimension-specific wrapper functions for Type 3
// =============================================================================

// 1D Type 3 wrapper (points_x + targets_s)
template <typename T>
ffi::Error nufft1d3_wrapper(T eps, int64_t iflag, int64_t n_tot, int64_t n_transf, int64_t n_j,
                            int64_t n_k_1, int64_t n_k_2, int64_t n_k_3, int64_t modeord,
                            int64_t debug, int64_t spread_debug, int64_t showwarn,
                            int64_t nthreads, int64_t fftw, int64_t spread_sort,
                            int64_t spread_kerevalmeth, int64_t spread_kerpad, double upsampfac,
                            int64_t spread_thread, int64_t maxbatchsize,
                            int64_t spread_nthr_atomic, int64_t spread_max_sp_size,
                            ffi::AnyBuffer source, ffi::AnyBuffer points_x,
                            ffi::AnyBuffer targets_s, ffi::Result<ffi::AnyBuffer> output) {
  auto opts = setup_opts<T>(modeord, debug, spread_debug, showwarn, nthreads, fftw, spread_sort,
                            spread_kerevalmeth, spread_kerpad, upsampfac, spread_thread,
                            maxbatchsize, spread_nthr_atomic, spread_max_sp_size);
  int64_t n_k[3] = {n_k_1, n_k_2, n_k_3};
  auto* c = reinterpret_cast<std::complex<T>*>(source.untyped_data());
  auto* x = reinterpret_cast<T*>(points_x.untyped_data());
  auto* s = reinterpret_cast<T*>(targets_s.untyped_data());
  auto* F = reinterpret_cast<std::complex<T>*>(output->untyped_data());
  return run_nufft<1, T, 3>(opts, eps, static_cast<int>(iflag), n_tot, static_cast<int>(n_transf),
                            n_j, n_k, x, nullptr, nullptr, c, s, nullptr, nullptr, F);
}

// 2D Type 3 wrapper (points_x, points_y + targets_s, targets_t)
template <typename T>
ffi::Error nufft2d3_wrapper(T eps, int64_t iflag, int64_t n_tot, int64_t n_transf, int64_t n_j,
                            int64_t n_k_1, int64_t n_k_2, int64_t n_k_3, int64_t modeord,
                            int64_t debug, int64_t spread_debug, int64_t showwarn,
                            int64_t nthreads, int64_t fftw, int64_t spread_sort,
                            int64_t spread_kerevalmeth, int64_t spread_kerpad, double upsampfac,
                            int64_t spread_thread, int64_t maxbatchsize,
                            int64_t spread_nthr_atomic, int64_t spread_max_sp_size,
                            ffi::AnyBuffer source, ffi::AnyBuffer points_x,
                            ffi::AnyBuffer points_y, ffi::AnyBuffer targets_s,
                            ffi::AnyBuffer targets_t, ffi::Result<ffi::AnyBuffer> output) {
  auto opts = setup_opts<T>(modeord, debug, spread_debug, showwarn, nthreads, fftw, spread_sort,
                            spread_kerevalmeth, spread_kerpad, upsampfac, spread_thread,
                            maxbatchsize, spread_nthr_atomic, spread_max_sp_size);
  int64_t n_k[3] = {n_k_1, n_k_2, n_k_3};
  auto* c = reinterpret_cast<std::complex<T>*>(source.untyped_data());
  auto* x = reinterpret_cast<T*>(points_x.untyped_data());
  auto* y = reinterpret_cast<T*>(points_y.untyped_data());
  auto* s = reinterpret_cast<T*>(targets_s.untyped_data());
  auto* t = reinterpret_cast<T*>(targets_t.untyped_data());
  auto* F = reinterpret_cast<std::complex<T>*>(output->untyped_data());
  return run_nufft<2, T, 3>(opts, eps, static_cast<int>(iflag), n_tot, static_cast<int>(n_transf),
                            n_j, n_k, x, y, nullptr, c, s, t, nullptr, F);
}

// 3D Type 3 wrapper (all point arrays)
template <typename T>
ffi::Error nufft3d3_wrapper(T eps, int64_t iflag, int64_t n_tot, int64_t n_transf, int64_t n_j,
                            int64_t n_k_1, int64_t n_k_2, int64_t n_k_3, int64_t modeord,
                            int64_t debug, int64_t spread_debug, int64_t showwarn,
                            int64_t nthreads, int64_t fftw, int64_t spread_sort,
                            int64_t spread_kerevalmeth, int64_t spread_kerpad, double upsampfac,
                            int64_t spread_thread, int64_t maxbatchsize,
                            int64_t spread_nthr_atomic, int64_t spread_max_sp_size,
                            ffi::AnyBuffer source, ffi::AnyBuffer points_x,
                            ffi::AnyBuffer points_y, ffi::AnyBuffer points_z,
                            ffi::AnyBuffer targets_s, ffi::AnyBuffer targets_t,
                            ffi::AnyBuffer targets_u, ffi::Result<ffi::AnyBuffer> output) {
  auto opts = setup_opts<T>(modeord, debug, spread_debug, showwarn, nthreads, fftw, spread_sort,
                            spread_kerevalmeth, spread_kerpad, upsampfac, spread_thread,
                            maxbatchsize, spread_nthr_atomic, spread_max_sp_size);
  int64_t n_k[3] = {n_k_1, n_k_2, n_k_3};
  auto* c = reinterpret_cast<std::complex<T>*>(source.untyped_data());
  auto* x = reinterpret_cast<T*>(points_x.untyped_data());
  auto* y = reinterpret_cast<T*>(points_y.untyped_data());
  auto* z = reinterpret_cast<T*>(points_z.untyped_data());
  auto* s = reinterpret_cast<T*>(targets_s.untyped_data());
  auto* t = reinterpret_cast<T*>(targets_t.untyped_data());
  auto* u = reinterpret_cast<T*>(targets_u.untyped_data());
  auto* F = reinterpret_cast<std::complex<T>*>(output->untyped_data());
  return run_nufft<3, T, 3>(opts, eps, static_cast<int>(iflag), n_tot, static_cast<int>(n_transf),
                            n_j, n_k, x, y, z, c, s, t, u, F);
}

// =============================================================================
// FFI Handler Binding Definitions
// =============================================================================

// Common attributes macro for all bindings
#define NUFFT_COMMON_ATTRS_FLOAT           \
  .Attr<float>("eps")                      \
      .Attr<int64_t>("iflag")              \
      .Attr<int64_t>("n_tot")              \
      .Attr<int64_t>("n_transf")           \
      .Attr<int64_t>("n_j")                \
      .Attr<int64_t>("n_k_1")              \
      .Attr<int64_t>("n_k_2")              \
      .Attr<int64_t>("n_k_3")              \
      .Attr<int64_t>("modeord")            \
      .Attr<int64_t>("debug")              \
      .Attr<int64_t>("spread_debug")       \
      .Attr<int64_t>("showwarn")           \
      .Attr<int64_t>("nthreads")           \
      .Attr<int64_t>("fftw")               \
      .Attr<int64_t>("spread_sort")        \
      .Attr<int64_t>("spread_kerevalmeth") \
      .Attr<int64_t>("spread_kerpad")      \
      .Attr<double>("upsampfac")           \
      .Attr<int64_t>("spread_thread")      \
      .Attr<int64_t>("maxbatchsize")       \
      .Attr<int64_t>("spread_nthr_atomic") \
      .Attr<int64_t>("spread_max_sp_size")

#define NUFFT_COMMON_ATTRS_DOUBLE          \
  .Attr<double>("eps")                     \
      .Attr<int64_t>("iflag")              \
      .Attr<int64_t>("n_tot")              \
      .Attr<int64_t>("n_transf")           \
      .Attr<int64_t>("n_j")                \
      .Attr<int64_t>("n_k_1")              \
      .Attr<int64_t>("n_k_2")              \
      .Attr<int64_t>("n_k_3")              \
      .Attr<int64_t>("modeord")            \
      .Attr<int64_t>("debug")              \
      .Attr<int64_t>("spread_debug")       \
      .Attr<int64_t>("showwarn")           \
      .Attr<int64_t>("nthreads")           \
      .Attr<int64_t>("fftw")               \
      .Attr<int64_t>("spread_sort")        \
      .Attr<int64_t>("spread_kerevalmeth") \
      .Attr<int64_t>("spread_kerpad")      \
      .Attr<double>("upsampfac")           \
      .Attr<int64_t>("spread_thread")      \
      .Attr<int64_t>("maxbatchsize")       \
      .Attr<int64_t>("spread_nthr_atomic") \
      .Attr<int64_t>("spread_max_sp_size")

// -----------------------------------------------------------------------------
// Dimension-specific bindings for Type 1/2 (1D: 1 point, 2D: 2 points, 3D: 3)
// -----------------------------------------------------------------------------

// 1D bindings (source + x)
inline auto MakeNufft1dBinding12Float() {
  return ffi::Ffi::Bind() NUFFT_COMMON_ATTRS_FLOAT
      .Arg<ffi::AnyBuffer>()   // source
      .Arg<ffi::AnyBuffer>()   // points_x
      .Ret<ffi::AnyBuffer>();  // output
}

inline auto MakeNufft1dBinding12Double() {
  return ffi::Ffi::Bind() NUFFT_COMMON_ATTRS_DOUBLE
      .Arg<ffi::AnyBuffer>()   // source
      .Arg<ffi::AnyBuffer>()   // points_x
      .Ret<ffi::AnyBuffer>();  // output
}

// 2D bindings (source + x + y)
inline auto MakeNufft2dBinding12Float() {
  return ffi::Ffi::Bind() NUFFT_COMMON_ATTRS_FLOAT
      .Arg<ffi::AnyBuffer>()   // source
      .Arg<ffi::AnyBuffer>()   // points_x
      .Arg<ffi::AnyBuffer>()   // points_y
      .Ret<ffi::AnyBuffer>();  // output
}

inline auto MakeNufft2dBinding12Double() {
  return ffi::Ffi::Bind() NUFFT_COMMON_ATTRS_DOUBLE
      .Arg<ffi::AnyBuffer>()   // source
      .Arg<ffi::AnyBuffer>()   // points_x
      .Arg<ffi::AnyBuffer>()   // points_y
      .Ret<ffi::AnyBuffer>();  // output
}

// 3D bindings (source + x + y + z)
inline auto MakeNufft3dBinding12Float() {
  return ffi::Ffi::Bind() NUFFT_COMMON_ATTRS_FLOAT
      .Arg<ffi::AnyBuffer>()   // source
      .Arg<ffi::AnyBuffer>()   // points_x
      .Arg<ffi::AnyBuffer>()   // points_y
      .Arg<ffi::AnyBuffer>()   // points_z
      .Ret<ffi::AnyBuffer>();  // output
}

inline auto MakeNufft3dBinding12Double() {
  return ffi::Ffi::Bind() NUFFT_COMMON_ATTRS_DOUBLE
      .Arg<ffi::AnyBuffer>()   // source
      .Arg<ffi::AnyBuffer>()   // points_x
      .Arg<ffi::AnyBuffer>()   // points_y
      .Arg<ffi::AnyBuffer>()   // points_z
      .Ret<ffi::AnyBuffer>();  // output
}

// -----------------------------------------------------------------------------
// Dimension-specific bindings for Type 3 (1D: x+s, 2D: x+y+s+t, 3D: x+y+z+s+t+u)
// -----------------------------------------------------------------------------

// 1D type 3 bindings (source + x + s)
inline auto MakeNufft1dBinding3Float() {
  return ffi::Ffi::Bind() NUFFT_COMMON_ATTRS_FLOAT
      .Arg<ffi::AnyBuffer>()   // source
      .Arg<ffi::AnyBuffer>()   // points_x
      .Arg<ffi::AnyBuffer>()   // targets_s
      .Ret<ffi::AnyBuffer>();  // output
}

inline auto MakeNufft1dBinding3Double() {
  return ffi::Ffi::Bind() NUFFT_COMMON_ATTRS_DOUBLE
      .Arg<ffi::AnyBuffer>()   // source
      .Arg<ffi::AnyBuffer>()   // points_x
      .Arg<ffi::AnyBuffer>()   // targets_s
      .Ret<ffi::AnyBuffer>();  // output
}

// 2D type 3 bindings (source + x + y + s + t)
inline auto MakeNufft2dBinding3Float() {
  return ffi::Ffi::Bind() NUFFT_COMMON_ATTRS_FLOAT
      .Arg<ffi::AnyBuffer>()   // source
      .Arg<ffi::AnyBuffer>()   // points_x
      .Arg<ffi::AnyBuffer>()   // points_y
      .Arg<ffi::AnyBuffer>()   // targets_s
      .Arg<ffi::AnyBuffer>()   // targets_t
      .Ret<ffi::AnyBuffer>();  // output
}

inline auto MakeNufft2dBinding3Double() {
  return ffi::Ffi::Bind() NUFFT_COMMON_ATTRS_DOUBLE
      .Arg<ffi::AnyBuffer>()   // source
      .Arg<ffi::AnyBuffer>()   // points_x
      .Arg<ffi::AnyBuffer>()   // points_y
      .Arg<ffi::AnyBuffer>()   // targets_s
      .Arg<ffi::AnyBuffer>()   // targets_t
      .Ret<ffi::AnyBuffer>();  // output
}

// 3D type 3 bindings (source + x + y + z + s + t + u)
inline auto MakeNufft3dBinding3Float() {
  return ffi::Ffi::Bind() NUFFT_COMMON_ATTRS_FLOAT
      .Arg<ffi::AnyBuffer>()   // source
      .Arg<ffi::AnyBuffer>()   // points_x
      .Arg<ffi::AnyBuffer>()   // points_y
      .Arg<ffi::AnyBuffer>()   // points_z
      .Arg<ffi::AnyBuffer>()   // targets_s
      .Arg<ffi::AnyBuffer>()   // targets_t
      .Arg<ffi::AnyBuffer>()   // targets_u
      .Ret<ffi::AnyBuffer>();  // output
}

inline auto MakeNufft3dBinding3Double() {
  return ffi::Ffi::Bind() NUFFT_COMMON_ATTRS_DOUBLE
      .Arg<ffi::AnyBuffer>()   // source
      .Arg<ffi::AnyBuffer>()   // points_x
      .Arg<ffi::AnyBuffer>()   // points_y
      .Arg<ffi::AnyBuffer>()   // points_z
      .Arg<ffi::AnyBuffer>()   // targets_s
      .Arg<ffi::AnyBuffer>()   // targets_t
      .Arg<ffi::AnyBuffer>()   // targets_u
      .Ret<ffi::AnyBuffer>();  // output
}

// =============================================================================
// FFI Handler Definitions using Macros
// =============================================================================

// Generic macro for defining FFI handlers
#define DEFINE_FFI_HANDLER(name, binding, impl)                                 \
  static constexpr XLA_FFI_Handler* name = +[](XLA_FFI_CallFrame* call_frame) { \
    static auto* handler = binding().To(impl).release();                        \
    return handler->Call(call_frame);                                           \
  }

// Type 1 handlers: non-uniform to uniform (dimension-specific bindings)
DEFINE_FFI_HANDLER(nufft1d1f, MakeNufft1dBinding12Float, (nufft1d1_wrapper<float>));
DEFINE_FFI_HANDLER(nufft1d1, MakeNufft1dBinding12Double, (nufft1d1_wrapper<double>));
DEFINE_FFI_HANDLER(nufft2d1f, MakeNufft2dBinding12Float, (nufft2d1_wrapper<float>));
DEFINE_FFI_HANDLER(nufft2d1, MakeNufft2dBinding12Double, (nufft2d1_wrapper<double>));
DEFINE_FFI_HANDLER(nufft3d1f, MakeNufft3dBinding12Float, (nufft3d1_wrapper<float>));
DEFINE_FFI_HANDLER(nufft3d1, MakeNufft3dBinding12Double, (nufft3d1_wrapper<double>));

// Type 2 handlers: uniform to non-uniform (dimension-specific bindings)
DEFINE_FFI_HANDLER(nufft1d2f, MakeNufft1dBinding12Float, (nufft1d2_wrapper<float>));
DEFINE_FFI_HANDLER(nufft1d2, MakeNufft1dBinding12Double, (nufft1d2_wrapper<double>));
DEFINE_FFI_HANDLER(nufft2d2f, MakeNufft2dBinding12Float, (nufft2d2_wrapper<float>));
DEFINE_FFI_HANDLER(nufft2d2, MakeNufft2dBinding12Double, (nufft2d2_wrapper<double>));
DEFINE_FFI_HANDLER(nufft3d2f, MakeNufft3dBinding12Float, (nufft3d2_wrapper<float>));
DEFINE_FFI_HANDLER(nufft3d2, MakeNufft3dBinding12Double, (nufft3d2_wrapper<double>));

// Type 3 handlers: non-uniform to non-uniform (dimension-specific bindings)
DEFINE_FFI_HANDLER(nufft1d3f, MakeNufft1dBinding3Float, (nufft1d3_wrapper<float>));
DEFINE_FFI_HANDLER(nufft1d3, MakeNufft1dBinding3Double, (nufft1d3_wrapper<double>));
DEFINE_FFI_HANDLER(nufft2d3f, MakeNufft2dBinding3Float, (nufft2d3_wrapper<float>));
DEFINE_FFI_HANDLER(nufft2d3, MakeNufft2dBinding3Double, (nufft2d3_wrapper<double>));
DEFINE_FFI_HANDLER(nufft3d3f, MakeNufft3dBinding3Float, (nufft3d3_wrapper<float>));
DEFINE_FFI_HANDLER(nufft3d3, MakeNufft3dBinding3Double, (nufft3d3_wrapper<double>));

#undef DEFINE_FFI_HANDLER

// =============================================================================
// Python Module Registration
// =============================================================================

template <typename T>
nb::capsule EncapsulateFfiHandler(T* fn) {
  static_assert(std::is_invocable_r_v<XLA_FFI_Error*, T, XLA_FFI_CallFrame*>,
                "FFI handler must have signature: XLA_FFI_Error*(XLA_FFI_CallFrame*)");
  return nb::capsule(reinterpret_cast<void*>(fn));
}

nb::dict Registrations() {
  nb::dict dict;

  // Type 1: non-uniform to uniform
  dict["nufft1d1f"] = EncapsulateFfiHandler(nufft1d1f);
  dict["nufft1d1"] = EncapsulateFfiHandler(nufft1d1);
  dict["nufft2d1f"] = EncapsulateFfiHandler(nufft2d1f);
  dict["nufft2d1"] = EncapsulateFfiHandler(nufft2d1);
  dict["nufft3d1f"] = EncapsulateFfiHandler(nufft3d1f);
  dict["nufft3d1"] = EncapsulateFfiHandler(nufft3d1);

  // Type 2: uniform to non-uniform
  dict["nufft1d2f"] = EncapsulateFfiHandler(nufft1d2f);
  dict["nufft1d2"] = EncapsulateFfiHandler(nufft1d2);
  dict["nufft2d2f"] = EncapsulateFfiHandler(nufft2d2f);
  dict["nufft2d2"] = EncapsulateFfiHandler(nufft2d2);
  dict["nufft3d2f"] = EncapsulateFfiHandler(nufft3d2f);
  dict["nufft3d2"] = EncapsulateFfiHandler(nufft3d2);

  // Type 3: non-uniform to non-uniform
  dict["nufft1d3f"] = EncapsulateFfiHandler(nufft1d3f);
  dict["nufft1d3"] = EncapsulateFfiHandler(nufft1d3);
  dict["nufft2d3f"] = EncapsulateFfiHandler(nufft2d3f);
  dict["nufft2d3"] = EncapsulateFfiHandler(nufft2d3);
  dict["nufft3d3f"] = EncapsulateFfiHandler(nufft3d3f);
  dict["nufft3d3"] = EncapsulateFfiHandler(nufft3d3);

  return dict;
}

}  // namespace cpu
}  // namespace jax_finufft

NB_MODULE(jax_finufft_cpu, m) {
  m.def("registrations", &jax_finufft::cpu::Registrations,
        "Get FFI handler registrations for all NUFFT operations");

  m.def("_omp_compile_check", []() {
#ifdef FINUFFT_USE_OPENMP
    return true;
#else
    return false;
#endif
  });

  m.attr("FFTW_ESTIMATE") = nb::int_(FFTW_ESTIMATE);
  m.attr("FFTW_MEASURE") = nb::int_(FFTW_MEASURE);
  m.attr("FFTW_PATIENT") = nb::int_(FFTW_PATIENT);
  m.attr("FFTW_EXHAUSTIVE") = nb::int_(FFTW_EXHAUSTIVE);
  m.attr("FFTW_WISDOM_ONLY") = nb::int_(FFTW_WISDOM_ONLY);
}
