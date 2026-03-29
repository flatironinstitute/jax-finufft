// CPU FFI bindings for jax-finufft using XLA typed FFI (api_version=4).
//
// This module exposes FINUFFT operations as XLA FFI custom calls with typed
// buffer interfaces, enabling seamless integration with JAX's compilation
// pipeline.

#include "jax_finufft_cpu.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>
#include <xla/ffi/api/ffi.h>

#include <complex>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace ffi = xla::ffi;
namespace nb = nanobind;

namespace jax_finufft {
namespace cpu {

// =============================================================================
// Core NUFFT execution logic
// =============================================================================

template <int ndim, typename T, int type>
ffi::Error run_nufft_masked(finufft_opts opts, T eps, int iflag, int64_t n_tot, int n_transf,
                            int64_t n_j, const int64_t* n_k, T* x, T* y, T* z, const int8_t* mask,
                            std::complex<T>* c, std::complex<T>* F) {
  int64_t n_k_total = 1;
  for (int d = 0; d < ndim; ++d) {
    n_k_total *= n_k[d];
  }

  typename plan_type<T>::type plan;
  int64_t n_k_mutable[3] = {n_k[0], n_k[1], n_k[2]};
  int ret = makeplan<T>(type, ndim, n_k_mutable, iflag, n_transf, eps, &plan, &opts);
  if (ret > 1) {
    return ffi::Error::Internal("FINUFFT makeplan failed with code " + std::to_string(ret));
  }

  for (int64_t index = 0; index < n_tot; ++index) {
    int64_t i_start = index * n_j;
    int64_t c_start = index * n_j * n_transf;
    int64_t k_start = index * n_k_total * n_transf;

    // 1. Determine Q_size first
    int64_t Q_size = 0;
    for (int64_t i = 0; i < n_j; ++i) {
      if (mask[i_start + i]) Q_size++;
    }

    // 2. Allocate exactly Q_size
    std::vector<T> Q_x(Q_size);
    std::vector<T> Q_y(ndim > 1 ? Q_size : 0);
    std::vector<T> Q_z(ndim > 2 ? Q_size : 0);
    std::vector<std::complex<T>> Q_c(Q_size * n_transf);

    // 3. Pack with correct PLANAR multi-transform indexing
    int64_t p = 0;
    for (int64_t i = 0; i < n_j; ++i) {
      if (mask[i_start + i]) {
        Q_x[p] = x[i_start + i];
        if constexpr (ndim > 1) Q_y[p] = y[i_start + i];
        if constexpr (ndim > 2) Q_z[p] = z[i_start + i];
        if constexpr (type != 2) {
          for (int t = 0; t < n_transf; ++t) {
            Q_c[t * Q_size + p] = c[c_start + t * n_j + i];
          }
        }
        p++;
      }
    }

    ret = setpts<T>(plan, Q_size, Q_x.data(), ndim > 1 ? Q_y.data() : nullptr,
                    ndim > 2 ? Q_z.data() : nullptr, 0, nullptr, nullptr, nullptr);
    if (ret != 0) {
      destroy<T>(plan);
      return ffi::Error::Internal("FINUFFT setpts failed with code " + std::to_string(ret));
    }

    ret = execute<T>(plan, Q_c.data(), &F[k_start]);
    if (ret != 0) {
      destroy<T>(plan);
      return ffi::Error::Internal("FINUFFT execute failed with code " + std::to_string(ret));
    }

    // 4. Unpack with correct PLANAR multi-transform indexing
    if constexpr (type == 2) {
      for (int t = 0; t < n_transf; ++t) {
        for (int64_t i = 0; i < n_j; ++i) {
          c[c_start + t * n_j + i] = {0.0, 0.0};
        }
      }
      p = 0;
      for (int64_t i = 0; i < n_j; ++i) {
        if (mask[i_start + i]) {
          for (int t = 0; t < n_transf; ++t) {
            c[c_start + t * n_j + i] = Q_c[t * Q_size + p];
          }
          p++;
        }
      }
    }
  }

  destroy<T>(plan);
  return ffi::Error::Success();
}

template <int ndim, typename T>
ffi::Error run_nufft_unmasked(finufft_opts opts, T eps, int iflag, int64_t n_tot, int n_transf,
                              int64_t n_j, const int64_t* n_k, T* x, T* y, T* z,
                              std::complex<T>* c, T* s, T* t, T* u, std::complex<T>* F) {
  int64_t n_k_total = n_k[0];

  typename plan_type<T>::type plan;
  int64_t n_k_mutable[3] = {n_k[0], n_k[1], n_k[2]};
  int ret = makeplan<T>(3, ndim, n_k_mutable, iflag, n_transf, eps, &plan, &opts);
  if (ret > 1) {
    return ffi::Error::Internal("FINUFFT makeplan failed with code " + std::to_string(ret));
  }

  for (int64_t index = 0; index < n_tot; ++index) {
    int64_t i_start = index * n_j;
    int64_t i_target = index * n_k_total;
    int64_t c_start = index * n_j * n_transf;
    int64_t k_start = index * n_k_total * n_transf;

    ret = setpts<T>(plan, n_j, &x[i_start], y_index<ndim, T>(y, i_start),
                    z_index<ndim, T>(z, i_start), n_k_total, &s[i_target],
                    y_index<ndim, T>(t, i_target), z_index<ndim, T>(u, i_target));
    if (ret != 0) {
      destroy<T>(plan);
      return ffi::Error::Internal("FINUFFT setpts failed with code " + std::to_string(ret));
    }

    ret = execute<T>(plan, &c[c_start], &F[k_start]);
    if (ret != 0) {
      destroy<T>(plan);
      return ffi::Error::Internal("FINUFFT execute failed with code " + std::to_string(ret));
    }
  }

  destroy<T>(plan);
  return ffi::Error::Success();
}

template <typename T>
finufft_opts build_opts(int64_t modeord, int64_t debug, int64_t spread_debug, int64_t showwarn,
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
                       ffi::AnyBuffer source, ffi::AnyBuffer points_mask, ffi::AnyBuffer points_x,
                       ffi::AnyBuffer points_y, ffi::AnyBuffer points_z,
                       ffi::Result<ffi::AnyBuffer> output) {
  finufft_opts opts =
      build_opts<T>(modeord, debug, spread_debug, showwarn, nthreads, fftw, spread_sort,
                    spread_kerevalmeth, spread_kerpad, upsampfac, spread_thread, maxbatchsize,
                    spread_nthr_atomic, spread_max_sp_size);
  int64_t n_k[3] = {n_k_1, n_k_2, n_k_3};

  auto* c = reinterpret_cast<std::complex<T>*>(source.untyped_data());
  auto* mask = reinterpret_cast<const int8_t*>(points_mask.untyped_data());
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

  return run_nufft_masked<ndim, T, 1>(opts, eps, static_cast<int>(iflag), n_tot,
                                      static_cast<int>(n_transf), n_j, n_k, x, y, z, mask, c, F);
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
                       ffi::AnyBuffer source, ffi::AnyBuffer points_mask, ffi::AnyBuffer points_x,
                       ffi::AnyBuffer points_y, ffi::AnyBuffer points_z,
                       ffi::Result<ffi::AnyBuffer> output) {
  finufft_opts opts =
      build_opts<T>(modeord, debug, spread_debug, showwarn, nthreads, fftw, spread_sort,
                    spread_kerevalmeth, spread_kerpad, upsampfac, spread_thread, maxbatchsize,
                    spread_nthr_atomic, spread_max_sp_size);
  int64_t n_k[3] = {n_k_1, n_k_2, n_k_3};

  auto* F = reinterpret_cast<std::complex<T>*>(source.untyped_data());
  auto* mask = reinterpret_cast<const int8_t*>(points_mask.untyped_data());
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

  return run_nufft_masked<ndim, T, 2>(opts, eps, static_cast<int>(iflag), n_tot,
                                      static_cast<int>(n_transf), n_j, n_k, x, y, z, mask, c, F);
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
                       ffi::AnyBuffer points_z, ffi::AnyBuffer target_x, ffi::AnyBuffer target_y,
                       ffi::AnyBuffer target_z, ffi::Result<ffi::AnyBuffer> output) {
  finufft_opts opts =
      build_opts<T>(modeord, debug, spread_debug, showwarn, nthreads, fftw, spread_sort,
                    spread_kerevalmeth, spread_kerpad, upsampfac, spread_thread, maxbatchsize,
                    spread_nthr_atomic, spread_max_sp_size);
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

  return run_nufft_unmasked<ndim, T>(opts, eps, static_cast<int>(iflag), n_tot,
                                     static_cast<int>(n_transf), n_j, n_k, x, y, z, c, s, t, u, F);
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

// 1D bindings (source + x)
inline auto MakeNufft1dBinding12Float() {
  return ffi::Ffi::Bind() NUFFT_COMMON_ATTRS_FLOAT.Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Ret<ffi::AnyBuffer>();
}

inline auto MakeNufft1dBinding12Double() {
  return ffi::Ffi::Bind() NUFFT_COMMON_ATTRS_DOUBLE.Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Ret<ffi::AnyBuffer>();
}

inline auto MakeNufft1dBinding3Float() {
  return ffi::Ffi::Bind() NUFFT_COMMON_ATTRS_FLOAT.Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Ret<ffi::AnyBuffer>();
}

inline auto MakeNufft1dBinding3Double() {
  return ffi::Ffi::Bind() NUFFT_COMMON_ATTRS_DOUBLE.Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Ret<ffi::AnyBuffer>();
}

inline auto MakeNufft2dBinding12Float() {
  return ffi::Ffi::Bind() NUFFT_COMMON_ATTRS_FLOAT.Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Ret<ffi::AnyBuffer>();
}

inline auto MakeNufft2dBinding12Double() {
  return ffi::Ffi::Bind() NUFFT_COMMON_ATTRS_DOUBLE.Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Ret<ffi::AnyBuffer>();
}

inline auto MakeNufft2dBinding3Float() {
  return ffi::Ffi::Bind() NUFFT_COMMON_ATTRS_FLOAT.Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Ret<ffi::AnyBuffer>();
}

inline auto MakeNufft2dBinding3Double() {
  return ffi::Ffi::Bind() NUFFT_COMMON_ATTRS_DOUBLE.Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Ret<ffi::AnyBuffer>();
}

inline auto MakeNufft3dBinding12Float() {
  return ffi::Ffi::Bind() NUFFT_COMMON_ATTRS_FLOAT.Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Ret<ffi::AnyBuffer>();
}

inline auto MakeNufft3dBinding12Double() {
  return ffi::Ffi::Bind() NUFFT_COMMON_ATTRS_DOUBLE.Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Ret<ffi::AnyBuffer>();
}

inline auto MakeNufft3dBinding3Float() {
  return ffi::Ffi::Bind() NUFFT_COMMON_ATTRS_FLOAT.Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Ret<ffi::AnyBuffer>();
}

inline auto MakeNufft3dBinding3Double() {
  return ffi::Ffi::Bind() NUFFT_COMMON_ATTRS_DOUBLE.Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Ret<ffi::AnyBuffer>();
}

ffi::Error nufft1d1f_1d_wrapper(float eps, int64_t iflag, int64_t n_tot, int64_t n_transf,
                                int64_t n_j, int64_t n_k_1, int64_t n_k_2, int64_t n_k_3,
                                int64_t modeord, int64_t debug, int64_t spread_debug,
                                int64_t showwarn, int64_t nthreads, int64_t fftw,
                                int64_t spread_sort, int64_t spread_kerevalmeth,
                                int64_t spread_kerpad, double upsampfac, int64_t spread_thread,
                                int64_t maxbatchsize, int64_t spread_nthr_atomic,
                                int64_t spread_max_sp_size, ffi::AnyBuffer source,
                                ffi::AnyBuffer points_mask, ffi::AnyBuffer points_x,
                                ffi::Result<ffi::AnyBuffer> output) {
  return nufft1_impl<1, float>(eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3, modeord,
                               debug, spread_debug, showwarn, nthreads, fftw, spread_sort,
                               spread_kerevalmeth, spread_kerpad, upsampfac, spread_thread,
                               maxbatchsize, spread_nthr_atomic, spread_max_sp_size, source,
                               points_mask, points_x, points_x, points_x, output);
}

ffi::Error nufft1d1_1d_wrapper(double eps, int64_t iflag, int64_t n_tot, int64_t n_transf,
                               int64_t n_j, int64_t n_k_1, int64_t n_k_2, int64_t n_k_3,
                               int64_t modeord, int64_t debug, int64_t spread_debug,
                               int64_t showwarn, int64_t nthreads, int64_t fftw,
                               int64_t spread_sort, int64_t spread_kerevalmeth,
                               int64_t spread_kerpad, double upsampfac, int64_t spread_thread,
                               int64_t maxbatchsize, int64_t spread_nthr_atomic,
                               int64_t spread_max_sp_size, ffi::AnyBuffer source,
                               ffi::AnyBuffer points_mask, ffi::AnyBuffer points_x,
                               ffi::Result<ffi::AnyBuffer> output) {
  return nufft1_impl<1, double>(eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3, modeord,
                                debug, spread_debug, showwarn, nthreads, fftw, spread_sort,
                                spread_kerevalmeth, spread_kerpad, upsampfac, spread_thread,
                                maxbatchsize, spread_nthr_atomic, spread_max_sp_size, source,
                                points_mask, points_x, points_x, points_x, output);
}

ffi::Error nufft1d2f_1d_wrapper(float eps, int64_t iflag, int64_t n_tot, int64_t n_transf,
                                int64_t n_j, int64_t n_k_1, int64_t n_k_2, int64_t n_k_3,
                                int64_t modeord, int64_t debug, int64_t spread_debug,
                                int64_t showwarn, int64_t nthreads, int64_t fftw,
                                int64_t spread_sort, int64_t spread_kerevalmeth,
                                int64_t spread_kerpad, double upsampfac, int64_t spread_thread,
                                int64_t maxbatchsize, int64_t spread_nthr_atomic,
                                int64_t spread_max_sp_size, ffi::AnyBuffer source,
                                ffi::AnyBuffer points_mask, ffi::AnyBuffer points_x,
                                ffi::Result<ffi::AnyBuffer> output) {
  return nufft2_impl<1, float>(eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3, modeord,
                               debug, spread_debug, showwarn, nthreads, fftw, spread_sort,
                               spread_kerevalmeth, spread_kerpad, upsampfac, spread_thread,
                               maxbatchsize, spread_nthr_atomic, spread_max_sp_size, source,
                               points_mask, points_x, points_x, points_x, output);
}

ffi::Error nufft1d2_1d_wrapper(double eps, int64_t iflag, int64_t n_tot, int64_t n_transf,
                               int64_t n_j, int64_t n_k_1, int64_t n_k_2, int64_t n_k_3,
                               int64_t modeord, int64_t debug, int64_t spread_debug,
                               int64_t showwarn, int64_t nthreads, int64_t fftw,
                               int64_t spread_sort, int64_t spread_kerevalmeth,
                               int64_t spread_kerpad, double upsampfac, int64_t spread_thread,
                               int64_t maxbatchsize, int64_t spread_nthr_atomic,
                               int64_t spread_max_sp_size, ffi::AnyBuffer source,
                               ffi::AnyBuffer points_mask, ffi::AnyBuffer points_x,
                               ffi::Result<ffi::AnyBuffer> output) {
  return nufft2_impl<1, double>(eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3, modeord,
                                debug, spread_debug, showwarn, nthreads, fftw, spread_sort,
                                spread_kerevalmeth, spread_kerpad, upsampfac, spread_thread,
                                maxbatchsize, spread_nthr_atomic, spread_max_sp_size, source,
                                points_mask, points_x, points_x, points_x, output);
}

ffi::Error nufft1d3f_1d_wrapper(float eps, int64_t iflag, int64_t n_tot, int64_t n_transf,
                                int64_t n_j, int64_t n_k_1, int64_t n_k_2, int64_t n_k_3,
                                int64_t modeord, int64_t debug, int64_t spread_debug,
                                int64_t showwarn, int64_t nthreads, int64_t fftw,
                                int64_t spread_sort, int64_t spread_kerevalmeth,
                                int64_t spread_kerpad, double upsampfac, int64_t spread_thread,
                                int64_t maxbatchsize, int64_t spread_nthr_atomic,
                                int64_t spread_max_sp_size, ffi::AnyBuffer source,
                                ffi::AnyBuffer points_x, ffi::AnyBuffer target_x,
                                ffi::Result<ffi::AnyBuffer> output) {
  return nufft3_impl<1, float>(eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3, modeord,
                               debug, spread_debug, showwarn, nthreads, fftw, spread_sort,
                               spread_kerevalmeth, spread_kerpad, upsampfac, spread_thread,
                               maxbatchsize, spread_nthr_atomic, spread_max_sp_size, source,
                               points_x, points_x, points_x, target_x, target_x, target_x, output);
}

ffi::Error nufft1d3_1d_wrapper(double eps, int64_t iflag, int64_t n_tot, int64_t n_transf,
                               int64_t n_j, int64_t n_k_1, int64_t n_k_2, int64_t n_k_3,
                               int64_t modeord, int64_t debug, int64_t spread_debug,
                               int64_t showwarn, int64_t nthreads, int64_t fftw,
                               int64_t spread_sort, int64_t spread_kerevalmeth,
                               int64_t spread_kerpad, double upsampfac, int64_t spread_thread,
                               int64_t maxbatchsize, int64_t spread_nthr_atomic,
                               int64_t spread_max_sp_size, ffi::AnyBuffer source,
                               ffi::AnyBuffer points_x, ffi::AnyBuffer target_x,
                               ffi::Result<ffi::AnyBuffer> output) {
  return nufft3_impl<1, double>(
      eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3, modeord, debug, spread_debug,
      showwarn, nthreads, fftw, spread_sort, spread_kerevalmeth, spread_kerpad, upsampfac,
      spread_thread, maxbatchsize, spread_nthr_atomic, spread_max_sp_size, source, points_x,
      points_x, points_x, target_x, target_x, target_x, output);
}

ffi::Error nufft2d1f_2d_wrapper(float eps, int64_t iflag, int64_t n_tot, int64_t n_transf,
                                int64_t n_j, int64_t n_k_1, int64_t n_k_2, int64_t n_k_3,
                                int64_t modeord, int64_t debug, int64_t spread_debug,
                                int64_t showwarn, int64_t nthreads, int64_t fftw,
                                int64_t spread_sort, int64_t spread_kerevalmeth,
                                int64_t spread_kerpad, double upsampfac, int64_t spread_thread,
                                int64_t maxbatchsize, int64_t spread_nthr_atomic,
                                int64_t spread_max_sp_size, ffi::AnyBuffer source,
                                ffi::AnyBuffer points_mask, ffi::AnyBuffer points_x,
                                ffi::AnyBuffer points_y, ffi::Result<ffi::AnyBuffer> output) {
  return nufft1_impl<2, float>(eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3, modeord,
                               debug, spread_debug, showwarn, nthreads, fftw, spread_sort,
                               spread_kerevalmeth, spread_kerpad, upsampfac, spread_thread,
                               maxbatchsize, spread_nthr_atomic, spread_max_sp_size, source,
                               points_mask, points_x, points_y, points_x, output);
}

ffi::Error nufft2d1_2d_wrapper(double eps, int64_t iflag, int64_t n_tot, int64_t n_transf,
                               int64_t n_j, int64_t n_k_1, int64_t n_k_2, int64_t n_k_3,
                               int64_t modeord, int64_t debug, int64_t spread_debug,
                               int64_t showwarn, int64_t nthreads, int64_t fftw,
                               int64_t spread_sort, int64_t spread_kerevalmeth,
                               int64_t spread_kerpad, double upsampfac, int64_t spread_thread,
                               int64_t maxbatchsize, int64_t spread_nthr_atomic,
                               int64_t spread_max_sp_size, ffi::AnyBuffer source,
                               ffi::AnyBuffer points_mask, ffi::AnyBuffer points_x,
                               ffi::AnyBuffer points_y, ffi::Result<ffi::AnyBuffer> output) {
  return nufft1_impl<2, double>(eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3, modeord,
                                debug, spread_debug, showwarn, nthreads, fftw, spread_sort,
                                spread_kerevalmeth, spread_kerpad, upsampfac, spread_thread,
                                maxbatchsize, spread_nthr_atomic, spread_max_sp_size, source,
                                points_mask, points_x, points_y, points_x, output);
}

ffi::Error nufft2d2f_2d_wrapper(float eps, int64_t iflag, int64_t n_tot, int64_t n_transf,
                                int64_t n_j, int64_t n_k_1, int64_t n_k_2, int64_t n_k_3,
                                int64_t modeord, int64_t debug, int64_t spread_debug,
                                int64_t showwarn, int64_t nthreads, int64_t fftw,
                                int64_t spread_sort, int64_t spread_kerevalmeth,
                                int64_t spread_kerpad, double upsampfac, int64_t spread_thread,
                                int64_t maxbatchsize, int64_t spread_nthr_atomic,
                                int64_t spread_max_sp_size, ffi::AnyBuffer source,
                                ffi::AnyBuffer points_mask, ffi::AnyBuffer points_x,
                                ffi::AnyBuffer points_y, ffi::Result<ffi::AnyBuffer> output) {
  return nufft2_impl<2, float>(eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3, modeord,
                               debug, spread_debug, showwarn, nthreads, fftw, spread_sort,
                               spread_kerevalmeth, spread_kerpad, upsampfac, spread_thread,
                               maxbatchsize, spread_nthr_atomic, spread_max_sp_size, source,
                               points_mask, points_x, points_y, points_x, output);
}

ffi::Error nufft2d2_2d_wrapper(double eps, int64_t iflag, int64_t n_tot, int64_t n_transf,
                               int64_t n_j, int64_t n_k_1, int64_t n_k_2, int64_t n_k_3,
                               int64_t modeord, int64_t debug, int64_t spread_debug,
                               int64_t showwarn, int64_t nthreads, int64_t fftw,
                               int64_t spread_sort, int64_t spread_kerevalmeth,
                               int64_t spread_kerpad, double upsampfac, int64_t spread_thread,
                               int64_t maxbatchsize, int64_t spread_nthr_atomic,
                               int64_t spread_max_sp_size, ffi::AnyBuffer source,
                               ffi::AnyBuffer points_mask, ffi::AnyBuffer points_x,
                               ffi::AnyBuffer points_y, ffi::Result<ffi::AnyBuffer> output) {
  return nufft2_impl<2, double>(eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3, modeord,
                                debug, spread_debug, showwarn, nthreads, fftw, spread_sort,
                                spread_kerevalmeth, spread_kerpad, upsampfac, spread_thread,
                                maxbatchsize, spread_nthr_atomic, spread_max_sp_size, source,
                                points_mask, points_x, points_y, points_x, output);
}

ffi::Error nufft2d3f_2d_wrapper(
    float eps, int64_t iflag, int64_t n_tot, int64_t n_transf, int64_t n_j, int64_t n_k_1,
    int64_t n_k_2, int64_t n_k_3, int64_t modeord, int64_t debug, int64_t spread_debug,
    int64_t showwarn, int64_t nthreads, int64_t fftw, int64_t spread_sort,
    int64_t spread_kerevalmeth, int64_t spread_kerpad, double upsampfac, int64_t spread_thread,
    int64_t maxbatchsize, int64_t spread_nthr_atomic, int64_t spread_max_sp_size,
    ffi::AnyBuffer source, ffi::AnyBuffer points_x, ffi::AnyBuffer points_y,
    ffi::AnyBuffer target_x, ffi::AnyBuffer target_y, ffi::Result<ffi::AnyBuffer> output) {
  return nufft3_impl<2, float>(eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3, modeord,
                               debug, spread_debug, showwarn, nthreads, fftw, spread_sort,
                               spread_kerevalmeth, spread_kerpad, upsampfac, spread_thread,
                               maxbatchsize, spread_nthr_atomic, spread_max_sp_size, source,
                               points_x, points_y, points_x, target_x, target_y, target_x, output);
}

ffi::Error nufft2d3_2d_wrapper(
    double eps, int64_t iflag, int64_t n_tot, int64_t n_transf, int64_t n_j, int64_t n_k_1,
    int64_t n_k_2, int64_t n_k_3, int64_t modeord, int64_t debug, int64_t spread_debug,
    int64_t showwarn, int64_t nthreads, int64_t fftw, int64_t spread_sort,
    int64_t spread_kerevalmeth, int64_t spread_kerpad, double upsampfac, int64_t spread_thread,
    int64_t maxbatchsize, int64_t spread_nthr_atomic, int64_t spread_max_sp_size,
    ffi::AnyBuffer source, ffi::AnyBuffer points_x, ffi::AnyBuffer points_y,
    ffi::AnyBuffer target_x, ffi::AnyBuffer target_y, ffi::Result<ffi::AnyBuffer> output) {
  return nufft3_impl<2, double>(
      eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3, modeord, debug, spread_debug,
      showwarn, nthreads, fftw, spread_sort, spread_kerevalmeth, spread_kerpad, upsampfac,
      spread_thread, maxbatchsize, spread_nthr_atomic, spread_max_sp_size, source, points_x,
      points_y, points_x, target_x, target_y, target_x, output);
}

ffi::Error nufft3d1f_3d_wrapper(
    float eps, int64_t iflag, int64_t n_tot, int64_t n_transf, int64_t n_j, int64_t n_k_1,
    int64_t n_k_2, int64_t n_k_3, int64_t modeord, int64_t debug, int64_t spread_debug,
    int64_t showwarn, int64_t nthreads, int64_t fftw, int64_t spread_sort,
    int64_t spread_kerevalmeth, int64_t spread_kerpad, double upsampfac, int64_t spread_thread,
    int64_t maxbatchsize, int64_t spread_nthr_atomic, int64_t spread_max_sp_size,
    ffi::AnyBuffer source, ffi::AnyBuffer points_mask, ffi::AnyBuffer points_x,
    ffi::AnyBuffer points_y, ffi::AnyBuffer points_z, ffi::Result<ffi::AnyBuffer> output) {
  return nufft1_impl<3, float>(eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3, modeord,
                               debug, spread_debug, showwarn, nthreads, fftw, spread_sort,
                               spread_kerevalmeth, spread_kerpad, upsampfac, spread_thread,
                               maxbatchsize, spread_nthr_atomic, spread_max_sp_size, source,
                               points_mask, points_x, points_y, points_z, output);
}

ffi::Error nufft3d1_3d_wrapper(
    double eps, int64_t iflag, int64_t n_tot, int64_t n_transf, int64_t n_j, int64_t n_k_1,
    int64_t n_k_2, int64_t n_k_3, int64_t modeord, int64_t debug, int64_t spread_debug,
    int64_t showwarn, int64_t nthreads, int64_t fftw, int64_t spread_sort,
    int64_t spread_kerevalmeth, int64_t spread_kerpad, double upsampfac, int64_t spread_thread,
    int64_t maxbatchsize, int64_t spread_nthr_atomic, int64_t spread_max_sp_size,
    ffi::AnyBuffer source, ffi::AnyBuffer points_mask, ffi::AnyBuffer points_x,
    ffi::AnyBuffer points_y, ffi::AnyBuffer points_z, ffi::Result<ffi::AnyBuffer> output) {
  return nufft1_impl<3, double>(eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3, modeord,
                                debug, spread_debug, showwarn, nthreads, fftw, spread_sort,
                                spread_kerevalmeth, spread_kerpad, upsampfac, spread_thread,
                                maxbatchsize, spread_nthr_atomic, spread_max_sp_size, source,
                                points_mask, points_x, points_y, points_z, output);
}

ffi::Error nufft3d2f_3d_wrapper(
    float eps, int64_t iflag, int64_t n_tot, int64_t n_transf, int64_t n_j, int64_t n_k_1,
    int64_t n_k_2, int64_t n_k_3, int64_t modeord, int64_t debug, int64_t spread_debug,
    int64_t showwarn, int64_t nthreads, int64_t fftw, int64_t spread_sort,
    int64_t spread_kerevalmeth, int64_t spread_kerpad, double upsampfac, int64_t spread_thread,
    int64_t maxbatchsize, int64_t spread_nthr_atomic, int64_t spread_max_sp_size,
    ffi::AnyBuffer source, ffi::AnyBuffer points_mask, ffi::AnyBuffer points_x,
    ffi::AnyBuffer points_y, ffi::AnyBuffer points_z, ffi::Result<ffi::AnyBuffer> output) {
  return nufft2_impl<3, float>(eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3, modeord,
                               debug, spread_debug, showwarn, nthreads, fftw, spread_sort,
                               spread_kerevalmeth, spread_kerpad, upsampfac, spread_thread,
                               maxbatchsize, spread_nthr_atomic, spread_max_sp_size, source,
                               points_mask, points_x, points_y, points_z, output);
}

ffi::Error nufft3d2_3d_wrapper(
    double eps, int64_t iflag, int64_t n_tot, int64_t n_transf, int64_t n_j, int64_t n_k_1,
    int64_t n_k_2, int64_t n_k_3, int64_t modeord, int64_t debug, int64_t spread_debug,
    int64_t showwarn, int64_t nthreads, int64_t fftw, int64_t spread_sort,
    int64_t spread_kerevalmeth, int64_t spread_kerpad, double upsampfac, int64_t spread_thread,
    int64_t maxbatchsize, int64_t spread_nthr_atomic, int64_t spread_max_sp_size,
    ffi::AnyBuffer source, ffi::AnyBuffer points_mask, ffi::AnyBuffer points_x,
    ffi::AnyBuffer points_y, ffi::AnyBuffer points_z, ffi::Result<ffi::AnyBuffer> output) {
  return nufft2_impl<3, double>(eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3, modeord,
                                debug, spread_debug, showwarn, nthreads, fftw, spread_sort,
                                spread_kerevalmeth, spread_kerpad, upsampfac, spread_thread,
                                maxbatchsize, spread_nthr_atomic, spread_max_sp_size, source,
                                points_mask, points_x, points_y, points_z, output);
}

ffi::Error nufft3d3f_3d_wrapper(
    float eps, int64_t iflag, int64_t n_tot, int64_t n_transf, int64_t n_j, int64_t n_k_1,
    int64_t n_k_2, int64_t n_k_3, int64_t modeord, int64_t debug, int64_t spread_debug,
    int64_t showwarn, int64_t nthreads, int64_t fftw, int64_t spread_sort,
    int64_t spread_kerevalmeth, int64_t spread_kerpad, double upsampfac, int64_t spread_thread,
    int64_t maxbatchsize, int64_t spread_nthr_atomic, int64_t spread_max_sp_size,
    ffi::AnyBuffer source, ffi::AnyBuffer points_x, ffi::AnyBuffer points_y,
    ffi::AnyBuffer points_z, ffi::AnyBuffer target_x, ffi::AnyBuffer target_y,
    ffi::AnyBuffer target_z, ffi::Result<ffi::AnyBuffer> output) {
  return nufft3_impl<3, float>(eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3, modeord,
                               debug, spread_debug, showwarn, nthreads, fftw, spread_sort,
                               spread_kerevalmeth, spread_kerpad, upsampfac, spread_thread,
                               maxbatchsize, spread_nthr_atomic, spread_max_sp_size, source,
                               points_x, points_y, points_z, target_x, target_y, target_z, output);
}

ffi::Error nufft3d3_3d_wrapper(
    double eps, int64_t iflag, int64_t n_tot, int64_t n_transf, int64_t n_j, int64_t n_k_1,
    int64_t n_k_2, int64_t n_k_3, int64_t modeord, int64_t debug, int64_t spread_debug,
    int64_t showwarn, int64_t nthreads, int64_t fftw, int64_t spread_sort,
    int64_t spread_kerevalmeth, int64_t spread_kerpad, double upsampfac, int64_t spread_thread,
    int64_t maxbatchsize, int64_t spread_nthr_atomic, int64_t spread_max_sp_size,
    ffi::AnyBuffer source, ffi::AnyBuffer points_x, ffi::AnyBuffer points_y,
    ffi::AnyBuffer points_z, ffi::AnyBuffer target_x, ffi::AnyBuffer target_y,
    ffi::AnyBuffer target_z, ffi::Result<ffi::AnyBuffer> output) {
  return nufft3_impl<3, double>(
      eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3, modeord, debug, spread_debug,
      showwarn, nthreads, fftw, spread_sort, spread_kerevalmeth, spread_kerpad, upsampfac,
      spread_thread, maxbatchsize, spread_nthr_atomic, spread_max_sp_size, source, points_x,
      points_y, points_z, target_x, target_y, target_z, output);
}

#define DEFINE_FFI_HANDLER(name, binding, impl)                                 \
  static constexpr XLA_FFI_Handler* name = +[](XLA_FFI_CallFrame* call_frame) { \
    static auto* handler = binding().To(impl).release();                        \
    return handler->Call(call_frame);                                           \
  }

DEFINE_FFI_HANDLER(nufft1d1f, MakeNufft1dBinding12Float, nufft1d1f_1d_wrapper);
DEFINE_FFI_HANDLER(nufft1d1, MakeNufft1dBinding12Double, nufft1d1_1d_wrapper);
DEFINE_FFI_HANDLER(nufft2d1f, MakeNufft2dBinding12Float, nufft2d1f_2d_wrapper);
DEFINE_FFI_HANDLER(nufft2d1, MakeNufft2dBinding12Double, nufft2d1_2d_wrapper);
DEFINE_FFI_HANDLER(nufft3d1f, MakeNufft3dBinding12Float, nufft3d1f_3d_wrapper);
DEFINE_FFI_HANDLER(nufft3d1, MakeNufft3dBinding12Double, nufft3d1_3d_wrapper);

DEFINE_FFI_HANDLER(nufft1d2f, MakeNufft1dBinding12Float, nufft1d2f_1d_wrapper);
DEFINE_FFI_HANDLER(nufft1d2, MakeNufft1dBinding12Double, nufft1d2_1d_wrapper);
DEFINE_FFI_HANDLER(nufft2d2f, MakeNufft2dBinding12Float, nufft2d2f_2d_wrapper);
DEFINE_FFI_HANDLER(nufft2d2, MakeNufft2dBinding12Double, nufft2d2_2d_wrapper);
DEFINE_FFI_HANDLER(nufft3d2f, MakeNufft3dBinding12Float, nufft3d2f_3d_wrapper);
DEFINE_FFI_HANDLER(nufft3d2, MakeNufft3dBinding12Double, nufft3d2_3d_wrapper);

DEFINE_FFI_HANDLER(nufft1d3f, MakeNufft1dBinding3Float, nufft1d3f_1d_wrapper);
DEFINE_FFI_HANDLER(nufft1d3, MakeNufft1dBinding3Double, nufft1d3_1d_wrapper);
DEFINE_FFI_HANDLER(nufft2d3f, MakeNufft2dBinding3Float, nufft2d3f_2d_wrapper);
DEFINE_FFI_HANDLER(nufft2d3, MakeNufft2dBinding3Double, nufft2d3_2d_wrapper);
DEFINE_FFI_HANDLER(nufft3d3f, MakeNufft3dBinding3Float, nufft3d3f_3d_wrapper);
DEFINE_FFI_HANDLER(nufft3d3, MakeNufft3dBinding3Double, nufft3d3_3d_wrapper);

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

NB_MODULE(jax_finufft_cpu, m) {
  m.def("registrations", &Registrations, "Get FFI handler registrations for all NUFFT operations");

  m.def("_omp_compile_check", []() {
#ifdef FINUFFT_USE_OPENMP
    return true;
#else
    return false;
#endif
  });

  m.attr("FFTW_ESTIMATE") = FFTW_ESTIMATE;
  m.attr("FFTW_MEASURE") = FFTW_MEASURE;
  m.attr("FFTW_PATIENT") = FFTW_PATIENT;
  m.attr("FFTW_EXHAUSTIVE") = FFTW_EXHAUSTIVE;
  m.attr("FFTW_WISDOM_ONLY") = FFTW_WISDOM_ONLY;
}
}  // namespace cpu
}  // namespace jax_finufft
