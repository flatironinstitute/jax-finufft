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
// FFI Handler Binding Definitions
// =============================================================================

// Build FFI binding for Type 1/2 operations with float eps
inline auto MakeNufftBinding12Float() {
  return ffi::Ffi::Bind()
      .Attr<float>("eps")
      .Attr<int64_t>("iflag")
      .Attr<int64_t>("n_tot")
      .Attr<int64_t>("n_transf")
      .Attr<int64_t>("n_j")
      .Attr<int64_t>("n_k_1")
      .Attr<int64_t>("n_k_2")
      .Attr<int64_t>("n_k_3")
      .Attr<int64_t>("modeord")
      .Attr<int64_t>("debug")
      .Attr<int64_t>("spread_debug")
      .Attr<int64_t>("showwarn")
      .Attr<int64_t>("nthreads")
      .Attr<int64_t>("fftw")
      .Attr<int64_t>("spread_sort")
      .Attr<int64_t>("spread_kerevalmeth")
      .Attr<int64_t>("spread_kerpad")
      .Attr<double>("upsampfac")
      .Attr<int64_t>("spread_thread")
      .Attr<int64_t>("maxbatchsize")
      .Attr<int64_t>("spread_nthr_atomic")
      .Attr<int64_t>("spread_max_sp_size")
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Ret<ffi::AnyBuffer>();
}

// Build FFI binding for Type 1/2 operations with double eps
inline auto MakeNufftBinding12Double() {
  return ffi::Ffi::Bind()
      .Attr<double>("eps")
      .Attr<int64_t>("iflag")
      .Attr<int64_t>("n_tot")
      .Attr<int64_t>("n_transf")
      .Attr<int64_t>("n_j")
      .Attr<int64_t>("n_k_1")
      .Attr<int64_t>("n_k_2")
      .Attr<int64_t>("n_k_3")
      .Attr<int64_t>("modeord")
      .Attr<int64_t>("debug")
      .Attr<int64_t>("spread_debug")
      .Attr<int64_t>("showwarn")
      .Attr<int64_t>("nthreads")
      .Attr<int64_t>("fftw")
      .Attr<int64_t>("spread_sort")
      .Attr<int64_t>("spread_kerevalmeth")
      .Attr<int64_t>("spread_kerpad")
      .Attr<double>("upsampfac")
      .Attr<int64_t>("spread_thread")
      .Attr<int64_t>("maxbatchsize")
      .Attr<int64_t>("spread_nthr_atomic")
      .Attr<int64_t>("spread_max_sp_size")
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Ret<ffi::AnyBuffer>();
}

// Build FFI binding for Type 3 operations with float eps
inline auto MakeNufftBinding3Float() {
  return ffi::Ffi::Bind()
      .Attr<float>("eps")
      .Attr<int64_t>("iflag")
      .Attr<int64_t>("n_tot")
      .Attr<int64_t>("n_transf")
      .Attr<int64_t>("n_j")
      .Attr<int64_t>("n_k_1")
      .Attr<int64_t>("n_k_2")
      .Attr<int64_t>("n_k_3")
      .Attr<int64_t>("modeord")
      .Attr<int64_t>("debug")
      .Attr<int64_t>("spread_debug")
      .Attr<int64_t>("showwarn")
      .Attr<int64_t>("nthreads")
      .Attr<int64_t>("fftw")
      .Attr<int64_t>("spread_sort")
      .Attr<int64_t>("spread_kerevalmeth")
      .Attr<int64_t>("spread_kerpad")
      .Attr<double>("upsampfac")
      .Attr<int64_t>("spread_thread")
      .Attr<int64_t>("maxbatchsize")
      .Attr<int64_t>("spread_nthr_atomic")
      .Attr<int64_t>("spread_max_sp_size")
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Ret<ffi::AnyBuffer>();
}

// Build FFI binding for Type 3 operations with double eps
inline auto MakeNufftBinding3Double() {
  return ffi::Ffi::Bind()
      .Attr<double>("eps")
      .Attr<int64_t>("iflag")
      .Attr<int64_t>("n_tot")
      .Attr<int64_t>("n_transf")
      .Attr<int64_t>("n_j")
      .Attr<int64_t>("n_k_1")
      .Attr<int64_t>("n_k_2")
      .Attr<int64_t>("n_k_3")
      .Attr<int64_t>("modeord")
      .Attr<int64_t>("debug")
      .Attr<int64_t>("spread_debug")
      .Attr<int64_t>("showwarn")
      .Attr<int64_t>("nthreads")
      .Attr<int64_t>("fftw")
      .Attr<int64_t>("spread_sort")
      .Attr<int64_t>("spread_kerevalmeth")
      .Attr<int64_t>("spread_kerpad")
      .Attr<double>("upsampfac")
      .Attr<int64_t>("spread_thread")
      .Attr<int64_t>("maxbatchsize")
      .Attr<int64_t>("spread_nthr_atomic")
      .Attr<int64_t>("spread_max_sp_size")
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Ret<ffi::AnyBuffer>();
}

// =============================================================================
// Explicit handler definitions without macros to avoid comma issues
// =============================================================================

// Type 1 handlers
static constexpr XLA_FFI_Handler* nufft1d1f = +[](XLA_FFI_CallFrame* call_frame) {
  static auto* handler = MakeNufftBinding12Float().To(nufft1_impl<1, float>).release();
  return handler->Call(call_frame);
};

static constexpr XLA_FFI_Handler* nufft1d1 = +[](XLA_FFI_CallFrame* call_frame) {
  static auto* handler = MakeNufftBinding12Double().To(nufft1_impl<1, double>).release();
  return handler->Call(call_frame);
};

static constexpr XLA_FFI_Handler* nufft2d1f = +[](XLA_FFI_CallFrame* call_frame) {
  static auto* handler = MakeNufftBinding12Float().To(nufft1_impl<2, float>).release();
  return handler->Call(call_frame);
};

static constexpr XLA_FFI_Handler* nufft2d1 = +[](XLA_FFI_CallFrame* call_frame) {
  static auto* handler = MakeNufftBinding12Double().To(nufft1_impl<2, double>).release();
  return handler->Call(call_frame);
};

static constexpr XLA_FFI_Handler* nufft3d1f = +[](XLA_FFI_CallFrame* call_frame) {
  static auto* handler = MakeNufftBinding12Float().To(nufft1_impl<3, float>).release();
  return handler->Call(call_frame);
};

static constexpr XLA_FFI_Handler* nufft3d1 = +[](XLA_FFI_CallFrame* call_frame) {
  static auto* handler = MakeNufftBinding12Double().To(nufft1_impl<3, double>).release();
  return handler->Call(call_frame);
};

// Type 2 handlers
static constexpr XLA_FFI_Handler* nufft1d2f = +[](XLA_FFI_CallFrame* call_frame) {
  static auto* handler = MakeNufftBinding12Float().To(nufft2_impl<1, float>).release();
  return handler->Call(call_frame);
};

static constexpr XLA_FFI_Handler* nufft1d2 = +[](XLA_FFI_CallFrame* call_frame) {
  static auto* handler = MakeNufftBinding12Double().To(nufft2_impl<1, double>).release();
  return handler->Call(call_frame);
};

static constexpr XLA_FFI_Handler* nufft2d2f = +[](XLA_FFI_CallFrame* call_frame) {
  static auto* handler = MakeNufftBinding12Float().To(nufft2_impl<2, float>).release();
  return handler->Call(call_frame);
};

static constexpr XLA_FFI_Handler* nufft2d2 = +[](XLA_FFI_CallFrame* call_frame) {
  static auto* handler = MakeNufftBinding12Double().To(nufft2_impl<2, double>).release();
  return handler->Call(call_frame);
};

static constexpr XLA_FFI_Handler* nufft3d2f = +[](XLA_FFI_CallFrame* call_frame) {
  static auto* handler = MakeNufftBinding12Float().To(nufft2_impl<3, float>).release();
  return handler->Call(call_frame);
};

static constexpr XLA_FFI_Handler* nufft3d2 = +[](XLA_FFI_CallFrame* call_frame) {
  static auto* handler = MakeNufftBinding12Double().To(nufft2_impl<3, double>).release();
  return handler->Call(call_frame);
};

// Type 3 handlers
static constexpr XLA_FFI_Handler* nufft1d3f = +[](XLA_FFI_CallFrame* call_frame) {
  static auto* handler = MakeNufftBinding3Float().To(nufft3_impl<1, float>).release();
  return handler->Call(call_frame);
};

static constexpr XLA_FFI_Handler* nufft1d3 = +[](XLA_FFI_CallFrame* call_frame) {
  static auto* handler = MakeNufftBinding3Double().To(nufft3_impl<1, double>).release();
  return handler->Call(call_frame);
};

static constexpr XLA_FFI_Handler* nufft2d3f = +[](XLA_FFI_CallFrame* call_frame) {
  static auto* handler = MakeNufftBinding3Float().To(nufft3_impl<2, float>).release();
  return handler->Call(call_frame);
};

static constexpr XLA_FFI_Handler* nufft2d3 = +[](XLA_FFI_CallFrame* call_frame) {
  static auto* handler = MakeNufftBinding3Double().To(nufft3_impl<2, double>).release();
  return handler->Call(call_frame);
};

static constexpr XLA_FFI_Handler* nufft3d3f = +[](XLA_FFI_CallFrame* call_frame) {
  static auto* handler = MakeNufftBinding3Float().To(nufft3_impl<3, float>).release();
  return handler->Call(call_frame);
};

static constexpr XLA_FFI_Handler* nufft3d3 = +[](XLA_FFI_CallFrame* call_frame) {
  static auto* handler = MakeNufftBinding3Double().To(nufft3_impl<3, double>).release();
  return handler->Call(call_frame);
};

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
