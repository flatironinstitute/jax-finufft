// GPU FFI bindings for jax-finufft using XLA typed FFI (api_version=4).
//
// This module exposes cuFINUFFT operations as XLA FFI custom calls with typed
// buffer interfaces, enabling seamless integration with JAX's compilation
// pipeline for CUDA execution.

#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>

#include <cstdint>

#include "cufinufft_wrapper.h"
#include "kernels.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
namespace nb = nanobind;

using namespace jax_finufft::gpu;

namespace {

// =============================================================================
// FFI Binding helpers - common attributes builder
// =============================================================================

// Common attributes for all NUFFT types
#define NUFFT_COMMON_ATTRS_FLOAT         \
  .Attr<float>("eps")                    \
      .Attr<int64_t>("iflag")            \
      .Attr<int64_t>("n_tot")            \
      .Attr<int64_t>("n_transf")         \
      .Attr<int64_t>("n_j")              \
      .Attr<int64_t>("n_k_1")            \
      .Attr<int64_t>("n_k_2")            \
      .Attr<int64_t>("n_k_3")            \
      .Attr<int64_t>("modeord")          \
      .Attr<double>("upsampfac")         \
      .Attr<int64_t>("gpu_method")       \
      .Attr<int64_t>("gpu_sort")         \
      .Attr<int64_t>("gpu_kerevalmeth")  \
      .Attr<int64_t>("gpu_maxbatchsize") \
      .Attr<int64_t>("debug")

#define NUFFT_COMMON_ATTRS_DOUBLE        \
  .Attr<double>("eps")                   \
      .Attr<int64_t>("iflag")            \
      .Attr<int64_t>("n_tot")            \
      .Attr<int64_t>("n_transf")         \
      .Attr<int64_t>("n_j")              \
      .Attr<int64_t>("n_k_1")            \
      .Attr<int64_t>("n_k_2")            \
      .Attr<int64_t>("n_k_3")            \
      .Attr<int64_t>("modeord")          \
      .Attr<double>("upsampfac")         \
      .Attr<int64_t>("gpu_method")       \
      .Attr<int64_t>("gpu_sort")         \
      .Attr<int64_t>("gpu_kerevalmeth")  \
      .Attr<int64_t>("gpu_maxbatchsize") \
      .Attr<int64_t>("debug")

// =============================================================================
// 1D NUFFT bindings (1 point array)
// =============================================================================

inline auto MakeNufft1dBinding12Float() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>() NUFFT_COMMON_ATTRS_FLOAT
      .Arg<ffi::AnyBuffer>()   // source
      .Arg<ffi::AnyBuffer>()   // points_x
      .Ret<ffi::AnyBuffer>();  // output
}

inline auto MakeNufft1dBinding12Double() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>() NUFFT_COMMON_ATTRS_DOUBLE
      .Arg<ffi::AnyBuffer>()   // source
      .Arg<ffi::AnyBuffer>()   // points_x
      .Ret<ffi::AnyBuffer>();  // output
}

inline auto MakeNufft1dBinding3Float() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>() NUFFT_COMMON_ATTRS_FLOAT
      .Arg<ffi::AnyBuffer>()   // source
      .Arg<ffi::AnyBuffer>()   // points_x
      .Arg<ffi::AnyBuffer>()   // target_x
      .Ret<ffi::AnyBuffer>();  // output
}

inline auto MakeNufft1dBinding3Double() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>() NUFFT_COMMON_ATTRS_DOUBLE
      .Arg<ffi::AnyBuffer>()   // source
      .Arg<ffi::AnyBuffer>()   // points_x
      .Arg<ffi::AnyBuffer>()   // target_x
      .Ret<ffi::AnyBuffer>();  // output
}

// =============================================================================
// 2D NUFFT bindings (2 point arrays)
// =============================================================================

inline auto MakeNufft2dBinding12Float() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>() NUFFT_COMMON_ATTRS_FLOAT
      .Arg<ffi::AnyBuffer>()   // source
      .Arg<ffi::AnyBuffer>()   // points_x
      .Arg<ffi::AnyBuffer>()   // points_y
      .Ret<ffi::AnyBuffer>();  // output
}

inline auto MakeNufft2dBinding12Double() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>() NUFFT_COMMON_ATTRS_DOUBLE
      .Arg<ffi::AnyBuffer>()   // source
      .Arg<ffi::AnyBuffer>()   // points_x
      .Arg<ffi::AnyBuffer>()   // points_y
      .Ret<ffi::AnyBuffer>();  // output
}

inline auto MakeNufft2dBinding3Float() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>() NUFFT_COMMON_ATTRS_FLOAT
      .Arg<ffi::AnyBuffer>()   // source
      .Arg<ffi::AnyBuffer>()   // points_x
      .Arg<ffi::AnyBuffer>()   // points_y
      .Arg<ffi::AnyBuffer>()   // target_x
      .Arg<ffi::AnyBuffer>()   // target_y
      .Ret<ffi::AnyBuffer>();  // output
}

inline auto MakeNufft2dBinding3Double() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>() NUFFT_COMMON_ATTRS_DOUBLE
      .Arg<ffi::AnyBuffer>()   // source
      .Arg<ffi::AnyBuffer>()   // points_x
      .Arg<ffi::AnyBuffer>()   // points_y
      .Arg<ffi::AnyBuffer>()   // target_x
      .Arg<ffi::AnyBuffer>()   // target_y
      .Ret<ffi::AnyBuffer>();  // output
}

// =============================================================================
// 3D NUFFT bindings (3 point arrays)
// =============================================================================

inline auto MakeNufft3dBinding12Float() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>() NUFFT_COMMON_ATTRS_FLOAT
      .Arg<ffi::AnyBuffer>()   // source
      .Arg<ffi::AnyBuffer>()   // points_x
      .Arg<ffi::AnyBuffer>()   // points_y
      .Arg<ffi::AnyBuffer>()   // points_z
      .Ret<ffi::AnyBuffer>();  // output
}

inline auto MakeNufft3dBinding12Double() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>() NUFFT_COMMON_ATTRS_DOUBLE
      .Arg<ffi::AnyBuffer>()   // source
      .Arg<ffi::AnyBuffer>()   // points_x
      .Arg<ffi::AnyBuffer>()   // points_y
      .Arg<ffi::AnyBuffer>()   // points_z
      .Ret<ffi::AnyBuffer>();  // output
}

inline auto MakeNufft3dBinding3Float() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>() NUFFT_COMMON_ATTRS_FLOAT
      .Arg<ffi::AnyBuffer>()   // source
      .Arg<ffi::AnyBuffer>()   // points_x
      .Arg<ffi::AnyBuffer>()   // points_y
      .Arg<ffi::AnyBuffer>()   // points_z
      .Arg<ffi::AnyBuffer>()   // target_x
      .Arg<ffi::AnyBuffer>()   // target_y
      .Arg<ffi::AnyBuffer>()   // target_z
      .Ret<ffi::AnyBuffer>();  // output
}

inline auto MakeNufft3dBinding3Double() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>() NUFFT_COMMON_ATTRS_DOUBLE
      .Arg<ffi::AnyBuffer>()   // source
      .Arg<ffi::AnyBuffer>()   // points_x
      .Arg<ffi::AnyBuffer>()   // points_y
      .Arg<ffi::AnyBuffer>()   // points_z
      .Arg<ffi::AnyBuffer>()   // target_x
      .Arg<ffi::AnyBuffer>()   // target_y
      .Arg<ffi::AnyBuffer>()   // target_z
      .Ret<ffi::AnyBuffer>();  // output
}

// =============================================================================
// 1D Wrapper implementations (use points_x as dummy for y and z)
// =============================================================================

// Type 1 1D wrappers
ffi::Error nufft1d1f_1d_wrapper(cudaStream_t stream, float eps, int64_t iflag, int64_t n_tot,
                                int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2,
                                int64_t n_k_3, int64_t modeord, double upsampfac,
                                int64_t gpu_method, int64_t gpu_sort, int64_t gpu_kerevalmeth,
                                int64_t gpu_maxbatchsize, int64_t debug, ffi::AnyBuffer source,
                                ffi::AnyBuffer points_x, ffi::Result<ffi::AnyBuffer> output) {
  return nufft1d1f_impl(stream, eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3, modeord,
                        upsampfac, gpu_method, gpu_sort, gpu_kerevalmeth, gpu_maxbatchsize, debug,
                        source, points_x, points_x, points_x, output);
}

ffi::Error nufft1d1_1d_wrapper(cudaStream_t stream, double eps, int64_t iflag, int64_t n_tot,
                               int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2,
                               int64_t n_k_3, int64_t modeord, double upsampfac,
                               int64_t gpu_method, int64_t gpu_sort, int64_t gpu_kerevalmeth,
                               int64_t gpu_maxbatchsize, int64_t debug, ffi::AnyBuffer source,
                               ffi::AnyBuffer points_x, ffi::Result<ffi::AnyBuffer> output) {
  return nufft1d1_impl(stream, eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3, modeord,
                       upsampfac, gpu_method, gpu_sort, gpu_kerevalmeth, gpu_maxbatchsize, debug,
                       source, points_x, points_x, points_x, output);
}

// Type 2 1D wrappers
ffi::Error nufft1d2f_1d_wrapper(cudaStream_t stream, float eps, int64_t iflag, int64_t n_tot,
                                int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2,
                                int64_t n_k_3, int64_t modeord, double upsampfac,
                                int64_t gpu_method, int64_t gpu_sort, int64_t gpu_kerevalmeth,
                                int64_t gpu_maxbatchsize, int64_t debug, ffi::AnyBuffer source,
                                ffi::AnyBuffer points_x, ffi::Result<ffi::AnyBuffer> output) {
  return nufft1d2f_impl(stream, eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3, modeord,
                        upsampfac, gpu_method, gpu_sort, gpu_kerevalmeth, gpu_maxbatchsize, debug,
                        source, points_x, points_x, points_x, output);
}

ffi::Error nufft1d2_1d_wrapper(cudaStream_t stream, double eps, int64_t iflag, int64_t n_tot,
                               int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2,
                               int64_t n_k_3, int64_t modeord, double upsampfac,
                               int64_t gpu_method, int64_t gpu_sort, int64_t gpu_kerevalmeth,
                               int64_t gpu_maxbatchsize, int64_t debug, ffi::AnyBuffer source,
                               ffi::AnyBuffer points_x, ffi::Result<ffi::AnyBuffer> output) {
  return nufft1d2_impl(stream, eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3, modeord,
                       upsampfac, gpu_method, gpu_sort, gpu_kerevalmeth, gpu_maxbatchsize, debug,
                       source, points_x, points_x, points_x, output);
}

// Type 3 1D wrappers
ffi::Error nufft1d3f_1d_wrapper(cudaStream_t stream, float eps, int64_t iflag, int64_t n_tot,
                                int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2,
                                int64_t n_k_3, int64_t modeord, double upsampfac,
                                int64_t gpu_method, int64_t gpu_sort, int64_t gpu_kerevalmeth,
                                int64_t gpu_maxbatchsize, int64_t debug, ffi::AnyBuffer source,
                                ffi::AnyBuffer points_x, ffi::AnyBuffer target_x,
                                ffi::Result<ffi::AnyBuffer> output) {
  return nufft1d3f_impl(stream, eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3, modeord,
                        upsampfac, gpu_method, gpu_sort, gpu_kerevalmeth, gpu_maxbatchsize, debug,
                        source, points_x, points_x, points_x, target_x, target_x, target_x,
                        output);
}

ffi::Error nufft1d3_1d_wrapper(cudaStream_t stream, double eps, int64_t iflag, int64_t n_tot,
                               int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2,
                               int64_t n_k_3, int64_t modeord, double upsampfac,
                               int64_t gpu_method, int64_t gpu_sort, int64_t gpu_kerevalmeth,
                               int64_t gpu_maxbatchsize, int64_t debug, ffi::AnyBuffer source,
                               ffi::AnyBuffer points_x, ffi::AnyBuffer target_x,
                               ffi::Result<ffi::AnyBuffer> output) {
  return nufft1d3_impl(stream, eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3, modeord,
                       upsampfac, gpu_method, gpu_sort, gpu_kerevalmeth, gpu_maxbatchsize, debug,
                       source, points_x, points_x, points_x, target_x, target_x, target_x, output);
}

// =============================================================================
// 2D Wrapper implementations (use points_x as dummy for z)
// =============================================================================

// Type 1 2D wrappers
ffi::Error nufft2d1f_2d_wrapper(cudaStream_t stream, float eps, int64_t iflag, int64_t n_tot,
                                int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2,
                                int64_t n_k_3, int64_t modeord, double upsampfac,
                                int64_t gpu_method, int64_t gpu_sort, int64_t gpu_kerevalmeth,
                                int64_t gpu_maxbatchsize, int64_t debug, ffi::AnyBuffer source,
                                ffi::AnyBuffer points_x, ffi::AnyBuffer points_y,
                                ffi::Result<ffi::AnyBuffer> output) {
  return nufft2d1f_impl(stream, eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3, modeord,
                        upsampfac, gpu_method, gpu_sort, gpu_kerevalmeth, gpu_maxbatchsize, debug,
                        source, points_x, points_y, points_x, output);
}

ffi::Error nufft2d1_2d_wrapper(cudaStream_t stream, double eps, int64_t iflag, int64_t n_tot,
                               int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2,
                               int64_t n_k_3, int64_t modeord, double upsampfac,
                               int64_t gpu_method, int64_t gpu_sort, int64_t gpu_kerevalmeth,
                               int64_t gpu_maxbatchsize, int64_t debug, ffi::AnyBuffer source,
                               ffi::AnyBuffer points_x, ffi::AnyBuffer points_y,
                               ffi::Result<ffi::AnyBuffer> output) {
  return nufft2d1_impl(stream, eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3, modeord,
                       upsampfac, gpu_method, gpu_sort, gpu_kerevalmeth, gpu_maxbatchsize, debug,
                       source, points_x, points_y, points_x, output);
}

// Type 2 2D wrappers
ffi::Error nufft2d2f_2d_wrapper(cudaStream_t stream, float eps, int64_t iflag, int64_t n_tot,
                                int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2,
                                int64_t n_k_3, int64_t modeord, double upsampfac,
                                int64_t gpu_method, int64_t gpu_sort, int64_t gpu_kerevalmeth,
                                int64_t gpu_maxbatchsize, int64_t debug, ffi::AnyBuffer source,
                                ffi::AnyBuffer points_x, ffi::AnyBuffer points_y,
                                ffi::Result<ffi::AnyBuffer> output) {
  return nufft2d2f_impl(stream, eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3, modeord,
                        upsampfac, gpu_method, gpu_sort, gpu_kerevalmeth, gpu_maxbatchsize, debug,
                        source, points_x, points_y, points_x, output);
}

ffi::Error nufft2d2_2d_wrapper(cudaStream_t stream, double eps, int64_t iflag, int64_t n_tot,
                               int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2,
                               int64_t n_k_3, int64_t modeord, double upsampfac,
                               int64_t gpu_method, int64_t gpu_sort, int64_t gpu_kerevalmeth,
                               int64_t gpu_maxbatchsize, int64_t debug, ffi::AnyBuffer source,
                               ffi::AnyBuffer points_x, ffi::AnyBuffer points_y,
                               ffi::Result<ffi::AnyBuffer> output) {
  return nufft2d2_impl(stream, eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3, modeord,
                       upsampfac, gpu_method, gpu_sort, gpu_kerevalmeth, gpu_maxbatchsize, debug,
                       source, points_x, points_y, points_x, output);
}

// Type 3 2D wrappers
ffi::Error nufft2d3f_2d_wrapper(cudaStream_t stream, float eps, int64_t iflag, int64_t n_tot,
                                int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2,
                                int64_t n_k_3, int64_t modeord, double upsampfac,
                                int64_t gpu_method, int64_t gpu_sort, int64_t gpu_kerevalmeth,
                                int64_t gpu_maxbatchsize, int64_t debug, ffi::AnyBuffer source,
                                ffi::AnyBuffer points_x, ffi::AnyBuffer points_y,
                                ffi::AnyBuffer target_x, ffi::AnyBuffer target_y,
                                ffi::Result<ffi::AnyBuffer> output) {
  return nufft2d3f_impl(stream, eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3, modeord,
                        upsampfac, gpu_method, gpu_sort, gpu_kerevalmeth, gpu_maxbatchsize, debug,
                        source, points_x, points_y, points_x, target_x, target_y, target_x,
                        output);
}

ffi::Error nufft2d3_2d_wrapper(cudaStream_t stream, double eps, int64_t iflag, int64_t n_tot,
                               int64_t n_transf, int64_t n_j, int64_t n_k_1, int64_t n_k_2,
                               int64_t n_k_3, int64_t modeord, double upsampfac,
                               int64_t gpu_method, int64_t gpu_sort, int64_t gpu_kerevalmeth,
                               int64_t gpu_maxbatchsize, int64_t debug, ffi::AnyBuffer source,
                               ffi::AnyBuffer points_x, ffi::AnyBuffer points_y,
                               ffi::AnyBuffer target_x, ffi::AnyBuffer target_y,
                               ffi::Result<ffi::AnyBuffer> output) {
  return nufft2d3_impl(stream, eps, iflag, n_tot, n_transf, n_j, n_k_1, n_k_2, n_k_3, modeord,
                       upsampfac, gpu_method, gpu_sort, gpu_kerevalmeth, gpu_maxbatchsize, debug,
                       source, points_x, points_y, points_x, target_x, target_y, target_x, output);
}

// =============================================================================
// FFI Handler definitions using macros
// =============================================================================

#define DEFINE_FFI_HANDLER(name, binding, impl)                                 \
  static constexpr XLA_FFI_Handler* name = +[](XLA_FFI_CallFrame* call_frame) { \
    static auto* handler = binding().To(impl).release();                        \
    return handler->Call(call_frame);                                           \
  }

// Type 1 handlers (non-uniform to uniform) - dimension-specific bindings
DEFINE_FFI_HANDLER(nufft1d1f, MakeNufft1dBinding12Float, nufft1d1f_1d_wrapper);
DEFINE_FFI_HANDLER(nufft1d1, MakeNufft1dBinding12Double, nufft1d1_1d_wrapper);
DEFINE_FFI_HANDLER(nufft2d1f, MakeNufft2dBinding12Float, nufft2d1f_2d_wrapper);
DEFINE_FFI_HANDLER(nufft2d1, MakeNufft2dBinding12Double, nufft2d1_2d_wrapper);
DEFINE_FFI_HANDLER(nufft3d1f, MakeNufft3dBinding12Float, nufft3d1f_impl);
DEFINE_FFI_HANDLER(nufft3d1, MakeNufft3dBinding12Double, nufft3d1_impl);

// Type 2 handlers (uniform to non-uniform) - dimension-specific bindings
DEFINE_FFI_HANDLER(nufft1d2f, MakeNufft1dBinding12Float, nufft1d2f_1d_wrapper);
DEFINE_FFI_HANDLER(nufft1d2, MakeNufft1dBinding12Double, nufft1d2_1d_wrapper);
DEFINE_FFI_HANDLER(nufft2d2f, MakeNufft2dBinding12Float, nufft2d2f_2d_wrapper);
DEFINE_FFI_HANDLER(nufft2d2, MakeNufft2dBinding12Double, nufft2d2_2d_wrapper);
DEFINE_FFI_HANDLER(nufft3d2f, MakeNufft3dBinding12Float, nufft3d2f_impl);
DEFINE_FFI_HANDLER(nufft3d2, MakeNufft3dBinding12Double, nufft3d2_impl);

// Type 3 handlers (non-uniform to non-uniform) - dimension-specific bindings
DEFINE_FFI_HANDLER(nufft1d3f, MakeNufft1dBinding3Float, nufft1d3f_1d_wrapper);
DEFINE_FFI_HANDLER(nufft1d3, MakeNufft1dBinding3Double, nufft1d3_1d_wrapper);
DEFINE_FFI_HANDLER(nufft2d3f, MakeNufft2dBinding3Float, nufft2d3f_2d_wrapper);
DEFINE_FFI_HANDLER(nufft2d3, MakeNufft2dBinding3Double, nufft2d3_2d_wrapper);
DEFINE_FFI_HANDLER(nufft3d3f, MakeNufft3dBinding3Float, nufft3d3f_impl);
DEFINE_FFI_HANDLER(nufft3d3, MakeNufft3dBinding3Double, nufft3d3_impl);

#undef DEFINE_FFI_HANDLER

// =============================================================================
// Python module registration
// =============================================================================

nb::dict Registrations() {
  nb::dict dict;

  // Type 1 (non-uniform to uniform)
  dict["nufft1d1f"] = nb::capsule(reinterpret_cast<void*>(nufft1d1f));
  dict["nufft1d1"] = nb::capsule(reinterpret_cast<void*>(nufft1d1));
  dict["nufft2d1f"] = nb::capsule(reinterpret_cast<void*>(nufft2d1f));
  dict["nufft2d1"] = nb::capsule(reinterpret_cast<void*>(nufft2d1));
  dict["nufft3d1f"] = nb::capsule(reinterpret_cast<void*>(nufft3d1f));
  dict["nufft3d1"] = nb::capsule(reinterpret_cast<void*>(nufft3d1));

  // Type 2 (uniform to non-uniform)
  dict["nufft1d2f"] = nb::capsule(reinterpret_cast<void*>(nufft1d2f));
  dict["nufft1d2"] = nb::capsule(reinterpret_cast<void*>(nufft1d2));
  dict["nufft2d2f"] = nb::capsule(reinterpret_cast<void*>(nufft2d2f));
  dict["nufft2d2"] = nb::capsule(reinterpret_cast<void*>(nufft2d2));
  dict["nufft3d2f"] = nb::capsule(reinterpret_cast<void*>(nufft3d2f));
  dict["nufft3d2"] = nb::capsule(reinterpret_cast<void*>(nufft3d2));

  // Type 3 (non-uniform to non-uniform)
  dict["nufft1d3f"] = nb::capsule(reinterpret_cast<void*>(nufft1d3f));
  dict["nufft1d3"] = nb::capsule(reinterpret_cast<void*>(nufft1d3));
  dict["nufft2d3f"] = nb::capsule(reinterpret_cast<void*>(nufft2d3f));
  dict["nufft2d3"] = nb::capsule(reinterpret_cast<void*>(nufft2d3));
  dict["nufft3d3f"] = nb::capsule(reinterpret_cast<void*>(nufft3d3f));
  dict["nufft3d3"] = nb::capsule(reinterpret_cast<void*>(nufft3d3));

  return dict;
}

NB_MODULE(jax_finufft_gpu, m) { m.def("registrations", &Registrations); }

}  // namespace
