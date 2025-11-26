// GPU FFI bindings for jax-finufft using XLA typed FFI (api_version=4).
//
// This module exposes cuFINUFFT operations as XLA FFI custom calls with typed
// buffer interfaces, enabling seamless integration with JAX's compilation
// pipeline for CUDA execution.

#include "cufinufft_wrapper.h"
#include "kernels.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>

#include <cstdint>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
namespace nb = nanobind;

using namespace jax_finufft::gpu;

namespace {

// =============================================================================
// FFI Binding helpers
// =============================================================================

// Helper to build bindings for type 1 and 2 NUFFTs (float version)
inline auto MakeNufftBinding12Float() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Attr<float>("eps")
      .Attr<int64_t>("iflag")
      .Attr<int64_t>("n_tot")
      .Attr<int64_t>("n_transf")
      .Attr<int64_t>("n_j")
      .Attr<int64_t>("n_k_1")
      .Attr<int64_t>("n_k_2")
      .Attr<int64_t>("n_k_3")
      .Attr<int64_t>("modeord")
      .Attr<double>("upsampfac")
      .Attr<int64_t>("gpu_method")
      .Attr<int64_t>("gpu_sort")
      .Attr<int64_t>("gpu_kerevalmeth")
      .Attr<int64_t>("gpu_maxbatchsize")
      .Attr<int64_t>("debug")
      .Arg<ffi::AnyBuffer>()   // source
      .Arg<ffi::AnyBuffer>()   // points_x
      .Arg<ffi::AnyBuffer>()   // points_y
      .Arg<ffi::AnyBuffer>()   // points_z
      .Ret<ffi::AnyBuffer>();  // output
}

// Helper to build bindings for type 1 and 2 NUFFTs (double version)
inline auto MakeNufftBinding12Double() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Attr<double>("eps")
      .Attr<int64_t>("iflag")
      .Attr<int64_t>("n_tot")
      .Attr<int64_t>("n_transf")
      .Attr<int64_t>("n_j")
      .Attr<int64_t>("n_k_1")
      .Attr<int64_t>("n_k_2")
      .Attr<int64_t>("n_k_3")
      .Attr<int64_t>("modeord")
      .Attr<double>("upsampfac")
      .Attr<int64_t>("gpu_method")
      .Attr<int64_t>("gpu_sort")
      .Attr<int64_t>("gpu_kerevalmeth")
      .Attr<int64_t>("gpu_maxbatchsize")
      .Attr<int64_t>("debug")
      .Arg<ffi::AnyBuffer>()   // source
      .Arg<ffi::AnyBuffer>()   // points_x
      .Arg<ffi::AnyBuffer>()   // points_y
      .Arg<ffi::AnyBuffer>()   // points_z
      .Ret<ffi::AnyBuffer>();  // output
}

// Helper to build bindings for type 3 NUFFTs (float version)
inline auto MakeNufftBinding3Float() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Attr<float>("eps")
      .Attr<int64_t>("iflag")
      .Attr<int64_t>("n_tot")
      .Attr<int64_t>("n_transf")
      .Attr<int64_t>("n_j")
      .Attr<int64_t>("n_k_1")
      .Attr<int64_t>("n_k_2")
      .Attr<int64_t>("n_k_3")
      .Attr<int64_t>("modeord")
      .Attr<double>("upsampfac")
      .Attr<int64_t>("gpu_method")
      .Attr<int64_t>("gpu_sort")
      .Attr<int64_t>("gpu_kerevalmeth")
      .Attr<int64_t>("gpu_maxbatchsize")
      .Attr<int64_t>("debug")
      .Arg<ffi::AnyBuffer>()   // source
      .Arg<ffi::AnyBuffer>()   // points_x
      .Arg<ffi::AnyBuffer>()   // points_y
      .Arg<ffi::AnyBuffer>()   // points_z
      .Arg<ffi::AnyBuffer>()   // target_x
      .Arg<ffi::AnyBuffer>()   // target_y
      .Arg<ffi::AnyBuffer>()   // target_z
      .Ret<ffi::AnyBuffer>();  // output
}

// Helper to build bindings for type 3 NUFFTs (double version)
inline auto MakeNufftBinding3Double() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Attr<double>("eps")
      .Attr<int64_t>("iflag")
      .Attr<int64_t>("n_tot")
      .Attr<int64_t>("n_transf")
      .Attr<int64_t>("n_j")
      .Attr<int64_t>("n_k_1")
      .Attr<int64_t>("n_k_2")
      .Attr<int64_t>("n_k_3")
      .Attr<int64_t>("modeord")
      .Attr<double>("upsampfac")
      .Attr<int64_t>("gpu_method")
      .Attr<int64_t>("gpu_sort")
      .Attr<int64_t>("gpu_kerevalmeth")
      .Attr<int64_t>("gpu_maxbatchsize")
      .Attr<int64_t>("debug")
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
// FFI Handler definitions
// =============================================================================

// Type 1 handlers (non-uniform to uniform)
static constexpr XLA_FFI_Handler* nufft1d1f =
    +[](XLA_FFI_CallFrame* call_frame) {
      static auto* handler =
          MakeNufftBinding12Float()
              .To(nufft1d1f_impl)
              .release();
      return handler->Call(call_frame);
    };

static constexpr XLA_FFI_Handler* nufft1d1 =
    +[](XLA_FFI_CallFrame* call_frame) {
      static auto* handler =
          MakeNufftBinding12Double()
              .To(nufft1d1_impl)
              .release();
      return handler->Call(call_frame);
    };

static constexpr XLA_FFI_Handler* nufft2d1f =
    +[](XLA_FFI_CallFrame* call_frame) {
      static auto* handler =
          MakeNufftBinding12Float()
              .To(nufft2d1f_impl)
              .release();
      return handler->Call(call_frame);
    };

static constexpr XLA_FFI_Handler* nufft2d1 =
    +[](XLA_FFI_CallFrame* call_frame) {
      static auto* handler =
          MakeNufftBinding12Double()
              .To(nufft2d1_impl)
              .release();
      return handler->Call(call_frame);
    };

static constexpr XLA_FFI_Handler* nufft3d1f =
    +[](XLA_FFI_CallFrame* call_frame) {
      static auto* handler =
          MakeNufftBinding12Float()
              .To(nufft3d1f_impl)
              .release();
      return handler->Call(call_frame);
    };

static constexpr XLA_FFI_Handler* nufft3d1 =
    +[](XLA_FFI_CallFrame* call_frame) {
      static auto* handler =
          MakeNufftBinding12Double()
              .To(nufft3d1_impl)
              .release();
      return handler->Call(call_frame);
    };

// Type 2 handlers (uniform to non-uniform)
static constexpr XLA_FFI_Handler* nufft1d2f =
    +[](XLA_FFI_CallFrame* call_frame) {
      static auto* handler =
          MakeNufftBinding12Float()
              .To(nufft1d2f_impl)
              .release();
      return handler->Call(call_frame);
    };

static constexpr XLA_FFI_Handler* nufft1d2 =
    +[](XLA_FFI_CallFrame* call_frame) {
      static auto* handler =
          MakeNufftBinding12Double()
              .To(nufft1d2_impl)
              .release();
      return handler->Call(call_frame);
    };

static constexpr XLA_FFI_Handler* nufft2d2f =
    +[](XLA_FFI_CallFrame* call_frame) {
      static auto* handler =
          MakeNufftBinding12Float()
              .To(nufft2d2f_impl)
              .release();
      return handler->Call(call_frame);
    };

static constexpr XLA_FFI_Handler* nufft2d2 =
    +[](XLA_FFI_CallFrame* call_frame) {
      static auto* handler =
          MakeNufftBinding12Double()
              .To(nufft2d2_impl)
              .release();
      return handler->Call(call_frame);
    };

static constexpr XLA_FFI_Handler* nufft3d2f =
    +[](XLA_FFI_CallFrame* call_frame) {
      static auto* handler =
          MakeNufftBinding12Float()
              .To(nufft3d2f_impl)
              .release();
      return handler->Call(call_frame);
    };

static constexpr XLA_FFI_Handler* nufft3d2 =
    +[](XLA_FFI_CallFrame* call_frame) {
      static auto* handler =
          MakeNufftBinding12Double()
              .To(nufft3d2_impl)
              .release();
      return handler->Call(call_frame);
    };

// Type 3 handlers (non-uniform to non-uniform)
static constexpr XLA_FFI_Handler* nufft1d3f =
    +[](XLA_FFI_CallFrame* call_frame) {
      static auto* handler =
          MakeNufftBinding3Float()
              .To(nufft1d3f_impl)
              .release();
      return handler->Call(call_frame);
    };

static constexpr XLA_FFI_Handler* nufft1d3 =
    +[](XLA_FFI_CallFrame* call_frame) {
      static auto* handler =
          MakeNufftBinding3Double()
              .To(nufft1d3_impl)
              .release();
      return handler->Call(call_frame);
    };

static constexpr XLA_FFI_Handler* nufft2d3f =
    +[](XLA_FFI_CallFrame* call_frame) {
      static auto* handler =
          MakeNufftBinding3Float()
              .To(nufft2d3f_impl)
              .release();
      return handler->Call(call_frame);
    };

static constexpr XLA_FFI_Handler* nufft2d3 =
    +[](XLA_FFI_CallFrame* call_frame) {
      static auto* handler =
          MakeNufftBinding3Double()
              .To(nufft2d3_impl)
              .release();
      return handler->Call(call_frame);
    };

static constexpr XLA_FFI_Handler* nufft3d3f =
    +[](XLA_FFI_CallFrame* call_frame) {
      static auto* handler =
          MakeNufftBinding3Float()
              .To(nufft3d3f_impl)
              .release();
      return handler->Call(call_frame);
    };

static constexpr XLA_FFI_Handler* nufft3d3 =
    +[](XLA_FFI_CallFrame* call_frame) {
      static auto* handler =
          MakeNufftBinding3Double()
              .To(nufft3d3_impl)
              .release();
      return handler->Call(call_frame);
    };

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

NB_MODULE(jax_finufft_gpu, m) {
  m.def("registrations", &Registrations);
}

}  // namespace
