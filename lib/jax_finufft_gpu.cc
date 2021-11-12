// This file defines the Python interface to the XLA custom call implemented on the CPU.
// It is exposed as a standard pybind11 module defining "capsule" objects containing our
// method. For simplicity, we export a separate capsule for each supported dtype.

#include "pybind11_kernel_helpers.h"
#include "jax_finufft_gpu.h"
#include "kernels.h"

using namespace jax_finufft;

namespace {

pybind11::dict Registrations() {
  pybind11::dict dict;

  // dict["nufft1d1f"] = encapsulate_function(nufft1d1f);
  // dict["nufft1d2f"] = encapsulate_function(nufft1d2f);
  dict["nufft2d1f"] = encapsulate_function(nufft2d1f);
  dict["nufft2d2f"] = encapsulate_function(nufft2d2f);
  dict["nufft3d1f"] = encapsulate_function(nufft3d1f);
  dict["nufft3d2f"] = encapsulate_function(nufft3d2f);

  // dict["nufft1d1"] = encapsulate_function(nufft1d1);
  // dict["nufft1d2"] = encapsulate_function(nufft1d2);
  dict["nufft2d1"] = encapsulate_function(nufft2d1);
  dict["nufft2d2"] = encapsulate_function(nufft2d2);
  dict["nufft3d1"] = encapsulate_function(nufft3d1);
  dict["nufft3d2"] = encapsulate_function(nufft3d2);

  return dict;
}

PYBIND11_MODULE(jax_finufft_gpu, m) {
  m.def("registrations", &Registrations);
}

}  // namespace
