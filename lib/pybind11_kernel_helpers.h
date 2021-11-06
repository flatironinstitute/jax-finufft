// This header extends kernel_helpers.h with the pybind11 specific interface to
// serializing descriptors. It also adds a pybind11 function for wrapping our
// custom calls in a Python capsule. This is separate from kernel_helpers so that
// the CUDA code itself doesn't include pybind11. I don't think that this is
// strictly necessary, but they do it in jaxlib, so let's do it here too.

#ifndef _JAX_FINUFFT_PYBIND11_KERNEL_HELPERS_H_
#define _JAX_FINUFFT_PYBIND11_KERNEL_HELPERS_H_

#include <pybind11/pybind11.h>

#include "jax_finufft.h"
#include "kernel_helpers.h"

namespace jax_finufft {

template <typename T>
pybind11::bytes pack_descriptor(const T& descriptor) {
  return pybind11::bytes(pack_descriptor_as_string(descriptor));
}

template <typename T>
pybind11::capsule encapsulate_function(T* fn) {
  return pybind11::capsule(bit_cast<void*>(fn), "xla._CUSTOM_CALL_TARGET");
}

template <typename T>
pybind11::bytes build_descriptor(T eps, int iflag, int64_t n_tot, int n_transf, int64_t n_j,
                                 int64_t n_k_1, int64_t n_k_2, int64_t n_k_3) {
  return pack_descriptor(
      NufftDescriptor<T>{eps, iflag, n_tot, n_transf, n_j, {n_k_1, n_k_2, n_k_3}});
}

}  // namespace jax_finufft

#endif
