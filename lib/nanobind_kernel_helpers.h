// This header extends kernel_helpers.h with the nanobind specific interface to
// serializing descriptors. It also adds a nanobind function for wrapping our
// custom calls in a Python capsule. This is separate from kernel_helpers so that
// the CUDA code itself doesn't include nanobind. I don't think that this is
// strictly necessary, but they do it in jaxlib, so let's do it here too.

#ifndef _JAX_FINUFFT_NANOBIND_KERNEL_HELPERS_H_
#define _JAX_FINUFFT_NANOBIND_KERNEL_HELPERS_H_

#include <nanobind/nanobind.h>

#include "kernel_helpers.h"

namespace jax_finufft {

template <typename T>
nanobind::bytes pack_descriptor(const T& descriptor) {
  return nanobind::bytes(bit_cast<const char*>(&descriptor), sizeof(T));
}

template <typename T>
nanobind::capsule encapsulate_function(T* fn) {
  return nanobind::capsule(bit_cast<void*>(fn), "xla._CUSTOM_CALL_TARGET");
}

}  // namespace jax_finufft

#endif
