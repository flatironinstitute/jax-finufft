// Nanobind-specific interface for serializing descriptors and wrapping custom
// calls in Python capsules. This is separate from kernel_helpers.h so that
// CUDA code doesn't need to include nanobind headers.

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
