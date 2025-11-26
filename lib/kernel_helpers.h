// Helper utilities for serializing descriptors used with XLA custom calls.
// This provides the infrastructure for the "opaque" parameter used by the GPU
// backend which still uses the legacy API.

#ifndef _JAX_FINUFFT_KERNEL_HELPERS_H_
#define _JAX_FINUFFT_KERNEL_HELPERS_H_

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace jax_finufft {

// https://en.cppreference.com/w/cpp/numeric/bit_cast
template <class To, class From>
typename std::enable_if<sizeof(To) == sizeof(From) &&
                            std::is_trivially_copyable<From>::value &&
                            std::is_trivially_copyable<To>::value,
                        To>::type
bit_cast(const From& src) noexcept {
  static_assert(std::is_trivially_constructible<To>::value,
                "This implementation additionally requires destination type "
                "to be trivially constructible");

  To dst;
  std::memcpy(&dst, &src, sizeof(To));
  return dst;
}

template <typename T>
const T* unpack_descriptor(const char* opaque, std::size_t opaque_len) {
  if (opaque_len != sizeof(T)) {
    throw std::runtime_error("Invalid opaque object size");
  }
  return bit_cast<const T*>(opaque);
}

}  // namespace jax_finufft

#endif
