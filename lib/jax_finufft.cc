// This file defines the Python interface to the XLA custom call implemented on the CPU.
// It is exposed as a standard pybind11 module defining "capsule" objects containing our
// method. For simplicity, we export a separate capsule for each supported dtype.

#include <pybind11/pybind11.h>

#include <complex>

#include "finufft.h"

// using namespace kepler_jax;

namespace {

// https://en.cppreference.com/w/cpp/numeric/bit_cast
template <class To, class From>
typename std::enable_if<sizeof(To) == sizeof(From) && std::is_trivially_copyable<From>::value &&
                            std::is_trivially_copyable<To>::value,
                        To>::type
bit_cast(const From &src) noexcept {
  static_assert(
      std::is_trivially_constructible<To>::value,
      "This implementation additionally requires destination type to be trivially constructible");

  To dst;
  memcpy(&dst, &src, sizeof(To));
  return dst;
}

template <typename T>
pybind11::capsule EncapsulateFunction(T *fn) {
  return pybind11::capsule(bit_cast<void *>(fn), "xla._CUSTOM_CALL_TARGET");
}

template <typename T>
void finufft1d1_(void *out, const void **in) {
  // Parse the inputs
  const int N = *reinterpret_cast<const int *>(in[0]);
  const int M = *reinterpret_cast<const int *>(in[1]);
  const int tol = *reinterpret_cast<const T *>(in[2]);
  const T *x = reinterpret_cast<const T *>(in[3]);
  const std::complex<T> *c = reinterpret_cast<const std::complex<T> *>(in[4]);

  std::complex<T> *F = reinterpret_cast<std::complex<T> *>(out);

  nufft_opts *opts = new nufft_opts;
  finufft_default_opts(opts);

  finufft1d1(M, const_cast<T *>(x), const_cast<std::complex<T> *>(c), 1, tol, N, F, opts);
}

pybind11::dict Registrations() {
  pybind11::dict dict;
  // dict["finufft1d1_f32"] = EncapsulateFunction(finufft1d1_<float>);
  dict["finufft1d1"] = EncapsulateFunction(finufft1d1_<double>);
  return dict;
}

PYBIND11_MODULE(jax_finufft, m) { m.def("registrations", &Registrations); }

}  // namespace