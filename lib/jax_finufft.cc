// This file defines the Python interface to the XLA custom call implemented on the CPU.
// It is exposed as a standard pybind11 module defining "capsule" objects containing our
// method. For simplicity, we export a separate capsule for each supported dtype.

#include <pybind11/pybind11.h>

#include <complex>

#include "finufft.h"

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
void finufft1d1_(void *out, void **in) {
  int N = *reinterpret_cast<int *>(in[0]);
  int M = *reinterpret_cast<int *>(in[1]);
  T tol = *reinterpret_cast<T *>(in[2]);
  T *x = reinterpret_cast<T *>(in[3]);
  std::complex<T> *c = reinterpret_cast<std::complex<T> *>(in[4]);
  std::complex<T> *F = reinterpret_cast<std::complex<T> *>(out);
  nufft_opts *opts = new nufft_opts;

#ifdef SINGLE
  finufftf_default_opts(opts);
  finufftf1d1(M, x, c, 1, tol, N, F, opts);
#else
  finufft_default_opts(opts);
  finufft1d1(M, x, c, 1, tol, N, F, opts);
#endif

  delete opts;
}

template <typename T>
void finufft1d2_(void *out, void **in) {
  int N = *reinterpret_cast<int *>(in[0]);
  int M = *reinterpret_cast<int *>(in[1]);
  T tol = *reinterpret_cast<T *>(in[2]);
  T *x = reinterpret_cast<T *>(in[3]);
  std::complex<T> *F = reinterpret_cast<std::complex<T> *>(in[4]);
  std::complex<T> *c = reinterpret_cast<std::complex<T> *>(out);
  nufft_opts *opts = new nufft_opts;

#ifdef SINGLE
  finufftf_default_opts(opts);
  finufftf1d2(M, x, c, 1, tol, N, F, opts);
#else
  finufft_default_opts(opts);
  finufft1d2(M, x, c, 1, tol, N, F, opts);
#endif

  delete opts;
}

pybind11::dict Registrations() {
  pybind11::dict dict;
#ifdef SINGLE
  dict["finufft1d1_single"] = EncapsulateFunction(finufft1d1_<float>);
  dict["finufft1d2_single"] = EncapsulateFunction(finufft1d2_<float>);
#else
  dict["finufft1d1"] = EncapsulateFunction(finufft1d1_<double>);
  dict["finufft1d2"] = EncapsulateFunction(finufft1d2_<double>);
#endif
  return dict;
}

#ifdef SINGLE
PYBIND11_MODULE(jax_finufft_single, m) { m.def("registrations", &Registrations); }
#else
PYBIND11_MODULE(jax_finufft, m) { m.def("registrations", &Registrations); }
#endif

}  // namespace
