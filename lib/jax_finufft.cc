// This file defines the Python interface to the XLA custom call implemented on the CPU.
// It is exposed as a standard pybind11 module defining "capsule" objects containing our
// method. For simplicity, we export a separate capsule for each supported dtype.

#include "pybind11_kernel_helpers.h"

using namespace jax_finufft;

namespace {

template <int ndim, typename T>
void run_nufft(int type, void *desc_in, T *x, T *y, T *z, std::complex<T> *c, std::complex<T> *F) {
  const NufftDescriptor<T> *descriptor = unpack_descriptor<NufftDescriptor<T>>(
      reinterpret_cast<const char *>(desc_in), sizeof(NufftDescriptor<T>));
  int64_t n_k = 1;
  for (int d = 0; d < ndim; ++d) n_k *= descriptor->n_k[d];

  nufft_opts *opts = new nufft_opts;
  typename plan_type<T>::type plan;
  default_opts<T>(opts);
  makeplan<T>(type, ndim, const_cast<int64_t *>(descriptor->n_k), descriptor->iflag,
              descriptor->n_transf, descriptor->eps, &plan, opts);
  for (int64_t index = 0; index < descriptor->n_tot; ++index) {
    int64_t j = index * descriptor->n_j * descriptor->n_transf;
    int64_t k = index * n_k * descriptor->n_transf;

    setpts<T>(plan, descriptor->n_j, &(x[j]), y_index<ndim, T>(y, j), z_index<ndim, T>(z, j), 0,
              NULL, NULL, NULL);

    execute<T>(plan, &c[j], &F[k]);
  }
  destroy<T>(plan);
  delete opts;
}

template <int ndim, typename T>
void nufft1(void *out, void **in) {
  std::complex<T> *c = reinterpret_cast<std::complex<T> *>(in[1]);
  T *x = reinterpret_cast<T *>(in[2]);
  T *y = NULL;
  T *z = NULL;
  if (ndim > 1) {
    y = reinterpret_cast<T *>(in[3]);
  }
  if (ndim > 2) {
    z = reinterpret_cast<T *>(in[4]);
  }
  std::complex<T> *F = reinterpret_cast<std::complex<T> *>(out);
  run_nufft<ndim, T>(1, in[0], x, y, z, c, F);
}

template <int ndim, typename T>
void nufft2(void *out, void **in) {
  std::complex<T> *F = reinterpret_cast<std::complex<T> *>(in[1]);
  T *x = reinterpret_cast<T *>(in[2]);
  T *y = NULL;
  T *z = NULL;
  if (ndim > 1) {
    y = reinterpret_cast<T *>(in[3]);
  }
  if (ndim > 2) {
    z = reinterpret_cast<T *>(in[4]);
  }
  std::complex<T> *c = reinterpret_cast<std::complex<T> *>(out);
  run_nufft<ndim, T>(2, in[0], x, y, z, c, F);
}

pybind11::dict Registrations() {
  pybind11::dict dict;

  dict["nufft1d1f"] = encapsulate_function(nufft1<1, float>);
  dict["nufft1d2f"] = encapsulate_function(nufft2<1, float>);
  dict["nufft2d1f"] = encapsulate_function(nufft1<2, float>);
  dict["nufft2d2f"] = encapsulate_function(nufft2<2, float>);
  dict["nufft3d1f"] = encapsulate_function(nufft1<3, float>);
  dict["nufft3d2f"] = encapsulate_function(nufft2<3, float>);

  dict["nufft1d1"] = encapsulate_function(nufft1<1, double>);
  dict["nufft1d2"] = encapsulate_function(nufft2<1, double>);
  dict["nufft2d1"] = encapsulate_function(nufft1<2, double>);
  dict["nufft2d2"] = encapsulate_function(nufft2<2, double>);
  dict["nufft3d1"] = encapsulate_function(nufft1<3, double>);
  dict["nufft3d2"] = encapsulate_function(nufft2<3, double>);

  return dict;
}

PYBIND11_MODULE(jax_finufft, m) {
  m.def("registrations", &Registrations);
  m.def("build_descriptorf", &build_descriptor<float>);
  m.def("build_descriptor", &build_descriptor<double>);
}

}  // namespace
