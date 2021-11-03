// This file defines the Python interface to the XLA custom call implemented on the CPU.
// It is exposed as a standard pybind11 module defining "capsule" objects containing our
// method. For simplicity, we export a separate capsule for each supported dtype.

#include "finufft.h"
#include "pybind11_kernel_helpers.h"

using namespace jax_finufft;

namespace {

template <typename T>
struct index_into {
  template <int ndim>
  static T *y(T *y, int64_t index) {
    return &(y[index]);
  }

  template <>
  static T *y<1>(T *y, int64_t index) {
    return NULL;
  }

  template <int ndim>
  static T *z(T *z, int64_t index) {
    return NULL;
  }

  template <>
  static T *z<3>(T *z, int64_t index) {
    return &(z[index]);
  }
};

template <int ndim, typename T>
void run_nufft(int type, void *desc_in, T *x, T *y, T *z, std::complex<T> *c, std::complex<T> *F) {
  const NufftDescriptor<ndim, T> *descriptor = unpack_descriptor<NufftDescriptor<ndim, T>>(
      reinterpret_cast<const char *>(desc_in), sizeof(NufftDescriptor<ndim, T>));
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
    setpts<T>(plan, descriptor->n_j, &(x[j]), index_into<T>::template y<ndim>(y, j),
              index_into<T>::template z<ndim>(z, j), 0, NULL, NULL, NULL);
    execute<T>(plan, &c[j], &F[k]);
  }
  destroy<T>(plan);
  delete opts;
}

template <typename T>
void nufft1d1(void *out, void **in) {
  std::complex<T> *c = reinterpret_cast<std::complex<T> *>(in[1]);
  T *x = reinterpret_cast<T *>(in[2]);
  T *y = NULL;
  T *z = NULL;

  std::complex<T> *F = reinterpret_cast<std::complex<T> *>(out);

  run_nufft<1, T>(1, in[0], x, y, z, c, F);
}

template <typename T>
void nufft1d2(void *out, void **in) {
  std::complex<T> *F = reinterpret_cast<std::complex<T> *>(in[1]);
  T *x = reinterpret_cast<T *>(in[2]);
  T *y = NULL;
  T *z = NULL;

  std::complex<T> *c = reinterpret_cast<std::complex<T> *>(out);

  run_nufft<1, T>(2, in[0], x, y, z, c, F);
}

pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["nufft1d1f"] = encapsulate_function(nufft1d1<float>);
  dict["nufft1d2f"] = encapsulate_function(nufft1d2<float>);

  dict["nufft1d1"] = encapsulate_function(nufft1d1<double>);
  dict["nufft1d2"] = encapsulate_function(nufft1d2<double>);

  return dict;
}

PYBIND11_MODULE(jax_finufft, m) {
  m.def("registrations", &Registrations);
  m.def("build_descriptor_1f", &build_descriptor_1<float>);
  m.def("build_descriptor_2f", &build_descriptor_2<float>);
  m.def("build_descriptor_3f", &build_descriptor_3<float>);
  m.def("build_descriptor_1", &build_descriptor_1<double>);
  m.def("build_descriptor_2", &build_descriptor_2<double>);
  m.def("build_descriptor_3", &build_descriptor_3<double>);
}

}  // namespace
