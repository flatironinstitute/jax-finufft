// This file defines the Python interface to the XLA custom call implemented on the CPU.
// It is exposed as a standard nanobind module defining "capsule" objects containing our
// method. For simplicity, we export a separate capsule for each supported dtype.

#include "jax_finufft_cpu.h"

#include "nanobind_kernel_helpers.h"

using namespace jax_finufft;
using namespace jax_finufft::cpu;
namespace nb = nanobind;

namespace {

template <int ndim, typename T, int type>
void run_nufft(void* desc_in, T* x, T* y, T* z, std::complex<T>* c, T* s, T* t, T* u,
               std::complex<T>* F) {
  const descriptor<T>* desc = unpack_descriptor<descriptor<T>>(
      reinterpret_cast<const char*>(desc_in), sizeof(descriptor<T>));
  int64_t n_k = 1;
  if constexpr (type != 3) {
    for (int d = 0; d < ndim; ++d) n_k *= desc->n_k[d];
  } else {
    // n_target is packed into n_k[0] for type 3 (n_k otherwise unused)
    n_k = desc->n_k[0];
  }
  finufft_opts opts = desc->opts;

  typename plan_type<T>::type plan;
  makeplan<T>(type, ndim, const_cast<int64_t*>(desc->n_k), desc->iflag, desc->n_transf, desc->eps,
              &plan, &opts);
  for (int64_t index = 0; index < desc->n_tot; ++index) {
    int64_t i = index * desc->n_j;
    int64_t j = i * desc->n_transf;
    int64_t k = index * n_k * desc->n_transf;

    if constexpr (type != 3) {
      setpts<T>(plan, desc->n_j, &(x[i]), y_index<ndim, T>(y, i), z_index<ndim, T>(z, i), 0, NULL,
                NULL, NULL);
    } else {
      int64_t i_target = index * n_k;
      setpts<T>(plan, desc->n_j, &(x[i]), y_index<ndim, T>(y, i), z_index<ndim, T>(z, i), n_k,
                &(s[i_target]), y_index<ndim, T>(t, i_target), z_index<ndim, T>(u, i_target));
    }
    execute<T>(plan, &c[j], &F[k]);  // type 1 & 3: c is input, F is output
  }
  destroy<T>(plan);
}

template <int ndim, typename T>
void nufft1(void* out, void** in) {
  std::complex<T>* c = reinterpret_cast<std::complex<T>*>(in[1]);
  T* x = reinterpret_cast<T*>(in[2]);
  T* y = NULL;
  T* z = NULL;
  if constexpr (ndim > 1) {
    y = reinterpret_cast<T*>(in[3]);
  }
  if constexpr (ndim > 2) {
    z = reinterpret_cast<T*>(in[4]);
  }
  std::complex<T>* F = reinterpret_cast<std::complex<T>*>(out);
  run_nufft<ndim, T, 1>(in[0], x, y, z, c, NULL, NULL, NULL, F);
}

template <int ndim, typename T>
void nufft2(void* out, void** in) {
  std::complex<T>* F = reinterpret_cast<std::complex<T>*>(in[1]);
  T* x = reinterpret_cast<T*>(in[2]);
  T* y = NULL;
  T* z = NULL;
  if constexpr (ndim > 1) {
    y = reinterpret_cast<T*>(in[3]);
  }
  if constexpr (ndim > 2) {
    z = reinterpret_cast<T*>(in[4]);
  }
  std::complex<T>* c = reinterpret_cast<std::complex<T>*>(out);
  run_nufft<ndim, T, 2>(in[0], x, y, z, c, NULL, NULL, NULL, F);
}

template <int ndim, typename T>
void nufft3(void* out, void** in) {
  // Note that nufft3 is like nufft1 in that `c` is the input and `F` the output.
  std::complex<T>* c = reinterpret_cast<std::complex<T>*>(in[1]);
  T* x = reinterpret_cast<T*>(in[2]);
  T* y = NULL;
  T* z = NULL;
  T* s = reinterpret_cast<T*>(in[2 + ndim]);
  T* t = NULL;
  T* u = NULL;

  if constexpr (ndim > 1) {
    y = reinterpret_cast<T*>(in[3]);
    t = reinterpret_cast<T*>(in[3 + ndim]);
  }
  if constexpr (ndim > 2) {
    z = reinterpret_cast<T*>(in[4]);
    u = reinterpret_cast<T*>(in[4 + ndim]);
  }

  std::complex<T>* F = reinterpret_cast<std::complex<T>*>(out);
  run_nufft<ndim, T, 3>(in[0], x, y, z, c, s, t, u, F);
}

template <typename T>
nb::bytes build_descriptor(T eps, int iflag, int64_t n_tot, int n_transf, int64_t n_j,
                           int64_t n_k_1, int64_t n_k_2, int64_t n_k_3, finufft_opts opts) {
  return pack_descriptor(
      descriptor<T>{eps, iflag, n_tot, n_transf, n_j, {n_k_1, n_k_2, n_k_3}, opts});
}

nb::dict Registrations() {
  nb::dict dict;

  dict["nufft1d1f"] = encapsulate_function(nufft1<1, float>);
  dict["nufft1d2f"] = encapsulate_function(nufft2<1, float>);
  dict["nufft1d3f"] = encapsulate_function(nufft3<1, float>);
  dict["nufft2d1f"] = encapsulate_function(nufft1<2, float>);
  dict["nufft2d2f"] = encapsulate_function(nufft2<2, float>);
  dict["nufft2d3f"] = encapsulate_function(nufft3<2, float>);
  dict["nufft3d1f"] = encapsulate_function(nufft1<3, float>);
  dict["nufft3d2f"] = encapsulate_function(nufft2<3, float>);
  dict["nufft3d3f"] = encapsulate_function(nufft3<3, float>);

  dict["nufft1d1"] = encapsulate_function(nufft1<1, double>);
  dict["nufft1d2"] = encapsulate_function(nufft2<1, double>);
  dict["nufft1d3"] = encapsulate_function(nufft3<1, double>);
  dict["nufft2d1"] = encapsulate_function(nufft1<2, double>);
  dict["nufft2d2"] = encapsulate_function(nufft2<2, double>);
  dict["nufft2d3"] = encapsulate_function(nufft3<2, double>);
  dict["nufft3d1"] = encapsulate_function(nufft1<3, double>);
  dict["nufft3d2"] = encapsulate_function(nufft2<3, double>);
  dict["nufft3d3"] = encapsulate_function(nufft3<3, double>);

  return dict;
}

NB_MODULE(jax_finufft_cpu, m) {
  m.def("registrations", &Registrations);
  m.def("build_descriptorf", &build_descriptor<float>);
  m.def("build_descriptor", &build_descriptor<double>);

  m.def("_omp_compile_check", []() {
#ifdef FINUFFT_USE_OPENMP
    return true;
#else
    return false;
#endif
  });

  m.attr("FFTW_ESTIMATE") = nb::int_(FFTW_ESTIMATE);
  m.attr("FFTW_MEASURE") = nb::int_(FFTW_MEASURE);
  m.attr("FFTW_PATIENT") = nb::int_(FFTW_PATIENT);
  m.attr("FFTW_EXHAUSTIVE") = nb::int_(FFTW_EXHAUSTIVE);
  m.attr("FFTW_WISDOM_ONLY") = nb::int_(FFTW_WISDOM_ONLY);

  nb::class_<finufft_opts> opts(m, "FinufftOpts");
  opts.def("__init__",
           [](finufft_opts* self, bool modeord, int debug, int spread_debug, bool showwarn,
              int nthreads, int fftw, int spread_sort, bool spread_kerevalmeth, bool spread_kerpad,
              double upsampfac, int spread_thread, int maxbatchsize, int spread_nthr_atomic,
              int spread_max_sp_size) {
             new (self) finufft_opts;
             default_opts<double>(self);
             self->modeord = int(modeord);
             self->debug = debug;
             self->spread_debug = spread_debug;
             self->showwarn = int(showwarn);
             self->nthreads = nthreads;
             self->fftw = fftw;
             self->spread_sort = spread_sort;
             self->spread_kerevalmeth = int(spread_kerevalmeth);
             self->spread_kerpad = int(spread_kerpad);
             self->upsampfac = upsampfac;
             self->spread_thread = int(spread_thread);
             self->maxbatchsize = maxbatchsize;
             self->spread_nthr_atomic = spread_nthr_atomic;
             self->spread_max_sp_size = spread_max_sp_size;
           });
}

}  // namespace
