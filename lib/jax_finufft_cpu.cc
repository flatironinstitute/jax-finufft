// This file defines the Python interface to the XLA custom call implemented on the CPU.
// It is exposed as a standard pybind11 module defining "capsule" objects containing our
// method. For simplicity, we export a separate capsule for each supported dtype.

#include "jax_finufft_cpu.h"

#include "pybind11_kernel_helpers.h"

using namespace jax_finufft;
namespace py = pybind11;

namespace {

template <int ndim, typename T>
void run_nufft(int type, void *desc_in, T *x, T *y, T *z, std::complex<T> *c, std::complex<T> *F) {
  const cpu::descriptor<T> *descriptor = unpack_descriptor<cpu::descriptor<T>>(
      reinterpret_cast<const char *>(desc_in), sizeof(cpu::descriptor<T>));
  int64_t n_k = 1;
  for (int d = 0; d < ndim; ++d) n_k *= descriptor->n_k[d];
  finufft_opts opts = descriptor->opts;

  typename cpu::plan_type<T>::type plan;
  cpu::makeplan<T>(type, ndim, const_cast<int64_t *>(descriptor->n_k), descriptor->iflag,
                   descriptor->n_transf, descriptor->eps, &plan, &opts);
  for (int64_t index = 0; index < descriptor->n_tot; ++index) {
    int64_t i = index * descriptor->n_j;
    int64_t j = i * descriptor->n_transf;
    int64_t k = index * n_k * descriptor->n_transf;

    cpu::setpts<T>(plan, descriptor->n_j, &(x[i]), cpu::y_index<ndim, T>(y, i),
                   cpu::z_index<ndim, T>(z, i), 0, NULL, NULL, NULL);
    cpu::execute<T>(plan, &c[j], &F[k]);
  }
  cpu::destroy<T>(plan);
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

template <typename T>
py::bytes build_descriptor(T eps, int iflag, int64_t n_tot, int n_transf, int64_t n_j,
                           int64_t n_k_1, int64_t n_k_2, int64_t n_k_3, finufft_opts opts) {
  return pack_descriptor(
      cpu::descriptor<T>{eps, iflag, n_tot, n_transf, n_j, {n_k_1, n_k_2, n_k_3}, opts});
}

template <typename T>
finufft_opts *build_opts(bool modeord, bool chkbnds, int debug, int spread_debug, bool showwarn,
                         int nthreads, int fftw, int spread_sort, bool spread_kerevalmeth,
                         bool spread_kerpad, double upsampfac, int spread_thread, int maxbatchsize,
                         int spread_nthr_atomic, int spread_max_sp_size) {
  finufft_opts *opts = new finufft_opts;
  cpu::default_opts<T>(opts);

  opts->modeord = int(modeord);
  opts->chkbnds = int(chkbnds);
  opts->debug = debug;
  opts->spread_debug = spread_debug;
  opts->showwarn = int(showwarn);
  opts->nthreads = nthreads;
  opts->fftw = fftw;
  opts->spread_sort = spread_sort;
  opts->spread_kerevalmeth = int(spread_kerevalmeth);
  opts->spread_kerpad = int(spread_kerpad);
  opts->upsampfac = upsampfac;
  opts->spread_thread = int(spread_thread);
  opts->maxbatchsize = maxbatchsize;
  opts->spread_nthr_atomic = spread_nthr_atomic;
  opts->spread_max_sp_size = spread_max_sp_size;

  return opts;
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

PYBIND11_MODULE(jax_finufft_cpu, m) {
  m.def("registrations", &Registrations);
  m.def("build_descriptorf", &build_descriptor<float>);
  m.def("build_descriptor", &build_descriptor<double>);

  m.attr("FFTW_ESTIMATE") = py::int_(FFTW_ESTIMATE);
  m.attr("FFTW_MEASURE") = py::int_(FFTW_MEASURE);
  m.attr("FFTW_PATIENT") = py::int_(FFTW_PATIENT);
  m.attr("FFTW_EXHAUSTIVE") = py::int_(FFTW_EXHAUSTIVE);
  m.attr("FFTW_WISDOM_ONLY") = py::int_(FFTW_WISDOM_ONLY);

  py::class_<finufft_opts> opts(m, "FinufftOpts");
  opts.def(py::init(&build_opts<double>), py::arg("modeord") = false, py::arg("chkbnds") = true,
           py::arg("debug") = 0, py::arg("spread_debug") = 0, py::arg("showwarn") = false,
           py::arg("nthreads") = 0, py::arg("fftw") = int(FFTW_ESTIMATE),
           py::arg("spread_sort") = 2, py::arg("spread_kerevalmeth") = true,
           py::arg("spread_kerpad") = true, py::arg("upsampfac") = 0.0,
           py::arg("spread_thread") = 0, py::arg("maxbatchsize") = 0,
           py::arg("spread_nthr_atomic") = -1, py::arg("spread_max_sp_size") = 0);
}

}  // namespace
