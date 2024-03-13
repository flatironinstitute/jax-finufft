// This file defines the Python interface to the XLA custom call implemented on the CPU.
// It is exposed as a standard pybind11 module defining "capsule" objects containing our
// method. For simplicity, we export a separate capsule for each supported dtype.

#include "jax_finufft_cpu.h"

#include "pybind11_kernel_helpers.h"

using namespace jax_finufft;
namespace py = pybind11;

namespace {

template <typename T>
py::bytes build_descriptor(T eps, int iflag, int64_t n_tot, int n_transf, int64_t n_j,
                           int64_t n_k_1, int64_t n_k_2, int64_t n_k_3, cpu::opts opts) {
  return pack_descriptor(
      cpu::descriptor<T>{eps, iflag, n_tot, n_transf, n_j, {n_k_1, n_k_2, n_k_3}, opts});
}

template <int ndim, typename T>
void run_nufft(int type, void *desc_in, T *x, T *y, T *z, std::complex<T> *c, std::complex<T> *F) {
  const cpu::descriptor<T> *descriptor = unpack_descriptor<cpu::descriptor<T>>(
      reinterpret_cast<const char *>(desc_in), sizeof(cpu::descriptor<T>));
  int64_t n_k = 1;
  for (int d = 0; d < ndim; ++d) n_k *= descriptor->n_k[d];
  finufft_opts opts = descriptor->opts.finufft_opts;

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

  py::class_<cpu::opts> opts(m, "Opts");

  py::enum_<cpu::opts::DebugLevel>(opts, "DebugLevel")
      .value("Silent", cpu::opts::DebugLevel::Silent)
      .value("Vebose", cpu::opts::DebugLevel::Verbose)
      .value("Noisy", cpu::opts::DebugLevel::Noisy);

  py::enum_<cpu::opts::FftwFlags>(opts, "FftwFlags")
      .value("Estimate", cpu::opts::FftwFlags::Estimate)
      .value("Measure", cpu::opts::FftwFlags::Measure)
      .value("Patient", cpu::opts::FftwFlags::Patient)
      .value("Exhaustive", cpu::opts::FftwFlags::Exhaustive)
      .value("WisdomOnly", cpu::opts::FftwFlags::WisdomOnly);

  py::enum_<cpu::opts::SpreadSort>(opts, "SpreadSort")
      .value("No", cpu::opts::SpreadSort::No)
      .value("Yes", cpu::opts::SpreadSort::Yes)
      .value("Heuristic", cpu::opts::SpreadSort::Heuristic);

  py::enum_<cpu::opts::SpreadThread>(opts, "SpreadThread")
      .value("Auto", cpu::opts::SpreadThread::Auto)
      .value("Seq", cpu::opts::SpreadThread::Seq)
      .value("Parallel", cpu::opts::SpreadThread::Parallel);

  opts.def(py::init<bool, bool, cpu::opts::DebugLevel, cpu::opts::DebugLevel, bool, int, int,
                    cpu::opts::SpreadSort, bool, bool, double, cpu::opts::SpreadThread, int, int,
                    int>(),
           py::arg("modeord") = false, py::arg("chkbnds") = true,
           py::arg("debug") = cpu::opts::DebugLevel::Silent,
           py::arg("spread_debug") = cpu::opts::DebugLevel::Silent, py::arg("showwarn") = false,
           py::arg("nthreads") = 0, py::arg("fftw") = int(FFTW_ESTIMATE),
           py::arg("spread_sort") = cpu::opts::SpreadSort::Heuristic,
           py::arg("spread_kerevalmeth") = true, py::arg("spread_kerpad") = true,
           py::arg("upsampfac") = 0.0, py::arg("spread_thread") = cpu::opts::SpreadThread::Auto,
           py::arg("maxbatchsize") = 0, py::arg("spread_nthr_atomic") = -1,
           py::arg("spread_max_sp_size") = 0);
}

}  // namespace
