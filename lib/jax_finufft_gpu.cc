// This file defines the Python interface to the XLA custom call implemented on the CPU.
// It is exposed as a standard pybind11 module defining "capsule" objects containing our
// method. For simplicity, we export a separate capsule for each supported dtype.

#include "kernels.h"
#include "pybind11_kernel_helpers.h"

using namespace jax_finufft;
using namespace jax_finufft::gpu;

namespace {

template <typename T>
py::bytes build_descriptor(T eps, int iflag, int64_t n_tot, int n_transf, int64_t n_j,
                           int64_t n_k_1, int64_t n_k_2, int64_t n_k_3, cufinufft_opts opts) {
  return pack_descriptor(
      descriptor<T>{eps, iflag, n_tot, n_transf, n_j, {n_k_1, n_k_2, n_k_3}, opts});
}

template <typename T>
cufinufft_opts *build_opts(double upsampfac, int gpu_method, bool gpu_sort, int gpu_binsizex,
                           int gpu_binsizey, int gpu_binsizez, int gpu_obinsizex,
                           int gpu_obinsizey, int gpu_obinsizez, int gpu_maxsubprobsize,
                           bool gpu_kerevalmeth, int gpu_spreadinterponly, int gpu_maxbatchsize) {
  cufinufft_opts *opts = new cufinufft_opts;
  default_opts<T>(opts);

  opts->upsampfac = upsampfac;
  opts->gpu_method = gpu_method;
  opts->gpu_sort = int(gpu_sort);
  opts->gpu_binsizex = gpu_binsizex;
  opts->gpu_binsizey = gpu_binsizey;
  opts->gpu_binsizez = gpu_binsizez;
  opts->gpu_obinsizex = gpu_obinsizex;
  opts->gpu_obinsizey = gpu_obinsizey;
  opts->gpu_obinsizez = gpu_obinsizez;
  opts->gpu_maxsubprobsize = gpu_maxsubprobsize;
  opts->gpu_kerevalmeth = gpu_kerevalmeth;
  opts->gpu_spreadinterponly = gpu_spreadinterponly;
  opts->gpu_maxbatchsize = gpu_maxbatchsize;

  return opts;
}

pybind11::dict Registrations() {
  pybind11::dict dict;

  // TODO: do we prefer to keep these names the same as the CPU version or prefix them with "cu"?
  // dict["nufft1d1f"] = encapsulate_function(nufft1d1f);
  // dict["nufft1d2f"] = encapsulate_function(nufft1d2f);
  dict["nufft2d1f"] = encapsulate_function(nufft2d1f);
  dict["nufft2d2f"] = encapsulate_function(nufft2d2f);
  dict["nufft3d1f"] = encapsulate_function(nufft3d1f);
  dict["nufft3d2f"] = encapsulate_function(nufft3d2f);

  // dict["nufft1d1"] = encapsulate_function(nufft1d1);
  // dict["nufft1d2"] = encapsulate_function(nufft1d2);
  dict["nufft2d1"] = encapsulate_function(nufft2d1);
  dict["nufft2d2"] = encapsulate_function(nufft2d2);
  dict["nufft3d1"] = encapsulate_function(nufft3d1);
  dict["nufft3d2"] = encapsulate_function(nufft3d2);

  return dict;
}

PYBIND11_MODULE(jax_finufft_gpu, m) {
  m.def("registrations", &Registrations);
  m.def("build_descriptorf", &build_descriptor<float>);
  m.def("build_descriptor", &build_descriptor<double>);

  py::class_<cufinufft_opts> opts(m, "CufinufftOpts");
  opts.def(py::init(&build_opts<double>), py::arg("upsampfac") = 2.0, py::arg("gpu_method") = 0,
           py::arg("gpu_sort") = true, py::arg("gpu_binsizex") = -1, py::arg("gpu_binsizey") = -1,
           py::arg("gpu_binsizez") = -1, py::arg("gpu_obinsizex") = -1,
           py::arg("gpu_obinsizey") = -1, py::arg("gpu_obinsizez") = -1,
           py::arg("gpu_maxsubprobsize") = 1024, py::arg("gpu_kerevalmeth") = true,
           py::arg("gpu_spreadinterponly") = 0, py::arg("gpu_maxbatchsize") = 0);
}

}  // namespace
