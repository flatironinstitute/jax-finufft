// This file defines the Python interface to the XLA custom call implemented on the CPU.
// It is exposed as a standard nanobind module defining "capsule" objects containing our
// method. For simplicity, we export a separate capsule for each supported dtype.

#include "cufinufft_wrapper.h"
#include "kernels.h"
#include "nanobind_kernel_helpers.h"

using namespace jax_finufft;
using namespace jax_finufft::gpu;
namespace nb = nanobind;

namespace {

template <typename T>
nb::bytes build_descriptor(T eps, int iflag, int64_t n_tot, int n_transf, int64_t n_j,
                           int64_t n_k_1, int64_t n_k_2, int64_t n_k_3, cufinufft_opts opts) {
  return pack_descriptor(
      descriptor<T>{eps, iflag, n_tot, n_transf, n_j, {n_k_1, n_k_2, n_k_3}, opts});
}

nb::dict Registrations() {
  nb::dict dict;

  // TODO: do we prefer to keep these names the same as the CPU version or prefix them with "cu"?
  dict["nufft1d1f"] = encapsulate_function(nufft1d1f);
  dict["nufft1d2f"] = encapsulate_function(nufft1d2f);
  dict["nufft1d3f"] = encapsulate_function(nufft1d3f);
  dict["nufft2d1f"] = encapsulate_function(nufft2d1f);
  dict["nufft2d2f"] = encapsulate_function(nufft2d2f);
  dict["nufft2d3f"] = encapsulate_function(nufft2d3f);
  dict["nufft3d1f"] = encapsulate_function(nufft3d1f);
  dict["nufft3d2f"] = encapsulate_function(nufft3d2f);
  dict["nufft3d3f"] = encapsulate_function(nufft3d3f);

  dict["nufft1d1"] = encapsulate_function(nufft1d1);
  dict["nufft1d2"] = encapsulate_function(nufft1d2);
  dict["nufft1d3"] = encapsulate_function(nufft1d3);
  dict["nufft2d1"] = encapsulate_function(nufft2d1);
  dict["nufft2d2"] = encapsulate_function(nufft2d2);
  dict["nufft2d3"] = encapsulate_function(nufft2d3);
  dict["nufft3d1"] = encapsulate_function(nufft3d1);
  dict["nufft3d2"] = encapsulate_function(nufft3d2);
  dict["nufft3d3"] = encapsulate_function(nufft3d3);

  return dict;
}

NB_MODULE(jax_finufft_gpu, m) {
  m.def("registrations", &Registrations);
  m.def("build_descriptorf", &build_descriptor<float>);
  m.def("build_descriptor", &build_descriptor<double>);

  nb::class_<cufinufft_opts> opts(m, "CufinufftOpts");
  opts.def("__init__",
           [](cufinufft_opts* self, bool modeord, double upsampfac, int gpu_method, bool gpu_sort,
              int gpu_binsizex, int gpu_binsizey, int gpu_binsizez, int gpu_obinsizex,
              int gpu_obinsizey, int gpu_obinsizez, int gpu_maxsubprobsize, bool gpu_kerevalmeth,
              int gpu_spreadinterponly, int gpu_maxbatchsize, int debug) {
             new (self) cufinufft_opts;
             default_opts<double>(self);

             self->modeord = int(modeord);
             self->upsampfac = upsampfac;
             self->gpu_method = gpu_method;
             self->gpu_sort = int(gpu_sort);
             self->gpu_binsizex = gpu_binsizex;
             self->gpu_binsizey = gpu_binsizey;
             self->gpu_binsizez = gpu_binsizez;
             self->gpu_obinsizex = gpu_obinsizex;
             self->gpu_obinsizey = gpu_obinsizey;
             self->gpu_obinsizez = gpu_obinsizez;
             self->gpu_maxsubprobsize = gpu_maxsubprobsize;
             self->gpu_kerevalmeth = gpu_kerevalmeth;
             self->gpu_spreadinterponly = gpu_spreadinterponly;
             self->gpu_maxbatchsize = gpu_maxbatchsize;
             self->debug = debug;
           });
}

}  // namespace
