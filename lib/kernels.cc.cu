#include "cufinufft_wrapper.h"
#include "kernel_helpers.h"
#include "kernels.h"

using namespace jax_finufft::gpu;

namespace jax_finufft {

void ThrowIfError(cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}

template <int ndim, typename T, int type>
void run_nufft(const descriptor<T>* descriptor, T* x, T* y, T* z, std::complex<T>* c, T* s, T* t,
               T* u, std::complex<T>* F, cudaStream_t stream) {
  int64_t n_k = 1;
  if constexpr (type != 3) {
    for (int d = 0; d < ndim; ++d) n_k *= descriptor->n_k[d];
  } else {
    n_k = descriptor->n_k[0];
  }

  cufinufft_opts opts = descriptor->opts;
  update_opts<T>(&opts, ndim, stream);
  device_type device;
  ThrowIfError(cudaGetDevice(&device.ordinal));
  update_opts<T>(&opts, ndim, device);

  typename plan_type<T>::type plan;
  makeplan<T>(type, ndim, descriptor->n_k, descriptor->iflag, descriptor->n_transf,
              descriptor->eps, &plan, &opts);
  for (int64_t index = 0; index < descriptor->n_tot; ++index) {
    int64_t i = index * descriptor->n_j;
    int64_t j = index * descriptor->n_j * descriptor->n_transf;
    int64_t k = index * n_k * descriptor->n_transf;

    if constexpr (type != 3) {
      setpts<T>(plan, descriptor->n_j, &(x[i]), y_index<ndim, T>(y, i), z_index<ndim, T>(z, i), 0,
                NULL, NULL, NULL);
    } else {
      int64_t i_target = index * n_k;
      setpts<T>(plan, descriptor->n_j, &(x[i]), y_index<ndim, T>(y, i), z_index<ndim, T>(z, i),
                n_k, &(s[i_target]), y_index<ndim, T>(t, i_target), z_index<ndim, T>(u, i_target));
    }

    execute<T>(plan, &c[j], &F[k]);
  }
  // Don't free resources like the cuFFT plan until the stream is done.
  cudaStreamSynchronize(stream);
  destroy<T>(plan);
}

template <int ndim, typename T>
void nufft1(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len) {
  const descriptor<T>* desc = unpack_descriptor<descriptor<T>>(opaque, opaque_len);

  std::complex<T>* c = reinterpret_cast<std::complex<T>*>(buffers[0]);
  T* x = reinterpret_cast<T*>(buffers[1]);
  T* y = NULL;
  T* z = NULL;
  int out_dim = 2;
  if (ndim > 1) {
    y = reinterpret_cast<T*>(buffers[2]);
    out_dim = 3;
  }
  if (ndim > 2) {
    z = reinterpret_cast<T*>(buffers[3]);
    out_dim = 4;
  }
  std::complex<T>* F = reinterpret_cast<std::complex<T>*>(buffers[out_dim]);

  run_nufft<ndim, T, 1>(desc, x, y, z, c, NULL, NULL, NULL, F, stream);

  ThrowIfError(cudaGetLastError());
}

template <int ndim, typename T>
void nufft2(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len) {
  const descriptor<T>* desc = unpack_descriptor<descriptor<T>>(opaque, opaque_len);

  std::complex<T>* F = reinterpret_cast<std::complex<T>*>(buffers[0]);
  T* x = reinterpret_cast<T*>(buffers[1]);
  T* y = NULL;
  T* z = NULL;
  int out_dim = 2;
  if (ndim > 1) {
    y = reinterpret_cast<T*>(buffers[2]);
    out_dim = 3;
  }
  if (ndim > 2) {
    z = reinterpret_cast<T*>(buffers[3]);
    out_dim = 4;
  }
  std::complex<T>* c = reinterpret_cast<std::complex<T>*>(buffers[out_dim]);

  run_nufft<ndim, T, 2>(desc, x, y, z, c, NULL, NULL, NULL, F, stream);

  ThrowIfError(cudaGetLastError());
}

template <int ndim, typename T>
void nufft3(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len) {
  const descriptor<T>* desc = unpack_descriptor<descriptor<T>>(opaque, opaque_len);

  std::complex<T>* c = reinterpret_cast<std::complex<T>*>(buffers[0]);
  T* x = reinterpret_cast<T*>(buffers[1]);
  T* y = NULL;
  T* z = NULL;
  T* s = reinterpret_cast<T*>(buffers[1 + ndim]);
  T* t = NULL;
  T* u = NULL;
  int out_dim = 3;
  if constexpr (ndim > 1) {
    y = reinterpret_cast<T*>(buffers[2]);
    t = reinterpret_cast<T*>(buffers[2 + ndim]);
    out_dim = 5;
  }
  if constexpr (ndim > 2) {
    z = reinterpret_cast<T*>(buffers[3]);
    u = reinterpret_cast<T*>(buffers[3 + ndim]);
    out_dim = 7;
  }
  std::complex<T>* F = reinterpret_cast<std::complex<T>*>(buffers[out_dim]);

  run_nufft<ndim, T, 3>(desc, x, y, z, c, s, t, u, F, stream);

  ThrowIfError(cudaGetLastError());
}

void nufft1d1(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len) {
  nufft1<1, double>(stream, buffers, opaque, opaque_len);
}

void nufft1d2(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len) {
  nufft2<1, double>(stream, buffers, opaque, opaque_len);
}

void nufft1d3(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len) {
  nufft3<1, double>(stream, buffers, opaque, opaque_len);
}

void nufft2d1(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len) {
  nufft1<2, double>(stream, buffers, opaque, opaque_len);
}

void nufft2d2(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len) {
  nufft2<2, double>(stream, buffers, opaque, opaque_len);
}

void nufft2d3(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len) {
  nufft3<2, double>(stream, buffers, opaque, opaque_len);
}

void nufft3d1(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len) {
  nufft1<3, double>(stream, buffers, opaque, opaque_len);
}

void nufft3d2(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len) {
  nufft2<3, double>(stream, buffers, opaque, opaque_len);
}

void nufft3d3(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len) {
  nufft3<3, double>(stream, buffers, opaque, opaque_len);
}

void nufft1d1f(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len) {
  nufft1<1, float>(stream, buffers, opaque, opaque_len);
}

void nufft1d2f(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len) {
  nufft2<1, float>(stream, buffers, opaque, opaque_len);
}

void nufft1d3f(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len) {
  nufft3<1, float>(stream, buffers, opaque, opaque_len);
}

void nufft2d1f(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len) {
  nufft1<2, float>(stream, buffers, opaque, opaque_len);
}

void nufft2d2f(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len) {
  nufft2<2, float>(stream, buffers, opaque, opaque_len);
}

void nufft2d3f(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len) {
  nufft3<2, float>(stream, buffers, opaque, opaque_len);
}

void nufft3d1f(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len) {
  nufft1<3, float>(stream, buffers, opaque, opaque_len);
}

void nufft3d2f(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len) {
  nufft2<3, float>(stream, buffers, opaque, opaque_len);
}

void nufft3d3f(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len) {
  nufft3<3, float>(stream, buffers, opaque, opaque_len);
}

}  // namespace jax_finufft
