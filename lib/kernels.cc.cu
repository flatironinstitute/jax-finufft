#include "jax_finufft_gpu.h"
#include "kernel_helpers.h"
#include "kernels.h"

namespace jax_finufft {

void ThrowIfError(cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}

template <int ndim, typename T>
void run_nufft(int type, const descriptor<T> *descriptor, T *x, T *y, T *z, std::complex<T> *c,
               std::complex<T> *F, cudaStream_t stream) {
  int64_t n_k = 1;
  for (int d = 0; d < ndim; ++d) n_k *= descriptor->n_k[d];

  cufinufft_opts opts = descriptor->opts;
  update_opts<T>(&opts, ndim, stream);

  typename plan_type<T>::type plan;
  makeplan<T>(type, ndim, descriptor->n_k, descriptor->iflag, descriptor->n_transf,
              descriptor->eps, &plan, &opts);
  for (int64_t index = 0; index < descriptor->n_tot; ++index) {
    int64_t i = index * descriptor->n_j;
    int64_t j = index * descriptor->n_j * descriptor->n_transf;
    int64_t k = index * n_k * descriptor->n_transf;

    setpts<T>(plan, descriptor->n_j, &(x[i]), y_index<ndim, T>(y, i), z_index<ndim, T>(z, i), 0,
              NULL, NULL, NULL);

    execute<T>(plan, &c[j], &F[k]);
  }
  // Don't free resources like the cuFFT plan until the stream is done.
  cudaStreamSynchronize(stream);
  destroy<T>(plan);

  delete opts;
}

template <int ndim, typename T>
void nufft1(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
  const descriptor<T> *descriptor = unpack_descriptor<descriptor<T>>(opaque, opaque_len);

  std::complex<T> *c = reinterpret_cast<std::complex<T> *>(buffers[0]);
  T *x = reinterpret_cast<T *>(buffers[1]);
  T *y = NULL;
  T *z = NULL;
  int out_dim = 2;
  if (ndim > 1) {
    y = reinterpret_cast<T *>(buffers[2]);
    out_dim = 3;
  }
  if (ndim > 2) {
    z = reinterpret_cast<T *>(buffers[3]);
    out_dim = 4;
  }
  std::complex<T> *F = reinterpret_cast<std::complex<T> *>(buffers[out_dim]);

  run_nufft<ndim, T>(1, descriptor, x, y, z, c, F, stream);

  ThrowIfError(cudaGetLastError());
}

template <int ndim, typename T>
void nufft2(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
  const descriptor<T> *descriptor = unpack_descriptor<descriptor<T>>(opaque, opaque_len);

  std::complex<T> *F = reinterpret_cast<std::complex<T> *>(buffers[0]);
  T *x = reinterpret_cast<T *>(buffers[1]);
  T *y = NULL;
  T *z = NULL;
  int out_dim = 2;
  if (ndim > 1) {
    y = reinterpret_cast<T *>(buffers[2]);
    out_dim = 3;
  }
  if (ndim > 2) {
    z = reinterpret_cast<T *>(buffers[3]);
    out_dim = 4;
  }
  std::complex<T> *c = reinterpret_cast<std::complex<T> *>(buffers[out_dim]);

  run_nufft<ndim, T>(2, descriptor, x, y, z, c, F, stream);

  ThrowIfError(cudaGetLastError());
}

void nufft2d1(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
  nufft1<2, double>(stream, buffers, opaque, opaque_len);
}

void nufft2d2(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
  nufft2<2, double>(stream, buffers, opaque, opaque_len);
}

void nufft3d1(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
  nufft1<3, double>(stream, buffers, opaque, opaque_len);
}

void nufft3d2(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
  nufft2<3, double>(stream, buffers, opaque, opaque_len);
}

void nufft2d1f(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
  nufft1<2, float>(stream, buffers, opaque, opaque_len);
}

void nufft2d2f(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
  nufft2<2, float>(stream, buffers, opaque, opaque_len);
}

void nufft3d1f(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
  nufft1<3, float>(stream, buffers, opaque, opaque_len);
}

void nufft3d2f(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
  nufft2<3, float>(stream, buffers, opaque, opaque_len);
}

}  // namespace jax_finufft
