#include "jax_finufft_gpu.h"
#include "kernels.h"
#include "kernel_helpers.h"

namespace jax_finufft {

void ThrowIfError(cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}
    
template <int ndim, typename T>
void run_nufft(int type, const NufftDescriptor<T>* descriptor, T *x, T *y, T *z, std::complex<T> *c, std::complex<T> *F) {
  int64_t n_k = 1;
  for (int d = 0; d < ndim; ++d) n_k *= descriptor->n_k[d];
    
  // TODO: okay to stack-allocate this?
  int nmodes32[ndim];
  for (int d = 0; d < ndim; ++d) nmodes32[d] = static_cast<int>(descriptor->n_k[d]);

  // TODO: does this need to be part of NufftDescriptor? It's GPU-specific.
  int maxbatchsize = 0;  // auto
  cufinufft_opts *opts = new cufinufft_opts;
  typename plan_type<T>::type plan;
  default_opts<T>(type, ndim, opts);
  makeplan<T>(type, ndim, nmodes32, descriptor->iflag,
              descriptor->n_transf, descriptor->eps, maxbatchsize, &plan, opts);
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
void nufft1(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len)  {
  const NufftDescriptor<T> *descriptor = unpack_descriptor<NufftDescriptor<T>>(opaque, opaque_len);
  
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
  
  // Call cuFINUFFT here...
  run_nufft<ndim, T>(1, descriptor, x, y, z, c, F);

  ThrowIfError(cudaGetLastError());
}

template <int ndim, typename T>
void nufft2(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len)  {
  const NufftDescriptor<T> *descriptor = unpack_descriptor<NufftDescriptor<T>>(opaque, opaque_len);
  
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
  
  // Call cuFINUFFT here...
  run_nufft<ndim, T>(1, descriptor, x, y, z, c, F);

  ThrowIfError(cudaGetLastError());
}

void nufft2d1(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len) {
  nufft1<2, double>(stream, buffers, opaque, opaque_len);
}

void nufft2d2(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len) {
  nufft2<2, double>(stream, buffers, opaque, opaque_len);
}

void nufft3d1(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len) {
  nufft1<3, double>(stream, buffers, opaque, opaque_len);
}

void nufft3d2(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len) {
  nufft2<3, double>(stream, buffers, opaque, opaque_len);
}

void nufft2d1f(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len) {
  nufft1<2, float>(stream, buffers, opaque, opaque_len);
}

void nufft2d2f(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len) {
  nufft2<2, float>(stream, buffers, opaque, opaque_len);
}

void nufft3d1f(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len) {
  nufft1<3, float>(stream, buffers, opaque, opaque_len);
}

void nufft3d2f(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len) {
  nufft2<3, float>(stream, buffers, opaque, opaque_len);
}

}