#include "jax_finufft.h"
#include "kernels.h"
#include "kernel_helpers.h"

namespace jax_finufft {

void ThrowIfError(cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
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
  // run_nufft<ndim, T>(1, in[0], x, y, z, c, F);

  ThrowIfError(cudaGetLastError());
}

template <int ndim, typename T>
void nufft2(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len)  {
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
  // run_nufft<ndim, T>(1, in[0], x, y, z, c, F);

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