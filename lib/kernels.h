#ifndef _JAX_FINUFFT_KERNELS_H_
#define _JAX_FINUFFT_KERNELS_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace jax_finufft {

template <typename T>
struct descriptor {
  T eps;
  int iflag;
  int64_t n_tot;
  int n_transf;
  int64_t n_j;
  int64_t n_k[3];
  cufinufft_opts opts;
};

void nufft1d1(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);
void nufft1d2(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);
void nufft1d3(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);
void nufft2d1(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);
void nufft2d2(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);
void nufft2d3(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);
void nufft3d1(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);
void nufft3d2(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);
void nufft3d3(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);

void nufft1d1f(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);
void nufft1d2f(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);
void nufft1d3f(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);
void nufft2d1f(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);
void nufft2d2f(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);
void nufft2d3f(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);
void nufft3d1f(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);
void nufft3d2f(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);
void nufft3d3f(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);

}  // namespace jax_finufft

#endif
