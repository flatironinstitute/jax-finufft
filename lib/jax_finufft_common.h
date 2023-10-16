#ifndef _JAX_FINUFFT_COMMON_H_
#define _JAX_FINUFFT_COMMON_H_

// This descriptor is common to both the jax_finufft and jax_finufft_gpu modules
// We will use the jax_finufft namespace for both

namespace jax_finufft {

template <typename T>
struct NufftDescriptor {
  T eps;
  int iflag;
  int64_t n_tot;
  int n_transf;
  int64_t n_j;
  int64_t n_k[3];
};
    
}

#endif
