#ifndef _CUFINUFFT_WRAPPER_H_
#define _CUFINUFFT_WRAPPER_H_

#include <complex>

#include "cufinufft.h"

namespace jax_finufft {

namespace gpu {

template <typename T>
struct plan_type;

template <>
struct plan_type<double> {
  typedef cufinufft_plan type;
};

template <>
struct plan_type<float> {
  typedef cufinufftf_plan type;
};

struct device_type {
  int ordinal;
};

template <typename T>
void default_opts(cufinufft_opts* opts);

template <typename T>
void update_opts(cufinufft_opts* opts, int dim, cudaStream_t stream);

template <typename T>
void update_opts(cufinufft_opts* opts, int dim, device_type device);

template <typename T>
int makeplan(int type, int dim, const int64_t nmodes[3], int iflag, int ntr, T eps,
             typename plan_type<T>::type* plan, cufinufft_opts* opts);

template <typename T>
int setpts(typename plan_type<T>::type plan, int64_t M, T* x, T* y, T* z, int64_t N, T* s, T* t,
           T* u);

template <typename T>
int execute(typename plan_type<T>::type plan, std::complex<T>* c, std::complex<T>* f);

template <typename T>
void destroy(typename plan_type<T>::type plan);

template <int ndim, typename T>
T* y_index(T* y, int64_t index) {
  return &(y[index]);
}

template <int ndim, typename T>
T* z_index(T* z, int64_t index) {
  return NULL;
}

template <>
double* y_index<1, double>(double* y, int64_t index);

template <>
float* y_index<1, float>(float* y, int64_t index);

template <>
double* z_index<3, double>(double* z, int64_t index);

template <>
float* z_index<3, float>(float* z, int64_t index);

}  // namespace gpu

}  // namespace jax_finufft

#endif
