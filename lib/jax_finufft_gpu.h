#ifndef _JAX_FINUFFT_GPU_H_
#define _JAX_FINUFFT_GPU_H_

#include <complex>

#include "cufinufft.h"

namespace jax_finufft {

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

template <typename T>
void default_opts(int type, int dim, cufinufft_opts* opts, cudaStream_t stream);

template <>
void default_opts<float>(int type, int dim, cufinufft_opts* opts, cudaStream_t stream) {
  cufinufft_default_opts(opts);
  opts->gpu_stream = stream;
}

template <>
void default_opts<double>(int type, int dim, cufinufft_opts* opts, cudaStream_t stream) {
  cufinufft_default_opts(opts);
  opts->gpu_stream = stream;
    
  // double precision in 3D blows out shared memory.
  // Fall back to a slower, non-shared memory algorithm
  // https://github.com/flatironinstitute/cufinufft/issues/58
  if(dim > 2){
      opts->gpu_method = 1;
  }
}

template <typename T>
int makeplan(int type, int dim, const int64_t nmodes[3], int iflag, int ntr, T eps,
             typename plan_type<T>::type* plan, cufinufft_opts* opts);

template <>
int makeplan<float>(int type, int dim, const int64_t nmodes[3], int iflag, int ntr, float eps,
                    typename plan_type<float>::type* plan, cufinufft_opts* opts) {
  int64_t tmp_nmodes[3] = {nmodes[0], nmodes[1], nmodes[2]};  // TODO: use const in cufinufftf_makeplan API
  return cufinufftf_makeplan(type, dim, tmp_nmodes, iflag, ntr, eps, plan, opts);
}

template <>
int makeplan<double>(int type, int dim, const int64_t nmodes[3], int iflag, int ntr, double eps,
                     typename plan_type<double>::type* plan, cufinufft_opts* opts) {
  int64_t tmp_nmodes[3] = {nmodes[0], nmodes[1], nmodes[2]};
  return cufinufft_makeplan(type, dim, tmp_nmodes, iflag, ntr, eps, plan, opts);
}

template <typename T>
int setpts(typename plan_type<T>::type plan, int64_t M, T* x, T* y, T* z, int64_t N, T* s, T* t,
           T* u);

template <>
int setpts<float>(typename plan_type<float>::type plan, int64_t M, float* x, float* y, float* z,
                  int64_t N, float* s, float* t, float* u) {
  return cufinufftf_setpts(plan, M, x, y, z, N, s, t, u);
}

template <>
int setpts<double>(typename plan_type<double>::type plan, int64_t M, double* x, double* y,
                   double* z, int64_t N, double* s, double* t, double* u) {
  return cufinufft_setpts(plan, M, x, y, z, N, s, t, u);
}

template <typename T>
int execute(typename plan_type<T>::type plan, std::complex<T>* c, std::complex<T>* f);

template <>
int execute<float>(typename plan_type<float>::type plan, std::complex<float>* c,
                   std::complex<float>* f) {
  cuFloatComplex* _c = reinterpret_cast<cuFloatComplex*>(c);
  cuFloatComplex* _f = reinterpret_cast<cuFloatComplex*>(f);
  return cufinufftf_execute(plan, _c, _f);
}

template <>
int execute<double>(typename plan_type<double>::type plan, std::complex<double>* c,
                    std::complex<double>* f) {
  cuDoubleComplex* _c = reinterpret_cast<cuDoubleComplex*>(c);
  cuDoubleComplex* _f = reinterpret_cast<cuDoubleComplex*>(f);
  return cufinufft_execute(plan, _c, _f);
}

template <typename T>
void destroy(typename plan_type<T>::type plan);

template <>
void destroy<float>(typename plan_type<float>::type plan) {
  cufinufftf_destroy(plan);
}

template <>
void destroy<double>(typename plan_type<double>::type plan) {
  cufinufft_destroy(plan);
}

template <int ndim, typename T>
T* y_index(T* y, int64_t index) {
  return &(y[index]);
}

template <>
double* y_index<1, double>(double* y, int64_t index) {
  return NULL;
}

template <>
float* y_index<1, float>(float* y, int64_t index) {
  return NULL;
}

template <int ndim, typename T>
T* z_index(T* z, int64_t index) {
  return NULL;
}

template <>
double* z_index<3, double>(double* z, int64_t index) {
  return &(z[index]);
}

template <>
float* z_index<3, float>(float* z, int64_t index) {
  return &(z[index]);
}

}  // namespace jax_finufft

#endif
