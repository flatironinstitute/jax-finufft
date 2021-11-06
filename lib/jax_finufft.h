#ifndef _JAX_FINUFFT_H_
#define _JAX_FINUFFT_H_

#include <complex>

#include "finufft.h"

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

template <typename T>
struct plan_type;

template <>
struct plan_type<double> {
  typedef finufft_plan type;
};

template <>
struct plan_type<float> {
  typedef finufftf_plan type;
};

template <typename T>
void default_opts(nufft_opts* opts);

template <>
void default_opts<float>(nufft_opts* opts) {
  finufftf_default_opts(opts);
}

template <>
void default_opts<double>(nufft_opts* opts) {
  finufft_default_opts(opts);
}

template <typename T>
int makeplan(int type, int dim, int64_t* nmodes, int iflag, int ntr, T eps,
             typename plan_type<T>::type* plan, nufft_opts* opts);

template <>
int makeplan<float>(int type, int dim, int64_t* nmodes, int iflag, int ntr, float eps,
                    typename plan_type<float>::type* plan, nufft_opts* opts) {
  return finufftf_makeplan(type, dim, nmodes, iflag, ntr, eps, plan, opts);
}

template <>
int makeplan<double>(int type, int dim, int64_t* nmodes, int iflag, int ntr, double eps,
                     typename plan_type<double>::type* plan, nufft_opts* opts) {
  return finufft_makeplan(type, dim, nmodes, iflag, ntr, eps, plan, opts);
}

template <typename T>
int setpts(typename plan_type<T>::type plan, int64_t M, T* x, T* y, T* z, int64_t N, T* s, T* t,
           T* u);

template <>
int setpts<float>(typename plan_type<float>::type plan, int64_t M, float* x, float* y, float* z,
                  int64_t N, float* s, float* t, float* u) {
  return finufftf_setpts(plan, M, x, y, z, N, s, t, u);
}

template <>
int setpts<double>(typename plan_type<double>::type plan, int64_t M, double* x, double* y,
                   double* z, int64_t N, double* s, double* t, double* u) {
  return finufft_setpts(plan, M, x, y, z, N, s, t, u);
}

template <typename T>
int execute(typename plan_type<T>::type plan, std::complex<T>* c, std::complex<T>* f);

template <>
int execute<float>(typename plan_type<float>::type plan, std::complex<float>* c,
                   std::complex<float>* f) {
  return finufftf_execute(plan, c, f);
}

template <>
int execute<double>(typename plan_type<double>::type plan, std::complex<double>* c,
                    std::complex<double>* f) {
  return finufft_execute(plan, c, f);
}

template <typename T>
void destroy(typename plan_type<T>::type plan);

template <>
void destroy<float>(typename plan_type<float>::type plan) {
  finufftf_destroy(plan);
}

template <>
void destroy<double>(typename plan_type<double>::type plan) {
  finufft_destroy(plan);
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
