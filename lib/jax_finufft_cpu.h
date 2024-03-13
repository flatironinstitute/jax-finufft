#ifndef _JAX_FINUFFT_H_
#define _JAX_FINUFFT_H_

#include <fftw3.h>

#include <complex>

#include "finufft.h"

namespace jax_finufft {

namespace cpu {

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
void default_opts(finufft_opts* opts);

template <>
void default_opts<float>(finufft_opts* opts) {
  finufftf_default_opts(opts);
}

template <>
void default_opts<double>(finufft_opts* opts) {
  finufft_default_opts(opts);
}

template <typename T>
int makeplan(int type, int dim, int64_t* nmodes, int iflag, int ntr, T eps,
             typename plan_type<T>::type* plan, finufft_opts* opts);

template <>
int makeplan<float>(int type, int dim, int64_t* nmodes, int iflag, int ntr, float eps,
                    typename plan_type<float>::type* plan, finufft_opts* opts) {
  return finufftf_makeplan(type, dim, nmodes, iflag, ntr, eps, plan, opts);
}

template <>
int makeplan<double>(int type, int dim, int64_t* nmodes, int iflag, int ntr, double eps,
                     typename plan_type<double>::type* plan, finufft_opts* opts) {
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

struct opts {
  enum DebugLevel { Silent = 0, Verbose, Noisy };
  enum FftwFlags {
    Estimate = FFTW_ESTIMATE,
    Measure = FFTW_MEASURE,
    Patient = FFTW_PATIENT,
    Exhaustive = FFTW_EXHAUSTIVE,
    WisdomOnly = FFTW_WISDOM_ONLY
  };
  enum SpreadSort { No = 0, Yes, Heuristic };
  enum SpreadThread { Auto = 0, Seq, Parallel };

  finufft_opts finufft_opts;

  opts(bool modeord,  // (type 1,2 only): 0 CMCL-style increasing mode order
                      //                  1 FFT-style mode order
       bool chkbnds,  // 0 don't check NU pts in [-3pi,3pi), 1 do (<few % slower)

       // diagnostic opts...
       DebugLevel debug,
       DebugLevel spread_debug,  // spreader: 0 silent, 1 some timing/debug, or 2 tonnes
       bool showwarn,            // 0 don't print warnings to stderr, 1 do

       // algorithm performance opts...
       int nthreads,                // number of threads to use, or 0 uses all available
       int fftw,                    // plan flags to FFTW (FFTW_ESTIMATE=64, FFTW_MEASURE=0,...)
       SpreadSort spread_sort,      // spreader: 0 don't sort, 1 do, or 2 heuristic choice
       bool spread_kerevalmeth,     // spreader: 0 exp(sqrt()), 1 Horner piecewise poly (faster)
       bool spread_kerpad,          // (exp(sqrt()) only): 0 don't pad kernel to 4n, 1 do
       double upsampfac,            // upsampling ratio sigma: 2.0 std, 1.25 small FFT, 0.0 auto
       SpreadThread spread_thread,  // (vectorized ntr>1 only): 0 auto, 1 seq multithreaded,
                                    //                          2 parallel single-thread spread
       int maxbatchsize,            // (vectorized ntr>1 only): max transform batch, 0 auto
       int spread_nthr_atomic,  // if >=0, threads above which spreader OMP critical goes atomic
       int spread_max_sp_size   // if >0, overrides spreader (dir=1) max subproblem size
  ) {
    default_opts<double>(&finufft_opts);

    finufft_opts.modeord = modeord;
    finufft_opts.chkbnds = chkbnds;

    finufft_opts.debug = int(debug);
    finufft_opts.spread_debug = int(spread_debug);
    finufft_opts.showwarn = int(showwarn);

    finufft_opts.nthreads = nthreads;
    finufft_opts.fftw = fftw;
    finufft_opts.spread_sort = spread_sort;
    finufft_opts.spread_kerevalmeth = int(spread_kerevalmeth);
    finufft_opts.spread_kerpad = int(spread_kerpad);
    finufft_opts.upsampfac = upsampfac;
    finufft_opts.spread_thread = int(spread_thread);
    finufft_opts.maxbatchsize = maxbatchsize;
    finufft_opts.spread_nthr_atomic = spread_nthr_atomic;
    finufft_opts.spread_max_sp_size = spread_max_sp_size;
  }
};

template <typename T>
struct descriptor {
  T eps;
  int iflag;
  int64_t n_tot;
  int n_transf;
  int64_t n_j;
  int64_t n_k[3];
  opts opts;
};

}  // namespace cpu
}  // namespace jax_finufft

#endif
