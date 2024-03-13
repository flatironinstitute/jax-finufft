#ifndef _JAX_FINUFFT_OPTS_H_
#define _JAX_FINUFFT_OPTS_H_

#include <fftw3.h>

#include "finufft.h"
#include "jax_finufft_cpu.h"

namespace jax_finufft {

struct jax_finufft_opts {
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

  finufft_opts opts;

  jax_finufft_opts(
      bool modeord,  // (type 1,2 only): 0 CMCL-style increasing mode order
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
      int spread_nthr_atomic,      // if >=0, threads above which spreader OMP critical goes atomic
      int spread_max_sp_size       // if >0, overrides spreader (dir=1) max subproblem size
  ) {
    default_opts<double>(&opts);

    opts.modeord = modeord;
    opts.chkbnds = chkbnds;

    opts.debug = int(debug);
    opts.spread_debug = int(spread_debug);
    opts.showwarn = int(showwarn);

    opts.nthreads = nthreads;
    opts.fftw = fftw;
    opts.spread_sort = spread_sort;
    opts.spread_kerevalmeth = int(spread_kerevalmeth);
    opts.spread_kerpad = int(spread_kerpad);
    opts.upsampfac = upsampfac;
    opts.spread_thread = int(spread_thread);
    opts.maxbatchsize = maxbatchsize;
    opts.spread_nthr_atomic = spread_nthr_atomic;
    opts.spread_max_sp_size = spread_max_sp_size;
  }
};

}  // namespace jax_finufft

#endif
