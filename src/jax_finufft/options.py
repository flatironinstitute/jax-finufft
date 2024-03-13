from enum import IntEnum

from pydantic.dataclasses import dataclass

from jax_finufft import jax_finufft_cpu


class DebugLevel(IntEnum):
    Silent = 0
    Verbose = 1
    Noisy = 2


class FftwFlags(IntEnum):
    Estimate = jax_finufft_cpu.FFTW_ESTIMATE
    Measure = jax_finufft_cpu.FFTW_MEASURE
    Patient = jax_finufft_cpu.FFTW_PATIENT
    Exhaustive = jax_finufft_cpu.FFTW_EXHAUSTIVE
    WisdomOnly = jax_finufft_cpu.FFTW_WISDOM_ONLY


class SpreadSort(IntEnum):
    NoSort = 0
    Sort = 1
    Heuristic = 2


class SpreadThread(IntEnum):
    Auto = 0
    Sequential = 1
    Parallel = 2


@dataclass(frozen=True)
class CPUOpts:
    modeord: bool = False
    chkbnds: bool = True
    debug: DebugLevel = DebugLevel.Silent
    spread_debug: DebugLevel = DebugLevel.Silent
    nthreads: int = 0
    fftw: int = FftwFlags.Estimate
    spread_sort: SpreadSort = SpreadSort.Heuristic
    spread_kerevalmeth: bool = True
    spread_kerpad: bool = True
    upsampfac: float = 0.0
    spread_thread: SpreadThread = SpreadThread.Auto
    maxbatchsize: int = 0
    spread_nthr_atomic: int = -1
    spread_max_sp_size: int = 0

    def to_finufft_opts(self) -> jax_finufft_cpu.FinufftOpts:
        return jax_finufft_cpu.FinufftOpts(
            modeord=self.modeord,
            chkbnds=self.chkbnds,
            debug=self.debug,
            spread_debug=self.spread_debug,
            nthreads=self.nthreads,
            fftw=self.fftw,
            spread_sort=self.spread_sort,
            spread_kerevalmeth=self.spread_kerevalmeth,
            spread_kerpad=self.spread_kerpad,
            upsampfac=self.upsampfac,
            spread_thread=self.spread_thread,
            maxbatchsize=self.maxbatchsize,
            spread_nthr_atomic=self.spread_nthr_atomic,
            spread_max_sp_size=self.spread_max_sp_size,
        )
