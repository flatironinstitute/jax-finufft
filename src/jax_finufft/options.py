from enum import IntEnum
from typing import Optional, Union

from pydantic.dataclasses import dataclass

from jax_finufft import jax_finufft_cpu


class DebugLevel(IntEnum):
    Silent = 0
    Verbose = 1
    Noisy = 2


class GpuDebugLevel(IntEnum):
    Silent = 0
    Verbose = 1


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


class GpuMethod(IntEnum):
    Auto = 0
    Driven = 1
    Shared = 2


@dataclass(frozen=True)
class Opts:

    # These correspond to the default cufinufft options
    # set in vendor/finufft/src/cuda/cufinufft.cu
    modeord: bool = False
    debug: DebugLevel = DebugLevel.Silent
    spread_debug: DebugLevel = DebugLevel.Silent
    showwarn: bool = False
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

    gpu_upsampfac: float = 2.0
    gpu_method: GpuMethod = 0
    gpu_sort: bool = True
    gpu_binsizex: int = 0
    gpu_binsizey: int = 0
    gpu_binsizez: int = 0
    gpu_obinsizex: int = 0
    gpu_obinsizey: int = 0
    gpu_obinsizez: int = 0
    gpu_maxsubprobsize: int = 1024
    gpu_kerevalmeth: bool = True
    gpu_spreadinterponly: bool = False
    gpu_maxbatchsize: int = 0
    gpu_debug: GpuDebugLevel = GpuDebugLevel.Silent

    def to_finufft_opts(self):
        compiled_with_omp = jax_finufft_cpu._omp_compile_check()
        return jax_finufft_cpu.FinufftOpts(
            self.modeord,
            int(self.debug),
            int(self.spread_debug),
            self.showwarn,
            self.nthreads if compiled_with_omp else 1,
            int(self.fftw),
            int(self.spread_sort),
            self.spread_kerevalmeth,
            self.spread_kerpad,
            self.upsampfac,
            int(self.spread_thread),
            self.maxbatchsize,
            self.spread_nthr_atomic,
            self.spread_max_sp_size,
        )

    def to_cufinufft_opts(self):
        from jax_finufft import jax_finufft_gpu

        return jax_finufft_gpu.CufinufftOpts(
            self.modeord,
            self.gpu_upsampfac,
            int(self.gpu_method),
            self.gpu_sort,
            self.gpu_binsizex,
            self.gpu_binsizey,
            self.gpu_binsizez,
            self.gpu_obinsizex,
            self.gpu_obinsizey,
            self.gpu_obinsizez,
            self.gpu_maxsubprobsize,
            self.gpu_kerevalmeth,
            self.gpu_spreadinterponly,
            self.gpu_maxbatchsize,
            int(self.gpu_debug),
        )


@dataclass(frozen=True)
class NestedOpts:
    type1: Optional[Opts] = None
    type2: Optional[Opts] = None
    type3: Optional[Opts] = None

    forward: Optional[Opts] = None
    backward: Optional[Union[Opts, "NestedOpts"]] = None


def unpack_opts(opts, finufft_type, forward):
    if opts is None or isinstance(opts, Opts):
        return opts

    if forward:
        if opts.forward is not None:
            return opts.forward
        elif finufft_type == 1:
            return opts.type1
        elif finufft_type == 2:
            return opts.type2
        elif finufft_type == 3:
            return opts.type3
    elif opts.backward is not None:
        return opts.backward

    return opts
