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
        """Return a simple object with native FINUFFT options for FFI."""
        compiled_with_omp = jax_finufft_cpu._omp_compile_check()

        class NativeOpts:
            pass

        opts = NativeOpts()
        opts.modeord = int(self.modeord)
        opts.debug = int(self.debug)
        opts.spread_debug = int(self.spread_debug)
        opts.showwarn = int(self.showwarn)
        opts.nthreads = self.nthreads if compiled_with_omp else 1
        opts.fftw = int(self.fftw)
        opts.spread_sort = int(self.spread_sort)
        opts.spread_kerevalmeth = int(self.spread_kerevalmeth)
        opts.spread_kerpad = int(self.spread_kerpad)
        opts.upsampfac = float(self.upsampfac)
        opts.spread_thread = int(self.spread_thread)
        opts.maxbatchsize = int(self.maxbatchsize)
        opts.spread_nthr_atomic = int(self.spread_nthr_atomic)
        opts.spread_max_sp_size = int(self.spread_max_sp_size)
        return opts

    def to_cufinufft_opts(self):
        """Return a simple object with native cuFINUFFT options for FFI."""

        class NativeOpts:
            pass

        opts = NativeOpts()
        opts.modeord = int(self.modeord)
        opts.upsampfac = float(self.gpu_upsampfac)
        opts.gpu_method = int(self.gpu_method)
        opts.gpu_sort = int(self.gpu_sort)
        opts.gpu_kerevalmeth = int(self.gpu_kerevalmeth)
        opts.gpu_maxbatchsize = int(self.gpu_maxbatchsize)
        opts.debug = int(self.gpu_debug)
        return opts


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
