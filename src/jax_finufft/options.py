import warnings
from enum import IntEnum
from typing import Union

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
    OutputDriven = 3


# FINUFFT retains these for ABI compatibility but ignores them; it warns at
# runtime when one is set away from its default, and we mirror that here.
DEPRECATED_OPTS = {
    "spread_thread": 0,
    "spread_kerevalmeth": True,
    "spread_kerpad": True,
}


@dataclass(frozen=True)
class Opts:
    # Defaults track finufft_default_opts (vendor/finufft/include/finufft/plan.hpp)
    # and cufinufft_default_opts (vendor/finufft/src/cuda/c_interface.cpp).
    modeord: bool = False
    debug: DebugLevel = DebugLevel.Silent
    spread_debug: DebugLevel = DebugLevel.Silent
    showwarn: bool = True
    nthreads: int = 0
    fftw: int = FftwFlags.Estimate
    spread_sort: SpreadSort = SpreadSort.Heuristic
    spread_kerevalmeth: bool = True
    spread_kerpad: bool = True
    spreadinterponly: bool = False
    upsampfac: float = 0.0
    spread_thread: SpreadThread = SpreadThread.Auto
    maxbatchsize: int = 0
    spread_nthr_atomic: int = -1
    spread_max_sp_size: int = 0
    allow_eps_too_small: bool = False

    gpu_upsampfac: float = 0.0
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
    gpu_np: int = 0
    gpu_debug: GpuDebugLevel = GpuDebugLevel.Silent

    def __post_init__(self):
        for name, default in DEPRECATED_OPTS.items():
            if getattr(self, name) != default:
                warnings.warn(
                    f"opts.{name} is deprecated and ignored by FINUFFT",
                    DeprecationWarning,
                    stacklevel=3,
                )

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
        opts.spreadinterponly = int(self.spreadinterponly)
        opts.upsampfac = float(self.upsampfac)
        opts.spread_thread = int(self.spread_thread)
        opts.maxbatchsize = int(self.maxbatchsize)
        opts.spread_nthr_atomic = int(self.spread_nthr_atomic)
        opts.spread_max_sp_size = int(self.spread_max_sp_size)
        opts.allow_eps_too_small = int(self.allow_eps_too_small)
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
        opts.gpu_binsizex = int(self.gpu_binsizex)
        opts.gpu_binsizey = int(self.gpu_binsizey)
        opts.gpu_binsizez = int(self.gpu_binsizez)
        opts.gpu_obinsizex = int(self.gpu_obinsizex)
        opts.gpu_obinsizey = int(self.gpu_obinsizey)
        opts.gpu_obinsizez = int(self.gpu_obinsizez)
        opts.gpu_maxsubprobsize = int(self.gpu_maxsubprobsize)
        opts.gpu_np = int(self.gpu_np)
        opts.gpu_kerevalmeth = int(self.gpu_kerevalmeth)
        opts.gpu_spreadinterponly = int(self.gpu_spreadinterponly)
        opts.gpu_maxbatchsize = int(self.gpu_maxbatchsize)
        opts.debug = int(self.gpu_debug)
        return opts


@dataclass(frozen=True)
class NestedOpts:
    type1: Opts | None = None
    type2: Opts | None = None
    type3: Opts | None = None

    forward: Opts | None = None
    backward: Union[Opts, "NestedOpts"] | None = None


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
