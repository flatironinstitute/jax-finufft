"""Lowering rules for jax-finufft primitives.

This module provides the MLIR lowering rules that compile JAX primitives
to XLA custom calls targeting the FINUFFT library using the typed FFI API.
Uses jax.ffi.ffi_lowering to avoid private MLIR imports.
"""

import numpy as np
import jax

from jax_finufft import options

from . import jax_finufft_cpu

try:
    from . import jax_finufft_gpu

    for _name, _value in jax_finufft_gpu.registrations().items():
        jax.ffi.register_ffi_target(_name, _value, platform="CUDA")
except ImportError:
    jax_finufft_gpu = None

# Register CPU FFI targets with api_version=4 (typed FFI)
for _name, _value in jax_finufft_cpu.registrations().items():
    jax.ffi.register_ffi_target(_name, _value, platform="cpu")


def lowering(
    ctx,
    *args,
    output_shape,
    iflag,
    eps,
    opts,
    nufft_type,
):
    """
    MLIR lowering rule for NUFFT primitives.

    Parameters
    ----------
    ctx : jax.core.LoweringRuleContext
        The MLIR lowering rule context.
    *args : tuple
        Arguments passed to the primitive.
    output_shape : tuple
        Shape of the output array.
    iflag : int
        Sign of the imaginary unit in the exponential.
    eps : float
        Requested precision tolerance.
    opts : Opts
        FINUFFT options.
    nufft_type : int
        Type of NUFFT.

    Returns
    -------
    Sequence
        Sequence of MLIR values representing the output.

    Raises
    ------
    ValueError
        If platform is unsupported or GPU support is unavailable.
    """
    if len(ctx.module_context.platforms) > 1:
        raise ValueError("Multi-platform lowering is not supported")
    platform = ctx.module_context.platforms[0]
    if platform not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported platform '{platform}'")
    if platform == "cuda" and jax_finufft_gpu is None:
        raise ValueError("jax-finufft was not compiled with GPU support")

    source_aval = ctx.avals_in[0]
    single = source_aval.dtype == np.complex64
    suffix = "f" if single else ""

    # Use aval shapes for computing dimensions - these should match MLIR value shapes
    source_shape = source_aval.shape
    n_tot = source_shape[0]
    n_transf = source_shape[1]

    # Dispatch to the correct custom call target depending on the dimension,
    # dtype, and NUFFT type.
    if nufft_type == 3:
        source = args[0]
        points = args[1:]
        ndim = len(points) // 2
        points_shape = tuple(x.shape for x in ctx.avals_in[1:])
        op_name = f"nufft{ndim}d3{suffix}"

        # Reverse points because backend uses Fortran order
        n_k = np.zeros(3, dtype=np.int64)
        n_k_full = np.zeros(3, dtype=np.int64)
        n_k_full[0] = points_shape[ndim][1]
        n_j = points_shape[0][1]
        points_fortran = points[:ndim][::-1] + points[ndim:][::-1]
        operands = [source] + list(points_fortran[:ndim]) + list(points_fortran[ndim:])
    else:
        source = args[0]
        points_mask = args[1]
        points = args[2:]
        ndim = len(points)
        points_shape = tuple(x.shape for x in ctx.avals_in[2:])
        n_j = points_shape[0][1]
        if nufft_type == 1:
            op_name = f"nufft{ndim}d1{suffix}"
            n_k = np.array(output_shape, dtype=np.int64)
        else:
            op_name = f"nufft{ndim}d2{suffix}"
            n_k = np.array(source_shape[2:], dtype=np.int64)

        # Reverse points because backend uses Fortran order
        n_k_full = np.zeros(3, dtype=np.int64)
        n_k_full[:ndim] = n_k[::-1]
        points_fortran = points[:ndim][::-1] + points[ndim:][::-1]
        operands = [source, points_mask] + list(points_fortran)

    assert 1 <= ndim <= 3

    if opts is None:
        opts = options.Opts()
    opts = options.unpack_opts(opts, nufft_type, True)
    assert isinstance(opts, options.Opts)

    eps_value = float(eps) if not single else np.float32(eps)

    if platform == "cpu":
        opts_native = opts.to_finufft_opts()

        ffi_kwargs = {
            "eps": eps_value,
            "iflag": np.int64(iflag),
            "n_tot": np.int64(n_tot),
            "n_transf": np.int64(n_transf),
            "n_j": np.int64(n_j),
            "n_k_1": np.int64(n_k_full[0]),
            "n_k_2": np.int64(n_k_full[1]),
            "n_k_3": np.int64(n_k_full[2]),
            "modeord": np.int64(opts_native.modeord),
            "debug": np.int64(opts_native.debug),
            "spread_debug": np.int64(opts_native.spread_debug),
            "showwarn": np.int64(opts_native.showwarn),
            "nthreads": np.int64(opts_native.nthreads),
            "fftw": np.int64(opts_native.fftw),
            "spread_sort": np.int64(opts_native.spread_sort),
            "spread_kerevalmeth": np.int64(opts_native.spread_kerevalmeth),
            "spread_kerpad": np.int64(opts_native.spread_kerpad),
            "upsampfac": float(opts_native.upsampfac),
            "spread_thread": np.int64(opts_native.spread_thread),
            "maxbatchsize": np.int64(opts_native.maxbatchsize),
            "spread_nthr_atomic": np.int64(opts_native.spread_nthr_atomic),
            "spread_max_sp_size": np.int64(opts_native.spread_max_sp_size),
        }
    else:
        opts_native = opts.to_cufinufft_opts()

        ffi_kwargs = {
            "eps": eps_value,
            "iflag": np.int64(iflag),
            "n_tot": np.int64(n_tot),
            "n_transf": np.int64(n_transf),
            "n_j": np.int64(n_j),
            "n_k_1": np.int64(n_k_full[0]),
            "n_k_2": np.int64(n_k_full[1]),
            "n_k_3": np.int64(n_k_full[2]),
            "modeord": np.int64(opts_native.modeord),
            "upsampfac": float(opts_native.upsampfac),
            "gpu_method": np.int64(opts_native.gpu_method),
            "gpu_sort": np.int64(opts_native.gpu_sort),
            "gpu_kerevalmeth": np.int64(opts_native.gpu_kerevalmeth),
            "gpu_maxbatchsize": np.int64(opts_native.gpu_maxbatchsize),
            "debug": np.int64(opts_native.debug),
        }

    lowering_rule = jax.ffi.ffi_lowering(op_name)
    return lowering_rule(ctx, *operands, **ffi_kwargs)
