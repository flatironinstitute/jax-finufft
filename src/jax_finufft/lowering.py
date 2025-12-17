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
    source,
    *points,
    output_shape,
    iflag,
    eps,
    opts,
    nufft_type,
):
    """MLIR lowering rule for NUFFT primitives.

    Uses jax.ffi.ffi_lowering to generate the lowering without private imports.

    Args:
        ctx: The MLIR lowering rule context.
        source: The source array (complex coefficients or frequency values).
        *points: Non-uniform point coordinates.
        output_shape: Shape of the output array (for type 1 transforms).
        iflag: Sign of the imaginary unit in the exponential (+1 or -1).
        eps: Requested precision tolerance.
        opts: FINUFFT options.
        nufft_type: Type of NUFFT (1, 2, or 3).

    Returns:
        Sequence of MLIR values representing the output.

    Raises:
        ValueError: If platform is unsupported or GPU support is unavailable.
    """
    if len(ctx.module_context.platforms) > 1:
        raise ValueError("Multi-platform lowering is not supported")
    platform = ctx.module_context.platforms[0]
    if platform not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported platform '{platform}'")
    if platform == "cuda" and jax_finufft_gpu is None:
        raise ValueError("jax-finufft was not compiled with GPU support")

    if nufft_type == 3:
        ndim = len(points) // 2
    else:
        ndim = len(points)
    assert 1 <= ndim <= 3

    source_aval = ctx.avals_in[0]
    single = source_aval.dtype == np.complex64
    suffix = "f" if single else ""

    # Use aval shapes for computing dimensions - these should match MLIR value shapes
    source_shape = source_aval.shape
    points_shape = tuple(x.shape for x in ctx.avals_in[1:])
    n_tot = source_shape[0]
    n_transf = source_shape[1]
    n_j = points_shape[0][1]

    # Dispatch to the correct custom call target depending on the dimension,
    # dtype, and NUFFT type.
    if nufft_type == 1:
        op_name = f"nufft{ndim}d1{suffix}"
        n_k = np.array(output_shape, dtype=np.int64)
    elif nufft_type == 2:
        op_name = f"nufft{ndim}d2{suffix}"
        n_k = np.array(source_shape[2:], dtype=np.int64)
    elif nufft_type == 3:
        op_name = f"nufft{ndim}d3{suffix}"
    else:
        raise ValueError("nufft_type must be 1, 2, or 3")

    # The backend expects the output shape in Fortran order, so we'll just
    # fake it here, by sending in n_k and x in the reverse order.
    n_k_full = np.zeros(3, dtype=np.int64)
    if nufft_type != 3:
        n_k_full[:ndim] = n_k[::-1]
    else:
        n_k_full[0] = points_shape[ndim][1]

    # Reverse points because backend uses Fortran order
    points_fortran = points[:ndim][::-1] + points[ndim:][::-1]

    if opts is None:
        opts = options.Opts()
    opts = options.unpack_opts(opts, nufft_type, True)
    assert isinstance(opts, options.Opts)

    # Get epsilon in the correct precision
    eps_value = float(eps) if not single else np.float32(eps)

    # Build operands list based on NUFFT type
    # Both CPU and GPU use dimension-specific bindings (no placeholders)
    if nufft_type == 3:
        # Type 3 has both source points and target points
        operands = [source] + list(points_fortran[:ndim]) + list(points_fortran[ndim:])
    else:
        # Type 1 and 2: source + points
        operands = [source] + list(points_fortran)

    if platform == "cpu":
        # Build FFI attributes dictionary for typed FFI
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
            # FINUFFT options as individual attributes
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
        # GPU path
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
            # cuFINUFFT options as individual attributes
            "modeord": np.int64(opts_native.modeord),
            "upsampfac": float(opts_native.upsampfac),
            "gpu_method": np.int64(opts_native.gpu_method),
            "gpu_sort": np.int64(opts_native.gpu_sort),
            "gpu_kerevalmeth": np.int64(opts_native.gpu_kerevalmeth),
            "gpu_maxbatchsize": np.int64(opts_native.gpu_maxbatchsize),
            "debug": np.int64(opts_native.debug),
        }

    # Use jax.ffi.ffi_lowering to create and call the lowering rule
    lowering_rule = jax.ffi.ffi_lowering(op_name)
    return lowering_rule(ctx, *operands, **ffi_kwargs)
