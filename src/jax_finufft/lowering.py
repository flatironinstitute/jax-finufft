"""Lowering rules for jax-finufft primitives.

This module provides the MLIR lowering rules that compile JAX primitives
to XLA custom calls targeting the FINUFFT library using the typed FFI API.
"""

from collections.abc import Sequence
from typing import Any

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


def default_layouts(*shapes: Sequence[int]) -> list[tuple[int, ...]]:
    """Compute default row-major layouts for the given shapes.

    Args:
        *shapes: Variable number of shape tuples.

    Returns:
        List of layout tuples in minor-to-major order (row-major).
    """
    return [tuple(range(len(shape) - 1, -1, -1)) for shape in shapes]


def lowering(
    ctx: Any,
    source: Any,
    *points: Any,
    output_shape: tuple[int, ...] | None,
    iflag: int,
    eps: float,
    opts: options.Opts | options.NestedOpts | None,
    nufft_type: int,
) -> Sequence[Any]:
    """MLIR lowering rule for NUFFT primitives.

    Lowers JAX NUFFT primitives to XLA custom calls targeting the FINUFFT
    library using the typed FFI API (api_version=4).

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

    if platform == "cpu":
        # Convert options to native finufft_opts for extracting values
        opts_native = opts.to_finufft_opts()

        # Build operands list based on NUFFT type
        # For type 1 and 2: source + points (x, y, z) - pad with placeholders
        # For type 3: source + source_points + target_points
        if nufft_type == 3:
            # Type 3 has both source points and target points
            # points_fortran contains: [x_rev, y_rev, z_rev, s_rev, t_rev, u_rev]
            # where some may be from padding
            operands = [source]
            # Add all point arrays (source and target), padding to 3D each
            for i in range(ndim):
                operands.append(points_fortran[i])  # source points
            # Pad source points to 3
            for i in range(ndim, 3):
                operands.append(points_fortran[0])  # placeholder
            for i in range(ndim):
                operands.append(points_fortran[ndim + i])  # target points
            # Pad target points to 3
            for i in range(ndim, 3):
                operands.append(points_fortran[ndim])  # placeholder
        else:
            # Type 1 and 2: source + points (padded to 3D)
            operands = [source]
            for i in range(ndim):
                operands.append(points_fortran[i])
            # Pad with placeholder arrays for unused dimensions
            for i in range(ndim, 3):
                operands.append(points_fortran[0])  # placeholder

        # Build FFI attributes dictionary
        ffi_attrs = {
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

        # Use jax.ffi.ffi_lowering for typed FFI (api_version=4)
        # Pass FFI attributes as kwargs when calling the lowering function
        # skip_ffi_layout_processing=True to use our explicit layouts
        return jax.ffi.ffi_lowering(
            op_name,
            operand_layouts=default_layouts(*(op.type.shape for op in operands)),
            result_layouts=default_layouts(ctx.avals_out[0].shape),
            skip_ffi_layout_processing=True,
        )(ctx, *operands, **ffi_attrs)

    else:
        # GPU path - using typed FFI (api_version=4)
        opts_native = opts.to_cufinufft_opts()

        # Build operands list based on NUFFT type
        # For type 1 and 2: source + points (x, y, z) - pad with placeholders
        # For type 3: source + source_points + target_points
        if nufft_type == 3:
            # Type 3 has both source points and target points
            operands = [source]
            # Add source points (padded to 3D)
            for i in range(ndim):
                operands.append(points_fortran[i])
            for i in range(ndim, 3):
                operands.append(points_fortran[0])  # placeholder
            # Add target points (padded to 3D)
            for i in range(ndim):
                operands.append(points_fortran[ndim + i])
            for i in range(ndim, 3):
                operands.append(points_fortran[ndim])  # placeholder
        else:
            # Type 1 and 2: source + points (padded to 3D)
            operands = [source]
            for i in range(ndim):
                operands.append(points_fortran[i])
            for i in range(ndim, 3):
                operands.append(points_fortran[0])  # placeholder

        # Build FFI attributes dictionary
        ffi_attrs = {
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

        # Use jax.ffi.ffi_lowering for typed FFI (api_version=4)
        # skip_ffi_layout_processing=True to use our explicit layouts
        return jax.ffi.ffi_lowering(
            op_name,
            operand_layouts=default_layouts(*(op.type.shape for op in operands)),
            result_layouts=default_layouts(ctx.avals_out[0].shape),
            skip_ffi_layout_processing=True,
        )(ctx, *operands, **ffi_attrs)
