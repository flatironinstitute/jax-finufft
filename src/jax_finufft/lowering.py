import numpy as np
import jax
from jax.interpreters import mlir
from jax.interpreters.mlir import ir

from jax_finufft import options

from . import jax_finufft_cpu

try:
    from . import jax_finufft_gpu

    for _name, _value in jax_finufft_gpu.registrations().items():
        jax.ffi.register_ffi_target(_name, _value, platform="CUDA", api_version=0)
except ImportError:
    jax_finufft_gpu = None

for _name, _value in jax_finufft_cpu.registrations().items():
    jax.ffi.register_ffi_target(_name, _value, platform="cpu", api_version=0)


def default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]


def lowering(
    ctx: mlir.LoweringRuleContext,
    source: ir.Value,
    *points,
    output_shape,
    iflag,
    eps,
    opts,
    nufft_type,
):
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
        op_name = f"nufft{ndim}d1{suffix}".encode("ascii")
        n_k = np.array(output_shape, dtype=np.int64)
    elif nufft_type == 2:
        op_name = f"nufft{ndim}d2{suffix}".encode("ascii")
        n_k = np.array(source_shape[2:], dtype=np.int64)
    elif nufft_type == 3:
        op_name = f"nufft{ndim}d3{suffix}".encode("ascii")
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

    if platform == "cpu":
        opts = opts.to_finufft_opts()
        descriptor_bytes = getattr(jax_finufft_cpu, f"build_descriptor{suffix}")(
            eps, iflag, n_tot, n_transf, n_j, *n_k_full, opts
        )
        descriptor = mlir.ir_constant(np.frombuffer(descriptor_bytes, dtype=np.uint8))
        return mlir.custom_call(
            op_name,
            result_types=[mlir.aval_to_ir_type(aval) for aval in ctx.avals_out],
            operands=[descriptor, source, *points_fortran],
            operand_layouts=default_layouts([0], source_shape, *points_shape[::-1]),
            result_layouts=default_layouts(ctx.avals_out[0].shape),
        ).results

    else:
        opts = opts.to_cufinufft_opts()
        descriptor_bytes = getattr(jax_finufft_gpu, f"build_descriptor{suffix}")(
            eps, iflag, n_tot, n_transf, n_j, *n_k_full, opts
        )
        return mlir.custom_call(
            op_name,
            result_types=[mlir.aval_to_ir_type(aval) for aval in ctx.avals_out],
            # Reverse points because backend uses Fortran order
            operands=[source, *points_fortran],
            backend_config=descriptor_bytes,
            operand_layouts=default_layouts(source_shape, *points_shape[::-1]),
            result_layouts=default_layouts(ctx.avals_out[0].shape),
        ).results
