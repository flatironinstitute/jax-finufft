__all__ = ["translation_rule"]

import numpy as np
from jax.lib import xla_client

from . import jax_finufft_cpu

try:
    from . import jax_finufft_gpu

    for _name, _value in jax_finufft_gpu.registrations().items():
        xla_client.register_custom_call_target(_name, _value, platform="cuda")
except ImportError:
    jax_finufft_gpu = None

for _name, _value in jax_finufft_cpu.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="cpu")

xops = xla_client.ops


def translation_rule(
    platform, ctx, avals_in, avals_out, source, *points, output_shape, iflag, eps
):
    if platform == "gpu" and jax_finufft_gpu is None:
        raise ValueError("jax-finufft was not compiled with GPU support")
    
    ndim = len(points)
    assert 1 <= ndim <= 3
    if platform == "gpu" and ndim == 1:
        raise ValueError("1-D transforms are not yet supported on the GPU")

    c = ctx.builder
    source_shape_info = c.get_shape(source)
    points_shape_info = list(map(c.get_shape, points))

    # Check supported and consistent dtypes
    source_dtype = source_shape_info.element_type()
    single = source_dtype == np.csingle and all(
        x.element_type() == np.single for x in points_shape_info
    )
    double = source_dtype == np.cdouble and all(
        x.element_type() == np.double for x in points_shape_info
    )
    assert single or double
    suffix = "f" if source_dtype == np.csingle else ""

    # Check shapes
    source_shape = source_shape_info.dimensions()
    points_shape = tuple(x.dimensions() for x in points_shape_info)
    n_tot = source_shape[0]
    n_transf = source_shape[1]
    n_j = points_shape[0][1]
    if output_shape is None:
        op_name = f"nufft{ndim}d2{suffix}".encode("ascii")
        n_k = np.array(source_shape[2:], dtype=np.int64)
        full_output_shape = source_shape[:2] + (n_j,)
    else:
        op_name = f"nufft{ndim}d1{suffix}".encode("ascii")
        n_k = np.array(output_shape, dtype=np.int64)
        full_output_shape = source_shape[:2] + tuple(output_shape)

    # The backend expects the output shape in Fortran order so we'll just
    # fake it here, by sending in n_k and x in the reverse order.
    n_k_full = np.zeros(3, dtype=np.int64)
    n_k_full[:ndim] = n_k[::-1]

    # Dispatch to the right op
    desc = getattr(jax_finufft_cpu, f"build_descriptor{suffix}")(
        eps, iflag, n_tot, n_transf, n_j, *n_k_full
    )

    # Set up most of the arguments
    operands = (
        source,
        *points[::-1],  # Reverse order because backend uses Fortran order
    )
    operand_shapes_with_layout = (
        xla_client.Shape.array_shape(
            source_dtype,
            source_shape,
            tuple(range(len(source_shape) - 1, -1, -1)),
        ),
    ) + tuple(
        xla_client.Shape.array_shape(
            x.element_type(),
            x.dimensions(),
            tuple(range(len(x.dimensions()) - 1, -1, -1)),
        )
        for x in points_shape_info[::-1]  # Reverse order, again
    )
    shape_with_layout = xla_client.Shape.array_shape(
        source_dtype,
        full_output_shape,
        tuple(range(len(full_output_shape) - 1, -1, -1)),
    )

    if platform == "cpu":
        return [
            xops.CustomCallWithLayout(
                c,
                op_name,
                operands=(xops.ConstantLiteral(c, np.frombuffer(desc, dtype=np.uint8)),)
                + operands,
                operand_shapes_with_layout=(
                    xla_client.Shape.array_shape(
                        np.dtype(np.uint8), (len(desc),), (0,)
                    ),
                )
                + operand_shapes_with_layout,
                shape_with_layout=shape_with_layout,
            )
        ]

    elif platform == "gpu":
        return [
            xops.CustomCallWithLayout(
                c,
                op_name,
                operands=operands,
                operand_shapes_with_layout=operand_shapes_with_layout,
                shape_with_layout=shape_with_layout,
                opaque=desc,
            )
        ]

    else:
        raise ValueError(f"Unrecognized platform '{platform}'")
