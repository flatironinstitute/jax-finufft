# -*- coding: utf-8 -*-

__all__ = ["nufft1", "nufft2"]

from functools import partial

import numpy as np
from jax import numpy as jnp
from jax.lib import xla_client
from jax import core, dtypes, lax
from jax.interpreters import ad, batching, xla
from jax.abstract_arrays import ShapedArray

from . import jax_finufft

for _name, _value in jax_finufft.registrations().items():
    xla_client.register_cpu_custom_call_target(_name, _value)

xops = xla_client.ops


def nufft1(output_shape, source, *points, iflag=1, eps=1e-6):
    iflag = int(iflag)
    eps = float(eps)
    ndim = len(points)
    if not 1 <= ndim <= 3:
        raise ValueError("Only 1-, 2-, and 3-dimensions are supported")

    # Support passing a scalar output_shape
    output_shape = np.atleast_1d(output_shape).astype(np.int64)
    if len(output_shape) != ndim:
        raise ValueError(f"output_shape must have shape: ({ndim},)")

    # Handle broadcasting
    expected_output_shape = source.shape[:-1] + tuple(output_shape)
    source, points = pad_shapes(1, source, *points)
    if points[0].shape[-1] != source.shape[-1]:
        raise ValueError("The final dimension of 'source' must match 'points'")

    prim = core.Primitive("nufft1")
    prim.def_impl(partial(xla.apply_primitive, prim))
    prim.def_abstract_eval(partial(abstract_eval, partial(type1_shapes, output_shape)))
    xla.backend_specific_translations["cpu"][prim] = partial(
        translation_rule,
        1,
        partial(type1_shapes, output_shape),
        eps=float(eps),
        iflag=int(iflag),
    )
    ad.primitive_jvps[prim] = partial(type1_jvp, prim)
    ad.primitive_transposes[prim] = partial(type1_transpose, eps=eps, iflag=iflag)
    batching.primitive_batchers[prim] = partial(batch, prim)

    return jnp.reshape(prim.bind(source, *points), expected_output_shape)


def nufft2(source, *points, iflag=-1, eps=1e-6):
    iflag = int(iflag)
    eps = float(eps)
    ndim = len(points)
    if not 1 <= ndim <= 3:
        raise ValueError("Only 1-, 2-, and 3-dimensions are supported")

    # Handle broadcasting
    expected_output_shape = source.shape[:-ndim]
    source, points = pad_shapes(ndim, source, *points)
    expected_output_shape = expected_output_shape + (points[0].shape[-1],)

    prim = core.Primitive("nufft2")
    prim.def_impl(partial(xla.apply_primitive, prim))
    prim.def_abstract_eval(partial(abstract_eval, type2_shapes))
    xla.backend_specific_translations["cpu"][prim] = partial(
        translation_rule,
        2,
        type2_shapes,
        eps=float(eps),
        iflag=int(iflag),
    )
    # ad.primitive_jvps[prim] = partial(type1_jvp, prim)
    # ad.primitive_transposes[prim] = transpose
    batching.primitive_batchers[prim] = partial(batch, prim)

    return jnp.reshape(prim.bind(source, *points), expected_output_shape)


def type1_shapes(output_shape, source_shape, *points_shape):
    ndim = len(points_shape)
    assert len(output_shape) == ndim
    assert len(points_shape[0]) >= 2
    assert all(
        x[-1] == source_shape[-1] and x[:-1] == source_shape[:-2] for x in points_shape
    )
    return tuple(source_shape[:-1]) + tuple(output_shape)


def type2_shapes(source_shape, *points_shape):
    ndim = len(points_shape)
    assert len(points_shape[0]) >= 2
    assert all(x[:-1] == source_shape[: -ndim - 1] for x in points_shape)
    return tuple(source_shape[:-ndim]) + (points_shape[0][-1],)


def abstract_eval(output_shape_func, source, *points):
    ndim = len(points)
    assert 1 <= ndim <= 3

    source_dtype = dtypes.canonicalize_dtype(source.dtype)
    points_dtype = [dtypes.canonicalize_dtype(x.dtype) for x in points]

    # Check supported and consistent dtypes
    single = source_dtype == np.csingle and all(x == np.single for x in points_dtype)
    double = source_dtype == np.cdouble and all(x == np.double for x in points_dtype)
    assert single or double

    return ShapedArray(
        output_shape_func(source.shape, *(x.shape for x in points)), source_dtype
    )


def translation_rule(type, output_shape_func, ctx, source, *points, eps=1e-6, iflag=1):
    assert type == 1 or type == 2
    ndim = len(points)
    assert 1 <= ndim <= 3

    source_shape_info = ctx.get_shape(source)
    points_shape_info = list(map(ctx.get_shape, points))

    # Check supported and consistent dtypes
    source_dtype = source_shape_info.element_type()
    single = source_dtype == np.csingle and all(
        x.element_type() == np.single for x in points_shape_info
    )
    double = source_dtype == np.cdouble and all(
        x.element_type() == np.double for x in points_shape_info
    )
    assert single or double

    # Check shapes
    source_shape = source_shape_info.dimensions()
    points_shape = tuple(x.dimensions() for x in points_shape_info)
    full_output_shape = output_shape_func(source_shape, *points_shape)

    # Work out the other dimenstions of the problem
    n_j = np.array(points_shape[0][-1]).astype(np.int64)
    if type == 1:
        n_tot = np.prod(source_shape[:-2]).astype(np.int64)
        n_transf = np.array(source_shape[-2]).astype(np.int32)
        n_k = np.array(full_output_shape[-ndim:], dtype=np.int64)
    else:
        n_tot = np.prod(source_shape[: -ndim - 1]).astype(np.int64)
        n_transf = np.array(source_shape[-ndim]).astype(np.int32)
        n_k = np.array(source_shape[-ndim:], dtype=np.int64)

    # The backend expects the output shape in Fortran order so we'll just
    # fake it here, by sending in n_k and x in the reverse order.
    n_k_full = np.zeros(3, dtype=np.int64)
    n_k_full[:ndim] = n_k[::-1]

    # Dispatch to the right op
    suffix = "f" if source_dtype == np.csingle else ""
    op_name = f"nufft{ndim}d{type}{suffix}".encode("ascii")
    desc = getattr(jax_finufft, f"build_descriptor{suffix}")(
        eps, iflag, n_tot, n_transf, n_j, *n_k_full
    )

    return xops.CustomCallWithLayout(
        ctx,
        op_name,
        # The inputs:
        operands=(
            xops.ConstantLiteral(ctx, np.frombuffer(desc, dtype=np.uint8)),
            source,
            *points[::-1],  # Reverse order because backend uses Fortran order
        ),
        # The input shapes:
        operand_shapes_with_layout=(
            xla_client.Shape.array_shape(np.dtype(np.uint8), (len(desc),), (0,)),
            xla_client.Shape.array_shape(
                source_dtype, source_shape, tuple(range(len(source_shape) - 1, -1, -1))
            ),
        )
        + tuple(
            xla_client.Shape.array_shape(
                x.element_type(),
                x.dimensions(),
                tuple(range(len(x.dimensions()) - 1, -1, -1)),
            )
            for x in points_shape_info[::-1]  # Reverse order, again
        ),
        # The output shapes:
        shape_with_layout=xla_client.Shape.array_shape(
            source_dtype,
            full_output_shape,
            tuple(range(len(full_output_shape) - 1, -1, -1)),
        ),
    )


def type1_jvp(prim, args, tangents):
    source, *points = args
    dsource, *dpoints = tangents
    f = prim.bind(source, *points)

    def zero_tangent(tan, val):
        return lax.zeros_like_array(val) if type(tan) is ad.Zero else tan

    # f(k) = sum[c(j) * exp[i k x(j)]]
    # df(k) = sum(dc(j) * exp[i k x(j)]) + k * sum(c(j) * exp[i k x(j)] * (i dx(j)))
    df = prim.bind(zero_tangent(dsource, source), *points)

    # The x gradient doesn't seem to work... hmmm
    # N = f.shape[0]
    # k = jnp.arange(-(N - N % 2) // 2, (N - 2 + N % 2) // 2 + 1)
    # k * prim.bind(x, 1.0j * c * zero_tangent(dx, x))
    assert all(type(dx) is ad.Zero for dx in dpoints)

    return f, df


def type1_transpose(ft, source, *points, eps=1e-6, iflag=1):
    assert ad.is_undefined_primal(source)
    assert not any(map(ad.is_undefined_primal, points))
    if type(ft) is ad.Zero:
        return tuple(None for _ in range(len(points) + 1))
    return (None, jnp.conj(nufft2(ft, *points, eps=eps, iflag=iflag)))


def batch(prim, args, axes):
    # We can't batch over the last two dimensions of source
    mx = args[0].ndim - 2
    assert all(a < mx for a in axes)
    assert all(a == axes[0] for a in axes[1:])
    return prim.bind(*args), axes[0]


def pad_shapes(output_dim, source, *points):
    points = jnp.broadcast_arrays(*points)
    if points[0].ndim == 0 or source.ndim == 0:
        raise ValueError(
            "0-dimensional arrays are not supported; are you vmap-ing somewhere "
            "where you don't want to?"
        )
    if points[0].ndim == source.ndim:
        source = source[..., None, :]
    if points[0].ndim != source.ndim - output_dim:
        raise ValueError("'source' must have one more dimension than 'points'")
    if source.ndim == output_dim + 1:
        source = source[None, ...]
        points = tuple(x[None, :] for x in points)

    return source, points
