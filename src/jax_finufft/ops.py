__all__ = ["nufft1", "nufft2"]

from functools import partial, reduce

import numpy as np
from jax import core, dtypes, jit
from jax import numpy as jnp
from jax.abstract_arrays import ShapedArray
from jax.interpreters import ad, batching, xla
from jax.lib import xla_client

from . import jax_finufft

for _name, _value in jax_finufft.registrations().items():
    xla_client.register_cpu_custom_call_target(_name, _value)

xops = xla_client.ops


@partial(jit, static_argnums=(0,), static_argnames=("iflag", "eps"))
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

    return jnp.reshape(
        nufft1_p.bind(source, *points, output_shape=output_shape, iflag=iflag, eps=eps),
        expected_output_shape,
    )


@partial(jit, static_argnames=("iflag", "eps"))
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

    return jnp.reshape(
        nufft2_p.bind(source, *points, output_shape=None, iflag=iflag, eps=eps),
        expected_output_shape,
    )


def get_output_shape(type_, source_shape, *points_shape, output_shape):
    if type_ == 1:
        ndim = len(points_shape)
        assert len(output_shape) == ndim
        assert len(points_shape[0]) >= 2
        assert all(
            x[-1] == source_shape[-1] and x[:-1] == source_shape[:-2]
            for x in points_shape
        )
        return tuple(source_shape[:-1]) + tuple(output_shape)

    elif type_ == 2:
        ndim = len(points_shape)
        assert len(points_shape[0]) >= 2
        assert all(x[:-1] == source_shape[: -ndim - 1] for x in points_shape)
        return tuple(source_shape[:-ndim]) + (points_shape[0][-1],)

    raise ValueError(f"Unsupported transformation type: {type_}")


def abstract_eval(type_, source, *points, output_shape, **_):
    ndim = len(points)
    assert 1 <= ndim <= 3

    source_dtype = dtypes.canonicalize_dtype(source.dtype)
    points_dtype = [dtypes.canonicalize_dtype(x.dtype) for x in points]

    # Check supported and consistent dtypes
    single = source_dtype == np.csingle and all(x == np.single for x in points_dtype)
    double = source_dtype == np.cdouble and all(x == np.double for x in points_dtype)
    assert single or double

    return ShapedArray(
        get_output_shape(
            type_, source.shape, *(x.shape for x in points), output_shape=output_shape
        ),
        source_dtype,
    )


def translation_rule(
    type_, ctx, avals_in, avals_out, source, *points, output_shape, iflag, eps
):
    ndim = len(points)
    assert 1 <= ndim <= 3

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

    # Check shapes
    source_shape = source_shape_info.dimensions()
    points_shape = tuple(x.dimensions() for x in points_shape_info)
    full_output_shape = get_output_shape(
        type_, source_shape, *points_shape, output_shape=output_shape
    )

    # Work out the other dimenstions of the problem
    n_j = np.array(points_shape[0][-1]).astype(np.int64)
    if type_ == 1:
        n_tot = np.prod(source_shape[:-2]).astype(np.int64)
        n_transf = np.array(source_shape[-2]).astype(np.int32)
        n_k = np.array(full_output_shape[-ndim:], dtype=np.int64)
    else:
        n_tot = np.prod(source_shape[: -ndim - 1]).astype(np.int64)
        n_transf = np.array(source_shape[-ndim - 1]).astype(np.int32)
        n_k = np.array(source_shape[-ndim:], dtype=np.int64)

    # The backend expects the output shape in Fortran order so we'll just
    # fake it here, by sending in n_k and x in the reverse order.
    n_k_full = np.zeros(3, dtype=np.int64)
    n_k_full[:ndim] = n_k[::-1]

    # Dispatch to the right op
    suffix = "f" if source_dtype == np.csingle else ""
    op_name = f"nufft{ndim}d{type_}{suffix}".encode("ascii")
    desc = getattr(jax_finufft, f"build_descriptor{suffix}")(
        eps, iflag, n_tot, n_transf, n_j, *n_k_full
    )

    return [
        xops.CustomCallWithLayout(
            c,
            op_name,
            # The inputs:
            operands=(
                xops.ConstantLiteral(c, np.frombuffer(desc, dtype=np.uint8)),
                source,
                *points[::-1],  # Reverse order because backend uses Fortran order
            ),
            # The input shapes:
            operand_shapes_with_layout=(
                xla_client.Shape.array_shape(np.dtype(np.uint8), (len(desc),), (0,)),
                xla_client.Shape.array_shape(
                    source_dtype,
                    source_shape,
                    tuple(range(len(source_shape) - 1, -1, -1)),
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
    ]


def points_jvp(type_, dim, prim, dpoints, source, *points, output_shape, iflag, eps):
    # Type 1:
    # f_k = sum_j c_j * exp(iflag * i * k * x_j)
    # df_k/dx_j = iflag * i * k * c_j * exp(iflag * i * k * x_j)

    # Type 2:
    # c_j = sum_k f_k * exp(iflag * i * k * x_j)
    # dc_j/dx_j = sum_k iflag * i * k * f_k * exp(iflag * i * k * x_j)

    ndim = len(points)
    n = output_shape[dim] if type_ == 1 else source.shape[-ndim + dim]

    shape = np.ones(ndim, dtype=int)
    shape[dim] = -1
    k = np.arange(-np.floor(n / 2), np.floor((n - 1) / 2 + 1))
    k = k.reshape(shape)
    factor = 1j * iflag * k

    if type_ == 1:
        return factor * prim.bind(
            source * dpoints,
            *points,
            output_shape=output_shape,
            iflag=iflag,
            eps=eps,
        )
    return dpoints * prim.bind(
        factor * source,
        *points,
        output_shape=output_shape,
        iflag=iflag,
        eps=eps,
    )


def jvp(type_, prim, args, tangents, *, output_shape, iflag, eps):
    # TODO: We could maybe speed this up by concatenating all the source terms and
    # then executing a single NUFFT since they all use the same NU points. The
    # bookkeeping might get a little ugly.

    source, *points = args
    dsource, *dpoints = tangents
    output = prim.bind(source, *points, output_shape=output_shape, iflag=iflag, eps=eps)

    output_tangents = []
    if type(dsource) is not ad.Zero:
        output_tangents.append(
            prim.bind(dsource, *points, output_shape=output_shape, iflag=iflag, eps=eps)
        )

    output_tangents += [
        points_jvp(
            type_,
            dim,
            prim,
            dx,
            source,
            *points,
            output_shape=output_shape,
            iflag=iflag,
            eps=eps,
        )
        for dim, dx in enumerate(dpoints)
    ]

    return output, reduce(ad.add_tangents, output_tangents, ad.Zero.from_value(output))


def transpose(type_, doutput, source, *points, output_shape, eps, iflag):
    assert ad.is_undefined_primal(source)
    assert not any(map(ad.is_undefined_primal, points))
    assert type(doutput) is not ad.Zero

    if type_ == 1:
        result = nufft2(doutput, *points, eps=eps, iflag=iflag)
    else:
        ndim = len(points)
        result = nufft1(
            source.aval.shape[-ndim:], doutput, *points, eps=eps, iflag=iflag
        )

    return (result,) + tuple(None for _ in range(len(points)))


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

    if points[0].ndim == source.ndim - output_dim + 1:
        new_shape = source.shape[:-output_dim] + (1,) + source.shape[-output_dim:]
        source = jnp.reshape(source, new_shape)
    if points[0].ndim != source.ndim - output_dim:
        raise ValueError(
            f"'source' must have {output_dim} more dimension than 'points'"
        )
    if source.ndim == output_dim + 1:
        source = source[None, ...]
        points = tuple(x[None, :] for x in points)

    return source, points


nufft1_p = core.Primitive("nufft1")
nufft1_p.def_impl(partial(xla.apply_primitive, nufft1_p))
nufft1_p.def_abstract_eval(partial(abstract_eval, 1))
xla.register_translation(nufft1_p, partial(translation_rule, 1), platform="cpu")
ad.primitive_jvps[nufft1_p] = partial(jvp, 1, nufft1_p)
ad.primitive_transposes[nufft1_p] = partial(transpose, 1)
batching.primitive_batchers[nufft1_p] = partial(batch, nufft1_p)


nufft2_p = core.Primitive("nufft2")
nufft2_p.def_impl(partial(xla.apply_primitive, nufft2_p))
nufft2_p.def_abstract_eval(partial(abstract_eval, 2))
xla.register_translation(nufft2_p, partial(translation_rule, 2), platform="cpu")
ad.primitive_jvps[nufft2_p] = partial(jvp, 2, nufft2_p)
ad.primitive_transposes[nufft2_p] = partial(transpose, 2)
batching.primitive_batchers[nufft2_p] = partial(batch, nufft2_p)
