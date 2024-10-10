__all__ = ["nufft1", "nufft2"]

from functools import partial, reduce

import numpy as np
from jax import core
from jax import jit
from jax import numpy as jnp
from jax.interpreters import ad, batching, xla, mlir

from jax_finufft import shapes, lowering, options


@partial(jit, static_argnums=(0,), static_argnames=("iflag", "eps", "opts"))
def nufft1(output_shape, source, *points, iflag=1, eps=1e-6, opts=None):
    iflag = int(iflag)
    eps = float(eps)
    ndim = len(points)
    if not 1 <= ndim <= 3:
        raise ValueError("Only 1-, 2-, and 3-dimensions are supported")

    # Support passing a scalar output_shape
    output_shape = np.atleast_1d(output_shape).astype(np.int64)
    if output_shape.shape != (ndim,):
        raise ValueError(f"output_shape must have shape: ({ndim},)")

    # Handle broadcasting and reshaping of inputs
    index, source, *points = shapes.broadcast_and_flatten_inputs(
        output_shape, source, *points
    )

    # Execute the transform primitive
    result = nufft1_p.bind(
        source, *points, output_shape=output_shape, iflag=iflag, eps=eps, opts=opts
    )

    # Move the axes back to their expected location
    return index.unflatten(result)


@partial(jit, static_argnames=("iflag", "eps", "opts"))
def nufft2(source, *points, iflag=-1, eps=1e-6, opts=None):
    iflag = int(iflag)
    eps = float(eps)
    ndim = len(points)
    if not 1 <= ndim <= 3:
        raise ValueError("Only 1-, 2-, and 3-dimensions are supported")

    # Handle broadcasting and reshaping of inputs
    index, source, *points = shapes.broadcast_and_flatten_inputs(None, source, *points)

    # Execute the transform primitive
    result = nufft2_p.bind(
        source, *points, output_shape=None, iflag=iflag, eps=eps, opts=opts
    )

    # Move the axes back to their expected location
    return index.unflatten(result)


def jvp(prim, args, tangents, *, output_shape, iflag, eps, opts):
    # Type 1:
    # f_k = sum_j c_j * exp(iflag * i * k * x_j)
    # df_k/dx_j = iflag * i * k * c_j * exp(iflag * i * k * x_j)

    # Type 2:
    # c_j = sum_k f_k * exp(iflag * i * k * x_j)
    # dc_j/dx_j = sum_k iflag * i * k * f_k * exp(iflag * i * k * x_j)

    source, *points = args
    dsource, *dpoints = tangents
    output = prim.bind(
        source, *points, output_shape=output_shape, iflag=iflag, eps=eps, opts=opts
    )

    # The JVP op can be written as a single transform of the same type with
    output_tangents = []
    ndim = len(points)
    scales = []
    arguments = []
    if type(dsource) is not ad.Zero:
        if output_shape is None:
            # It might look like we could combine this with the single transform at
            # the end, but then we'd be mixing tangents and concrete values
            output_tangents.append(
                prim.bind(
                    dsource,
                    *points,
                    output_shape=output_shape,
                    iflag=iflag,
                    eps=eps,
                    opts=opts,
                )
            )
        else:
            scales.append(1.0)
            arguments.append(dsource)

    for dim, dx in enumerate(dpoints):
        if type(dx) is ad.Zero:
            continue

        n = source.shape[-ndim + dim] if output_shape is None else output_shape[dim]
        shape = np.ones(ndim, dtype=int)
        shape[dim] = -1
        k = np.arange(-np.floor(n / 2), np.floor((n - 1) / 2 + 1))
        k = k.reshape(shape)
        factor = 1j * iflag * k
        dx = dx[:, None, :]

        if output_shape is None:
            scales.append(dx)
            arguments.append(factor * source)
        else:
            scales.append(factor)
            arguments.append(dx * source)

    if len(scales):
        func = nufft2 if output_shape is None else partial(nufft1, tuple(output_shape))
        argument = jnp.stack(arguments, axis=2)
        output_tangent = func(
            argument, *(p[:, None] for p in points), iflag=iflag, eps=eps, opts=opts
        )
        output_tangents += [s * output_tangent[:, :, n] for n, s in enumerate(scales)]

    if jax.version.__version_info__ < (0, 4, 34):
        zero = ad.Zero.from_value(output)
    else:
        zero = ad.Zero.from_primal_value(output)

    return output, reduce(ad.add_tangents, output_tangents, zero)


def transpose(doutput, source, *points, output_shape, eps, iflag, opts):
    assert ad.is_undefined_primal(source)
    assert not any(map(ad.is_undefined_primal, points))
    assert type(doutput) is not ad.Zero

    if output_shape is None:
        ndim = len(points)
        result = nufft1(
            source.aval.shape[-ndim:],
            doutput,
            *points,
            eps=eps,
            iflag=iflag,
            opts=options.unpack_opts(opts, 1, False),
        )
    else:
        result = nufft2(
            doutput,
            *points,
            eps=eps,
            iflag=iflag,
            opts=options.unpack_opts(opts, 2, False),
        )

    return (result,) + tuple(None for _ in range(len(points)))


def batch(args, axes, *, output_shape, **kwargs):
    source, *points = args
    bsource, *bpoints = axes

    # If none of the points are being mapped, we can get a faster computation using
    # a single transform with num_transforms * num_repeats
    if all(bx is batching.not_mapped for bx in bpoints):
        assert bsource is not batching.not_mapped
        source = batching.moveaxis(source, bsource, 0)
        mapped_points = tuple(p[None] for p in points)

    else:
        # Otherwise move the batching dimension to the front and repeat the arrays
        # to the right shape
        if bsource is None:
            assert any(bx is not batching.not_mapped for bx in bpoints)
            num_repeats = next(
                x.shape[bx]
                for x, bx in zip(points, bpoints)
                if bx is not batching.not_mapped
            )
            source = jnp.repeat(source[jnp.newaxis], num_repeats, axis=0)
        else:
            num_repeats = source.shape[bsource]
            source = batching.moveaxis(source, bsource, 0)

        mapped_points = []
        for x, bx in zip(points, bpoints):
            if bx is batching.not_mapped:
                mapped_points.append(jnp.repeat(x[None], num_repeats, axis=0))
            else:
                mapped_points.append(batching.moveaxis(x, bx, 0))

    if output_shape is None:
        return nufft2(source, *mapped_points, **kwargs), 0
    else:
        return nufft1(tuple(output_shape), source, *mapped_points, **kwargs), 0


nufft1_p = core.Primitive("nufft1")
nufft1_p.def_impl(partial(xla.apply_primitive, nufft1_p))
nufft1_p.def_abstract_eval(shapes.abstract_eval)
mlir.register_lowering(nufft1_p, lowering.lowering, platform="cpu")
if lowering.jax_finufft_gpu is not None:
    mlir.register_lowering(nufft1_p, lowering.lowering, platform="cuda")
ad.primitive_jvps[nufft1_p] = partial(jvp, nufft1_p)
ad.primitive_transposes[nufft1_p] = transpose
batching.primitive_batchers[nufft1_p] = batch


nufft2_p = core.Primitive("nufft2")
nufft2_p.def_impl(partial(xla.apply_primitive, nufft2_p))
nufft2_p.def_abstract_eval(shapes.abstract_eval)
mlir.register_lowering(nufft2_p, lowering.lowering, platform="cpu")
if lowering.jax_finufft_gpu is not None:
    mlir.register_lowering(nufft2_p, lowering.lowering, platform="cuda")
ad.primitive_jvps[nufft2_p] = partial(jvp, nufft2_p)
ad.primitive_transposes[nufft2_p] = transpose
batching.primitive_batchers[nufft2_p] = batch
