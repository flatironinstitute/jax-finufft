# -*- coding: utf-8 -*-

__all__ = ["finufft1d1"]

from functools import partial

import numpy as np
from jax import numpy as jnp
from jax.lib import xla_client
from jax import core, dtypes, lax
from jax.interpreters import ad, batching, xla
from jax.abstract_arrays import ShapedArray

# Register the CPU XLA custom calls
from . import jax_finufft, jax_finufft_single

for _name, _value in jax_finufft.registrations().items():
    xla_client.register_cpu_custom_call_target(_name, _value)
for _name, _value in jax_finufft_single.registrations().items():
    xla_client.register_cpu_custom_call_target(_name, _value)

xops = xla_client.ops


def finufft1d1(N, x, c, *, tol=None):
    x = jnp.asarray(x)
    if tol is None:
        tol = 1e-6
    N = int(N)
    assert N > 0

    def abstract_eval(x, c):
        x_dtype = dtypes.canonicalize_dtype(x.dtype)
        c_dtype = dtypes.canonicalize_dtype(c.dtype)
        assert (x_dtype == jnp.double and c_dtype == jnp.cdouble) or (
            x_dtype == jnp.single and c_dtype == jnp.csingle
        )
        assert x.shape == c.shape
        return ShapedArray((N,), c_dtype)

    def translation(ctx, x, c):
        # The inputs have "shapes" that provide both the shape and the dtype
        x_shape = ctx.get_shape(x)
        c_shape = ctx.get_shape(c)

        # Extract the dtype and shape
        x_dtype = x_shape.element_type()
        c_dtype = c_shape.element_type()
        dims = x_shape.dimensions()
        assert len(dims) == 1
        assert c_shape.dimensions() == dims
        assert (x_dtype == jnp.double and c_dtype == jnp.cdouble) or (
            x_dtype == jnp.single and c_dtype == jnp.csingle
        )

        # We dispatch a different call depending on the dtype
        if x_dtype == jnp.single:
            op_name = b"finufft1d1_single"
        elif x_dtype == jnp.double:
            op_name = b"finufft1d1"
        else:
            raise NotImplementedError(f"Unsupported dtype {x_dtype}")

        # On the CPU, we pass the size of the data as a the first input
        # argument
        return xops.CustomCallWithLayout(
            ctx,
            op_name,
            # The inputs:
            operands=(
                xops.ConstantLiteral(ctx, jnp.int32(N)),
                xops.ConstantLiteral(ctx, jnp.int32(dims[0])),
                xops.ConstantLiteral(ctx, jnp.array(tol, dtype=x_dtype)),
                x,
                c,
            ),
            # The input shapes:
            operand_shapes_with_layout=(
                xla_client.Shape.array_shape(jnp.dtype(jnp.int32), (), ()),
                xla_client.Shape.array_shape(jnp.dtype(jnp.int32), (), ()),
                xla_client.Shape.array_shape(jnp.dtype(x_dtype), (), ()),
                xla_client.Shape.array_shape(jnp.dtype(x_dtype), dims, (0,)),
                xla_client.Shape.array_shape(jnp.dtype(c_dtype), dims, (0,)),
            ),
            # The output shapes:
            shape_with_layout=xla_client.Shape.array_shape(
                np.dtype(c_dtype), (N,), (0,)
            ),
        )

    def jvp(args, tangents):
        x, c = args
        dx, dc = tangents
        f = prim.bind(x, c)

        def zero_tangent(tan, val):
            return lax.zeros_like_array(val) if type(tan) is ad.Zero else tan

        df = prim.bind(x, zero_tangent(dc, c)) + jnp.arange(f.shape[0]) * prim.bind(
            x, 1j * c * zero_tangent(dx, x)
        )
        return f, df

    def transpose(ct, x, c):
        print(ct, x, c)
        assert 0

    prim = core.Primitive(f"finufft1d1_{N}")
    prim.def_impl(partial(xla.apply_primitive, prim))
    prim.def_abstract_eval(abstract_eval)
    xla.backend_specific_translations["cpu"][prim] = translation
    ad.primitive_jvps[prim] = jvp
    ad.primitive_transposes[prim] = transpose

    return prim.bind(x, c)
