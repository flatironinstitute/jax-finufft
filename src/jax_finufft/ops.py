# -*- coding: utf-8 -*-

__all__ = ["finufft1d1", "finufft1d2"]

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
        x_shape = ctx.get_shape(x)
        c_shape = ctx.get_shape(c)
        x_dtype = x_shape.element_type()
        c_dtype = c_shape.element_type()
        dims = x_shape.dimensions()
        assert len(dims) == 1
        assert c_shape.dimensions() == dims
        assert (x_dtype == jnp.double and c_dtype == jnp.cdouble) or (
            x_dtype == jnp.single and c_dtype == jnp.csingle
        )

        if x_dtype == jnp.single:
            op_name = b"finufft1d1_single"
        elif x_dtype == jnp.double:
            op_name = b"finufft1d1"
        else:
            raise NotImplementedError(f"Unsupported dtype {x_dtype}")

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

    def transpose(ft, x, c):
        assert ad.is_undefined_primal(c)
        assert not ad.is_undefined_primal(x)
        if type(ft) is ad.Zero:
            return None, ad.Zero(c.aval)
        return None, finufft1d2(x, ft, tol=tol)

    prim = core.Primitive(f"finufft1d1_{N}")
    prim.def_impl(partial(xla.apply_primitive, prim))
    prim.def_abstract_eval(abstract_eval)
    xla.backend_specific_translations["cpu"][prim] = translation
    ad.primitive_jvps[prim] = jvp
    ad.primitive_transposes[prim] = transpose

    return prim.bind(x, c)


def finufft1d2(x, f, *, tol=None):
    x = jnp.asarray(x)
    if tol is None:
        tol = 1e-6

    def abstract_eval(x, f):
        x_dtype = dtypes.canonicalize_dtype(x.dtype)
        f_dtype = dtypes.canonicalize_dtype(f.dtype)
        assert (x_dtype == jnp.double and f_dtype == jnp.cdouble) or (
            x_dtype == jnp.single and f_dtype == jnp.csingle
        )
        return ShapedArray(x.shape, f_dtype)

    def translation(ctx, x, f):
        x_shape = ctx.get_shape(x)
        f_shape = ctx.get_shape(f)
        x_dtype = x_shape.element_type()
        f_dtype = f_shape.element_type()
        x_dims = x_shape.dimensions()
        f_dims = f_shape.dimensions()
        assert len(x_dims) == 1
        assert len(f_dims) == 1
        assert (x_dtype == jnp.double and f_dtype == jnp.cdouble) or (
            x_dtype == jnp.single and f_dtype == jnp.csingle
        )

        # We dispatch a different call depending on the dtype
        if x_dtype == jnp.single:
            op_name = b"finufft1d2_single"
        elif x_dtype == jnp.double:
            op_name = b"finufft1d2"
        else:
            raise NotImplementedError(f"Unsupported dtype {x_dtype}")

        # On the CPU, we pass the size of the data as a the first input
        # argument
        return xops.CustomCallWithLayout(
            ctx,
            op_name,
            # The inputs:
            operands=(
                xops.ConstantLiteral(ctx, jnp.int32(f_dims[0])),
                xops.ConstantLiteral(ctx, jnp.int32(x_dims[0])),
                xops.ConstantLiteral(ctx, jnp.array(tol, dtype=x_dtype)),
                x,
                f,
            ),
            # The input shapes:
            operand_shapes_with_layout=(
                xla_client.Shape.array_shape(jnp.dtype(jnp.int32), (), ()),
                xla_client.Shape.array_shape(jnp.dtype(jnp.int32), (), ()),
                xla_client.Shape.array_shape(jnp.dtype(x_dtype), (), ()),
                xla_client.Shape.array_shape(jnp.dtype(x_dtype), x_dims, (0,)),
                xla_client.Shape.array_shape(jnp.dtype(f_dtype), f_dims, (0,)),
            ),
            # The output shapes:
            shape_with_layout=xla_client.Shape.array_shape(
                np.dtype(f_dtype), x_dims, (0,)
            ),
        )

    def jvp(args, tangents):
        assert 0

    def transpose(ct, x, f):
        assert 0

    prim = core.Primitive(f"finufft1d2")
    prim.def_impl(partial(xla.apply_primitive, prim))
    prim.def_abstract_eval(abstract_eval)
    xla.backend_specific_translations["cpu"][prim] = translation
    ad.primitive_jvps[prim] = jvp
    ad.primitive_transposes[prim] = transpose

    return prim.bind(x, f)
