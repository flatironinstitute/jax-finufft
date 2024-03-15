import numpy as np
from jax.interpreters.mlir import ir
from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call as _custom_call
from jax_finufft import options

try:
    from jaxlib.hlo_helpers import hlo_const
except ImportError:
    # Copied from jaxlib/hlo_helpers.py for old versions of jax
    from functools import partial

    import jaxlib.mlir.dialects.stablehlo as hlo

    _dtype_to_ir_type_factory = {
        np.dtype(np.bool_): partial(ir.IntegerType.get_signless, 1),
        np.dtype(np.int8): partial(ir.IntegerType.get_signless, 8),
        np.dtype(np.int16): partial(ir.IntegerType.get_signless, 16),
        np.dtype(np.int32): partial(ir.IntegerType.get_signless, 32),
        np.dtype(np.int64): partial(ir.IntegerType.get_signless, 64),
        np.dtype(np.uint8): partial(ir.IntegerType.get_unsigned, 8),
        np.dtype(np.uint16): partial(ir.IntegerType.get_unsigned, 16),
        np.dtype(np.uint32): partial(ir.IntegerType.get_unsigned, 32),
        np.dtype(np.uint64): partial(ir.IntegerType.get_unsigned, 64),
        np.dtype(np.float16): ir.F16Type.get,
        np.dtype(np.float32): ir.F32Type.get,
        np.dtype(np.float64): ir.F64Type.get,
        np.dtype(np.complex64): lambda: ir.ComplexType.get(ir.F32Type.get()),
        np.dtype(np.complex128): lambda: ir.ComplexType.get(ir.F64Type.get()),
    }

    def dtype_to_ir_type(dtype) -> ir.Type:
        return _dtype_to_ir_type_factory[np.dtype(dtype)]()

    def hlo_const(x):
        assert isinstance(x, np.ndarray)
        return hlo.ConstantOp(
            ir.DenseElementsAttr.get(x, type=dtype_to_ir_type(x.dtype))
        ).result


from . import jax_finufft_cpu

try:
    from . import jax_finufft_gpu

    for _name, _value in jax_finufft_gpu.registrations().items():
        xla_client.register_custom_call_target(_name, _value, platform="gpu")
except ImportError:
    jax_finufft_gpu = None

for _name, _value in jax_finufft_cpu.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="cpu")


# Handle old versions of jax which had a different syntax for custom_call
def custom_call(*args, **kwargs):
    try:
        return _custom_call(*args, **kwargs).results
    except TypeError:
        kwargs["out_types"] = kwargs.pop("result_types")
        return (_custom_call(*args, **kwargs),)


def default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]


def lowering(platform, ctx, source, *points, output_shape, iflag, eps, opts):
    del ctx

    if platform not in ["cpu", "gpu"]:
        raise ValueError(f"Unrecognized platform '{platform}'")

    if platform == "gpu" and jax_finufft_gpu is None:
        raise ValueError("jax-finufft was not compiled with GPU support")

    ndim = len(points)
    assert 1 <= ndim <= 3
    if platform == "gpu" and ndim == 1:
        raise ValueError("1-D transforms are not yet supported on the GPU")

    source_type = ir.RankedTensorType(source.type)
    points_type = [ir.RankedTensorType(x.type) for x in points]

    # Check supported and consistent dtypes
    f32 = ir.F32Type.get()
    f64 = ir.F64Type.get()
    source_dtype = source_type.element_type
    single = source_dtype == ir.ComplexType.get(f32) and all(
        x.element_type == f32 for x in points_type
    )
    double = source_dtype == ir.ComplexType.get(f64) and all(
        x.element_type == f64 for x in points_type
    )
    assert single or double
    suffix = "f" if single else ""

    # Check shapes
    source_shape = source_type.shape
    points_shape = tuple(x.shape for x in points_type)
    n_tot = source_shape[0]
    n_transf = source_shape[1]
    n_j = points_shape[0][1]
    if output_shape is None:
        op_name = f"nufft{ndim}d2{suffix}".encode("ascii")
        n_k = np.array(source_shape[2:], dtype=np.int64)
        full_output_shape = tuple(source_shape[:2]) + (n_j,)
    else:
        op_name = f"nufft{ndim}d1{suffix}".encode("ascii")
        n_k = np.array(output_shape, dtype=np.int64)
        full_output_shape = tuple(source_shape[:2]) + tuple(output_shape)

    # The backend expects the output shape in Fortran order, so we'll just
    # fake it here, by sending in n_k and x in the reverse order.
    n_k_full = np.zeros(3, dtype=np.int64)
    n_k_full[:ndim] = n_k[::-1]

    if opts is None:
        opts = options.Opts()
    opts = options.unpack_opts(opts, 2 if output_shape is None else 1, True)
    assert isinstance(opts, options.Opts)

    if platform == "cpu":
        opts = opts.to_finufft_opts()
        opaque = getattr(jax_finufft_cpu, f"build_descriptor{suffix}")(
            eps, iflag, n_tot, n_transf, n_j, *n_k_full, opts
        )
        opaque_arg = hlo_const(np.frombuffer(opaque, dtype=np.uint8))
        opaque_shape = ir.RankedTensorType(opaque_arg.type).shape
        return custom_call(
            op_name,
            result_types=[
                ir.RankedTensorType.get(full_output_shape, source_type.element_type)
            ],
            # Reverse points because backend uses Fortran order
            operands=[opaque_arg, source, *points[::-1]],
            operand_layouts=default_layouts(
                opaque_shape, source_shape, *points_shape[::-1]
            ),
            result_layouts=default_layouts(full_output_shape),
        )

    else:
        opts = opts.to_cufinufft_opts()
        opaque = getattr(jax_finufft_gpu, f"build_descriptor{suffix}")(
            eps, iflag, n_tot, n_transf, n_j, *n_k_full, opts
        )
        return custom_call(
            op_name,
            result_types=[
                ir.RankedTensorType.get(full_output_shape, source_type.element_type)
            ],
            # Reverse points because backend uses Fortran order
            operands=[source, *points[::-1]],
            backend_config=opaque,
            operand_layouts=default_layouts(source_shape, *points_shape[::-1]),
            result_layouts=default_layouts(full_output_shape),
        )
