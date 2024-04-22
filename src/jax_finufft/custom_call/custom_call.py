import dataclasses
from typing import Any, Dict

from jax import core, tree_util
from jax.interpreters import mlir
from jax.interpreters.mlir import ir
from jax.lib import xla_client
import jaxlib.mlir.dialects.stablehlo as hlo

from jax_finufft.custom_call import custom_call_lib

xla_client.register_custom_call_target(
    "_custom_call", custom_call_lib.get_custom_call_wrapper(), platform="cpu"
)

custom_call_p = core.Primitive("custom_call")
custom_call_p.multiple_results = True


def _custom_call_abstract_eval(*avals, out_avals, name, **_):
    del avals
    return out_avals


custom_call_p.def_abstract_eval(_custom_call_abstract_eval)


def _custom_call_lowering_rule(
    ctx: mlir.LoweringRuleContext, *in_nodes, out_avals, name, descriptor
):
    platform = ctx.module_context.platform
    custom_call = _CUSTOM_CALL_REGISTRY[name]
    function_ptr = custom_call._registry[platform]
    custom_call_descriptor, keepalive = custom_call_lib.make_descriptor(
        function_ptr, descriptor
    )
    ctx.module_context.add_keepalive(keepalive)
    ctx.module_context.add_keepalive(descriptor)
    out_type = ir.TupleType.get_tuple(
        [
            ir.RankedTensorType.get(
                out_aval.shape, mlir.dtype_to_ir_type(out_aval.dtype)
            )
            for out_aval in out_avals
        ]
    )
    i32_type = ir.IntegerType.get_signless(32)
    descriptor_operand = mlir.ir_constant(
        custom_call_descriptor, canonicalize_types=False
    )
    in_nodes = [descriptor_operand, *in_nodes]
    out = hlo.CustomCallOp(
        [out_type],
        in_nodes,
        call_target_name=ir.StringAttr.get("jax_ffi_call"),
        has_side_effect=ir.BoolAttr.get(False),
        backend_config=ir.StringAttr.get(""),
        api_version=ir.IntegerAttr.get(i32_type, 2),
        called_computations=ir.ArrayAttr.get([]),
    )
    return [
        hlo.GetTupleElementOp(out, mlir.i32_attr(i)).result
        for i in range(len(out_avals))
    ]


mlir.register_lowering(custom_call_p, _custom_call_lowering_rule, platform="cpu")


@dataclasses.dataclass
class CustomCall:
    name: str
    _registry: Dict[str, Any] = dataclasses.field(init=False, default_factory=dict)

    def register(self, function_ptr, *, platform):
        platform = platform.lower()
        if platform != "cpu":
            raise NotImplementedError(platform)
        self._registry[platform] = function_ptr

    def __call__(self, *args, out_shape_dtype, descriptor=None):
        flat_out_shape_dtype, out_tree = tree_util.tree_flatten(out_shape_dtype)
        out_avals = [core.ShapedArray(a.shape, a.dtype) for a in flat_out_shape_dtype]
        out_flat = custom_call_p.bind(
            *args, out_avals=out_avals, descriptor=descriptor, name=self.name
        )
        return tree_util.tree_unflatten(out_tree, out_flat)


_CUSTOM_CALL_REGISTRY: Dict[str, CustomCall] = {}


def custom_call(name: str):
    if name in _CUSTOM_CALL_REGISTRY:
        raise ValueError(f"Custom call name already used: {name}")
    _CUSTOM_CALL_REGISTRY[name] = cc = CustomCall(name)
    return cc
