__all__ = ["abstract_eval", "broadcast_and_flatten_inputs"]

from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp
import numpy as np
from jax import dtypes
from jax.abstract_arrays import ShapedArray


@dataclass
class BroadcastIndex:
    broadcast_from: Sequence[int]
    broadcast_to: Sequence[int]
    expected_output_shape: Sequence[int]

    def unflatten(self, result):
        if len(self.broadcast_to):
            return jnp.moveaxis(
                jnp.reshape(result, self.expected_output_shape),
                self.broadcast_to,
                self.broadcast_from,
            )
        return jnp.reshape(result, self.expected_output_shape)


def broadcast_and_flatten_inputs(output_shape, source, *points):
    num_dim = len(points)
    assert num_dim

    # Coerce the points into the appropriate shape
    points = jnp.broadcast_arrays(*points)
    *input_shape, num_points = points[0].shape

    # Handle unpadded points
    if (
        output_shape is None
        and source.ndim == len(input_shape) + num_dim + 1
        or (output_shape is not None and source.ndim == len(input_shape) + 2)
    ):
        input_shape = tuple(input_shape) + (1,)
        points = tuple(p[..., None, :] for p in points)

    # Work out a consistent shape for the broadcastable dimensions
    target_shape = jnp.broadcast_shapes(source.shape[: len(input_shape)], input_shape)

    # Find the broadcasted dimensions
    broadcast_from = tuple(
        n
        for n, (input_dim, target_dim) in enumerate(zip(input_shape, target_shape))
        if input_dim != target_dim
    )
    broadcast_to = tuple(
        len(target_shape) - len(broadcast_from) + n for n in range(len(broadcast_from))
    )

    # We expect all the broadcasted dimensions to have size 1; do we need to check?
    assert all(input_shape[n] == 1 for n in broadcast_from)

    # Broadcast the source term appropriately
    source = jnp.broadcast_to(source, target_shape + source.shape[len(target_shape) :])

    # Move the broadcasted dimensions to the middle
    if len(broadcast_to):
        source = jnp.moveaxis(source, broadcast_from, broadcast_to)
        points = tuple(jnp.moveaxis(p, broadcast_from, broadcast_to) for p in points)

    # Compute some dimensions of the flattened problem
    num_in = len(target_shape)
    num_axes = num_in - len(broadcast_from)
    size_in = np.prod(source.shape[:num_axes], dtype=int)
    size_bcast = np.prod(source.shape[num_axes:num_in], dtype=int)

    if output_shape is None:
        # Type 2
        assert source.ndim == num_in + num_dim
        expected_output_shape = source.shape[:num_in] + (num_points,)
        source_extra_shape = source.shape[num_in:]

    else:
        # Type 1
        assert source.ndim == len(target_shape) + 1
        assert source.shape[-1] == num_points
        expected_output_shape = source.shape[:num_in] + tuple(output_shape)
        source_extra_shape = (num_points,)

    # Flatten all the inputs
    source = jnp.reshape(source, (size_in, size_bcast) + source_extra_shape)
    points = tuple(jnp.reshape(p, (size_in, num_points)) for p in points)

    return (
        BroadcastIndex(
            broadcast_from=broadcast_from,
            broadcast_to=broadcast_to,
            expected_output_shape=expected_output_shape,
        ),
        source,
        *points,
    )


def abstract_eval(source, *points, output_shape, **_):
    ndim = len(points)
    assert 1 <= ndim <= 3

    source_dtype = dtypes.canonicalize_dtype(source.dtype)
    points_dtype = [dtypes.canonicalize_dtype(x.dtype) for x in points]

    # Check supported and consistent dtypes
    single = source_dtype == np.csingle and all(x == np.single for x in points_dtype)
    double = source_dtype == np.cdouble and all(x == np.double for x in points_dtype)
    assert single or double

    # Check that the inputs have the right shapes
    assert all(p.ndim == 2 for p in points)
    assert all(p.shape == points[0].shape for p in points[1:])
    assert source.shape[0] == points[0].shape[0]
    if output_shape is None:
        assert source.ndim == 2 + ndim
        return ShapedArray(source.shape[:2] + (points[0].shape[-1],), source_dtype)
    else:
        assert source.ndim == 3
        assert source.shape[2] == points[0].shape[1]
        return ShapedArray(source.shape[:2] + tuple(output_shape), source_dtype)
