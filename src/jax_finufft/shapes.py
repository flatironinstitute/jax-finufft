__all__ = ["abstract_eval", "broadcast_and_flatten_inputs"]

from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp
import numpy as np
from jax import dtypes


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


def broadcast_and_flatten_inputs(nufft_type, output_shape, source, *points):
    # This function searches for "points" dimensions that are broadcast over
    # "source" dimensions. These dimensions are packed together so they can use
    # finufft batch mode.

    if nufft_type == 3:
        num_dim = len(points) // 2
        points3 = points[num_dim:]  # type-3 target points
        points = points[:num_dim]
    else:
        num_dim = len(points)
    assert num_dim

    # Coerce the points into the appropriate shape
    points = jnp.broadcast_arrays(*points)
    *input_shape, num_points = points[0].shape

    # Broadcast source points and target points, except the number of points
    # is allowed to be different
    if nufft_type == 3:
        points3 = jnp.broadcast_arrays(*points3)
        *input_shape3, num_points3 = points3[0].shape
        input_shape = jnp.broadcast_shapes(input_shape, input_shape3)
        del input_shape3
        points = [jnp.broadcast_to(p, input_shape + (num_points,)) for p in points]
        points3 = [jnp.broadcast_to(p, input_shape + (num_points3,)) for p in points3]
    else:
        points3 = []

    # Handle unpadded points
    if (
        nufft_type == 2
        and source.ndim == len(input_shape) + num_dim + 1
        or (nufft_type in (1, 3) and source.ndim == len(input_shape) + 2)
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
        points3 = tuple(jnp.moveaxis(p, broadcast_from, broadcast_to) for p in points3)

    # Compute some dimensions of the flattened problem
    num_in = len(target_shape)
    num_axes = num_in - len(broadcast_from)
    size_in = np.prod(source.shape[:num_axes], dtype=int)
    size_bcast = np.prod(source.shape[num_axes:num_in], dtype=int)

    if nufft_type == 3:
        assert source.ndim == len(target_shape) + 1
        assert source.shape[-1] == num_points
        expected_output_shape = source.shape[:num_in] + (num_points3,)
        source_extra_shape = (num_points,)
    elif nufft_type == 2:
        assert source.ndim == num_in + num_dim
        expected_output_shape = source.shape[:num_in] + (num_points,)
        source_extra_shape = source.shape[num_in:]
    elif nufft_type == 1:
        assert source.ndim == len(target_shape) + 1
        assert source.shape[-1] == num_points
        assert output_shape is not None
        expected_output_shape = source.shape[:num_in] + tuple(output_shape)
        source_extra_shape = (num_points,)

    # Flatten all the inputs
    source = jnp.reshape(source, (size_in, size_bcast) + source_extra_shape)
    points = tuple(jnp.reshape(p, (size_in, num_points)) for p in points)
    points3 = tuple(jnp.reshape(p, (size_in, num_points3)) for p in points3)

    return (
        BroadcastIndex(
            broadcast_from=broadcast_from,
            broadcast_to=broadcast_to,
            expected_output_shape=expected_output_shape,
        ),
        source,
        *points,
        *points3,
    )


def abstract_eval(source, *points, output_shape, nufft_type, **_):
    if nufft_type == 3:
        ndim = len(points) // 2
    else:
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
    assert all(p.shape == points[0].shape for p in points[1:ndim])
    assert source.shape[0] == points[0].shape[0]

    if nufft_type == 3:
        assert source.ndim == 3
        assert all(p.shape == points[ndim].shape for p in points[ndim + 1 :])
        assert all(
            p.shape[:-1] == p3.shape[:-1]
            for (p, p3) in zip(points[:ndim], points[ndim:])
        )
        return source.update(
            shape=source.shape[:2] + (points[ndim].shape[-1],), dtype=source_dtype
        )

    elif nufft_type == 2:
        assert source.ndim == 2 + ndim
        return source.update(
            shape=source.shape[:2] + (points[0].shape[-1],), dtype=source_dtype
        )
    elif nufft_type == 1:
        assert source.ndim == 3
        assert source.shape[2] == points[0].shape[1]
        return source.update(
            shape=source.shape[:2] + tuple(output_shape), dtype=source_dtype
        )
