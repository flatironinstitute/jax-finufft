import jax.numpy as jnp

from jax_finufft import shapes


def test_broadcast_inputs():
    index, source, *points = shapes.broadcast_and_flatten_inputs(
        (10,), jnp.empty((50, 5, 1, 7, 12, 5)), jnp.empty((50, 1, 6, 1, 12, 5))
    )
    assert source.shape == (50 * 6 * 12, 5 * 7, 5)
    assert len(points) == 1
    assert points[0].shape == (50 * 6 * 12, 5)
    assert index.broadcast_from == (1, 3)
    assert index.broadcast_to == (3, 4)


def test_vector_inputs():
    index, source, *points = shapes.broadcast_and_flatten_inputs(
        None, jnp.empty((5, 6)), jnp.empty((10,)), jnp.empty((10,))
    )
    assert source.shape == (1, 1, 5, 6)
    assert len(points) == 2
    assert points[0].shape == (1, 10)
    assert points[1].shape == (1, 10)
    assert index.expected_output_shape == (10,)
    assert index.broadcast_from == ()
    assert index.broadcast_to == ()


def test_unpadded_points():
    index, source, *points = shapes.broadcast_and_flatten_inputs(
        None, jnp.empty((3, 8, 5, 6)), jnp.empty((3, 10)), jnp.empty((3, 10))
    )
    assert source.shape == (3, 8, 5, 6)
    assert len(points) == 2
    assert points[0].shape == (3, 10)
    assert points[1].shape == (3, 10)
    assert index.expected_output_shape == (
        3,
        8,
        10,
    )
    assert index.broadcast_from == (1,)
    assert index.broadcast_to == (1,)
