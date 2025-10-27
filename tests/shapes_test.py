import jax.numpy as jnp

from jax_finufft import shapes


def test_broadcast_inputs():
    index, source, *points = shapes.broadcast_and_flatten_inputs(
        1, (10,), jnp.empty((50, 5, 1, 7, 12, 5)), jnp.empty((50, 1, 6, 1, 12, 5))
    )
    assert source.shape == (50 * 6 * 12, 5 * 7, 5)
    assert len(points) == 1
    assert points[0].shape == (50 * 6 * 12, 5)
    assert index.broadcast_from == (1, 3)
    assert index.broadcast_to == (3, 4)
    assert index.expected_output_shape == (50, 6, 12, 5, 7, 10)


def test_broadcast_inputs_type2():
    index, source, *points = shapes.broadcast_and_flatten_inputs(
        2, None, jnp.empty((50, 5, 1, 7, 12, 11)), jnp.empty((50, 1, 6, 1, 12, 5))
    )
    assert source.shape == (50 * 6 * 12, 5 * 7, 11)
    assert len(points) == 1
    assert points[0].shape == (50 * 6 * 12, 5)
    assert index.broadcast_from == (1, 3)
    assert index.broadcast_to == (3, 4)
    assert index.expected_output_shape == (50, 6, 12, 5, 7, 5)


def test_vector_inputs():
    index, source, *points = shapes.broadcast_and_flatten_inputs(
        2, None, jnp.empty((5, 6)), jnp.empty((10,)), jnp.empty((10,))
    )
    assert source.shape == (1, 1, 5, 6)
    assert len(points) == 2
    assert points[0].shape == (1, 10)
    assert points[1].shape == (1, 10)
    assert index.expected_output_shape == (10,)
    assert index.broadcast_from == ()
    assert index.broadcast_to == ()


def test_unpadded_points_type2():
    index, source, *points = shapes.broadcast_and_flatten_inputs(
        2, None, jnp.empty((3, 8, 5, 6)), jnp.empty((3, 10)), jnp.empty((3, 10))
    )
    assert source.shape == (3, 8, 5, 6)
    assert len(points) == 2
    assert points[0].shape == (3, 10)
    assert points[1].shape == (3, 10)
    assert index.expected_output_shape == (3, 8, 10)
    assert index.broadcast_from == (1,)
    assert index.broadcast_to == (1,)


def test_unpadded_points_type1():
    index, source, *points = shapes.broadcast_and_flatten_inputs(
        1, (17, 13), jnp.empty((3, 5, 6)), jnp.empty((1, 6)), jnp.empty((1, 6))
    )
    assert source.shape == (1, 3 * 5, 6)
    assert len(points) == 2
    assert points[0].shape == (1, 6)
    assert points[1].shape == (1, 6)
    assert index.expected_output_shape == (3, 5, 17, 13)
    assert index.broadcast_from == (0, 1)
    assert index.broadcast_to == (0, 1)


def test_unpadded_points_type3():
    index, source, *points = shapes.broadcast_and_flatten_inputs(
        3,
        None,
        jnp.empty((3, 5, 6)),
        jnp.empty((3, 6)),
        jnp.empty((3, 6)),
        jnp.empty((3, 11)),
        jnp.empty((3, 11)),
    )
    assert source.shape == (3, 5, 6)
    assert len(points) == 4
    assert points[0].shape == (3, 6)
    assert points[1].shape == (3, 6)
    assert points[2].shape == (3, 11)
    assert points[3].shape == (3, 11)
    assert index.expected_output_shape == (3, 5, 11)
    assert index.broadcast_from == (1,)
    assert index.broadcast_to == (1,)


def test_type3():
    # fmt: off
    index, source, *points = shapes.broadcast_and_flatten_inputs(
        3,
        None,
        jnp.empty((13, 3, 5, 7, 11)),
        jnp.empty((13, 1, 5, 1, 11)),
        jnp.empty((    1, 5, 1, 23)),
    )
    # fmt: on

    assert source.shape == (13 * 5, 3 * 7, 11)
    assert len(points) == 2
    assert points[0].shape == (13 * 5, 11)
    assert points[1].shape == (13 * 5, 23)
    assert index.expected_output_shape == (13, 5, 3, 7, 23)
    assert index.broadcast_from == (1, 3)
    assert index.broadcast_to == (2, 3)


def test_type3_2d():
    # fmt: off
    index, source, *points = shapes.broadcast_and_flatten_inputs(
        3,
        None,
        jnp.empty((13, 3, 5, 7, 11)),
        jnp.empty((13, 1, 5, 1, 11)),
        jnp.empty((    1, 5, 1, 11)),
        jnp.empty((       5, 1, 23)),
        jnp.empty((          1, 23)),
    )
    # fmt: on

    assert source.shape == (13 * 5, 3 * 7, 11)
    assert len(points) == 4
    assert points[0].shape == (13 * 5, 11)
    assert points[1].shape == (13 * 5, 11)
    assert points[2].shape == (13 * 5, 23)
    assert points[3].shape == (13 * 5, 23)
    assert index.expected_output_shape == (13, 5, 3, 7, 23)
    assert index.broadcast_from == (1, 3)
    assert index.broadcast_to == (2, 3)
