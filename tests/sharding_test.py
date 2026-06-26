"""Correctness of the NUFFT transforms under ``jax.shard_map``.

These tests exercise the data-parallel patterns over the nonuniform
points: the points (and coefficients) are sharded across the mesh while the
uniform grid is replicated, and the result is reduced across devices with
``jax.lax.psum``. The forward transform shards trivially -- each device runs a
local transform on its slice of points -- but the gradient of a *replicated*
input requires a cross-device ``psum`` that is only inserted when JAX tracks
varying manual axes (vma) or registers a rep rule. Without it the forward is
correct but the gradient is silently wrong.
"""

from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import test_util as jtu
from jax.sharding import NamedSharding, PartitionSpec as P

from jax_finufft import nufft1, nufft2, nufft3

# Must run before the backend is initialized, so keep it at import time.
jtu.request_cpu_devices(2)

try:
    from jax import shard_map  # jax >= 0.10
except ImportError:
    from jax.experimental.shard_map import shard_map


def check_close(a, b, **kwargs):
    kwargs["rtol"] = kwargs.get("rtol", {"complex128": 1e-7, "complex64": 1e-4})
    return jtu.check_close(a, b, **kwargs)


if jax.version.__version_info__ < (0, 8, 0):
    enable_x64 = jax.experimental.enable_x64
else:
    enable_x64 = jax.enable_x64


# All tests in this module need a multi-device mesh.
pytestmark = pytest.mark.skipif(jax.device_count() < 2, reason="requires >=2 devices")


def _mesh():
    # Plain Mesh (Auto axis types); shard_map then enters manual mode. Avoid
    # jax.make_mesh, which now defaults to Explicit axes (sharding-in-types).
    return jax.sharding.Mesh(np.array(jax.devices()), ("s",))


def _check(single, sharded, primal, primal_spec, mesh):
    """Compare value_and_grad of a single-device loss against its shard_map'd
    counterpart, differentiating with respect to ``primal``."""
    v0, g0 = jax.value_and_grad(single)(primal)
    primal = jax.device_put(primal, NamedSharding(mesh, primal_spec))
    v1, g1 = jax.value_and_grad(sharded)(primal)
    check_close(v1, v0)  # forward (correct even without the fix)
    check_close(g1, g0)  # gradient (wrong without the fix)


@pytest.mark.parametrize("ndim, iflag", product([1, 2, 3], [-1, 1]))
def test_nufft2_shard_map_grad(ndim, iflag):
    # Data-parallel type 2: grid replicated, points + data sharded (the #226 repro).
    # Differentiate w.r.t. the *replicated* grid -- this is the reported bug.
    random = np.random.default_rng(657)
    eps = 1e-10
    n = jax.device_count()
    num_nonuniform = 32 * n
    num_uniform = tuple(4 + np.arange(ndim))

    x = [
        jnp.asarray(random.uniform(-np.pi, np.pi, size=num_nonuniform))
        for _ in range(ndim)
    ]
    grid = jnp.asarray(
        random.normal(size=num_uniform) + 1j * random.normal(size=num_uniform)
    )
    data = jnp.asarray(
        random.normal(size=num_nonuniform) + 1j * random.normal(size=num_nonuniform)
    )

    def chi(g, *pts, d):
        return jnp.sum(jnp.abs(nufft2(g, *pts, iflag=iflag, eps=eps) - d) ** 2)

    with enable_x64():
        mesh = _mesh()
        rows = NamedSharding(mesh, P("s"))
        xs = [jax.device_put(xi, rows) for xi in x]
        ds = jax.device_put(data, rows)

        def local(g, *pts_and_data):
            *pts, d = pts_and_data
            return jax.lax.psum(chi(g, *pts, d=d), "s")

        sm = shard_map(
            local,
            mesh=mesh,
            in_specs=(P(),) + (P("s"),) * (ndim + 1),
            out_specs=P(),
        )

        _check(
            single=lambda g: chi(g, *x, d=data),
            sharded=lambda g: sm(g, *xs, ds),
            primal=grid,
            primal_spec=P(),
            mesh=mesh,
        )


@pytest.mark.parametrize("ndim, iflag", product([1, 2, 3], [-1, 1]))
def test_nufft1_shard_map_grad(ndim, iflag):
    # Data-parallel type 1: coeffs + points sharded; the type-1 sum over points is
    # additive across shards, so the local results are combined with psum. The
    # transpose of type 1 is a replicated->sharded type 2, exercising the vma path
    # in the reverse direction. Differentiate w.r.t. the sharded coeffs.
    random = np.random.default_rng(657)
    eps = 1e-10
    n = jax.device_count()
    num_nonuniform = 32 * n
    num_uniform = tuple(4 + np.arange(ndim))

    x = [
        jnp.asarray(random.uniform(-np.pi, np.pi, size=num_nonuniform))
        for _ in range(ndim)
    ]
    c = jnp.asarray(
        random.normal(size=num_nonuniform) + 1j * random.normal(size=num_nonuniform)
    )
    target = jnp.asarray(
        random.normal(size=num_uniform) + 1j * random.normal(size=num_uniform)
    )

    with enable_x64():
        mesh = _mesh()
        rows = NamedSharding(mesh, P("s"))
        xs = [jax.device_put(xi, rows) for xi in x]

        def single(c):
            f = nufft1(num_uniform, c, *x, iflag=iflag, eps=eps)
            return jnp.sum(jnp.abs(f - target) ** 2)

        def local(c, *pts):
            f = jax.lax.psum(nufft1(num_uniform, c, *pts, iflag=iflag, eps=eps), "s")
            return jnp.sum(jnp.abs(f - target) ** 2)

        sm = shard_map(
            local,
            mesh=mesh,
            in_specs=(P("s"),) * (ndim + 1),
            out_specs=P(),
        )

        _check(
            single=single,
            sharded=lambda c: sm(c, *xs),
            primal=c,
            primal_spec=P("s"),
            mesh=mesh,
        )


@pytest.mark.parametrize("ndim, iflag", product([1, 2, 3], [-1, 1]))
def test_nufft3_shard_map_grad(ndim, iflag):
    # Data-parallel type 3: source coeffs + source points sharded, target points
    # replicated. Each shard contributes all targets; combined with psum. The
    # replicated target points force a pvary up to the sharded axis inside the
    # transform. Differentiate w.r.t. the sharded coeffs.
    random = np.random.default_rng(657)
    eps = 1e-10
    n = jax.device_count()
    num_source = 32 * n
    num_target = 20

    x = [jnp.asarray(random.uniform(-1.0, 1.0, size=num_source)) for _ in range(ndim)]
    s = [jnp.asarray(random.uniform(-1.0, 1.0, size=num_target)) for _ in range(ndim)]
    c = jnp.asarray(
        random.normal(size=num_source) + 1j * random.normal(size=num_source)
    )
    target = jnp.asarray(
        random.normal(size=num_target) + 1j * random.normal(size=num_target)
    )

    with enable_x64():
        mesh = _mesh()
        rows = NamedSharding(mesh, P("s"))
        repl = NamedSharding(mesh, P())
        xs = [jax.device_put(xi, rows) for xi in x]
        ss = [jax.device_put(si, repl) for si in s]

        def single(c):
            f = nufft3(c, *x, *s, iflag=iflag, eps=eps)
            return jnp.sum(jnp.abs(f - target) ** 2)

        def local(c, *pts):
            xp, sp = pts[:ndim], pts[ndim:]
            f = jax.lax.psum(nufft3(c, *xp, *sp, iflag=iflag, eps=eps), "s")
            return jnp.sum(jnp.abs(f - target) ** 2)

        sm = shard_map(
            local,
            mesh=mesh,
            in_specs=(P("s"),) + (P("s"),) * ndim + (P(),) * ndim,
            out_specs=P(),
        )

        _check(
            single=single,
            sharded=lambda c: sm(c, *xs, *ss),
            primal=c,
            primal_spec=P("s"),
            mesh=mesh,
        )
