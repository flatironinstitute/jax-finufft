from functools import partial
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import test_util as jtu
from utils import TARGET_EPS, check_accurate, check_close, conditioned_eps

from jax_finufft import nufft1, nufft2, nufft3
from jax_finufft.options import Opts

if jax.version.__version_info__ < (0, 8, 0):
    enable_x64 = jax.experimental.enable_x64
else:
    enable_x64 = jax.enable_x64


@pytest.mark.parametrize("iflag", [-1, 1], ids=["im1", "ip1"])
@pytest.mark.parametrize("num_uniform", [75], ids=["N75"])
@pytest.mark.parametrize("num_nonnuniform", [50], ids=["M50"])
@pytest.mark.parametrize("x64", [False, True], ids=["f32", "f64"])
@pytest.mark.parametrize("ndim", [1, 2, 3], ids=["1D", "2D", "3D"])
def test_nufft1_forward(ndim, x64, num_nonnuniform, num_uniform, iflag):
    random = np.random.default_rng(657)

    dtype = np.double if x64 else np.single
    cdtype = np.cdouble if x64 else np.csingle

    num_uniform = tuple(num_uniform // ndim + 5 * np.arange(ndim))
    eps = conditioned_eps(x64, num_uniform)
    ks = [np.arange(-np.floor(n / 2), np.floor((n - 1) / 2 + 1)) for n in num_uniform]

    x = random.uniform(-np.pi, np.pi, size=(ndim, num_nonnuniform)).astype(dtype)
    c = random.normal(size=num_nonnuniform) + 1j * random.normal(size=num_nonnuniform)
    c = c.astype(cdtype)
    f_expect = np.zeros(num_uniform, dtype=cdtype)
    for coords in product(*map(range, num_uniform)):
        k_vec = np.array([k[n] for (n, k) in zip(coords, ks)])
        f_expect[coords] = np.sum(c * np.exp(iflag * 1j * np.dot(k_vec, x)))

    with enable_x64(x64):
        f_calc = nufft1(num_uniform, c, *x, eps=eps, iflag=iflag)
        check_accurate(f_calc, f_expect, eps)

        f_calc = jax.jit(nufft1, static_argnums=(0,), static_argnames=("eps", "iflag"))(
            num_uniform, c, *x, eps=eps, iflag=iflag
        )
        check_accurate(f_calc, f_expect, eps)


@pytest.mark.parametrize("iflag", [-1, 1], ids=["im1", "ip1"])
@pytest.mark.parametrize("num_uniform", [75], ids=["N75"])
@pytest.mark.parametrize("num_nonnuniform", [50], ids=["M50"])
@pytest.mark.parametrize("x64", [False, True], ids=["f32", "f64"])
@pytest.mark.parametrize("ndim", [1, 2, 3], ids=["1D", "2D", "3D"])
def test_nufft2_forward(ndim, x64, num_nonnuniform, num_uniform, iflag):
    random = np.random.default_rng(657)

    dtype = np.double if x64 else np.single
    cdtype = np.cdouble if x64 else np.csingle

    num_uniform = tuple(num_uniform // ndim + 5 * np.arange(ndim))
    eps = conditioned_eps(x64, num_uniform)
    ks = [np.arange(-np.floor(n / 2), np.floor((n - 1) / 2 + 1)) for n in num_uniform]
    x = [
        random.uniform(-np.pi, np.pi, size=num_nonnuniform).astype(dtype)
        for _ in range(ndim)
    ]
    f = random.normal(size=num_uniform) + 1j * random.normal(size=num_uniform)
    f = f.astype(cdtype)

    c_expect = np.zeros(num_nonnuniform, dtype=cdtype)
    for n in range(num_nonnuniform):
        arg = np.copy(f)
        for i, k in enumerate(ks):
            coords = [None for _ in range(ndim)]
            coords[i] = slice(None)
            arg *= np.exp(iflag * 1j * k * x[i][n])[tuple(coords)]
        c_expect[n] = np.sum(arg)

    with enable_x64(x64):
        c_calc = nufft2(f, *x, eps=eps, iflag=iflag)
        check_accurate(c_calc, c_expect, eps)

        c_calc = jax.jit(nufft2, static_argnames=("eps", "iflag"))(
            f, *x, eps=eps, iflag=iflag
        )
        check_accurate(c_calc, c_expect, eps)


@pytest.mark.parametrize("iflag", [-1, 1], ids=["im1", "ip1"])
@pytest.mark.parametrize("num_targets", [20], ids=["N20"])
@pytest.mark.parametrize("num_sources", [25], ids=["M25"])
@pytest.mark.parametrize("x64", [False, True], ids=["f32", "f64"])
@pytest.mark.parametrize("ndim", [1, 2, 3], ids=["1D", "2D", "3D"])
def test_nufft3_forward(ndim, x64, num_sources, num_targets, iflag):
    random = np.random.default_rng(657)

    # The conditioning rule keys off the space-bandwidth product, but these
    # points span so little of it that FINUFFT's minimum fine grid sets the
    # floor instead, so pin a per-precision value the way upstream does.
    eps = TARGET_EPS if x64 else 1e-5
    dtype = np.double if x64 else np.single
    cdtype = np.cdouble if x64 else np.csingle

    x = [random.uniform(-1.0, 1.0, size=num_sources).astype(dtype) for _ in range(ndim)]
    s = [random.uniform(-1.0, 1.0, size=num_targets).astype(dtype) for _ in range(ndim)]
    c = random.normal(size=num_sources) + 1j * random.normal(size=num_sources)
    c = c.astype(cdtype)

    f_expect = np.zeros(num_targets, dtype=cdtype)
    for k in range(num_targets):
        s_vec = np.array([s_dim[k] for s_dim in s])
        x_mat = np.array([x_dim for x_dim in x])
        f_expect[k] = np.sum(c * np.exp(iflag * 1j * np.dot(s_vec, x_mat)))

    with enable_x64(x64):
        f_calc = nufft3(c, *x, *s, eps=eps, iflag=iflag)
        check_accurate(f_calc, f_expect, eps)

        f_calc = jax.jit(nufft3, static_argnames=("eps", "iflag"))(
            c, *x, *s, eps=eps, iflag=iflag
        )
        check_accurate(f_calc, f_expect, eps)


@pytest.mark.parametrize("iflag", [-1, 1], ids=["im1", "ip1"])
@pytest.mark.parametrize("num_uniform", [35], ids=["N35"])
@pytest.mark.parametrize("num_nonnuniform", [50], ids=["M50"])
@pytest.mark.parametrize("ndim", [1, 2, 3], ids=["1D", "2D", "3D"])
def test_nufft1_grad(ndim, num_nonnuniform, num_uniform, iflag):
    random = np.random.default_rng(657)

    eps = TARGET_EPS
    dtype = np.double
    cdtype = np.cdouble

    num_uniform = tuple(num_uniform // ndim + 5 * np.arange(ndim))

    x = [
        random.uniform(-np.pi, np.pi, size=num_nonnuniform).astype(dtype)
        for _ in range(ndim)
    ]
    c = random.normal(size=num_nonnuniform) + 1j * random.normal(size=num_nonnuniform)
    c = c.astype(cdtype)

    with enable_x64():
        func = partial(nufft1, num_uniform, eps=eps, iflag=iflag)
        jtu.check_grads(func, (c, *x), 1, modes=("fwd", "rev"))

        def scalar_func(*args):
            return jnp.linalg.norm(func(*args))

        expect = jax.grad(scalar_func, argnums=tuple(range(len(x) + 1)))(c, *x)
        for n, g in enumerate(expect):
            check_close(jax.grad(scalar_func, argnums=(n,))(c, *x)[0], g)


@pytest.mark.parametrize("iflag", [-1, 1], ids=["im1", "ip1"])
@pytest.mark.parametrize("num_uniform", [35], ids=["N35"])
@pytest.mark.parametrize("num_nonnuniform", [50], ids=["M50"])
@pytest.mark.parametrize("ndim", [1, 2, 3], ids=["1D", "2D", "3D"])
def test_nufft2_grad(ndim, num_nonnuniform, num_uniform, iflag):
    random = np.random.default_rng(657)

    eps = TARGET_EPS
    dtype = np.double
    cdtype = np.cdouble

    num_uniform = tuple(num_uniform // ndim + 5 * np.arange(ndim))

    x = [
        random.uniform(-np.pi, np.pi, size=num_nonnuniform).astype(dtype)
        for _ in range(ndim)
    ]
    f = random.normal(size=num_uniform) + 1j * random.normal(size=num_uniform)
    f = f.astype(cdtype)

    with enable_x64():
        func = partial(nufft2, eps=eps, iflag=iflag)
        jtu.check_grads(func, (f, *x), 1, modes=("fwd", "rev"))

        def scalar_func(*args):
            return jnp.linalg.norm(func(*args))

        expect = jax.grad(scalar_func, argnums=tuple(range(len(x) + 1)))(f, *x)
        for n, g in enumerate(expect):
            check_close(jax.grad(scalar_func, argnums=(n,))(f, *x)[0], g)


@pytest.mark.parametrize("iflag", [-1, 1], ids=["im1", "ip1"])
@pytest.mark.parametrize("num_target", [35], ids=["N35"])
@pytest.mark.parametrize("num_source", [50], ids=["M50"])
@pytest.mark.parametrize("ndim", [1, 2, 3], ids=["1D", "2D", "3D"])
def test_nufft3_grad(ndim, num_source, num_target, iflag):
    random = np.random.default_rng(657)

    eps = TARGET_EPS
    dtype = np.double
    cdtype = np.cdouble

    x = [random.uniform(-1.0, 1.0, size=num_source).astype(dtype) for _ in range(ndim)]
    s = [random.uniform(-1.0, 1.0, size=num_target).astype(dtype) for _ in range(ndim)]
    c = random.normal(size=num_source) + 1j * random.normal(size=num_source)
    c = c.astype(cdtype)

    with enable_x64():
        func = partial(nufft3, eps=eps, iflag=iflag)
        jtu.check_grads(func, (c, *x, *s), 1, modes=("fwd", "rev"))

        def scalar_func(*args):
            return jnp.linalg.norm(func(*args))

        expect = jax.grad(scalar_func, argnums=tuple(range(len(x) + len(s) + 1)))(
            c, *x, *s
        )
        for n, g in enumerate(expect):
            check_close(jax.grad(scalar_func, argnums=(n,))(c, *x, *s)[0], g)


@pytest.mark.parametrize("iflag", [-1, 1], ids=["im1", "ip1"])
@pytest.mark.parametrize("num_uniform", [35], ids=["N35"])
@pytest.mark.parametrize("num_nonnuniform", [50], ids=["M50"])
@pytest.mark.parametrize("ndim", [1, 2, 3], ids=["1D", "2D", "3D"])
def test_nufft1_vmap(ndim, num_nonnuniform, num_uniform, iflag):
    random = np.random.default_rng(657)

    dtype = np.double
    cdtype = np.cdouble

    num_repeat = 5
    num_uniform = tuple(num_uniform // ndim + 5 * np.arange(ndim))

    x = [
        random.uniform(-np.pi, np.pi, size=(num_repeat, num_nonnuniform)).astype(dtype)
        for _ in range(ndim)
    ]
    c = random.normal(size=(num_repeat, num_nonnuniform)) + 1j * random.normal(
        size=(num_repeat, num_nonnuniform)
    )
    c = c.astype(cdtype)
    func = partial(nufft1, num_uniform, iflag=iflag)

    with enable_x64():
        # Start by checking the full basic vmap
        calc = jax.vmap(func)(c, *x)
        for n in range(num_repeat):
            check_close(calc[n], func(c[n], *(x_[n] for x_ in x)))

        # With different in_axes
        calc_ax = jax.vmap(func, in_axes=(1,) + (0,) * ndim)(jnp.moveaxis(c, 0, 1), *x)
        check_close(calc_ax, calc)

        # With unmapped source axis
        calc_unmap = jax.vmap(func, in_axes=(None,) + (0,) * ndim)(c[0], *x)
        for n in range(num_repeat):
            check_close(calc_unmap[n], func(c[0], *(x_[n] for x_ in x)))

        # With unmapped points axis
        calc_unmap_pt = jax.vmap(func, in_axes=(0,) + (0,) * (ndim - 1) + (None,))(
            c, *x[:-1], x[-1][0]
        )
        for n in range(num_repeat):
            check_close(
                calc_unmap_pt[n], func(c[n], *(x_[n] for x_ in x[:-1]), x[-1][0])
            )


@pytest.mark.parametrize("iflag", [-1, 1], ids=["im1", "ip1"])
@pytest.mark.parametrize("num_uniform", [35], ids=["N35"])
@pytest.mark.parametrize("num_nonnuniform", [50], ids=["M50"])
@pytest.mark.parametrize("ndim", [1, 2, 3], ids=["1D", "2D", "3D"])
def test_nufft2_vmap(ndim, num_nonnuniform, num_uniform, iflag):
    random = np.random.default_rng(657)

    dtype = np.double
    cdtype = np.cdouble

    num_repeat = 5
    num_uniform = tuple(num_uniform // ndim + 5 * np.arange(ndim))

    x = [
        random.uniform(-np.pi, np.pi, size=(num_repeat, num_nonnuniform)).astype(dtype)
        for _ in range(ndim)
    ]
    f = random.normal(size=(num_repeat,) + num_uniform) + 1j * random.normal(
        size=(num_repeat,) + num_uniform
    )
    f = f.astype(cdtype)
    func = partial(nufft2, iflag=iflag)

    with enable_x64():
        # Start by checking the full basic vmap
        calc = jax.vmap(func)(f, *x)
        for n in range(num_repeat):
            check_close(calc[n], func(f[n], *(x_[n] for x_ in x)))

        # With different in_axes
        calc_ax = jax.vmap(func, in_axes=(1,) + (0,) * ndim)(jnp.moveaxis(f, 0, 1), *x)
        check_close(calc_ax, calc)

        # With unmapped source axis
        calc_unmap = jax.vmap(func, in_axes=(None,) + (0,) * ndim)(f[0], *x)
        for n in range(num_repeat):
            check_close(calc_unmap[n], func(f[0], *(x_[n] for x_ in x)))

        # With unmapped points axis
        calc_unmap_pt = jax.vmap(func, in_axes=(0,) + (0,) * (ndim - 1) + (None,))(
            f, *x[:-1], x[-1][0]
        )
        for n in range(num_repeat):
            check_close(
                calc_unmap_pt[n], func(f[n], *(x_[n] for x_ in x[:-1]), x[-1][0])
            )


@pytest.mark.parametrize("iflag", [-1, 1], ids=["im1", "ip1"])
@pytest.mark.parametrize("num_target", [35], ids=["N35"])
@pytest.mark.parametrize("num_source", [50], ids=["M50"])
@pytest.mark.parametrize("ndim", [1, 2, 3], ids=["1D", "2D", "3D"])
def test_nufft3_vmap(ndim, num_source, num_target, iflag):
    random = np.random.default_rng(657)

    dtype = np.double
    cdtype = np.cdouble

    num_repeat = 5

    x = [
        random.uniform(-np.pi, np.pi, size=(num_repeat, num_source)).astype(dtype)
        for _ in range(ndim)
    ]
    s = [
        random.uniform(-1.0, 1.0, size=(num_repeat, num_target)).astype(dtype)
        for _ in range(ndim)
    ]
    c = random.normal(size=(num_repeat, num_source)) + 1j * random.normal(
        size=(num_repeat, num_source)
    )
    c = c.astype(cdtype)
    func = partial(nufft3, iflag=iflag)

    with enable_x64():
        # Start by checking the full basic vmap
        calc = jax.vmap(func)(c, *x, *s)
        for n in range(num_repeat):
            check_close(calc[n], func(c[n], *(x_[n] for x_ in x), *(s_[n] for s_ in s)))

        # With different in_axes
        calc_ax = jax.vmap(func, in_axes=(1,) + (0,) * 2 * ndim)(
            jnp.moveaxis(c, 0, 1), *x, *s
        )
        check_close(calc_ax, calc)

        # With unmapped source axis
        calc_unmap = jax.vmap(func, in_axes=(None,) + (0,) * 2 * ndim)(c[0], *x, *s)
        for n in range(num_repeat):
            check_close(
                calc_unmap[n], func(c[0], *(x_[n] for x_ in x), *(s_[n] for s_ in s))
            )

        # With unmapped points axis
        calc_unmap_pt = jax.vmap(
            func, in_axes=(0,) + 2 * ((0,) * (ndim - 1) + (None,))
        )(c, *x[:-1], x[-1][0], *s[:-1], s[-1][0])
        for n in range(num_repeat):
            check_close(
                calc_unmap_pt[n],
                func(
                    c[n],
                    *(x_[n] for x_ in x[:-1]),
                    x[-1][0],
                    *(s_[n] for s_ in s[:-1]),
                    s[-1][0],
                ),
            )


def test_multi_transform():
    random = np.random.default_rng(314)

    n_tot, n_tr, n_j, n_k, n_target = 5, 1, 50, 12, 35
    f_shape = (n_tot, n_tr, n_k)
    c_shape = (n_tot, n_tr, n_j)
    f = random.standard_normal(size=f_shape) + 1j * random.standard_normal(size=f_shape)
    c = random.standard_normal(size=c_shape) + 1j * random.standard_normal(size=c_shape)
    x = random.uniform(-np.pi, np.pi, (n_tot, n_j))
    x_target = random.uniform(-1.0, 1.0, (n_tot, n_target))

    calc1 = nufft1(n_k, c, x)
    calc2 = nufft2(f, x)
    calc3 = nufft3(c, x, x_target, eps=1e-5)
    for n in range(n_tr):
        check_close(calc1[:, n], nufft1(n_k, c[:, n], x), rtol=1e-4)
        check_close(calc2[:, n], nufft2(f[:, n], x), rtol=1e-4)
        check_close(calc3[:, n], nufft3(c[:, n], x, x_target, eps=1e-5), rtol=1e-4)


def test_gh14():
    M = 100
    N = 200

    random = np.random.default_rng(14)
    x = 2 * np.pi * random.uniform(size=M)
    c = random.normal(size=M) + 1j * random.normal(size=M)

    def norm_nufft1(c, x):
        f = nufft1(N, c, x, eps=1e-4, iflag=1)
        return jnp.linalg.norm(f)

    def norm_nufft2(c, x):
        f = nufft2(c, x, eps=1e-4, iflag=1)
        return jnp.linalg.norm(f)

    jax.block_until_ready(jax.grad(norm_nufft2, argnums=(1))(c, x))
    jax.block_until_ready(jax.grad(norm_nufft1, argnums=(0,))(c, x))
    jax.block_until_ready(jax.grad(norm_nufft1, argnums=(0, 1))(c, x))
    jax.block_until_ready(jax.grad(norm_nufft1, argnums=(1,))(c, x))


def test_gh37():
    @partial(jax.vmap, in_axes=(0, 0, None))
    def cconv_test(f, xs, kernel):
        # f.shape = (n_grid, in_features)
        # x.shape = (n_grid, ndim)
        # kernel.shape = (*k_grid, in_features, out_features)

        ndim = xs.shape[-1]
        k_grid_shape = kernel.shape[:-2]

        f_ = f.astype(
            {np.float32: np.complex64, np.float64: np.complex128}[f.dtype.type]
        ).transpose()
        coords = [xs[..., i] for i in range(ndim)]

        f_hat = nufft1(k_grid_shape, f_, *coords, iflag=-1, eps=1e-5)
        c_hat = jnp.einsum("a...,...ab->b...", f_hat, kernel)
        return nufft2(c_hat, *coords, iflag=1, eps=1e-5)

    kernel = jnp.array(np.random.randn(32, 32, 32, 16, 16))
    f = jnp.array(np.random.randn(8, 100, 16))
    x = jnp.array(np.random.uniform(low=-np.pi, high=np.pi, size=(8, 100, 3)))

    a = cconv_test(f, x, kernel)
    b = jax.jit(cconv_test)(f, x, kernel)

    assert np.isfinite(a).all()
    assert np.isfinite(b).all()

    check_close(a, b)


def test_gh54():
    @jax.vmap
    def aux(f, x):
        f_hat = nufft1((32, 32, 32), f, *x, iflag=-1, eps=1e-5)
        return jnp.real(f_hat).mean()

    def test(f, x):
        return aux(f, x).mean()

    f = np.random.randn(8, 1000).astype(jnp.complex_)
    x = np.random.randn(8, 3, 1000)

    # Shapes are known from tracing alone, so without blocking on the values a
    # transform that fails at runtime would still satisfy these assertions.
    primal, _ = jax.block_until_ready(
        jax.jvp(partial(test, f), (x,), (jnp.ones_like(x),))
    )
    assert jax.block_until_ready(test(f, x)).shape == primal.shape
    assert jax.block_until_ready(jax.grad(test, argnums=0)(f, x)).shape == f.shape
    assert jax.block_until_ready(jax.grad(test, argnums=1)(f, x)).shape == x.shape


def test_gh54_type2():
    @jax.vmap
    def aux(f, x):
        f_hat = nufft2(f, *x, iflag=-1, eps=1e-5)
        return jnp.real(f_hat).mean()

    def test(f, x):
        return aux(f, x).mean()

    f = np.random.randn(8, 32, 32, 32).astype(jnp.complex_)
    x = np.random.randn(8, 3, 1000)

    # Shapes are known from tracing alone, so without blocking on the values a
    # transform that fails at runtime would still satisfy these assertions.
    primal, _ = jax.block_until_ready(
        jax.jvp(partial(test, f), (x,), (jnp.ones_like(x),))
    )
    assert jax.block_until_ready(test(f, x)).shape == primal.shape
    assert jax.block_until_ready(jax.grad(test, argnums=0)(f, x)).shape == f.shape
    assert jax.block_until_ready(jax.grad(test, argnums=1)(f, x)).shape == x.shape


@pytest.mark.parametrize("modeord", [0, 1], ids=["mo0", "mo1"])
@pytest.mark.parametrize("Nf", [16, 17], ids=["even", "odd"])
@pytest.mark.parametrize("ndim", [1, 2, 3], ids=["1D", "2D", "3D"])
@pytest.mark.parametrize("nufft_type", [1, 2], ids=["t1", "t2"])
def test_modeord(modeord, Nf, ndim, nufft_type):
    random = np.random.default_rng(657)

    iflag = 1
    num_uniform = tuple(Nf // ndim + 5 * np.arange(ndim))
    num_nonnuniform = 50
    eps = TARGET_EPS
    dtype = np.double
    cdtype = np.cdouble

    if modeord == 0:
        ks = [
            np.arange(-np.floor(n / 2), np.floor((n - 1) / 2 + 1)) for n in num_uniform
        ]
    else:
        ks = [
            np.concatenate(
                (
                    (
                        np.arange(0, np.floor(n / 2))
                        if n % 2 == 0
                        else np.arange(0, np.floor(n / 2) + 1)
                    ),
                    np.arange(-np.floor(n / 2), 0),
                )
            )
            for n in num_uniform
        ]

    if nufft_type == 1:
        x = random.uniform(-np.pi, np.pi, size=(ndim, num_nonnuniform)).astype(dtype)
        c = random.normal(size=num_nonnuniform) + 1j * random.normal(
            size=num_nonnuniform
        )
        c = c.astype(cdtype)
        f_expect = np.zeros(num_uniform, dtype=cdtype)
        for coords in product(*map(range, num_uniform)):
            k_vec = np.array([k[n] for (n, k) in zip(coords, ks)])
            f_expect[coords] = np.sum(c * np.exp(iflag * 1j * np.dot(k_vec, x)))

        opts = Opts(modeord=modeord)

        with enable_x64():
            f_calc = nufft1(num_uniform, c, *x, eps=eps, iflag=iflag, opts=opts)
            check_accurate(f_calc, f_expect, eps)

            func = partial(nufft1, num_uniform, eps=eps, iflag=iflag, opts=opts)
            jtu.check_grads(func, (c, *x), 1, modes=("fwd", "rev"))
    else:
        x = random.uniform(-np.pi, np.pi, size=(ndim, num_nonnuniform)).astype(dtype)
        f = random.normal(size=num_uniform) + 1j * random.normal(size=num_uniform)
        f = f.astype(cdtype)

        c_expect = np.zeros(num_nonnuniform, dtype=cdtype)
        for n in range(num_nonnuniform):
            arg = np.copy(f)
            for i, k in enumerate(ks):
                coords = [None for _ in range(ndim)]
                coords[i] = slice(None)
                arg *= np.exp(iflag * 1j * k * x[i][n])[tuple(coords)]
            c_expect[n] = np.sum(arg)

        opts = Opts(modeord=modeord)

        with enable_x64():
            c_calc = nufft2(f, *x, eps=eps, iflag=iflag, opts=opts)
            check_accurate(c_calc, c_expect, eps)

            func = partial(nufft2, eps=eps, iflag=iflag, opts=opts)
            jtu.check_grads(func, (f, *x), 1, modes=("fwd", "rev"))


def test_batched_points_grad():
    M = 100
    N = 200
    D = 3
    B = 4

    random = np.random.default_rng(123)
    xyz = 2 * np.pi * random.uniform(size=(B, D, M))
    stu = random.uniform(-1.0, 1.0, size=(B, D, N))
    c = random.normal(size=(B, M)) + 1j * random.normal(size=(B, M))

    def norm_nufft3(c, xyz, stu):
        f = nufft3(
            c,
            xyz[:, 0, :],
            xyz[:, 1, :],
            xyz[:, 2, :],
            stu[:, 0, :],
            stu[:, 1, :],
            stu[:, 2, :],
            eps=1e-5,
        )
        return jnp.linalg.norm(f)

    jax.block_until_ready(jax.grad(norm_nufft3, argnums=0)(c, xyz, stu))
    jax.block_until_ready(jax.grad(norm_nufft3, argnums=1)(c, xyz, stu))
    jax.block_until_ready(jax.grad(norm_nufft3, argnums=2)(c, xyz, stu))


@pytest.mark.parametrize("Nf", [32, 33], ids=["even", "odd"])
@pytest.mark.parametrize("ndim", [1, 2, 3], ids=["1D", "2D", "3D"])
@pytest.mark.parametrize("nufft_type", [1, 2], ids=["t1", "t2"])
def test_spreadinterponly_grad(Nf, ndim, nufft_type):
    random = np.random.default_rng(657)

    iflag = 1
    num_uniform = tuple(Nf + 5 * np.arange(ndim))
    num_nonnuniform = 50
    eps = 1e-10
    dtype = np.double
    cdtype = np.cdouble

    if nufft_type == 1:
        x = random.uniform(-np.pi, np.pi, size=(ndim, num_nonnuniform)).astype(dtype)
        c = random.normal(size=num_nonnuniform) + 1j * random.normal(
            size=num_nonnuniform
        )
        c = c.astype(cdtype)

        opts = Opts(spreadinterponly=1, gpu_spreadinterponly=1)

        with enable_x64():
            func = partial(nufft1, num_uniform, eps=eps, iflag=iflag, opts=opts)
            jtu.check_grads(
                lambda source: func(source, *x), (c,), 1, modes=("fwd", "rev")
            )

            for mode in ("fwd", "rev"):
                with pytest.raises(
                    NotImplementedError,
                    match=(
                        "Point derivatives are not supported when spreadinterponly "
                        "is enabled"
                    ),
                ):
                    jtu.check_grads(
                        lambda *points: func(c, *points), tuple(x), 1, modes=(mode,)
                    )
    else:
        x = random.uniform(-np.pi, np.pi, size=(ndim, num_nonnuniform)).astype(dtype)
        f = random.normal(size=num_uniform) + 1j * random.normal(size=num_uniform)
        f = f.astype(cdtype)

        opts = Opts(spreadinterponly=1, gpu_spreadinterponly=1)

        with enable_x64():
            func = partial(nufft2, eps=eps, iflag=iflag, opts=opts)
            jtu.check_grads(
                lambda source: func(source, *x), (f,), 1, modes=("fwd", "rev")
            )

            for mode in ("fwd", "rev"):
                with pytest.raises(
                    NotImplementedError,
                    match=(
                        "Point derivatives are not supported when spreadinterponly "
                        "is enabled"
                    ),
                ):
                    jtu.check_grads(
                        lambda *points: func(f, *points), tuple(x), 1, modes=(mode,)
                    )


@pytest.mark.parametrize("Nf", [100, 201], ids=["even", "odd"])
@pytest.mark.parametrize("ndim", [1, 2, 3], ids=["1D", "2D", "3D"])
@pytest.mark.parametrize("nufft_type", [1, 2, 3], ids=["t1", "t2", "t3"])
def test_spreadinterponly(Nf, ndim, nufft_type):
    # see https://github.com/flatironinstitute/finufft/blob/master/test/spreadinterp1d_test.cpp
    random = np.random.default_rng(657)

    iflag = 1
    num_uniform = tuple(Nf + 5 * np.arange(ndim))
    num_nonnuniform = 50
    eps = 1e-10
    dtype = np.double
    cdtype = np.cdouble
    opts = Opts(spreadinterponly=1, gpu_spreadinterponly=1)

    with enable_x64():
        if nufft_type == 1:
            func = partial(nufft1, num_uniform, eps=eps, iflag=iflag, opts=opts)
            # Compute kernel mass (spread a delta pulse at the origin)
            unit_x = np.zeros((ndim, 1), dtype=dtype)
            unit_c = np.ones(1, dtype=cdtype)
            kernel_mass = np.sum(func(unit_c, *unit_x))
            # Spread
            x = random.uniform(-np.pi, np.pi, size=(ndim, num_nonnuniform)).astype(
                dtype
            )
            c = random.normal(size=num_nonnuniform) + 1j * random.normal(
                size=num_nonnuniform
            )
            c = c.astype(cdtype)
            output = func(c, *x)
            # Check that sum of spreaded values is equal to kernel mass times sum of source strengths.
            assert np.allclose(np.sum(output), kernel_mass * np.sum(c))

        elif nufft_type == 2:
            # Compute kernel mass (spread a delta pulse at the origin)
            func = partial(nufft1, num_uniform, eps=eps, iflag=iflag, opts=opts)
            unit_x = np.zeros((ndim, 1), dtype=dtype)
            unit_c = np.ones(1, dtype=cdtype)
            kernel_mass = np.sum(func(unit_c, *unit_x))
            # Interpolate constant 1 function at random points
            func = partial(nufft2, eps=eps, iflag=iflag, opts=opts)
            x = random.uniform(-np.pi, np.pi, size=(ndim, num_nonnuniform)).astype(
                dtype
            )
            f = np.ones(num_uniform, dtype=cdtype)
            output = func(f, *x)
            # Check that interpolated values are equal to kernel mass everywhere
            assert np.allclose(output, kernel_mass)
        else:
            num_uniform = tuple(n + (n % 2) for n in num_uniform)

            x = random.uniform(-np.pi, np.pi, size=(ndim, num_nonnuniform)).astype(
                dtype
            )
            c = random.normal(size=num_nonnuniform) + 1j * random.normal(
                size=num_nonnuniform
            )
            s = random.uniform(-1.0, 1.0, size=(ndim, *num_uniform)).astype(dtype)
            func = partial(nufft3, eps=eps, iflag=iflag, opts=opts)
            # Check if type 3 raises an error when spreadinterponly is enabled, since it is not supported.
            with pytest.raises(
                ValueError,
                match="spreadinterponly is not supported for nufft3",
            ):
                func(c, *x, *s)
