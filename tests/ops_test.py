from functools import partial
from itertools import product

import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import test_util as jtu

from jax_finufft import nufft1, nufft2


def check_close(a, b, **kwargs):
    kwargs["rtol"] = kwargs.get("rtol", {"complex128": 1e-7, "complex64": 1e-4})
    return jtu.check_close(a, b, **kwargs)


@pytest.mark.parametrize(
    "ndim, x64, num_nonnuniform, num_uniform, iflag",
    product([1, 2, 3], [False, True], [50], [75], [-1, 1]),
)
def test_nufft1_forward(ndim, x64, num_nonnuniform, num_uniform, iflag):
    if ndim == 1 and jax.default_backend() != "cpu":
        pytest.skip("1D transforms not implemented on GPU")

    random = np.random.default_rng(657)

    eps = 1e-10 if x64 else 1e-7
    dtype = np.double if x64 else np.single
    cdtype = np.cdouble if x64 else np.csingle

    num_uniform = tuple(num_uniform // ndim + 5 * np.arange(ndim))
    ks = [np.arange(-np.floor(n / 2), np.floor((n - 1) / 2 + 1)) for n in num_uniform]

    x = random.uniform(-np.pi, np.pi, size=(ndim, num_nonnuniform)).astype(dtype)
    c = random.normal(size=num_nonnuniform) + 1j * random.normal(size=num_nonnuniform)
    c = c.astype(cdtype)
    f_expect = np.zeros(num_uniform, dtype=cdtype)
    for coords in product(*map(range, num_uniform)):
        k_vec = np.array([k[n] for (n, k) in zip(coords, ks)])
        f_expect[coords] = np.sum(c * np.exp(iflag * 1j * np.dot(k_vec, x)))

    with jax.experimental.enable_x64(x64):
        f_calc = nufft1(num_uniform, c, *x, eps=eps, iflag=iflag)
        check_close(f_calc, f_expect)

        f_calc = jax.jit(nufft1, static_argnums=(0,), static_argnames=("eps", "iflag"))(
            num_uniform, c, *x, eps=eps, iflag=iflag
        )
        check_close(f_calc, f_expect)


@pytest.mark.parametrize(
    "ndim, x64, num_nonnuniform, num_uniform, iflag",
    product([1, 2, 3], [False, True], [50], [75], [-1, 1]),
)
def test_nufft2_forward(ndim, x64, num_nonnuniform, num_uniform, iflag):
    if ndim == 1 and jax.default_backend() != "cpu":
        pytest.skip("1D transforms not implemented on GPU")

    random = np.random.default_rng(657)

    eps = 1e-10 if x64 else 1e-7
    dtype = np.double if x64 else np.single
    cdtype = np.cdouble if x64 else np.csingle

    num_uniform = tuple(num_uniform // ndim + 5 * np.arange(ndim))
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

    with jax.experimental.enable_x64(x64):
        c_calc = nufft2(f, *x, eps=eps, iflag=iflag)
        check_close(c_calc, c_expect)

        c_calc = jax.jit(nufft2, static_argnames=("eps", "iflag"))(
            f, *x, eps=eps, iflag=iflag
        )
        check_close(c_calc, c_expect)


@pytest.mark.parametrize(
    "ndim, num_nonnuniform, num_uniform, iflag",
    product([1, 2, 3], [50], [35], [-1, 1]),
)
def test_nufft1_grad(ndim, num_nonnuniform, num_uniform, iflag):
    if ndim == 1 and jax.default_backend() != "cpu":
        pytest.skip("1D transforms not implemented on GPU")

    random = np.random.default_rng(657)

    eps = 1e-10
    dtype = np.double
    cdtype = np.cdouble

    num_uniform = tuple(num_uniform // ndim + 5 * np.arange(ndim))

    x = [
        random.uniform(-np.pi, np.pi, size=num_nonnuniform).astype(dtype)
        for _ in range(ndim)
    ]
    c = random.normal(size=num_nonnuniform) + 1j * random.normal(size=num_nonnuniform)
    c = c.astype(cdtype)

    with jax.experimental.enable_x64():
        func = partial(nufft1, num_uniform, eps=eps, iflag=iflag)
        jtu.check_grads(func, (c, *x), 1, modes=("fwd", "rev"))

        def scalar_func(*args):
            return jnp.linalg.norm(func(*args))

        expect = jax.grad(scalar_func, argnums=tuple(range(len(x) + 1)))(c, *x)
        for n, g in enumerate(expect):
            check_close(jax.grad(scalar_func, argnums=(n,))(c, *x)[0], g)


@pytest.mark.parametrize(
    "ndim, num_nonnuniform, num_uniform, iflag",
    product([1, 2, 3], [50], [35], [-1, 1]),
)
def test_nufft2_grad(ndim, num_nonnuniform, num_uniform, iflag):
    if ndim == 1 and jax.default_backend() != "cpu":
        pytest.skip("1D transforms not implemented on GPU")

    random = np.random.default_rng(657)

    eps = 1e-10
    dtype = np.double
    cdtype = np.cdouble

    num_uniform = tuple(num_uniform // ndim + 5 * np.arange(ndim))

    x = [
        random.uniform(-np.pi, np.pi, size=num_nonnuniform).astype(dtype)
        for _ in range(ndim)
    ]
    f = random.normal(size=num_uniform) + 1j * random.normal(size=num_uniform)
    f = f.astype(cdtype)

    with jax.experimental.enable_x64():
        func = partial(nufft2, eps=eps, iflag=iflag)
        jtu.check_grads(func, (f, *x), 1, modes=("fwd", "rev"))

        def scalar_func(*args):
            return jnp.linalg.norm(func(*args))

        expect = jax.grad(scalar_func, argnums=tuple(range(len(x) + 1)))(f, *x)
        for n, g in enumerate(expect):
            check_close(jax.grad(scalar_func, argnums=(n,))(f, *x)[0], g)


@pytest.mark.parametrize(
    "ndim, num_nonnuniform, num_uniform, iflag",
    product([1, 2, 3], [50], [35], [-1, 1]),
)
def test_nufft1_vmap(ndim, num_nonnuniform, num_uniform, iflag):
    if ndim == 1 and jax.default_backend() != "cpu":
        pytest.skip("1D transforms not implemented on GPU")

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

    with jax.experimental.enable_x64():
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


@pytest.mark.parametrize(
    "ndim, num_nonnuniform, num_uniform, iflag",
    product([1, 2, 3], [50], [35], [-1, 1]),
)
def test_nufft2_vmap(ndim, num_nonnuniform, num_uniform, iflag):
    if ndim == 1 and jax.default_backend() != "cpu":
        pytest.skip("1D transforms not implemented on GPU")

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

    with jax.experimental.enable_x64():
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


def test_multi_transform():
    # TODO: is there a 2D or 3D version of this test?
    if jax.default_backend() != "cpu":
        pytest.skip("1D transforms not implemented on GPU")

    random = np.random.default_rng(314)

    n_tot, n_tr, n_j, n_k = 4, 10, 100, 12
    f_shape = (n_tot, n_tr, n_k)
    c_shape = (n_tot, n_tr, n_j)
    f = random.standard_normal(size=f_shape) + 1j * random.standard_normal(size=f_shape)
    c = random.standard_normal(size=c_shape) + 1j * random.standard_normal(size=c_shape)
    x = random.uniform(-np.pi, np.pi, (n_tot, n_j))

    calc1 = nufft1(n_k, c, x)
    calc2 = nufft2(f, x)
    for n in range(n_tr):
        check_close(calc1[:, n], nufft1(n_k, c[:, n], x), rtol=1e-4)
        check_close(calc2[:, n], nufft2(f[:, n], x), rtol=1e-4)


def test_issue14():
    if jax.default_backend() != "cpu":
        pytest.skip("1D transforms not implemented on GPU")

    M = 100
    N = 200

    random = np.random.default_rng(14)
    x = 2 * np.pi * random.uniform(size=M)
    c = random.normal(size=M) + 1j * random.normal(size=M)

    def norm_nufft1(c, x):
        f = nufft1(N, c, x, eps=1e-6, iflag=1)
        return jnp.linalg.norm(f)

    def norm_nufft2(c, x):
        f = nufft2(c, x, eps=1e-6, iflag=1)
        return jnp.linalg.norm(f)

    jax.grad(norm_nufft2, argnums=(1))(c, x)
    jax.grad(norm_nufft1, argnums=(0,))(c, x)
    jax.grad(norm_nufft1, argnums=(0, 1))(c, x)
    jax.grad(norm_nufft1, argnums=(1,))(c, x)


def test_issue37():
    if jax.default_backend() != "cpu":
        pytest.skip("1D transforms not implemented on GPU")

    @jax.jit
    @partial(jax.vmap, in_axes=(0, 0, None))
    def cconv_test(f, xs, kernel):

        # f.shape = (n_grid, in_features)
        # x.shape = (n_grid, ndim)
        # kernel.shape = (*k_grid, in_features, out_features)

        ndim = xs.shape[-1]
        k_grid_shape = kernel.shape[:-2]

        f_ = f.transpose().astype(complex)
        coords = [xs[...,i] for i in range(ndim)]

        f_hat = nufft1(k_grid_shape, f_, *coords, iflag=-1)
        c_hat = jnp.einsum('a...,...ab->b...', f_hat, kernel)
        return nufft2(c_hat, *coords, iflag=1)

    kernel = jnp.array(np.random.randn(32, 32, 32, 16, 16))
    f = jnp.array(np.random.randn(8, 100, 16))
    x = jnp.array(np.random.uniform(low=-np.pi, high=np.pi, size=(8, 100, 3)))

    fconv = cconv_test(f, x, kernel)
