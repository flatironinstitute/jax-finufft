from functools import partial
from itertools import product

import jax
import numpy as np
import pytest
from jax.test_util import check_grads

from jax_finufft import nufft1, nufft2


@pytest.mark.parametrize(
    "ndim, x64, num_nonnuniform, num_uniform, iflag",
    product([1, 2, 3], [False, True], [50], [75], [-1, 1]),
)
def test_nufft1_forward(ndim, x64, num_nonnuniform, num_uniform, iflag):
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
    x_vec = np.array(x)
    c = random.normal(size=num_nonnuniform) + 1j * random.normal(size=num_nonnuniform)
    c = c.astype(cdtype)
    f_expect = np.zeros(num_uniform, dtype=cdtype)
    for coords in product(*map(range, num_uniform)):
        k_vec = np.array([k[n] for (n, k) in zip(coords, ks)])
        f_expect[coords] = np.sum(c * np.exp(iflag * 1j * np.dot(k_vec, x_vec)))

    with jax.experimental.enable_x64(x64):
        f_calc = nufft1(num_uniform, c, *x, eps=eps, iflag=iflag)
        np.testing.assert_allclose(f_calc, f_expect, rtol=5e-7 if x64 else 5e-2)

        f_calc = jax.jit(nufft1, static_argnums=(0,), static_argnames=("eps", "iflag"))(
            num_uniform, c, *x, eps=eps, iflag=iflag
        )
        np.testing.assert_allclose(f_calc, f_expect, rtol=5e-7 if x64 else 5e-2)


@pytest.mark.parametrize(
    "ndim, x64, num_nonnuniform, num_uniform, iflag",
    product([1, 2, 3], [False, True], [50], [75], [-1, 1]),
)
def test_nufft2_forward(ndim, x64, num_nonnuniform, num_uniform, iflag):
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
        np.testing.assert_allclose(c_calc, c_expect, rtol=5e-7 if x64 else 5e-2)

        c_calc = jax.jit(nufft2, static_argnames=("eps", "iflag"))(
            f, *x, eps=eps, iflag=iflag
        )
        np.testing.assert_allclose(c_calc, c_expect, rtol=5e-7 if x64 else 5e-2)


@pytest.mark.parametrize(
    "ndim, num_nonnuniform, num_uniform, iflag",
    product([1, 2, 3], [50], [35], [-1, 1]),
)
def test_nufft1_grad(ndim, num_nonnuniform, num_uniform, iflag):
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
        check_grads(func, (c, *x), 1, modes=("fwd", "rev"))


@pytest.mark.parametrize(
    "ndim, num_nonnuniform, num_uniform, iflag",
    product([1, 2, 3], [50], [35], [-1, 1]),
)
def test_nufft2_grad(ndim, num_nonnuniform, num_uniform, iflag):
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
        check_grads(func, (f, *x), 1, modes=("fwd", "rev"))
