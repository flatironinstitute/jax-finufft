import jax
import numpy as np
from jax_finufft import nufft1
import pytest
from itertools import product


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
