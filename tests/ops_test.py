import jax
import numpy as np
from jax_finufft import nufft1
import pytest
from itertools import product


@pytest.mark.parametrize(
    "ndim, x64, num_nonnuniform, num_uniform, iflag",
    product([1, 2], [False, True], [500], [600], [-1, 1]),
)
def test_nufft1(ndim, x64, num_nonnuniform, num_uniform, iflag):
    random = np.random.default_rng(657)

    eps = 1e-10 if x64 else 1e-7
    dtype = np.double if x64 else np.single
    cdtype = np.cdouble if x64 else np.csingle

    num_uniform = tuple(num_uniform + np.arange(ndim))
    ks = [np.arange(-np.floor(n / 2), np.floor((n - 1) / 2 + 1)) for n in num_uniform]

    x = [
        random.uniform(-np.pi, np.pi, size=num_nonnuniform).astype(dtype)
        for _ in range(ndim)
    ]
    c = random.normal(size=num_nonnuniform) + 1j * random.normal(size=num_nonnuniform)
    c = c.astype(cdtype)
    f_expect = np.empty(num_uniform, dtype=cdtype)
    if ndim == 1:
        for i, vals in enumerate(ks):
            for j, k in enumerate(vals):
                f_expect[j] = np.sum(c * np.exp(iflag * 1j * k * x[i]), axis=-1)

    with jax.experimental.enable_x64(x64):
        f_calc = nufft1(num_uniform, c, *x, eps=eps, iflag=iflag)
        np.testing.assert_allclose(f_calc, f_expect, rtol=1e-7 if x64 else 5e-2)

        f_calc = jax.jit(nufft1, static_argnums=(0,), static_argnames=("eps", "iflag"))(
            num_uniform, c, *x, eps=eps, iflag=iflag
        )
        np.testing.assert_allclose(f_calc, f_expect, rtol=1e-7 if x64 else 5e-2)
