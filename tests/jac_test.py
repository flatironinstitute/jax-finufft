"""Test interpolation of real valued maps and their JVPs."""

import jax.numpy as jnp
from jax_finufft import nufft2, options
import pytest
import numpy as np


from jax import config as jax_config
from jax import value_and_grad

jax_config.update("jax_enable_x64", True)


def nufft2r_1d(f, x, domain=(0, 2 * jnp.pi), vec=False, eps=1e-6):
    """Non-uniform real fast Forier transform of second type.

    Notes
    -----
    Vectorization with the following signatures are supported.
     - ``(b,f),(x)->(b,x)``

    Parameters
    ----------
    f : jnp.ndarray
        Fourier coefficients fₙ of the map x ↦ c(x) such that c(x) = ∑ₙ fₙ exp(i n x)
        where n >= 0.
    x : jnp.ndarray
        Real query points of coordinate in ``domain`` where interpolation is desired.
        The coordinates stored here must be the same coordinate enumerated across
        axis ``-1`` of ``f``.
    domain : tuple[float]
        Domain of coordinate specified by x over which samples were taken.
    vec : bool
        If set to ``True``, then it is assumed that multiple Fourier series are
        to be evaluated at the same non-uniform points. In that case, this flag
        must be set to retain the function signature of ``(b,f),(x)->(b,x)``.
    eps : float
        Precision requested. Default is ``1e-6``.

    Returns
    -------
    c(x) : jnp.ndarray
        Real function value at query points.

    """
    s = 2 * jnp.pi / (domain[1] - domain[0])
    x = (x - domain[0]) * s

    s = f.shape[-1] // 2
    s = jnp.exp(1j * s * x)
    s = s[..., jnp.newaxis, :] if vec else s

    opts = options.Opts(modeord=0)
    return (nufft2(f, x, iflag=1, eps=eps, opts=opts) * s).real


def nufft2r_2d(
    f,
    x0,
    x1,
    domain0=(0, 2 * jnp.pi),
    domain1=(0, 2 * jnp.pi),
    rfft_axis=-1,
    vec=False,
    eps=1e-6,
):
    """Non-uniform real fast Forier transform of second type.

    Notes
    -----
    Vectorization with the following signatures are supported.
     - ``(b,f0,f1),(x),(x)->(b,x)``

    Parameters
    ----------
    f : jnp.ndarray
        Fourier coefficients fₘₙ of the map x₀,x₁ ↦ c(x₀,x₁) such that
        c(x₀,x₁) = ∑ₘₙ fₘₙ exp(i m x₀) exp(i n x₁).
    x0 : jnp.ndarray
        Real query points of coordinate in ``domain0`` where interpolation is desired.
        The coordinates stored here must be the same coordinate
        enumerated across axis ``-2`` of ``f``.
    x1 : jnp.ndarray
        Real query points of coordinate in ``domain1`` where interpolation is desired.
        The coordinates stored here must be the same coordinate
        enumerated across axis ``-1`` of ``f``.
    domain0 : tuple[float]
        Domain of coordinate specified by x₀ over which samples were taken.
    domain1 : tuple[float]
        Domain of coordinate specified by x₁ over which samples were taken.
    rfft_axis : int
        Axis along which real FFT was performed.
        If -1 (-2), assumes c(x₀,x₁) = ∑ₘₙ fₘₙ exp(i m x₀) exp(i n x₁) where
            n ( m) >= 0, respectively.
    vec : bool
        If set to ``True``, then it is assumed that multiple Fourier series are
        to be evaluated at the same non-uniform points. In that case, this flag
        must be set to retain the function signature of ``(b,f0,f1),(x),(x)->(b,x)``.
    eps : float
        Precision requested. Default is ``1e-6``.

    Returns
    -------
    c(x₀,x₁) : jnp.ndarray
        Real function value at query points.

    """
    s0 = 2 * jnp.pi / (domain0[1] - domain0[0])
    s1 = 2 * jnp.pi / (domain1[1] - domain1[0])
    x0 = (x0 - domain0[0]) * s0
    x1 = (x1 - domain1[0]) * s1

    if rfft_axis is None:
        s = 1
    elif rfft_axis != -1 and rfft_axis != -2:
        raise NotImplementedError(f"rfft_axis must be -1 or -2, but got {rfft_axis}.")
    else:
        s = f.shape[rfft_axis] // 2
        s = jnp.exp(1j * s * (x1 if rfft_axis == -1 else x0))
        s = s[..., jnp.newaxis, :] if vec else s
        f = jnp.fft.ifftshift(f, rfft_axis)

    opts = options.Opts(modeord=1)
    return (nufft2(f, x0, x1, iflag=1, eps=eps, opts=opts) * s).real


def _c_1d(x):
    """Test function for 1D FFT."""
    return jnp.cos(7 * x) + jnp.sin(x) - 33.2


def _c_1d_nyquist_freq():
    return 7


def _c_2d(x, y):
    """Test function for 2D FFT."""
    x_freq, y_freq = 3, 5
    return (
        # something that's not separable
        jnp.cos(x_freq * x) * jnp.sin(2 * x + y)
        + jnp.sin(y_freq * y) * jnp.cos(x + 3 * y)
        - 33.2
        + jnp.cos(x)
        + jnp.cos(y)
    )


def _c_2d_nyquist_freq():
    x_freq, y_freq = 3, 5
    x_freq_nyquist = x_freq + 2
    y_freq_nyquist = y_freq + 3
    return x_freq_nyquist, y_freq_nyquist


_test_inputs_1D = [
    (_c_1d, 2 * _c_1d_nyquist_freq() + 1, (0, 2 * jnp.pi)),
    (_c_1d, 2 * _c_1d_nyquist_freq(), (0, 2 * jnp.pi)),
    (_c_1d, 2 * _c_1d_nyquist_freq() + 1, (-jnp.pi, jnp.pi)),
    (_c_1d, 2 * _c_1d_nyquist_freq(), (-jnp.pi, jnp.pi)),
    (lambda x: jnp.cos(7 * x), 2, (-jnp.pi / 7, jnp.pi / 7)),
    (lambda x: jnp.sin(7 * x), 3, (-jnp.pi / 7, jnp.pi / 7)),
]


@pytest.mark.parametrize("func, n, domain", _test_inputs_1D)
def test_non_uniform_real_FFT(func, n, domain):
    """Test non-uniform real FFT interpolation."""
    x = jnp.linspace(domain[0], domain[1], n, endpoint=False)
    c = func(x)
    xq = jnp.array([7.34, 1.10134, 2.28])

    f = 2 * jnp.fft.rfft(c, norm="forward")
    f = f.at[..., (0, -1) if (n % 2 == 0) else 0].divide(2)
    np.testing.assert_allclose(nufft2r_1d(f, xq, domain), func(xq))

    @value_and_grad
    def fun1(xq):
        return nufft2r_1d(f, xq, domain, eps=1e-7).sum()

    @value_and_grad
    def truth(xq):
        return func(xq).sum()

    v, g = truth(xq)
    f1v, f1g = fun1(xq)
    np.testing.assert_allclose(f1v, v)
    np.testing.assert_allclose(f1g, g)


@pytest.mark.parametrize(
    "func, m, n, domain_x, domain_y",
    [
        (
            _c_2d,
            2 * _c_2d_nyquist_freq()[0] + 1,
            2 * _c_2d_nyquist_freq()[1] + 1,
            (0, 2 * jnp.pi),
            (0, 2 * jnp.pi),
        ),
        (
            _c_2d,
            2 * _c_2d_nyquist_freq()[0] + 1,
            2 * _c_2d_nyquist_freq()[1] + 1,
            (-jnp.pi / 3, 5 * jnp.pi / 3),
            (jnp.pi, 3 * jnp.pi),
        ),
        (
            lambda x, y: jnp.cos(30 * x) + jnp.sin(y) ** 2 + 1,
            2 * 30 // 30 + 1,
            2 * 2 + 1,
            (0, 2 * jnp.pi / 30),
            (jnp.pi, 3 * jnp.pi),
        ),
    ],
)
def test_non_uniform_real_FFT_2D(func, m, n, domain_x, domain_y):
    """Test non-uniform real FFT 2D interpolation."""
    x = jnp.linspace(domain_x[0], domain_x[1], m, endpoint=False)
    y = jnp.linspace(domain_y[0], domain_y[1], n, endpoint=False)
    x, y = map(jnp.ravel, tuple(jnp.meshgrid(x, y, indexing="ij")))
    c = func(x, y).reshape(m, n)

    xq = jnp.array([7.34, 1.10134, 2.28, 1e3 * jnp.e])
    yq = jnp.array([1.1, 3.78432, 8.542, 0])

    f1 = 2 * jnp.fft.rfft2(c, norm="forward")
    f1 = f1.at[..., (0, -1) if (n % 2 == 0) else 0].divide(2)
    f2 = jnp.fft.fft2(c, norm="forward")

    v = func(xq, yq)
    np.testing.assert_allclose(nufft2r_2d(f1, xq, yq, domain_x, domain_y), v)
    np.testing.assert_allclose(nufft2r_2d(f2, xq, yq, domain_x, domain_y, None), v)

    @value_and_grad
    def fun1(xq, yq):
        return nufft2r_2d(f1, xq, yq, domain_x, domain_y, eps=1e-7).sum()

    @value_and_grad
    def fun2(xq, yq):
        return nufft2r_2d(f2, xq, yq, domain_x, domain_y, None, eps=1e-7).sum()

    @value_and_grad
    def truth(xq, yq):
        return func(xq, yq).sum()

    v, g = truth(xq, yq)
    f1v, f1g = fun1(xq, yq)
    f2v, f2g = fun2(xq, yq)

    np.testing.assert_allclose(f1v, v)
    np.testing.assert_allclose(f2v, v)

    np.testing.assert_allclose(f1g, f2g, err_msg="Something wrong with modeord")
    np.testing.assert_allclose(f1g, g)
    np.testing.assert_allclose(f2g, g)


def test_nufft2_vec():
    """Test vectorized JAX-finufft interpolation.

    https://github.com/flatironinstitute/jax-finufft/issues/155.
    """
    func_1, n, domain = _test_inputs_1D[0]
    func_2 = lambda x: -77 * jnp.sin(7 * x) + 18 * jnp.cos(x) + 100  # noqa: E731
    x = jnp.linspace(domain[0], domain[1], n, endpoint=False)
    c = jnp.stack([func_2(x), func_2(x)])

    xq = jnp.array([7.34, 1.10134, 2.28])
    cq = jnp.stack([func_2(xq), func_2(xq)])

    f = 2 * jnp.fft.rfft(c, norm="forward")
    f = f.at[..., (0, -1) if (c.shape[-1] % 2 == 0) else 0].divide(2)

    def _nufft2r(a, xq, vec=False):
        return nufft2r_1d(a, xq, domain, vec)

    # multiple (2) fourier series evaluated at the same (3) points.
    # Much more efficient than naive vectorization for large # of points.
    np.testing.assert_allclose(_nufft2r(f, xq), cq)

    xq = jnp.stack([xq, xq**2, xq**3, xq**4])[jnp.newaxis]
    f = jnp.stack([f, -f, 2 * f, 3 * f])[jnp.newaxis]

    # vectorized over batch of size (1, 4), evaluating
    # multiple (2) fourier series evaluated at the same (3) points
    np.testing.assert_allclose(
        _nufft2r(f, xq, vec=True),
        jnp.vectorize(_nufft2r, signature="(b,f),(x)->(b,x)")(f, xq),
    )
