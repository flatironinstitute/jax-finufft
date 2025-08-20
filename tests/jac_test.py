import jax.numpy as jnp
from jax_finufft import nufft2, options
import pytest
import numpy as np


from jax import config as jax_config
from jax import grad

jax_config.update("jax_enable_x64", True)


def nufft2r(
    f,
    x,
    y,
    domain_x=(0, 2 * jnp.pi),
    domain_y=(0, 2 * jnp.pi),
    rfft_axis=-1,
    vec=False,
    eps=1e-6,
):
    """Non-uniform real fast Fourier transform of second type.

    Notes
    -----
    Vectorization with the following signatures are supported.
     - ``(f,c0,c1),(x),(x)->(f,x)``

    Parameters
    ----------
    c : jnp.ndarray
        Fourier coefficients cₙ of the map x₀,x₁ ↦ c(x₀,x₁) such that
        c(x₀,x₁) = ∑ₘₙ cₘₙ exp(i m x₀) exp(i n x₁).
    x0 : jnp.ndarray
        Real query points of coordinate in ``domain0`` where interpolation is desired.
        The coordinates stored here must be the same coordinate enumerated across axis
        ``-2`` of ``c``.
    x1 : jnp.ndarray
        Real query points of coordinate in ``domain1`` where interpolation is desired.
        The coordinates stored here must be the same coordinate enumerated across axis
        ``-1`` of ``c``.
    domain0 : tuple[float]
        Domain of coordinate specified by x₀ over which samples were taken.
    domain1 : tuple[float]
        Domain of coordinate specified by x₁ over which samples were taken.
    rfft_axis : int
        Axis along which real FFT was performed.
        If -1 (-2), assumes c(x₀,x₁) = ∑ₘₙ cₘₙ exp(i m x₀) exp(i n x₁) where
            n ( m) >= 0, respectively.
        If ``None``, then assumes no real FFT was performed.
    vec : bool
        If set to ``True``, then it is assumed that multiple Fourier series are
        to be evaluated at the same non-uniform points. In that case, this flag
        must be set to retain the function signature of ``(f,c0,c1),(x),(x)->(f,x)``.
    eps : float
        Precision requested. Default is ``1e-6``.

    Returns
    -------
    c(x₀,x₁) : jnp.ndarray
        Real function value at query points.

    """
    scale_x = 2 * jnp.pi / (domain_x[1] - domain_x[0])
    scale_y = 2 * jnp.pi / (domain_y[1] - domain_y[0])
    x = (x - domain_x[0]) * scale_x
    y = (y - domain_y[0]) * scale_y

    if rfft_axis is None:
        s = 1
    else:
        if rfft_axis != -1 and rfft_axis != -2:
            raise NotImplementedError(
                f"rfft_axis must be -1 or -2, but got {rfft_axis}"
            )
        s = f.shape[rfft_axis] // 2
        s = jnp.exp(1j * s * (y if rfft_axis == -1 else x))
        s = s[..., jnp.newaxis, :] if vec else s
        f = jnp.fft.ifftshift(f, rfft_axis)

    opts = options.Opts(modeord=1)
    return (nufft2(f, x, y, iflag=1, eps=eps, opts=opts) * s).real


def _f_2d(x, y):
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


def _f_2d_nyquist_freq():
    x_freq, y_freq = 3, 5
    x_freq_nyquist = x_freq + 2
    y_freq_nyquist = y_freq + 3
    return x_freq_nyquist, y_freq_nyquist


_2d_inputs = [
    (
        _f_2d,
        2 * _f_2d_nyquist_freq()[0] + 1,
        2 * _f_2d_nyquist_freq()[1] + 1,
        (0, 2 * jnp.pi),
        (0, 2 * jnp.pi),
    ),
    (
        _f_2d,
        2 * _f_2d_nyquist_freq()[0] + 1,
        2 * _f_2d_nyquist_freq()[1] + 1,
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
]


@pytest.mark.parametrize("func, m, n, domain_x, domain_y", _2d_inputs)
def test_non_uniform_real_FFT_2D(func, m, n, domain_x, domain_y):
    """Test non-uniform real FFT 2D interpolation."""
    x = jnp.linspace(domain_x[0], domain_x[1], m, endpoint=False)
    y = jnp.linspace(domain_y[0], domain_y[1], n, endpoint=False)
    x, y = map(jnp.ravel, tuple(jnp.meshgrid(x, y, indexing="ij")))
    c = func(x, y).reshape(m, n)

    xq = jnp.array([7.34, 1.10134, 2.28, 1e3 * jnp.e])
    yq = jnp.array([1.1, 3.78432, 8.542, 0])

    f = 2 * jnp.fft.rfft2(c, norm="forward")
    f = f.at[..., (0, -1) if (n % 2 == 0) else 0].divide(2)
    np.testing.assert_allclose(
        nufft2r(f, xq, yq, domain_x, domain_y),
        func(xq, yq),
    )

    f = jnp.fft.fft2(c, norm="forward")
    np.testing.assert_allclose(
        nufft2r(f, xq, yq, domain_x, domain_y, rfft_axis=None),
        func(xq, yq),
    )


@pytest.mark.parametrize("func, m, n, domain_x, domain_y", _2d_inputs)
def test_non_uniform_real_FFT_2D_jac(func, m, n, domain_x, domain_y):
    """Test non-uniform real FFT 2D interpolation Jacobian."""
    x = jnp.linspace(domain_x[0], domain_x[1], m, endpoint=False)
    y = jnp.linspace(domain_y[0], domain_y[1], n, endpoint=False)
    x, y = map(jnp.ravel, tuple(jnp.meshgrid(x, y, indexing="ij")))
    c = func(x, y).reshape(m, n)

    xq = jnp.array([7.34, 1.10134, 2.28, 1e3 * jnp.e])
    yq = jnp.array([1.1, 3.78432, 8.542, 0])

    f = 2 * jnp.fft.rfft2(c, norm="forward")
    f = f.at[..., (0, -1) if (n % 2 == 0) else 0].divide(2)

    g = jnp.fft.fft2(c, norm="forward")

    def fun1(xq, yq):
        return nufft2r(f, xq, yq, domain_x, domain_y).sum()

    def fun2(xq, yq):
        return nufft2r(g, xq, yq, domain_x, domain_y, rfft_axis=None).sum()

    def truth(xq, yq):
        return func(xq, yq).sum()

    np.testing.assert_allclose(grad(fun1)(xq, yq), grad(fun2)(xq, yq))
    np.testing.assert_allclose(grad(fun1)(xq, yq), grad(truth)(xq, yq))
    np.testing.assert_allclose(grad(fun2)(xq, yq), grad(truth)(xq, yq))
