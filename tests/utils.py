import numpy as np
from jax._src import test_util as jtu

# Tightest tolerance we ever request. Might request looser via conditioned_eps(),
# especially for large transforms or float32.
TARGET_EPS = 1e-10


def conditioned_eps(x64, num_uniform):
    """Tightest tolerance worth requesting for a transform onto this grid.

    NUFFT conditioning grows like the largest mode count, so rounding limits the
    relative output error to ~max(N) * eps_mach no matter what algorithm is used
    (finufft docs/trouble.rst).

    This is an approximate, but motivated, way of avoiding FINUFFT's "eps too small"
    error.
    """
    eps_mach = np.finfo(np.double if x64 else np.single).eps
    return max(TARGET_EPS, eps_mach * max(num_uniform))


def check_close(a, b, **kwargs):
    """
    Element-wise accuracy check. Not a property that finufft guarantees, so we'll
    apply loose tolerances.
    """
    default_rtol = {
        "complex128": 1e-7,
        "complex64": 1e-4,
        "float64": 1e-7,
        "float32": 1e-4,
    }
    kwargs["rtol"] = kwargs.get("rtol", default_rtol)
    return jtu.check_close(a, b, **kwargs)


def check_accurate(calc, expect, eps, factor=10):
    """Assert a transform met its requested tolerance to within `factor`.

    FINUFFT bounds the error norm-wise rather than mode by mode.
    """
    expect = np.asarray(expect)
    rel_err = np.linalg.norm(np.asarray(calc) - expect) / np.linalg.norm(expect)
    assert rel_err < factor * eps, f"relative error {rel_err:.3e} > {factor * eps:.3e}"
