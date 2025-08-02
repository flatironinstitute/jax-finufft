import concurrent.futures
import sys
import sysconfig

import numpy as np
import pytest
from ops_test import check_close

import jax_finufft


def nogil_or_exit():
    freethreaded_build = sysconfig.get_config_var("Py_GIL_DISABLED")
    try:
        gil_enabled = sys._is_gil_enabled()
    except AttributeError:
        gil_enabled = True

    if freethreaded_build:
        if gil_enabled:
            pytest.fail(
                "Python was built with free-threading but is not running in free-threaded mode."
            )
    else:
        pytest.skip("Python not built in free-threaded mode")


def test_threaded_pool_nufft1(N_transforms=1000, N_points=1000):
    """
    Test concurrent computation of nufft1 transforms using ThreadPoolExecutor.
    """
    nogil_or_exit()

    rng = np.random.default_rng(42)
    modes = 64
    x64 = False
    eps = 1e-10 if x64 else 1e-7

    x_values = rng.uniform(-np.pi, np.pi, size=(N_transforms, N_points))
    c_values = rng.normal(size=(N_transforms, N_points)) + 1j * rng.normal(
        size=(N_transforms, N_points)
    )

    def compute_transform(idx):
        """Compute a single nufft1 transform."""
        return np.array(
            jax_finufft.nufft1(modes, c_values[idx], x_values[idx], eps=eps)
        )

    # Sequential computation
    sequential_results = [compute_transform(i) for i in range(N_transforms)]

    # Parallel computation using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(compute_transform, i) for i in range(N_transforms)]
        parallel_results = [future.result() for future in futures]

    # Verify results are the same
    assert len(sequential_results) == N_transforms
    assert len(parallel_results) == N_transforms

    for seq_res, par_res in zip(sequential_results, parallel_results):
        check_close(seq_res, par_res, rtol=1e-3)
