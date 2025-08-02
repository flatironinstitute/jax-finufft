import concurrent.futures
import os
import sys

import numpy as np
import pytest
import jax

import jax_finufft

JAX_FINUFFT_FORCE_FREETHREADED_TEST = (
    os.environ.get('JAX_FINUFFT_FORCE_FREETHREADED_TEST', '0') == '1'
)


def nogil_or_exit():
    try:
        gil_enabled = sys._is_gil_enabled()
    except AttributeError:
        gil_enabled = True

    if gil_enabled:
        if JAX_FINUFFT_FORCE_FREETHREADED_TEST:
            pytest.fail(
                'JAX_FINUFFT_FORCE_FREETHREADED_TEST is set, but Python is not running in free-threaded mode.'
            )
        else:
            pytest.skip('Python is not running in free-threaded mode')


def test_threaded_pool_nufft1(N_transforms=1000, N_points=1000):
    """
    Test concurrent computation of nufft1 transforms using ThreadPoolExecutor.
    """
    nogil_or_exit()

    rng = np.random.default_rng(42)
    modes = 64

    x_values = rng.uniform(-np.pi, np.pi, size=(N_transforms, N_points))
    c_values = rng.normal(size=(N_transforms, N_points)) + 1j * rng.normal(size=(N_transforms, N_points))
    
    def compute_transform(idx):
        """Compute a single nufft1 transform."""
        return np.array(jax_finufft.nufft1(
            modes, c_values[idx], x_values[idx], eps=1e-6
        ))
    
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
        np.testing.assert_allclose(seq_res, par_res, rtol=1e-4)
