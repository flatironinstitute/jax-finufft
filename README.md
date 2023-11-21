# JAX bindings to FINUFFT

[![GitHub Tests](https://github.com/flatironinstitute/jax-finufft/actions/workflows/tests.yml/badge.svg)](https://github.com/flatironinstitute/jax-finufft/actions/workflows/tests.yml)
[![Jenkins Tests](https://jenkins.flatironinstitute.org/buildStatus/icon?job=jax-finufft%2Fmain&subject=Jenkins%20Tests)](https://jenkins.flatironinstitute.org/job/jax-finufft/job/main/)

This package provides a [JAX](https://github.com/google/jax) interface to (a
subset of) the [Flatiron Institute Non-uniform Fast Fourier Transform (FINUFFT)
library](https://github.com/flatironinstitute/finufft). Take a look at the
[FINUFFT docs](https://finufft.readthedocs.io) for all the necessary
definitions, conventions, and more information about the algorithms and their
implementation. This package uses a low-level interface to directly expose the
FINUFFT library to JAX's XLA backend, as well as implementing differentiation
rules for the transforms.

## Included features

This library includes CPU and GPU (CUDA) support. GPU support is implemented
through the [cuFINUFFT interface](https://finufft.readthedocs.io/en/latest/c_gpu.html)
of the FINUFFT library.

[Type 1 and 2](https://finufft.readthedocs.io/en/latest/math.html) transforms
are supported in 1-, 2-, and 3-dimensions. All of these functions support
forward, reverse, and higher-order differentiation, as well as batching using
`vmap`.

## Installation

_For now, only a source build is supported._

For building, you should only need a recent version of Python (>3.6) and
[FFTW](https://www.fftw.org/). GPU-enabled builds also require a working CUDA
compiler (i.e. the CUDA Toolkit), CUDA >= 11.8, and a compatible cuDNN (older versions of CUDA may work but
are untested). At runtime, you'll need `numpy`, `scipy`, and `jax`.

First, clone the repo and `cd` into the repo root (don't forget the `--recursive` flag because FINUFFT is included as a submodule):
```bash
git clone --recursive https://github.com/dfm/jax-finufft
cd jax-finufft
```

Then, you can use `conda` to set up a build environment (but you're welcome to
use whatever workflow works for you!). For example, for a CPU build, you can use:

```bash
conda create -n jax-finufft -c conda-forge python=3.10 numpy scipy fftw cxx-compiler
conda activate jax-finufft
export CPATH=$CONDA_PREFIX/include:$CPATH
python -m pip install "jax[cpu]"
python -m pip install .
```

The `CPATH` export is needed so that the build can find the headers for libraries like FFTW installed through conda.

For a GPU build, while the CUDA libraries and compiler are nominally available through conda,
our experience trying to install them through conda suggests that the "traditional"
way of obtaining the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) directly
from NVIDIA may work best (see [related advice for Horovod](https://horovod.readthedocs.io/en/stable/conda_include.html)). After installing the CUDA Toolkit, one can set up the rest of the dependencies with:

```bash
conda create -n gpu-jax-finufft -c conda-forge python=3.10 numpy scipy fftw 'gxx<12'
conda activate gpu-jax-finufft
export CPATH=$CONDA_PREFIX/include:$CPATH
python -m pip install "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
CMAKE_ARGS=-DCMAKE_CUDA_ARCHITECTURES=70 python -m pip install .
```

In the last line, you'll need to select the CUDA architecture(s) you wish to compile for. See the [FINUFFT docs](https://finufft.readthedocs.io/en/latest/install_gpu.html#cmake-installation).

At runtime, you may also need:
```bash
export LD_LIBRARY_PATH="$CUDA_PATH/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
```
If `CUDA_PATH` isn't set, you'll need to replace it with the path to your CUDA installation in the above line, often something like `/usr/local/cuda`.

For Flatiron users, the following environment setup script can be used instead of conda:
<details>
<summary>Environment script</summary>

```bash
ml modules/2.2
ml gcc
ml python/3.11
ml fftw
ml cuda/11
ml cudnn
ml nccl

export LD_LIBRARY_PATH=$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=60;70;80;90"
```
</details>

## Usage

This library provides two high-level functions (and these should be all that you
generally need to interact with): `nufft1` and `nufft2` (for the two "types" of
transforms). If you're already familiar with the [Python
interface](https://finufft.readthedocs.io/en/latest/python.html) to FINUFFT,
_please note that the function signatures here are different_!

For example, here's how you can do a 1-dimensional type 1 transform:

```python
import numpy as np
from jax_finufft import nufft1

M = 100000
N = 200000

x = 2 * np.pi * np.random.uniform(size=M)
c = np.random.standard_normal(size=M) + 1j * np.random.standard_normal(size=M)
f = nufft1(N, c, x, eps=1e-6, iflag=1)
```

Noting that the `eps` and `iflag` are optional, and that (for good reason, I
promise!) the order of the positional arguments is reversed from the `finufft`
Python package.

The syntax for a 2-, or 3-dimensional transform is:

```python
f = nufft1((Nx, Ny), c, x, y)  # 2D
f = nufft1((Nx, Ny, Nz), c, x, y, z)  # 3D
```

The syntax for a type 2 transform is (also allowing optional `iflag` and `eps`
parameters):

```python
c = nufft2(f, x)  # 1D
c = nufft2(f, x, y)  # 2D
c = nufft2(f, x, y, z)  # 3D
```

## Similar libraries

- [finufft](https://finufft.readthedocs.io/en/latest/python.html): The
  "official" Python bindings to FINUFFT. A good choice if you're not already
  using JAX and if you don't need to differentiate through your transform.
- [mrphys/tensorflow-nufft](https://github.com/mrphys/tensorflow-nufft):
  TensorFlow bindings for FINUFFT and cuFINUFFT.

## License & attribution

This package, developed by Dan Foreman-Mackey is licensed under the Apache
License, Version 2.0, with the following copyright:

Copyright 2021, 2022, 2023 The Simons Foundation, Inc.

If you use this software, please cite the primary references listed on the
[FINUFFT docs](https://finufft.readthedocs.io/en/latest/refs.html).
