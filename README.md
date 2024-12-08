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
are supported in 1, 2, and 3 dimensions on the CPU, and 2 and 3 dimensions on the GPU.
All of these functions support forward, reverse, and higher-order differentiation,
as well as batching using `vmap`.

> [!NOTE]
> The GPU backend does not currently support 1D (#125).

## Installation

The easiest ways to install jax-finufft is to install a pre-compiled binary from
PyPI or conda-forge, but if you need GPU support or want to get tuned
performance, you'll want to follow the instructions to install from source as
described below.

### Install binary from PyPI

> [!NOTE]
> Only the CPU-enabled build of jax-finufft is available as a binary wheel on
> PyPI. For a GPU-enabled build, you'll need to build from source as described
> below.

To install a binary wheel from [PyPI](https://pypi.org/project/jax-finufft/)
using pip, run the following commands:

```bash
python -m pip install "jax[cpu]"
python -m pip install jax-finufft
```

If this fails, you may need to use a conda-forge binary, or install from source.

### Install binary from conda-forge

> [!NOTE]
> Only the CPU-enabled build of jax-finufft is available as a binary from
> conda-forge. For a GPU-enabled build, you'll need to build from source as
> described below.

To install using [mamba](https://github.com/mamba-org/mamba) (or
[conda](https://docs.conda.io)), run:

```bash
mamba install -c conda-forge jax-finufft
```

### Install from source

#### Dependencies

Unsurprisingly, a key dependency is JAX, which can be installed following the
directions in [the JAX
documentation](https://jax.readthedocs.io/en/latest/installation.html). If
you're going to want to run on a GPU, make sure that you install the appropriate
JAX build.

The non-Python dependencies that you'll need are:

- [FFTW](https://www.fftw.org),
- [OpenMP](https://www.openmp.org) (for CPU, optional),
- CUDA (for GPU, >= 11.8), and

Older versions of CUDA may work, but they are untested.

Below we provide some example workflows for installing the required dependencies:

<details>
<summary>Install CPU dependencies with mamba or conda</summary>

```bash
mamba create -n jax-finufft -c conda-forge python jax fftw cxx-compiler
mamba activate jax-finufft
```
</details>

<details>
<summary>Install GPU dependencies with mamba or conda</summary>

For a GPU build, while the CUDA libraries and compiler are nominally available
through conda, our experience trying to install them this way suggests that the
"traditional" way of obtaining the [CUDA
Toolkit](https://developer.nvidia.com/cuda-downloads) directly from NVIDIA may
work best (see [related advice for
Horovod](https://horovod.readthedocs.io/en/stable/conda_include.html)). After
installing the CUDA Toolkit, one can set up the rest of the dependencies with:

```bash
mamba create -n gpu-jax-finufft -c conda-forge python numpy scipy fftw 'gxx<12'
mamba activate gpu-jax-finufft
export CMAKE_PREFIX_PATH=$CONDA_PREFIX:$CMAKE_PREFIX_PATH
python -m pip install "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Other ways of installing JAX are given on the JAX website; the ["local CUDA"
install
methods](https://jax.readthedocs.io/en/latest/installation.html#pip-installation-gpu-cuda-installed-locally-harder)
are preferred for jax-finufft as this ensures the CUDA extensions are compiled
with the same Toolkit version as the CUDA runtime. However, this is not required
as long as both JAX and jax-finufft use CUDA with the same major version.
</details>

<details>
<summary>Install GPU dependencies using Flatiron module system</summary>

```bash
ml modules/2.3 \
   gcc \
   python/3.11 \
   fftw \
   cuda/12

export CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CUDA_ARCHITECTURES=60;70;80;90 -DJAX_FINUFFT_USE_CUDA=ON"
```
</details>


#### Configuring the build
There are several important CMake variables that control aspects of the jax-finufft and (cu)finufft builds. These include:

- **`JAX_FINUFFT_USE_CUDA`** [disabled by default]: build with GPU support
- **`CMAKE_CUDA_ARCHITECTURES`** [default `native`]: the target GPU architecture. `native` means the GPU arch of the build system.
- **`FINUFFT_ARCH_FLAGS`** [default `-march=native`]: the target CPU architecture. `native` means the CPU arch of the build system.

Each of these can be set as `-Ccmake.define.NAME=VALUE` arguments to `pip install`. For example,
to build with GPU support from the repo root, run:

```bash
pip install -Ccmake.define.JAX_FINUFFT_USE_CUDA=ON .
```

Use multiple `-C` arguments to set multiple variables. The `-C` argument will work with any of the source installation methods (e.g. PyPI source dist, GitHub, etc).

Build options can also be set with the `CMAKE_ARGS` environment variable. For example:

```bash
export CMAKE_ARGS="$CMAKE_ARGS -DJAX_FINUFFT_USE_CUDA=ON"
```

#### GPU build configuration
Building with GPU support requires passing `JAX_FINUFFT_USE_CUDA=ON` to CMake. See [Configuring the build](#configuring-the-build).

By default, jax-finufft will build for the GPU of the build machine. If you need to target
a different compute capability, such as 8.0 for Ampere, set `CMAKE_CUDA_ARCHITECTURES` as a CMake define:

```bash
pip install -Ccmake.define.JAX_FINUFFT_USE_CUDA=ON -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=80 .
```

`CMAKE_CUDA_ARCHITECTURES` also takes a semicolon-separated list.

To detect the arch for a specific GPU, one can run:
```bash
$ nvidia-smi --query-gpu=compute_cap --format=csv,noheader
8.0
```

The values are also listed on the [NVIDIA website](https://developer.nvidia.com/cuda-gpus).

In some cases, you may also need the following at runtime:

```bash
export LD_LIBRARY_PATH="$CUDA_PATH/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
```

If `CUDA_PATH` isn't set, you'll need to replace it with the path to your CUDA
installation in the above line, often something like `/usr/local/cuda`.

#### Install source from PyPI

The source code for all released versions of jax-finufft are available on PyPI,
and this can be installed using:

```bash
python -m pip install --no-binary jax-finufft
```

#### Install source from GitHub

Alternatively, you can check out the source repository from GitHub:

```bash
git clone --recurse-submodules https://github.com/flatironinstitute/jax-finufft
cd jax-finufft
```

> [!NOTE]
> Don't forget the `--recurse-submodules` argument when cloning the repo because
> the upstream FINUFFT library is included as a git submodule. If you do forget,
> you can run `git submodule update --init --recursive` in your local copy to
> checkout the submodule after the initial clone.

After cloning the repository, you can install the local copy using:

```bash
python -m pip install -e .
```

where the `-e` flag optionally runs an "editable" install.

As yet another alternative, the latest development version from GitHub can be
installed directly (i.e. without cloning first) with

```bash
python -m pip install git+https://github.com/flatironinstitute/jax-finufft.git
```

## Usage

This library provides two high-level functions (and these should be all that you
generally need to interact with): `nufft1` and `nufft2` (for the two "types" of
transforms). If you're already familiar with the [Python
interface](https://finufft.readthedocs.io/en/latest/python.html) to FINUFFT,
_please note that the function signatures here are different_!

For example, here's how you can do a 1-dimensional type 1 transform
(only works on CPU):

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

The syntax for a 2-, or 3-dimensional transform (CPU or GPU) is:

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

All of these functions support batching using `vmap`, and forward and reverse
mode differentiation.

## Selecting a platform
If you compiled jax-finufft with GPU support, you can force it to use a particular
backend by setting the environment variable `JAX_PLATFORMS=cpu` or `JAX_PLATFORMS=cuda`.

## Advanced usage

The tuning parameters for the library can be set using the `opts` parameter to
`nufft1` and `nufft2`. For example, to explicitly set the CPU [up-sampling
factor](https://finufft.readthedocs.io/en/latest/opts.html) that FINUFFT should
use, you can update the example from above as follows:

```python
from jax_finufft import options

opts = options.Opts(upsampfac=2.0)
nufft1(N, c, x, opts=opts)
```

The corresponding option for the GPU is `gpu_upsampfac`. In fact, all options
for the GPU are prefixed with `gpu_`.

One complication here is that the [vector-Jacobian
product](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#vector-jacobian-products-vjps-aka-reverse-mode-autodiff)
for a NUFFT requires evaluating a NUFFT of a different type. This means that you
might want to separately tune the options for the forward and backward pass.
This can be achieved using the `options.NestedOpts` interface. For example, to
use a different up-sampling factor for the forward and backward passes, the code
from above becomes:

```python
import jax

opts = options.NestedOpts(
  forward=options.Opts(upsampfac=2.0),
  backward=options.Opts(upsampfac=1.25),
)
jax.grad(lambda args: nufft1(N, *args, opts=opts).real.sum())((c, x))
```

or, in this case equivalently:

```python
opts = options.NestedOpts(
  type1=options.Opts(upsampfac=2.0),
  type2=options.Opts(upsampfac=1.25),
)
```

See [the FINUFFT docs](https://finufft.readthedocs.io/en/latest/opts.html) for
descriptions of all the CPU tuning parameters. The corresponding GPU parameters
are currently only listed in source code form in
[`cufinufft_opts.h`](https://github.com/flatironinstitute/finufft/blob/master/include/cufinufft_opts.h).

## Similar libraries

- [finufft](https://finufft.readthedocs.io/en/latest/python.html): The
  "official" Python bindings to FINUFFT. A good choice if you're not already
  using JAX and if you don't need to differentiate through your transform.
- [mrphys/tensorflow-nufft](https://github.com/mrphys/tensorflow-nufft):
  TensorFlow bindings for FINUFFT and cuFINUFFT.

## License & attribution

This package, developed by Dan Foreman-Mackey is licensed under the Apache
License, Version 2.0, with the following copyright:

Copyright 2021-2024 The Simons Foundation, Inc.

If you use this software, please cite the primary references listed on the
[FINUFFT docs](https://finufft.readthedocs.io/en/latest/refs.html).
