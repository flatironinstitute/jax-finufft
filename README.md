# JAX bindings to FINUFFT

[![GitHub Tests](https://github.com/flatironinstitute/jax-finufft/actions/workflows/tests.yml/badge.svg)](https://github.com/flatironinstitute/jax-finufft/actions/workflows/tests.yml)
[![Jenkins Tests](https://jenkins.flatironinstitute.org/buildStatus/icon?job=jax-finufft%2Fmain&subject=Jenkins%20Tests)](https://jenkins.flatironinstitute.org/job/jax-finufft/job/main/)

This package provides a [JAX](https://github.com/google/jax) interface to the [Flatiron Institute Non-uniform Fast Fourier Transform (FINUFFT)
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

[Type 1, 2, and 3](https://finufft.readthedocs.io/en/latest/math.html) transforms
are supported in 1, 2, and 3 dimensions on the CPU and GPU.
All of these functions support forward, reverse, and higher-order differentiation,
as well as batching using `vmap`.

The [FINUFFT plan interface](https://finufft.readthedocs.io/en/latest/c.html#guru-plan-interface)
is not directly exposed, although within a given jax-finufft call, plans are reused where possible,
and transforms sharing the same non-uniform points are stacked/vectorized. All of the tuning options
one can set in the plan interface are available through the `opts` argument of the jax-finufft API
(see [Advanced Usage](#advanced-usage)).

## Installation

The easiest way to install jax-finufft is from a pre-compiled binary on
PyPI or conda-forge. Only CPU binaries currently are available on PyPI, while
conda-forge has both CPU and GPU binaries. If you want GPU support without using
conda, you can install jax-finufft from source as detailed below. This is also
useful when you want to build finufft optimized for your hardware.

Currently only `jax<0.8` is supported.

### Install binary from PyPI

> [!NOTE]
> Only the CPU-enabled build of jax-finufft is available as a binary wheel on
> PyPI. For a GPU-enabled build, you'll need to build from source as described
> below or use conda-forge.

To install a binary wheel from [PyPI](https://pypi.org/project/jax-finufft/)
using [uv](https://docs.astral.sh/uv/), run the following command in a venv:

```bash
uv pip install jax-finufft
```

To install with `pip` instead of `uv`, simply drop `uv` from that command.

### Install binary from conda-forge
To install a CPU build using [mamba](https://github.com/mamba-org/mamba) (or
[conda](https://docs.conda.io)), run:

```bash
mamba install -c conda-forge jax-finufft
```

To install a GPU-enabled build, run:

```bash
mamba install -c conda-forge 'jax-finufft=*=cuda*'
```

Make note of the installed package version, like `conda-forge/linux-64::jax-finufft-1.1.0-cuda129py312h8ad7275_1`.
The `cuda129` substring indicates the package was built for CUDA 12.9. Your
NVIDIA driver will need to support this version of CUDA. Only one CUDA
build per major CUDA version is provided at present.

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
- CUDA (for GPU, >= 11.8)

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

```bash
mamba create -n gpu-jax-finufft -c conda-forge python fftw cxx-compiler jax 'jaxlib=*=*cuda*'
mamba activate gpu-jax-finufft
mamba install cuda libcufft-static -c nvidia
export CMAKE_PREFIX_PATH=$CONDA_PREFIX:$CMAKE_PREFIX_PATH
```
</details>

<details>
<summary>Install GPU dependencies using Flatiron module system</summary>

```bash
ml modules/2.4 \
   gcc \
   python \
   uv \
   fftw \
   cuda/12.8 \
   cudnn/9

export CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CUDA_ARCHITECTURES=80;90;120 -DJAX_FINUFFT_USE_CUDA=ON"
```
</details>

Other ways of installing JAX are given on the JAX website; the
["local CUDA" install methods](https://jax.readthedocs.io/en/latest/installation.html#pip-installation-gpu-cuda-installed-locally-harder)
are preferred for jax-finufft as this ensures the CUDA extensions are compiled
with the same Toolkit version as the CUDA runtime. However, in theory, this is not required
as long as both JAX and jax-finufft use CUDA with the same major version.

#### Notes on CUDA versions
While jax-finufft may build with a wide range of CUDA
versions, the resulting binaries may not be compatible with JAX (resulting in
odd runtime errors, like failed cuDNN or cuBLAS initialization). For the greatest
chance of success, we recommend building with the same version as JAX was built with.
To discover that, one can look at the requirements in [JAX's `build` directory](https://github.com/jax-ml/jax/tree/main/build)
(be sure to select the git tag for your version of JAX). Similarly, when installing from PyPI, we encourage using
`jax[cuda12-local]` so JAX and jax-finufft use the same CUDA libraries.

Depending on how challenging the installation is, users might want to run jax-finufft in a container. The [`.devcontainer`](./.devcontainer) directory is a good starting point for this.

#### Configuring the build
There are several important CMake variables that control aspects of the jax-finufft and (cu)finufft builds. These include:

- **`JAX_FINUFFT_USE_CUDA`** [disabled by default]: build with GPU support
- **`CMAKE_CUDA_ARCHITECTURES`** [default `native`]: the target GPU architecture. `native` means the GPU arch of the build system.
- **`FINUFFT_ARCH_FLAGS`** [default `-march=native`]: the target CPU architecture. The default is the native CPU arch of the build system.

Each of these can be set as `-Ccmake.define.NAME=VALUE` arguments to `pip install` or `uv pip install`. For example,
to build with GPU support from the repo root, run:

```bash
uv pip install -Ccmake.define.JAX_FINUFFT_USE_CUDA=ON .
```

Use multiple `-C` arguments to set multiple variables. The `-C` argument will work with any of the source installation methods (e.g. PyPI source dist, GitHub, `pip install`, `uv pip install`, `uv sync`, etc).

Build options can also be set with the `CMAKE_ARGS` environment variable. For example:

```bash
export CMAKE_ARGS="$CMAKE_ARGS -DJAX_FINUFFT_USE_CUDA=ON"
```

#### GPU build configuration
Building with GPU support requires passing `JAX_FINUFFT_USE_CUDA=ON` to CMake. See [Configuring the build](#configuring-the-build).

By default, jax-finufft will build for the GPU of the build machine. If you need to target
a different compute capability, such as 8.0 for Ampere, set `CMAKE_CUDA_ARCHITECTURES` as a CMake define:

```bash
uv pip install -Ccmake.define.JAX_FINUFFT_USE_CUDA=ON -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=80 .
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
export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
```

If `CUDA_HOME` isn't set, you'll need to replace it with the path to your CUDA
installation in the above line, often something like `/usr/local/cuda`.

#### Install source from PyPI

The source code for all released versions of jax-finufft are available on PyPI,
and this can be installed using:

```bash
uv pip install jax-finufft --no-binary jax-finufft
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

After cloning the repository, you can install the local copy using the uv ["project interface"](https://docs.astral.sh/uv/guides/projects/):

```bash
uv sync
```

or using the pip interface:

```bash
uv pip install -e .
```

where the `-e` flag optionally runs an "editable" install.

As yet another alternative, the latest development version from GitHub can be
installed directly (i.e. without cloning first) with

```bash
uv pip install git+https://github.com/flatironinstitute/jax-finufft.git
```

## Usage

This library provides three high-level functions (and these should be all that you
generally need to interact with): `nufft1`, `nufft2`, and `nufft3` (for the three "types" of
transforms). If you're already familiar with the [Python
interface](https://finufft.readthedocs.io/en/latest/python.html) to FINUFFT,
_please note that the function signatures here are different_!

For example, here's how you can do a 1-dimensional type 1 transform:

```python
import numpy as np

from jax_finufft import nufft1

M = 100000
N = 200000

rng = np.random.default_rng(123)
x = 2 * np.pi * rng.random(M)
c = rng.standard_normal(M) + 1j * rng.standard_normal(M)
f = nufft1(N, c, x, eps=1e-6, iflag=1)
```

Noting that the `eps` and `iflag` are optional, and that (for good reason, we
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

The syntax for a type 3 transform with "source points" `x`, `y`, `z` and
"target points" `s`, `t`, `u` is:
```python
f = nufft3(c, x, s)  # 1D
f = nufft3(c, x, y, s, t)  # 2D
f = nufft3(c, x, y, z, s, t, u)  # 3D
```

All of these functions support batching using `vmap`, and forward and reverse
mode differentiation.

### Stacked Transforms and Broadcasting

A "stacked", or "vectorized", finufft transform is one where the same non-uniform points are reused for multiple sets of source strengths. In the JAX interface, this is achieved by broadcasting. In the following example, only one finufft plan is created and one `setpts` call made, with a stack of 32 source strengths:

```python
import numpy as np

from jax_finufft import nufft1

M = 100000
N = 200000
S = 32

rng = np.random.default_rng(123)
x = 2 * np.pi * rng.random(M)
c = rng.standard_normal((S, M)) + 1j * rng.standard_normal((S, M))
f = nufft1(N, c, x)
```

To verify that a stacked transform is being used, see [Inspecting the finufft calls](#inspecting-the-finufft-calls).

Note that the broadcasting occurs because an implicit axis of length 1 is inserted in the second-to-last dimension of `x`. Currently, this is the only style of broadcasting that is supported when the strengths and points have unequal numbers of non-core dimensions. For other styles of broadcasting, insert axes of length 1 into the inputs. Any broadcast axes (even non-consecutive ones) are grouped and stacked in the transform.

Matched, but not broadcast, axes will be executed as separate transforms, each with their own `setpts` calls (but a single shared plan). In the following example (which continues from the previous), 1 plan is created and 4 `setpts` and 4 `execute` calls are made, each executing a stack of 32 transforms:

```python
P = 4

x = 2 * np.pi * rng.random((P, 1, M))
c = rng.standard_normal((P, S, M)) + 1j * rng.standard_normal((P, S, M))
f = nufft1(N, c, x)
```


## Selecting a platform
If you compiled jax-finufft with GPU support, you can force it to use a particular
backend by setting the environment variable `JAX_PLATFORMS=cpu` or `JAX_PLATFORMS=cuda`.

## Advanced usage

### Options

The tuning parameters for the library can be set using the `opts` parameter to
`nufft1`, `nufft2`, and `nufft3`. For example, to explicitly set the CPU [up-sampling
factor](https://finufft.readthedocs.io/en/latest/opts.html) that FINUFFT should
use, you can update the example from above as follows:

```python
from jax_finufft import options

opts = options.Opts(upsampfac=2.0)
nufft1(N, c, x, opts=opts)
```

The corresponding option for the GPU is `gpu_upsampfac`. In fact, all options
for the GPU are prefixed with `gpu_`, with the exception of `modeord`.

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

For descriptions of the options, see these pages in the FINUFFT docs:
- CPU: https://finufft.readthedocs.io/en/latest/opts.html
- GPU: https://finufft.readthedocs.io/en/latest/c_gpu.html#options-for-gpu-code

### Inspecting the finufft calls
When evaluating a single NUFFT, it's fairly obvious that jax-finufft will execute one
finufft transform under the hood. However, when evaluating a stacked NUFFT, or taking
the gradients of a NUFFT, the sequence of calls may be less obvious. One way to inspect
exactly what finufft calls are being made is to enable finufft's debug output by
passing `opts=Opts(debug=True)` or `opts=Opts(gpu_debug=True)`.

For example, taking the [Stacked Transforms](#stacked-transforms-and-broadcasting) example and enabling
debug output, we see the following:

```python-repl
>>> f = nufft1(N, c, x, eps=1e-6, iflag=1, opts=Opts(debug=True))
[FINUFFT_PLAN_T] new plan: FINUFFT version 2.4.1 .................
[FINUFFT_PLAN_T] 1d1: (ms,mt,mu)=(200000,1,1) (nf1,nf2,nf3)=(400000,1,1)
               ntrans=32 nthr=16 batchSize=16  spread_thread=2
[FINUFFT_PLAN_T] kernel fser (ns=7):            0.000765 s
[FINUFFT_PLAN_T] fwBatch 0.05GB alloc:          0.00703 s
[FINUFFT_PLAN_T] FFT plan (mode 64, nthr=16):   0.00892 s
[setpts] sort (didSort=1):              0.00327 s
[execute] start ntrans=32 (2 batches, bsize=16)...
[execute] done. tot spread:             0.0236 s
               tot FFT:                         0.0164 s
               tot deconvolve:                  0.00191 s
```

Evidently, we are creating a single plan with 32 transforms, and finufft has chosen to
batch them into two sets of 16. `setpts` is only called once, as is `execute`, as we
would expect for a stacked transform.

## Notes on the Implementation of the Gradients
The NUFFT gradients are implemented as [Jacobian-vector products](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html#jacobian-vector-products-jvps-aka-forward-mode-autodiff) (JVP, i.e. forward-mode autodiff), with associated transpose rules that implement the vector-Jacobian product (VJP, reverse mode). These are found in [`ops.py`](./src/jax_finufft/ops.py), in the `jvp` and `transpose` functions.

The JVP of a D-dimensional type 1 or 2 NUFFT requires D transforms of the same type in D dimensions (considering just the gradients with respect to the non-uniform locations). Each transform is weighted by the frequencies (as a overall scaling for type 1, and at the Fourier strength level for type 2). These transforms are fully stacked, and finufft plans are reused where possible.

Furthermore, the JAX `jvp` evaluates the function in addition to its JVP, so 1 more transform is necessary. This transform is not stacked with the JVP transforms. Likewise, 1 more is needed when the gradient with respect to the source or Fourier strengths is requested. However, this transform is stacked with the JVP.

In reverse mode, the VJP of a type 1 NUFFT requires type 2 transforms, and type 2 requires type 1. In either case, the function evaluation returned under JAX's `vjp` still requires an NUFFT of the original type (which cannot be stacked with the VJP transforms, as they are of a different type).

For type 3, the JVP requires `2*D` type 3 transforms of dimension D to evaluate the gradients with respect to both the source and target locations. The strengths of each transform are weighted by the source or target locations. The source and target transforms are stacked separately. As with type 1 and 2, the strengths gradient transform is stacked with the source locations and the function evaluation transform is not stacked.

The VJP of a type 3 NUFFT also uses type 3 NUFFTs, but with the source and target points swapped.

In all of the above, whenever a user requests [stacked transforms via broadcasting](#stacked-transforms-and-broadcasting), this does not introduce new plans or finufft calls—the stacks simply get deeper. New sets of non-uniform points necessarily introduce new `setpts` and new executions, but not new plans.

To see all of the stacking behavior in action, take a look at [Inspecting the finufft calls](#inspecting-the-finufft-calls).

## Similar libraries

- [finufft](https://finufft.readthedocs.io/en/latest/python.html): The
  "official" Python bindings to FINUFFT. A good choice if you're not already
  using JAX and if you don't need to differentiate through your transform.
- A list of other finufft binding libraries (e.g. for Julia, TensorFlow, PyTorch) is maintained at https://finufft.readthedocs.io/en/latest/users.html#other-wrappers-to-cu-finufft

## License & attribution

This package, developed by Dan Foreman-Mackey is licensed under the Apache
License, Version 2.0, with the following copyright:

Copyright 2021-2025 The Simons Foundation, Inc.

If you use this software, please cite the primary references listed on the
[FINUFFT docs](https://finufft.readthedocs.io/en/latest/refs.html).
