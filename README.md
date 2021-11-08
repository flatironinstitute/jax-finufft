# JAX bindings to FINUFFT

This package provides a [JAX](https://github.com/google/jax) interface to (a
subset of) the [Flatiron Institute Non-uniform Fast Fourier Transform (FINUFFT)
library](https://github.com/flatironinstitute/finufft). Take a look at the
[FINUFFT docs](https://finufft.readthedocs.io) for all the necessary
definitions, conventions, and more information about the algorithms and their
implementation. This package uses a low-level interface to directly expose the
FINUFFT library to JAX's XLA backend, as well as implementing differentiation
rules for the transforms.

## Included features

_This library is currently CPU-only, but GPU support is in the works using the
[cuFINUFFT library](https://github.com/flatironinstitute/cufinufft)._

[Type 1 and 2](https://finufft.readthedocs.io/en/latest/math.html) transforms
are supported in 1-, 2-, and 3-dimensions. All of these functions support
forward, reverse, and higher-order differentiation, as well as batching using
`vmap`.

## Installation

_For now, only a source build is supported._

For building, you should only need a recent version of Python (>3.6) and
[FFTW](https://www.fftw.org/). At runtime, you'll need `numpy`, `scipy`, and
`jax`. To set up such an environment, you can use `conda` (but you're welcome to
use whatever workflow works for you!):

```bash
conda create -n jax-finufft -c conda-forge python=3.9 numpy scipy fftw
python -m pip install "jax[cpu]"
```

Then you can install from source using (don't forget the `--recursive` flag
because FINUFFT is included as a submodule):

```bash
git clone --recursive https://github.com/dfm/jax-finufft
cd jax-finufft
python -m pip install .
```

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

Copyright 2021 The Simons Foundation, Inc.

If you use this software, please cite the primary references listed on the
[FINUFFT docs](https://finufft.readthedocs.io/en/latest/refs.html).
