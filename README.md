```bash
conda create -n jax-finufft python=3.9 numpy scipy
python -m pip install "jax[cpu]"

git clone --recursive https://github.com/dfm/jax-finufft
cd jax-finufft
python -m pip install .
```

Then:

```python
import numpy as np
from jax_finufft import finufft1d1

M = 100000
x = 2 * np.pi * np.random.uniform(size=M)
c = np.random.standard_normal(size=M) + 1j * np.random.standard_normal(size=M)
N = 200000
f = finufft1d1(N, x, c)
```
