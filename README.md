# numpy-einsum-newdtype-guard

This repository has been consolidated into [`numpy-correctness-guards`](https://github.com/zhuhroscar-tech/numpy-correctness-guards).

Use the shared package instead:

```bash
python -m pip install git+https://github.com/zhuhroscar-tech/numpy-correctness-guards.git
numpy-guard run einsum-newdtype detect --json
```

Python API:

```python
from numpy_correctness_guards.guards.einsum_newdtype import safe_einsum

result = safe_einsum("ij,jk->ik", a, b)
```

The original functionality from this repository was migrated into the umbrella package as `numpy_correctness_guards.guards.einsum_newdtype` and verified there with the migrated test suite.

This repository is archived as a read-only historical pointer.
