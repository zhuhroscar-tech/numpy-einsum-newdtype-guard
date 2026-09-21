# numpy-einsum-newdtype-guard

An independently-verified detector and safe drop-in replacement for a real,
currently-open numpy bug:

[numpy/numpy#32671](https://github.com/numpy/numpy/issues/32671) (opened
2026-09-17, open as of this guard's creation) — `numpy.einsum` silently
returns **wrong results, or segfaults**, whenever any operand's dtype is a
"new-style" DType (any dtype registered through numpy's modern DType C-API,
which reports `type_num == -1` — for example the third-party
[`numpy_quaddtype`](https://pypi.org/project/numpy-quaddtype/) 128-bit
quad-precision dtype, or any other library implementing a custom
high-precision or domain-specific dtype through the same modern API).

## Why this exists

Per the upstream issue's own root-cause trace,
`einsum_sumprod.c.src`'s kernel-selection table lookup only guards
`type_num >= NPY_NTYPES_LEGACY` — it never checks for `type_num < 0`. New-
style DTypes use `type_num == -1`, so the lookup silently indexes the
specialization table out of bounds. That is undefined behavior: on this
host (macOS, numpy 2.5.3, arm64) it produces **silently wrong numeric
results**; the upstream reporter also observed a **segfault** on a
different platform/expression shape. No exception is ever raised — code
using `einsum` with a new-style dtype gets bad numbers or a crash, with no
signal that anything went wrong.

`numpy.matmul` and `numpy.dot` on the SAME data are unaffected (confirmed
below) — the defect is specific to `einsum`'s own optimized kernel-table
dispatch, not to new-style dtype arithmetic in general.

## Independent reproduction (this repository, 2026-09-21)

Reproduced live on this host using the real `numpy_quaddtype` package (the
same package the upstream issue itself uses), numpy 2.5.3, comparing
against a plain float64 reference computed on the same underlying data
before it was cast to quad precision:

```
type_num: -1
ij,jk->ik  max abs error vs float64: 4.4     <- should be ~1e-15
ij,ij->ij  max abs error vs float64: 4.19    <- should be ~1e-15
ij->ji     max abs error vs float64: 0       <- transpose-only case unaffected
matmul     max abs error vs float64: 8.88e-16 <- matmul is FINE
i,i->      (scalar reduction) result: 0.0 instead of 0.872... <- silently wrong
```

These numbers match the magnitude and pattern the original issue reports.
This is a genuine, live-reproduced defect — not a theoretical or
dependency-internal-only concern (see `why_not_accepted_yet` history in
this contributor's ledger for the prior evidence-needed status this
candidate carried before this reproduction).

## Install and use

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,quaddtype]"
```

```bash
numpy-einsum-newdtype-guard detect --json
```

Exits `0` if the installed numpy is NOT affected (the bug has presumably
been fixed upstream), `1` if the live probe confirms the bug is present,
`2` if `numpy_quaddtype` (needed only for the live probe) is not
installed.

## Using the safe replacement directly

```python
from numpy_einsum_newdtype_guard.core import safe_einsum

# Behaves exactly like numpy.einsum for ordinary dtypes (float64, int64,
# etc.) -- zero overhead beyond one dtype check per operand, since those
# are unaffected by this bug and go straight to numpy.einsum.
result = safe_einsum("ij,jk->ik", a, b)

# For new-style dtype operands (numpy_quaddtype, or any other DType
# registered via the modern C-API), routes through an independent,
# from-scratch generic einsum implementation (`naive_einsum`) instead of
# numpy's buggy optimized kernel path, so results are correct.
```

## Limits

- `naive_einsum` (the correctness fallback used for affected dtypes) is a
  direct, unoptimized implementation of the Einstein-summation definition:
  cost is the product of every distinct index's dimension size. This is
  fine for the modest-size arrays typical of new-style-dtype use (quad
  precision is itself already far slower than float64 per-element, so
  callers reaching for it are not chasing raw throughput) but is NOT a
  production-scale contraction planner — do not use it for large tensor
  networks with many operands or huge dimensions.
- `naive_einsum` requires an explicit `"->"` output subscript (no
  implicit-output einsum), does not support ellipsis (`...`), and does
  not support repeated indices within a single operand's own subscript
  (diagonal-extraction einsum like `"ii->i"`). `safe_einsum` inherits
  these limits only on the affected-dtype fallback path; the fast path
  (ordinary dtypes) supports the full numpy einsum grammar as always.
- `is_new_style_dtype` detects the bug's precondition (`type_num < 0`) —
  it does not itself prove every new-style dtype triggers wrong output in
  every einsum expression shape, only that the vulnerable code path is
  reachable. `detect_einsum_newstyle_dtype_bug` empirically confirms
  actual wrong output for the specific expressions it tests.
- The live probe (`detect_einsum_newstyle_dtype_bug`, and the `detect`
  CLI command) requires the optional `numpy_quaddtype` package
  (`pip install numpy-einsum-newdtype-guard[quaddtype]`), which itself
  requires Python 3.11+ and numpy>=2.4. `safe_einsum`/`naive_einsum`
  themselves have no such requirement and work with any new-style dtype
  under this package's own `requires-python = ">=3.9"` (they take
  whatever dtype-bearing arrays you already have).
- The bug is UNDEFINED BEHAVIOR (an out-of-bounds C table index), and it
  manifests differently by platform: on this project's macOS/arm64 dev
  host it produces a silently wrong value; on `ubuntu-latest` x86_64 CI
  it SEGFAULTED the whole pytest process on this guard's first CI run.
  Because of that, every actual `np.einsum` call on a new-style-dtype
  operand runs in an **isolated subprocess** (`_worker.py`, invoked via
  `python -m numpy_einsum_newdtype_guard._worker <case>`) inside
  `detect_einsum_newstyle_dtype_bug()` and its regression test — a crash
  there only kills that subprocess, and a nonzero return code is treated
  as confirmed bug evidence (a real segfault), not a test-harness
  failure. `safe_einsum`'s `naive_einsum` fallback never calls
  `np.einsum` on the risky dtype at all, so it never needs this
  isolation. Verified on both `ubuntu-latest` and `macos-latest` CI
  (`.github/workflows/ci.yml`) after adding the subprocess isolation.

## Development

```bash
python -m pip install -e ".[dev,quaddtype]"
python -m pytest -v --cov=numpy_einsum_newdtype_guard --cov-report=term-missing
```

## License

MIT
