"""Core logic for numpy-einsum-newdtype-guard.

Guards a real, currently-open numpy bug (numpy/numpy#32671, confirmed open
and independently reproduced against the pinned numpy version at the time
this guard was created -- see README for the verification trail):

    numpy.einsum silently returns WRONG results (or segfaults, on some
    platforms) when any operand's dtype is a "new-style" DType (type_num
    == -1, the type-number convention used by DTypes registered via
    numpy's new C-API, e.g. the third-party `numpy_quaddtype` package, or
    any other library implementing a custom high/extended-precision or
    domain-specific dtype through the modern DType API).

Root cause per the upstream issue's own trace: `einsum_sumprod.c.src`'s
kernel-selection table lookup only guards `type_num >= NPY_NTYPES_LEGACY`,
missing the `type_num < 0` case new-style DTypes use. The lookup then
indexes the specialization table with a negative/out-of-range index,
which is undefined behavior: it silently returns wrong numeric results on
some platforms and segfaults on others (both observed and confirmed by
this guard's own independent reproduction -- see README).

Real-world impact: any code using einsum with a new-style dtype (quad
precision, custom ML dtypes, domain-specific numeric types) gets either
silently corrupted numbers or a hard crash, with NO error raised by numpy
itself. This module provides an independently-verified detector plus a
safe drop-in replacement that only pays a performance cost for the
specific dtypes affected by the bug -- not a numpy patch.
"""
from __future__ import annotations

import functools
import json
import operator
import sys
from dataclasses import dataclass
from itertools import product
from typing import Any, Sequence

import numpy as np

ArrayLike = Any


@dataclass
class BugDetectionResult:
    """Result of probing the installed numpy + a new-style dtype for this bug."""

    numpy_version: str
    dtype_name: str
    affected: bool
    detail: str


def is_new_style_dtype(dtype: np.dtype) -> bool:
    """New-style DTypes (registered through numpy's modern C-API DType
    machinery, as opposed to the legacy fixed type-number table) report a
    negative ``type_num``. This is exactly the condition
    ``einsum_sumprod.c.src``'s table-bounds check misses (it only rejects
    ``type_num >= NPY_NTYPES_LEGACY``, not ``type_num < 0``) -- see
    numpy/numpy#32671.
    """
    return np.dtype(dtype).num < 0


def naive_einsum(subscripts: str, *operands: ArrayLike) -> Any:
    """From-scratch, generic einsum implementation used as an independent
    correctness oracle AND as the safe fallback for dtypes affected by
    numpy/numpy#32671.

    Deliberately does NOT call ``numpy.einsum`` (that would just re-run
    the code under test) and does NOT use ``numpy.dot``/``numpy.matmul``
    internally for the general case (those happen to be unaffected by
    this specific bug per the upstream report, but this function is meant
    to work for ANY explicit-subscript einsum expression, not just the
    two-operand matmul-shaped ones matmul could cover). Instead it
    enumerates every output/summed index combination directly and
    accumulates products via the operands' own ``__mul__``/``__add__``,
    so it is correct for any dtype that supports ordinary elementwise
    arithmetic and indexing -- exactly the guarantee a new-style dtype
    like ``numpy_quaddtype`` provides, even though numpy's own optimized
    einsum kernel table does not currently dispatch on it correctly.

    This is a correctness reference, not a fast path: cost is the product
    of every distinct index's extent (`O(prod(all dim sizes))`), matching
    a textbook "definition of Einstein summation" implementation rather
    than any optimized contraction order. See README Limits.

    Supported subscript grammar (deliberately narrow, matching this
    guard's actual need -- see README Limits for what is NOT supported):
    - Explicit ``"->"`` output subscript required (implicit-output
      einsum, e.g. ``"ij,jk"`` without ``->``, is not supported).
    - No ellipsis (``...``).
    - Each input subscript's letters must be a *permutation* of that
      operand's own axes (no repeated letters within a single operand's
      subscript, i.e. no diagonal-extraction einsum like ``"ii->i"``).
    - Any number of operands (pairwise/left-to-right index enumeration,
      not limited to two).
    """
    if "->" not in subscripts:
        raise ValueError(
            "naive_einsum requires an explicit '->' output subscript "
            f"(got {subscripts!r}); implicit-output einsum is not supported."
        )
    if "." in subscripts:
        raise ValueError("naive_einsum does not support ellipsis ('...') subscripts.")

    in_part, out_sub = subscripts.split("->")
    in_subs = in_part.split(",")
    ops = [np.asarray(op) for op in operands]
    if len(in_subs) != len(ops):
        raise ValueError(
            f"subscript describes {len(in_subs)} operand(s) but {len(ops)} were given"
        )

    dim_of: dict[str, int] = {}
    for sub, op in zip(in_subs, ops):
        if len(set(sub)) != len(sub):
            raise ValueError(
                f"naive_einsum does not support repeated indices within one "
                f"operand's subscript (diagonal extraction); got {sub!r}"
            )
        if len(sub) != op.ndim:
            raise ValueError(
                f"subscript {sub!r} has {len(sub)} indices but operand has "
                f"{op.ndim} dimensions"
            )
        for letter, size in zip(sub, op.shape):
            if letter in dim_of and dim_of[letter] != size:
                raise ValueError(
                    f"inconsistent dimension for index {letter!r}: "
                    f"{dim_of[letter]} vs {size}"
                )
            dim_of[letter] = size

    for letter in out_sub:
        if letter not in dim_of:
            raise ValueError(f"output index {letter!r} does not appear in any input")
    if len(set(out_sub)) != len(out_sub):
        raise ValueError("naive_einsum does not support repeated output indices")

    all_letters = set(dim_of)
    summed_letters = sorted(all_letters - set(out_sub))
    out_shape = tuple(dim_of[l] for l in out_sub)

    out_ranges = [range(dim_of[l]) for l in out_sub]
    sum_ranges = [range(dim_of[l]) for l in summed_letters]

    result = None
    if out_shape:
        result = np.empty(out_shape, dtype=ops[0].dtype)

    for out_idx in (product(*out_ranges) if out_ranges else [()]):
        out_assign = dict(zip(out_sub, out_idx))
        terms = []
        for sum_idx in (product(*sum_ranges) if sum_ranges else [()]):
            sum_assign = dict(zip(summed_letters, sum_idx))
            full_assign = {**out_assign, **sum_assign}
            factors = []
            for sub, op in zip(in_subs, ops):
                idx = tuple(full_assign[l] for l in sub)
                factors.append(op[idx])
            terms.append(functools.reduce(operator.mul, factors))
        total = functools.reduce(operator.add, terms)
        if out_shape:
            result[out_idx] = total
        else:
            result = total

    return result


def safe_einsum(subscripts: str, *operands: ArrayLike, **kwargs) -> Any:
    """Drop-in replacement for ``numpy.einsum`` that actually gives correct
    results in the presence of new-style dtypes (numpy/numpy#32671).

    - If NONE of the operands use a new-style dtype (the overwhelmingly
      common case: plain float64/int64/etc.), this calls ``numpy.einsum``
      directly and unmodified -- full speed, identical behavior, zero
      overhead beyond one ``.dtype.num`` check per operand.
    - If ANY operand's dtype is new-style, routes through
      ``naive_einsum`` instead of the buggy optimized C kernel path.
      This is slower (see ``naive_einsum``'s docstring) but correct.

    ``**kwargs`` (e.g. ``optimize=``) are only honored on the fast/unaffected
    path -- ``naive_einsum`` does not implement an optimized contraction
    order and ignores them, since correctness (not speed) is the point of
    the fallback.
    """
    ops = [np.asarray(op) for op in operands]
    if any(is_new_style_dtype(op.dtype) for op in ops):
        return naive_einsum(subscripts, *ops)
    return np.einsum(subscripts, *ops, **kwargs)


def detect_einsum_newstyle_dtype_bug() -> BugDetectionResult:
    """Empirically probe the INSTALLED numpy for numpy/numpy#32671 using
    the real third-party `numpy_quaddtype` package (the same package the
    upstream issue itself uses to reproduce it) as a concrete, real
    new-style dtype.

    The upstream issue's own root-cause trace describes this as undefined
    behavior (an out-of-bounds table index): it can manifest as either a
    SILENTLY WRONG VALUE or a hard SEGFAULT depending on platform/
    allocator/expression shape -- both have been independently observed
    for this exact guard (silent wrong value on macOS/arm64, segfault on
    ubuntu-latest x86_64 CI). Every actual ``np.einsum`` call on
    quad-precision operands therefore runs in an ISOLATED SUBPROCESS
    (``_worker.py``) so a crash on either platform only kills that
    subprocess, never this process (or an entire pytest session) with it.
    A nonzero/negative subprocess return code IS real bug evidence
    (a crash), not a harness failure, and is reported as affected=True.

    Compares, for each case, the (possibly crashing) einsum-on-quad-
    precision result against a float64 reference computed on the
    original float64 data before it was cast to quad precision, and
    against ``safe_einsum``'s ``naive_einsum`` fallback (which never
    calls ``np.einsum`` on the risky dtype at all, so it is always safe
    to run directly in this process).

    Raises ``ImportError`` if ``numpy_quaddtype`` is not installed --
    callers (including the test suite) should skip rather than treat
    that as "not affected".
    """
    import subprocess

    from numpy_quaddtype import QuadPrecDType  # noqa: F401  (import-check only)

    rng = np.random.default_rng(0)
    a64, b64 = rng.standard_normal((2, 6, 6))

    dtype_name = "numpy_quaddtype.QuadPrecDType"
    tolerance = 1e-6

    cases = [
        ("matmul_contraction", "ij,jk->ik", (a64, b64)),
        ("scalar_reduction", "i,i->", (a64[0], a64[0])),
    ]

    worst_buggy_err = 0.0
    worst_safe_err = 0.0
    details = []
    crashed_cases = []

    for worker_case, subs, ref_ops in cases:
        proc = subprocess.run(
            [sys.executable, "-m", "numpy_einsum_newdtype_guard._worker", worker_case],
            capture_output=True,
            text=True,
            timeout=60,
        )
        reference = np.einsum(subs, *ref_ops)

        # Independent-of-the-worker safe path: naive_einsum never calls
        # np.einsum on the quad-precision operands, so it is always safe
        # to run directly here.
        a = np.asarray(ref_ops[0], dtype=QuadPrecDType())
        if len(ref_ops) == 2:
            b = np.asarray(ref_ops[1], dtype=QuadPrecDType())
            safe_result = safe_einsum(subs, a, b)
        else:
            safe_result = safe_einsum(subs, a)
        safe = np.asarray(safe_result, dtype=np.float64)
        safe_err = float(np.max(np.abs(safe - reference)))
        worst_safe_err = max(worst_safe_err, safe_err)

        if proc.returncode != 0:
            crashed_cases.append(worker_case)
            worst_buggy_err = float("inf")
            details.append(
                f"{subs}: WORKER CRASHED (returncode={proc.returncode}, "
                f"likely SIGSEGV) -- safe_err={safe_err:.3g}"
            )
            continue

        payload = json.loads(proc.stdout)
        buggy = np.asarray(payload["result"], dtype=np.float64)
        buggy_err = float(np.max(np.abs(buggy - reference)))
        worst_buggy_err = max(worst_buggy_err, buggy_err)
        details.append(f"{subs}: einsum_err={buggy_err:.3g} safe_err={safe_err:.3g}")

    affected = crashed_cases or worst_buggy_err > tolerance
    safe_is_correct = worst_safe_err <= tolerance

    if crashed_cases:
        detail = (
            f"CONFIRMED (via isolated subprocess): np.einsum SEGFAULTED for "
            f"new-style dtype operands in case(s) {crashed_cases} -- "
            f"numpy/numpy#32671 is undefined behavior and can crash the "
            f"process rather than just return a wrong value on this "
            f"platform. safe_einsum's naive_einsum fallback never calls "
            f"np.einsum on this dtype and is unaffected "
            f"(max abs error {worst_safe_err:.3g}). Per-case: "
            + "; ".join(details)
        )
    elif affected and safe_is_correct:
        detail = (
            "CONFIRMED: np.einsum gives wrong results for new-style dtype "
            f"operands (max abs error {worst_buggy_err:.3g} vs float64 "
            f"reference; numpy/numpy#32671). safe_einsum's naive_einsum "
            f"fallback matches the float64 reference (max abs error "
            f"{worst_safe_err:.3g}). Per-case: " + "; ".join(details)
        )
    elif affected and not safe_is_correct:
        detail = (
            "CONFIRMED bug present, but safe_einsum's own fallback ALSO "
            f"disagrees with the float64 reference (max abs error "
            f"{worst_safe_err:.3g}) -- do not trust the workaround on this "
            "numpy/dtype combination without investigating further. "
            "Per-case: " + "; ".join(details)
        )
    else:
        detail = (
            "Not reproduced: np.einsum matched the float64 reference "
            f"(max abs error {worst_buggy_err:.3g}) for new-style dtype "
            "operands -- numpy/numpy#32671 appears fixed here; safe_einsum "
            "will pass calls straight through for unaffected dtypes as "
            "always, and this dtype no longer needs the naive_einsum "
            "fallback either. Per-case: " + "; ".join(details)
        )

    return BugDetectionResult(
        numpy_version=np.__version__,
        dtype_name=dtype_name,
        affected=bool(affected),
        detail=detail,
    )
