"""Tests for numpy_einsum_newdtype_guard.core.

These tests are written to distinguish three real scenarios rather than
assume the bug's presence, and to work whether or not the optional
`numpy_quaddtype` dependency is installed:

1. `naive_einsum` (the independent oracle/fallback) is correct on its own,
   verified against plain numpy.einsum on ordinary float64 arrays for a
   variety of subscript shapes -- this does NOT require numpy_quaddtype
   and always runs.
2. `safe_einsum` passes plain-dtype calls straight through to numpy.einsum
   unmodified (fast path) -- always runs.
3. The LIVE bug reproduction against the real numpy_quaddtype package
   (numpy/numpy#32671) -- skipped if numpy_quaddtype is not installed,
   never silently treated as "not affected".
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from numpy_einsum_newdtype_guard.core import (
    BugDetectionResult,
    is_new_style_dtype,
    naive_einsum,
    safe_einsum,
)

try:
    import numpy_quaddtype  # noqa: F401

    HAVE_QUADDTYPE = True
except ImportError:
    HAVE_QUADDTYPE = False

requires_quaddtype = pytest.mark.skipif(
    not HAVE_QUADDTYPE, reason="numpy_quaddtype not installed"
)


# --- naive_einsum correctness (independent oracle), no special dtype needed ---


@pytest.mark.parametrize(
    "subs,shapes",
    [
        ("ij,jk->ik", [(3, 4), (4, 5)]),
        ("ij,ij->ij", [(3, 4), (3, 4)]),
        ("ij->ji", [(3, 5)]),
        ("i,i->", [(6,), (6,)]),
        ("ijk,klmn,nop,jmp->ilo", [(2, 3, 4), (4, 2, 3, 5), (5, 6, 2), (3, 3, 2)]),
        ("ab,bc,cd->ad", [(2, 3), (3, 4), (4, 2)]),
    ],
)
def test_naive_einsum_matches_numpy_on_float64(subs, shapes):
    rng = np.random.default_rng(0)
    operands = [rng.standard_normal(shape) for shape in shapes]
    expected = np.einsum(subs, *operands)
    got = naive_einsum(subs, *operands)
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=1e-10, atol=1e-10)


def test_naive_einsum_requires_explicit_output_subscript():
    with pytest.raises(ValueError, match="explicit"):
        naive_einsum("ij,jk", np.eye(2), np.eye(2))


def test_naive_einsum_rejects_ellipsis():
    with pytest.raises(ValueError, match="ellipsis"):
        naive_einsum("...ij->...ji", np.eye(2))


def test_naive_einsum_rejects_repeated_index_within_one_operand():
    with pytest.raises(ValueError, match="repeated indices"):
        naive_einsum("ii->i", np.eye(3))


def test_naive_einsum_rejects_mismatched_operand_count():
    with pytest.raises(ValueError, match="operand"):
        naive_einsum("ij,jk->ik", np.eye(2))


def test_naive_einsum_rejects_inconsistent_dimension():
    a = np.ones((3, 4))
    b = np.ones((5, 6))  # 'j' would be 4 from a, 5 from b -- inconsistent
    with pytest.raises(ValueError, match="inconsistent dimension"):
        naive_einsum("ij,jk->ik", a, b)


def test_naive_einsum_rejects_unknown_output_index():
    with pytest.raises(ValueError, match="does not appear"):
        naive_einsum("ij->k", np.eye(2))


# --- is_new_style_dtype ---


def test_is_new_style_dtype_false_for_ordinary_dtypes():
    assert is_new_style_dtype(np.float64) is False
    assert is_new_style_dtype(np.int32) is False
    assert is_new_style_dtype(np.dtype("complex128")) is False


@requires_quaddtype
def test_is_new_style_dtype_true_for_quadprec():
    from numpy_quaddtype import QuadPrecDType

    assert is_new_style_dtype(QuadPrecDType()) is True


# --- safe_einsum fast-path passthrough (no special dtype needed) ---


def test_safe_einsum_passes_through_for_ordinary_dtypes():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((4, 5))
    b = rng.standard_normal((5, 6))
    expected = np.einsum("ij,jk->ik", a, b)
    got = safe_einsum("ij,jk->ik", a, b)
    np.testing.assert_array_equal(got, expected)


def test_safe_einsum_honors_optimize_kwarg_on_fast_path():
    rng = np.random.default_rng(2)
    ops = [rng.standard_normal((3, 3)) for _ in range(3)]
    expected = np.einsum("ij,jk,kl->il", *ops, optimize=True)
    got = safe_einsum("ij,jk,kl->il", *ops, optimize=True)
    np.testing.assert_allclose(got, expected)


# --- live bug reproduction against the real third-party package ---


@requires_quaddtype
def test_live_reproduction_of_numpy_32671():
    """Direct, no-mocking reproduction using the exact package and shapes
    the upstream issue itself uses. Does not hard-assert affected=True:
    if a numpy release fixes this, that is real, desirable information
    this guard should surface truthfully. Asserts internal consistency
    and, critically, that safe_einsum always agrees with numpy's own
    unaffected matmul path on the same data regardless of bug status."""
    from numpy_quaddtype import QuadPrecDType

    from numpy_einsum_newdtype_guard.core import detect_einsum_newstyle_dtype_bug

    result = detect_einsum_newstyle_dtype_bug()
    assert isinstance(result, BugDetectionResult)
    assert result.numpy_version == np.__version__
    assert isinstance(result.affected, bool)
    assert len(result.detail) > 0

    rng = np.random.default_rng(0)
    a64, b64 = rng.standard_normal((2, 6, 6))
    a = np.asarray(a64, dtype=QuadPrecDType())
    b = np.asarray(b64, dtype=QuadPrecDType())

    # safe_einsum must ALWAYS match the float64 reference for this dtype,
    # regardless of whether numpy's own einsum is currently buggy --
    # that is the entire point of the guard.
    safe_result = np.asarray(safe_einsum("ij,jk->ik", a, b), dtype=np.float64)
    reference = np.einsum("ij,jk->ik", a64, b64)
    np.testing.assert_allclose(safe_result, reference, rtol=1e-6, atol=1e-6)


@requires_quaddtype
def test_regression_this_exact_case_was_wrong_or_crashes_before_the_fix():
    """Regression test for the exact minimal case from numpy/numpy#32671,
    run through the isolated-subprocess worker (never in-process -- the
    upstream root cause is undefined behavior and has been independently
    observed to SEGFAULT on ubuntu-latest x86_64 CI for this exact guard,
    even though it only produces a silently wrong value on macOS/arm64;
    calling the risky np.einsum directly in this test process crashed the
    entire pytest session on first push -- see v23 in the fleet's
    adaptive prompt history).

    Either outcome (worker crash, OR worker survives but returns a value
    that disagrees with the float64 reference by more than trivial
    rounding error) counts as reproducing the bug. If numpy ever fixes
    this upstream, the worker will survive AND match the reference,
    which is real, desirable information -- this test would then fail
    loudly, which is the correct signal to revisit this guard's
    'affected' framing, not a guard bug."""
    import subprocess

    rng = np.random.default_rng(0)
    a64, b64 = rng.standard_normal((2, 6, 6))
    reference = np.einsum("ij,jk->ik", a64, b64)

    proc = subprocess.run(
        [sys.executable, "-m", "numpy_einsum_newdtype_guard._worker", "matmul_contraction"],
        capture_output=True,
        text=True,
        timeout=60,
    )

    from numpy_quaddtype import QuadPrecDType

    a = np.asarray(a64, dtype=QuadPrecDType())
    b = np.asarray(b64, dtype=QuadPrecDType())
    safe = np.asarray(safe_einsum("ij,jk->ik", a, b), dtype=np.float64)
    safe_err = float(np.max(np.abs(safe - reference)))
    assert safe_err < 1e-6

    if proc.returncode != 0:
        # A crash IS the bug reproduced (undefined behavior manifesting
        # as a segfault rather than a wrong value on this platform).
        return

    payload = json.loads(proc.stdout)
    buggy = np.asarray(payload["result"], dtype=np.float64)
    buggy_err = float(np.max(np.abs(buggy - reference)))

    # This assertion documents the bug as observed at guard-creation time
    # (2026-09-21, numpy 2.5.3): silently wrong by a large margin, not
    # trivial floating-point noise.
    assert buggy_err > 1e-3, (
        "np.einsum matched the float64 reference for QuadPrecDType operands "
        "-- numpy/numpy#32671 may be fixed upstream; re-evaluate this guard's "
        "'affected' framing rather than treating this failure as a guard bug."
    )
