"""Subprocess worker: run ONE potentially-crashing np.einsum call, on
new-style dtype operands, in an isolated child process.

numpy/numpy#32671's own root cause (an out-of-bounds table index from
type_num == -1) is undefined behavior: the upstream report itself notes
it can manifest as either a SILENTLY WRONG VALUE (observed on this
project's macOS/arm64 dev host) or a hard SEGFAULT (observed on
ubuntu-latest x86_64 CI for this exact project, and separately noted by
the upstream reporter on Windows for a different expression). Running
each risky einsum call in its own subprocess means a crash only kills
that subprocess -- this is the only reliable way to test a bug that can
corrupt process memory without taking down the caller (`detect_*`, the
CLI, or an entire pytest session) with it.

Invoked as ``python -m numpy_einsum_newdtype_guard._worker <case>``.
Never import/call this module's functions expecting them to be
crash-safe when called directly in-process -- the whole point is that
they run in a throwaway child.
"""
from __future__ import annotations

import json
import sys

# Same fixed seed/shapes as core.py's reference computation, so the
# worker's quad-precision operands are constructed identically to what
# the caller compares against.
_SEED = 0
_SHAPE = (2, 6, 6)

_CASES = ("matmul_contraction", "scalar_reduction")


def _make_operands():
    import numpy as np
    from numpy_quaddtype import QuadPrecDType

    rng = np.random.default_rng(_SEED)
    a64, b64 = rng.standard_normal(_SHAPE)
    a = np.asarray(a64, dtype=QuadPrecDType())
    b = np.asarray(b64, dtype=QuadPrecDType())
    return a, b


def _run_case(case: str) -> dict:
    import numpy as np

    a, b = _make_operands()
    if case == "matmul_contraction":
        result = np.einsum("ij,jk->ik", a, b)  # may crash -- isolated by design
        as_list = np.asarray(result, dtype=np.float64).tolist()
    elif case == "scalar_reduction":
        result = np.einsum("i,i->", a[0], a[0])  # may crash -- isolated by design
        as_list = float(np.asarray(result, dtype=np.float64))
    else:
        raise SystemExit(f"unknown case: {case!r} (expected one of {_CASES})")
    return {"case": case, "result": as_list}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        raise SystemExit(f"usage: python -m numpy_einsum_newdtype_guard._worker <case>")
    result = _run_case(argv[0])
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
