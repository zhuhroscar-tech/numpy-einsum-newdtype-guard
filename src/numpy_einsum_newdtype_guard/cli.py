"""numpy-einsum-newdtype-guard CLI: probe the installed numpy for the
einsum new-style-dtype bug (numpy/numpy#32671) and demonstrate the safe
fallback against a float64 reference.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import __version__
from .core import BugDetectionResult, detect_einsum_newstyle_dtype_bug
from .style import print_fields, resolve_style, status_headline


def cmd_detect(args: argparse.Namespace) -> int:
    try:
        result = detect_einsum_newstyle_dtype_bug()
    except ImportError as exc:
        payload = {
            "error": (
                "numpy_quaddtype is required for the live probe "
                f"(pip install numpy_quaddtype): {exc}"
            )
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            style = resolve_style(args.no_color)
            print(status_headline(style, "warn", "cannot probe: numpy_quaddtype not installed"))
            print_fields([("detail", payload["error"])])
        return 2

    if args.json:
        print(json.dumps(result.__dict__, indent=2, sort_keys=True))
        return 1 if result.affected else 0

    style = resolve_style(args.no_color)
    level = "fail" if result.affected else "ok"
    print(status_headline(style, level, "numpy einsum new-style-dtype probe"))
    print_fields(
        [
            ("numpy version", result.numpy_version),
            ("probe dtype", result.dtype_name),
            ("affected", "yes" if result.affected else "no"),
            ("detail", result.detail),
        ]
    )
    return 1 if result.affected else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="numpy-einsum-newdtype-guard",
        description=(
            "Detect and work around numpy/numpy#32671: np.einsum silently "
            "returns wrong results (or crashes) for new-style dtype operands."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"numpy-einsum-newdtype-guard {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    detect = sub.add_parser("detect", help="probe the installed numpy for the bug")
    detect.add_argument("--json", action="store_true")
    detect.add_argument("--no-color", action="store_true")
    detect.set_defaults(func=cmd_detect)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
