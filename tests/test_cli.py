"""CLI tests for numpy-einsum-newdtype-guard."""
from __future__ import annotations

import json

import pytest

from numpy_einsum_newdtype_guard.cli import main

try:
    import numpy_quaddtype  # noqa: F401

    HAVE_QUADDTYPE = True
except ImportError:
    HAVE_QUADDTYPE = False


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "numpy-einsum-newdtype-guard" in out


def test_no_subcommand_requires_a_command():
    with pytest.raises(SystemExit):
        main([])


def test_detect_without_quaddtype_reports_clean_error(capsys, monkeypatch):
    """When numpy_quaddtype is not importable, detect must exit 2 with a
    clear, structured error -- never silently claim affected=False."""
    import numpy_einsum_newdtype_guard.cli as cli_mod

    def _raise_import_error():
        raise ImportError("no module named numpy_quaddtype (simulated)")

    # cli.py does `from .core import detect_einsum_newstyle_dtype_bug`, binding
    # the name directly into its own module namespace -- patch it there, not
    # on the core module object, or the cli's already-bound reference is
    # untouched and this test silently exercises the real live probe instead.
    monkeypatch.setattr(cli_mod, "detect_einsum_newstyle_dtype_bug", _raise_import_error)
    rc = main(["detect", "--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert "error" in payload
    assert rc == 2


@pytest.mark.skipif(not HAVE_QUADDTYPE, reason="numpy_quaddtype not installed")
def test_detect_json_has_required_fields(capsys):
    rc = main(["detect", "--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert "numpy_version" in payload
    assert "affected" in payload
    assert "detail" in payload
    assert rc in (0, 1)
    assert rc == (1 if payload["affected"] else 0)


@pytest.mark.skipif(not HAVE_QUADDTYPE, reason="numpy_quaddtype not installed")
def test_detect_text_mode_prints_status_headline(capsys):
    rc = main(["detect", "--no-color"])
    out = capsys.readouterr().out
    assert "numpy einsum new-style-dtype probe" in out
    assert "numpy version" in out
    assert rc in (0, 1)
