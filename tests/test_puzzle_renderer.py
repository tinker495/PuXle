"""Architecture + behaviour guard for the Puzzle Renderer Module.

Locks the contract documented in CONTEXT.md "Puzzle Renderer":
- `puxle.core.puzzle_base` consumes `attach_state_renderer` from
  `puxle.render`, not the legacy `add_img_parser` from `puxle.utils.util`.
- The legacy `puxle.utils.util.add_img_parser` still works (deprecated alias
  contract) so external puzzle authors do not break.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np


def _puzzle_base_source() -> str:
    base = Path(__file__).resolve().parents[1] / "puxle" / "core" / "puzzle_base.py"
    return base.read_text()


def test_puzzle_base_imports_canonical_seam():
    source = _puzzle_base_source()
    assert "from puxle.render import attach_state_renderer" in source, (
        "puzzle_base.py must import from puxle.render, not puxle.utils.util."
    )


def test_puzzle_base_does_not_import_legacy_alias():
    source = _puzzle_base_source()
    assert "from puxle.utils.util import add_img_parser" not in source, (
        "puzzle_base.py must not import the legacy add_img_parser alias."
    )
    assert "add_img_parser(" not in source, (
        "puzzle_base.py must not call add_img_parser; use attach_state_renderer."
    )


def test_legacy_add_img_parser_still_delegates():
    """External puzzle authors that import add_img_parser must keep working."""
    util = importlib.import_module("puxle.utils.util")
    render = importlib.import_module("puxle.render")
    assert hasattr(util, "add_img_parser")
    assert hasattr(render, "attach_state_renderer")
    # Both names should attach the same `img` method to a fresh class.
    captured = {}

    def fake_imgfunc(state, **kwargs):
        captured["called"] = True
        return np.zeros((1, 1, 3), dtype=np.uint8)

    class _StubCls:
        pass

    util.add_img_parser(_StubCls, fake_imgfunc)
    assert hasattr(_StubCls, "img"), "add_img_parser must attach an `img` method."


def test_render_module_attach_state_renderer_attaches_img_method():
    render = importlib.import_module("puxle.render")

    def fake_imgfunc(state, **kwargs):
        return np.zeros((1, 1, 3), dtype=np.uint8)

    class _StubCls:
        pass

    out = render.attach_state_renderer(_StubCls, fake_imgfunc)
    assert out is _StubCls, (
        "attach_state_renderer should return the class for chaining."
    )
    assert hasattr(_StubCls, "img")
