"""Architecture + behaviour guard for the Puzzle Renderer Module.

Locks the contracts documented in CONTEXT.md "Puzzle Renderer":

- ``puxle.core.puzzle_base`` consumes ``attach_state_renderer`` from
  ``puxle.render``, not the removed ``add_img_parser`` alias from
  ``puxle.utils.util``.
- Migrated puzzle modules render directly with cv2 and no longer depend on
  the deleted ``Cv2Backend`` wrapper.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

_MIGRATED_PUZZLES = (
    "slidepuzzle",
    "lightsout",
    "maze",
    "dotknot",
    "hanoi",
    "pancake",
    "topspin",
    "tsp",
    "rubikscube",
)


def _puzzle_base_source() -> str:
    base = Path(__file__).resolve().parents[1] / "puxle" / "core" / "puzzle_base.py"
    return base.read_text()


def _puzzle_module_source(name: str) -> str:
    path = Path(__file__).resolve().parents[1] / "puxle" / "puzzles" / f"{name}.py"
    return path.read_text()


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


def test_legacy_add_img_parser_alias_is_removed():
    util = importlib.import_module("puxle.utils.util")
    utils = importlib.import_module("puxle.utils")
    render = importlib.import_module("puxle.render")
    assert not hasattr(util, "add_img_parser")
    assert not hasattr(utils, "add_img_parser")
    assert hasattr(render, "attach_state_renderer")


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


def test_render_module_exposes_only_attachment_seam():
    render = importlib.import_module("puxle.render")
    assert render.__all__ == ["attach_state_renderer"]
    assert not hasattr(render, "Cv2Backend")


def test_migrated_puzzle_modules_do_not_import_cv2_backend():
    """Migrated puzzles must not depend on the deleted Cv2Backend wrapper."""
    for name in _MIGRATED_PUZZLES:
        source = _puzzle_module_source(name)
        assert "Cv2Backend" not in source
        assert "puxle.render.backends" not in source


def test_migrated_puzzle_imgfuncs_produce_valid_images():
    """End-to-end smoke for the migrated puzzles' ``get_img_parser`` results."""
    import jax

    from puxle.puzzles.dotknot import DotKnot
    from puxle.puzzles.hanoi import TowerOfHanoi
    from puxle.puzzles.lightsout import LightsOut
    from puxle.puzzles.maze import Maze
    from puxle.puzzles.pancake import PancakeSorting
    from puxle.puzzles.rubikscube import RubiksCube
    from puxle.puzzles.slidepuzzle import SlidePuzzle
    from puxle.puzzles.topspin import TopSpin
    from puxle.puzzles.tsp import TSP

    key = jax.random.PRNGKey(0)

    sp = SlidePuzzle(size=3)
    sp_sc, sp_state = sp.get_inits(key)
    sp_img = sp.get_img_parser()(sp_state)
    assert sp_img.ndim == 3 and sp_img.shape[-1] == 3 and sp_img.dtype == np.uint8

    lo = LightsOut(size=3, initial_shuffle=2)
    lo_sc, lo_state = lo.get_inits(key)
    lo_img = lo.get_img_parser()(lo_state)
    assert lo_img.ndim == 3 and lo_img.shape[-1] == 3 and lo_img.dtype == np.uint8

    mz = Maze(size=11)
    mz_sc, mz_state = mz.get_inits(key)
    mz_img = mz.get_img_parser()(mz_state, solve_config=mz_sc)
    assert mz_img.ndim == 3 and mz_img.shape[-1] == 3 and mz_img.dtype == np.uint8

    dk = DotKnot()
    dk_sc, dk_state = dk.get_inits(key)
    dk_img = dk.get_img_parser()(dk_state)
    assert dk_img.ndim == 3 and dk_img.shape[-1] == 3 and dk_img.dtype == np.uint8

    th = TowerOfHanoi()
    th_sc, th_state = th.get_inits(key)
    th_img = th.get_img_parser()(th_state)
    assert th_img.ndim == 3 and th_img.shape[-1] == 3 and th_img.dtype == np.uint8

    pc = PancakeSorting()
    pc_sc, pc_state = pc.get_inits(key)
    pc_img = pc.get_img_parser()(pc_state)
    assert pc_img.ndim == 3 and pc_img.shape[-1] == 3 and pc_img.dtype == np.uint8

    ts = TopSpin()
    ts_sc, ts_state = ts.get_inits(key)
    ts_img = ts.get_img_parser()(ts_state)
    assert ts_img.ndim == 3 and ts_img.shape[-1] == 3 and ts_img.dtype == np.uint8

    tsp = TSP()
    tsp_sc, tsp_state = tsp.get_inits(key)
    tsp_img = tsp.get_img_parser()(tsp_state, solve_config=tsp_sc)
    assert tsp_img.ndim == 3 and tsp_img.shape[-1] == 3 and tsp_img.dtype == np.uint8

    rc = RubiksCube(size=3)
    rc_sc, rc_state = rc.get_inits(key)
    rc_img = rc.get_img_parser()(rc_state)
    assert rc_img.ndim == 3 and rc_img.shape[-1] == 3 and rc_img.dtype == np.uint8
