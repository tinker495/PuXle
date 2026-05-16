"""Architecture + behaviour guard for the Puzzle Renderer Module.

Locks the contracts documented in CONTEXT.md "Puzzle Renderer" and
"Puzzle Render Backend":

- ``puxle.core.puzzle_base`` consumes ``attach_state_renderer`` from
  ``puxle.render``, not the legacy ``add_img_parser`` from
  ``puxle.utils.util``.
- The legacy ``puxle.utils.util.add_img_parser`` still works (deprecated
  alias contract) so external puzzle authors do not break.
- ``puxle.render.Cv2Backend`` exposes the eight-method Puzzle Render
  Backend Interface and produces valid ``(H, W, 3)`` uint8 BGR images.
- Migrated puzzle modules (``slidepuzzle``, ``lightsout``, ``maze``) route
  every cv2 primitive through ``Cv2Backend`` and do not import cv2
  directly inside their own module bodies. Adding a new puzzle to the
  migrated set means appending it to ``_MIGRATED_PUZZLES`` below.
"""

from __future__ import annotations

import importlib
import inspect
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

_BACKEND_INTERFACE_METHODS = (
    "canvas",
    "canvas_from_mono",
    "rect",
    "circle",
    "line",
    "text",
    "text_size",
    "text_centered",
    "ellipse",
    "fill_poly",
    "polylines",
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


def test_legacy_add_img_parser_still_delegates():
    """External puzzle authors that import add_img_parser must keep working."""
    util = importlib.import_module("puxle.utils.util")
    render = importlib.import_module("puxle.render")
    assert hasattr(util, "add_img_parser")
    assert hasattr(render, "attach_state_renderer")

    def fake_imgfunc(state, **kwargs):
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


def test_render_module_exposes_cv2_backend():
    render = importlib.import_module("puxle.render")
    assert hasattr(render, "Cv2Backend"), (
        "puxle.render must export Cv2Backend so puzzles can `from puxle.render "
        "import Cv2Backend`."
    )


def test_cv2_backend_exposes_full_interface():
    from puxle.render import Cv2Backend

    backend = Cv2Backend()
    for method_name in _BACKEND_INTERFACE_METHODS:
        method = getattr(backend, method_name, None)
        assert callable(method), (
            f"Cv2Backend must define `{method_name}` per the eight-method "
            "Puzzle Render Backend Interface in CONTEXT.md."
        )

    public_methods = {
        name
        for name, value in inspect.getmembers(backend, predicate=callable)
        if not name.startswith("_")
    }
    extra = public_methods - set(_BACKEND_INTERFACE_METHODS)
    assert not extra, (
        "Cv2Backend exposes unexpected public methods beyond the documented "
        f"Puzzle Render Backend Interface: {sorted(extra)}. Update CONTEXT.md "
        "or remove the method to keep the seam small."
    )


def test_cv2_backend_canvas_round_trip():
    from puxle.render import Cv2Backend

    backend = Cv2Backend()
    img = backend.canvas(size=(8, 8), fill_bgr=(10, 20, 30))
    assert img.shape == (8, 8, 3)
    assert img.dtype == np.uint8
    assert tuple(img[0, 0]) == (10, 20, 30)

    img = backend.rect(img, top_left=(1, 1), bottom_right=(4, 4), color_bgr=(255, 0, 0))
    assert tuple(img[2, 2]) == (255, 0, 0)


def test_migrated_puzzle_modules_do_not_import_cv2_directly():
    """Migrated puzzles must consume Cv2Backend rather than importing cv2.

    This guard prevents regressions where a contributor adds a new cv2 call
    inside an already-migrated puzzle's ``imgfunc`` body. Puzzles still
    pending migration are allowed to import cv2 inside their
    ``get_img_parser`` bodies; they are intentionally absent from
    ``_MIGRATED_PUZZLES``.
    """
    for name in _MIGRATED_PUZZLES:
        source = _puzzle_module_source(name)
        assert "import cv2" not in source, (
            f"Migrated puzzle `{name}` must not import cv2 directly; consume "
            "puxle.render.Cv2Backend instead. If you intentionally need a cv2 "
            "primitive that is not yet on Cv2Backend, extend the backend "
            "Interface and document the addition in CONTEXT.md."
        )
        assert "from puxle.render import Cv2Backend" in source, (
            f"Migrated puzzle `{name}` should import Cv2Backend from "
            "puxle.render to render through the canonical seam."
        )


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
