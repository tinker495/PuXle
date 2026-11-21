"""Architecture and behaviour guard for the Puzzle Renderer Module.

Locks the contracts documented in CONTEXT.md "Puzzle Renderer":

- ``puxle.core.puzzle_base`` consumes ``attach_state_renderer`` from
  ``puxle.render``, never an ``add_img_parser`` alias in ``puxle.utils.util``.
- Puzzle modules render directly with cv2 without a ``Cv2Backend`` wrapper.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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


def test_render_module_exposes_only_canonical_helpers():
    """Author: tinker495.

    Why: Locks the canonical ``puxle.render`` public surface
    (``attach_state_renderer``, ``normalize_rgb_image``, ``img_to_colored_str``)
    and forbids exposing ``Cv2Backend``.
    """
    render = importlib.import_module("puxle.render")
    assert render.__all__ == [
        "attach_state_renderer",
        "normalize_rgb_image",
        "img_to_colored_str",
    ]
    assert not hasattr(render, "Cv2Backend")


def test_normalize_rgb_image_enforces_host_uint8_and_optional_resize():
    """Author: tinker495.

    Why: Ensures ``normalize_rgb_image`` always produces a host NumPy ``uint8``
    ``(H, W, 3)`` RGB frame (converting float ``[0, 1]`` and JAX arrays),
    supports ``size=(width, height)`` resizing, and rejects invalid shapes or
    non-positive target dimensions loudly.
    """
    from puxle.render import normalize_rgb_image

    float_frame = jnp.array([[[0.0, 0.5, 1.0], [1.0, 0.0, 0.25]]], dtype=jnp.float32)
    out = normalize_rgb_image(float_frame)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.uint8
    assert out.shape == (1, 2, 3)
    assert out[0, 0].tolist() == [0, 128, 255]
    assert out[0, 1].tolist() == [255, 0, 64]

    resized = normalize_rgb_image(out, size=(6, 4))
    assert resized.shape == (4, 6, 3)
    assert resized.dtype == np.uint8

    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        normalize_rgb_image(np.zeros((2, 2, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="positive"):
        normalize_rgb_image(out, size=(0, 4))


def test_img_to_colored_str_renders_24bit_ansi_rows():
    """Author: tinker495.

    Why: Verifies the shared ``puxle.render`` terminal RGB preview helper
    formats 24-bit ANSI ``██`` rows consistently for both world models and
    adapters.
    """
    from puxle.render import img_to_colored_str

    img = np.array([[[255, 0, 10], [0, 128, 64]]], dtype=np.uint8)
    rendered = img_to_colored_str(img)
    assert rendered == ("\x1b[38;2;255;0;10m██\x1b[0m\x1b[38;2;0;128;64m██\x1b[0m")


def test_attach_state_renderer_centralizes_str_and_img_on_single_and_batch():
    """Author: tinker495.

    Why: Verifies that ``Puzzle.__init__`` binds both ``str``/``__str__`` and
    ``img(size=...)`` centrally through ``attach_state_renderer`` for both
    single and batched states without per-puzzle ``__str__`` boilerplate.
    """
    from puxle.puzzles.slidepuzzle import SlidePuzzle

    puzzle = SlidePuzzle(size=3)
    sc, state = puzzle.get_inits(jax.random.PRNGKey(0))
    neighbours, _ = puzzle.get_neighbours(sc, state)

    assert str(state) == state.str()
    assert isinstance(neighbours.str(solve_config=sc), str)
    assert "Batched" in neighbours.str(solve_config=sc)

    single_img = state.img(size=(32, 24))
    assert isinstance(single_img, np.ndarray)
    assert single_img.shape == (24, 32, 3)
    assert single_img.dtype == np.uint8

    batched_img = neighbours.img(size=(16, 16))
    assert isinstance(batched_img, np.ndarray)
    assert batched_img.shape == (puzzle.action_size, 16, 16, 3)
    assert batched_img.dtype == np.uint8


def test_migrated_puzzle_modules_do_not_import_cv2_backend():
    """Puzzle modules must render directly with cv2 without importing Cv2Backend."""
    for name in _MIGRATED_PUZZLES:
        source = _puzzle_module_source(name)
        assert "Cv2Backend" not in source
        assert "puxle.render.backends" not in source


def test_migrated_puzzle_imgfuncs_produce_valid_images():
    """Verify each cv2-rendered puzzle's ``get_img_parser`` returns a valid uint8 RGB image."""
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
