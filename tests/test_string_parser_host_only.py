"""String parsers must render on the host.

Every parser walks its board element by element in Python. If it does so over a
device array, each element read costs a device synchronisation — one 23x23 maze
state took 1.2 s to render, so a 79-step solution path spent ~97 s in the
terminal visualiser alone.

`jax.transfer_guard("disallow")` rejects implicit transfers while still allowing
an explicit `jax.device_get`, so it fails exactly on the per-element pattern and
passes on "copy the struct to the host once, then use numpy".
"""

import jax
import pytest

import puxle.puzzles
from puxle.core.puzzle_base import Puzzle


def _puzzle_classes():
    # The package exports lazily, so walk `__all__` rather than its module dict.
    exported = (getattr(puxle.puzzles, name) for name in sorted(puxle.puzzles.__all__))
    return [
        attr
        for attr in exported
        if isinstance(attr, type) and issubclass(attr, Puzzle) and attr is not Puzzle
    ]


@pytest.mark.parametrize(
    "puzzle_class", _puzzle_classes(), ids=lambda cls: cls.__name__
)
def test_string_parser_does_not_touch_the_device(puzzle_class):
    if puzzle_class is puxle.puzzles.CayleyPuzzle:
        pytest.skip("CayleyPuzzle requires a graph argument")
    if issubclass(puzzle_class, puxle.puzzles.CayleyPuzzle):
        pytest.importorskip("cayleypy")

    puzzle = puzzle_class()

    solve_config, state = puzzle.get_inits(jax.random.PRNGKey(42))

    with jax.transfer_guard("disallow"):
        rendered = state.str(solve_config=solve_config)

    assert isinstance(rendered, str)
