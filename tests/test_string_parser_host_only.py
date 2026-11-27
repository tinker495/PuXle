"""String and image parsers must render on the host.

Every parser walks its board element by element in Python. If it does so over a
device array, each element read costs a device synchronisation — one 23x23 maze
state took 1.2 s to render, so a 79-step solution path spent ~97 s in the
terminal visualiser alone.

`jax.transfer_guard("disallow")` rejects implicit transfers while still allowing
an explicit `jax.device_get`, so it fails exactly on the per-element pattern and
passes on "copy the struct to the host once, then use numpy".
"""

import jax
import jax.numpy as jnp
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


def _instantiate_puzzle(puzzle_class: type[Puzzle]) -> Puzzle:
    if puzzle_class is puxle.puzzles.CayleyPuzzle:
        pytest.importorskip("cayleypy")
        # Author: tinker495 — CayleyPuzzle requires a graph target; verify host-only
        # rendering via CayleyPresets rather than skipping coverage.
        from puxle import CayleyPresets

        return puzzle_class(**CayleyPresets.PANCAKE_7)
    if puzzle_class is puxle.puzzles.PDDL:
        # Author: tinker495 — PDDL requires (domain, problem); verify host-only
        # rendering via PDDLPresets rather than skipping coverage.
        from puxle import PDDLPresets

        return puzzle_class(**PDDLPresets.BLOCKSWORLD_S01)
    if puzzle_class is puxle.puzzles.WorldModelPuzzle:
        # Author: tinker495 — WorldModelPuzzle requires checkpoint/dataset config;
        # instantiate an offline tiny subclass to verify host-only string parsing.
        class _TinyWorldModel(puzzle_class):
            def data_init(self):
                self.inits = jnp.zeros((1, 2), dtype=jnp.uint8)
                self.targets = self.inits
                self.init_state_size = 1

        return _TinyWorldModel(
            data_path="unused",
            data_shape=(2,),
            latent_shape=(8,),
            action_size=2,
            init_params=True,
            str_parse_img=False,
        )
    return puzzle_class()


@pytest.mark.parametrize(
    "puzzle_class", _puzzle_classes(), ids=lambda cls: cls.__name__
)
def test_string_parser_does_not_touch_the_device(puzzle_class):
    puzzle = _instantiate_puzzle(puzzle_class)
    solve_config, state = puzzle.get_inits(jax.random.PRNGKey(42))

    with jax.transfer_guard("disallow"):
        rendered = state.str(solve_config=solve_config)

    assert isinstance(rendered, str)


@pytest.mark.parametrize(
    "puzzle_class",
    [cls for cls in _puzzle_classes() if cls is not puxle.puzzles.WorldModelPuzzle],
    ids=lambda cls: cls.__name__,
)
def test_img_parser_does_not_touch_the_device(puzzle_class):
    """Author: tinker495.

    Why: Verifies that ``attach_state_renderer`` copies inputs to host before
    invoking ``get_img_parser()`` so ``state.img(solve_config=...)`` triggers
    zero implicit host/device transfers across all host-rendered puzzles.
    """
    puzzle = _instantiate_puzzle(puzzle_class)
    solve_config, state = puzzle.get_inits(jax.random.PRNGKey(42))

    with jax.transfer_guard("disallow"):
        rendered = state.img(solve_config=solve_config)

    assert rendered.ndim == 3 and rendered.shape[2] == 3
