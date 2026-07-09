"""Base-class contract test for the filled/inf-cost action invariant.

Puzzles implement only the pure transition ``_apply``; ``Puzzle.get_actions``
composes the contract shared by every puzzle: a move is taken only when it is
intrinsically valid (finite ``_apply`` cost) *and* ``filled``, otherwise the
original state is returned with ``jnp.inf`` cost. One test guarantees that
invariant for every puzzle routed through the base seam — a class is picked up
automatically as soon as it drops its own ``get_actions`` in favour of ``_apply``.
"""

import inspect

import jax
import jax.numpy as jnp
import pytest

import puxle.puzzles as puzzles_module
from puxle.core.puzzle_base import Puzzle


def _seam_puzzles():
    """Default-constructible puzzles that route through the base get_actions seam."""
    out = []
    for name in dir(puzzles_module):
        if name == "CayleyPuzzle":  # abstract base, needs ctor args
            continue
        cls = getattr(puzzles_module, name)
        if (
            inspect.isclass(cls)
            and issubclass(cls, Puzzle)
            and cls is not Puzzle
            and cls.get_actions is Puzzle.get_actions  # routed through the base seam
        ):
            out.append(cls)
    return out


@pytest.mark.parametrize("puzzle_class", _seam_puzzles(), ids=lambda c: c.__name__)
def test_action_contract(puzzle_class):
    key = jax.random.PRNGKey(0)
    try:
        puzzle = puzzle_class()
    except ModuleNotFoundError as e:  # optional deps (e.g. cayleypy) absent
        pytest.skip(f"{puzzle_class.__name__} requires missing dependency: {e.name}")
    solve_config = puzzle.get_solve_config(key=key)
    state = puzzle.get_initial_state(solve_config, key=key)
    name = puzzle_class.__name__

    # filled=False blocks every action: inf cost and the original state, unchanged.
    blocked_states, blocked_costs = puzzle.get_neighbours(
        solve_config, state, filled=False
    )
    assert jnp.all(jnp.isinf(blocked_costs)), f"{name}: filled=False must be all inf"
    for i in range(puzzle.action_size):
        blocked_i = jax.tree_util.tree_map(lambda x: x[i], blocked_states)
        assert blocked_i == state, (
            f"{name}: filled=False must preserve state (action {i})"
        )

    # filled=True: get_actions must agree with get_neighbours, and some move must exist.
    n_states, n_costs = puzzle.get_neighbours(solve_config, state, filled=True)
    assert jnp.any(jnp.isfinite(n_costs)), f"{name}: at least one move must be possible"
    for i in range(puzzle.action_size):
        s_i, c_i = puzzle.get_actions(solve_config, state, jnp.asarray(i), filled=True)
        expected = jax.tree_util.tree_map(lambda x: x[i], n_states)
        assert s_i == expected, f"{name}: get_actions state mismatch at action {i}"
        assert jnp.array_equal(c_i, n_costs[i]), (
            f"{name}: get_actions cost mismatch at action {i}"
        )
