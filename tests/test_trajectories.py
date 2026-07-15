import importlib
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from puxle.core.trajectory import (
    create_hindsight_target_shuffled_path,
    create_hindsight_target_triangular_shuffled_path,
    create_target_shuffled_path,
)
from puxle.puzzles.cayley_puzzle import CayleyPuzzle
from puxle.puzzles.lightsout import LightsOut
from puxle.puzzles.rubikscube import RubiksCube
from puxle.puzzles.slidepuzzle import SlidePuzzle
from puxle.puzzles.sokoban import Sokoban
from puxle.puzzles.topspin import TopSpin

scramble = importlib.import_module("puxle.core.scramble")


def _assert_tree_equal(left, right):
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves):
        assert jnp.array_equal(left_leaf, right_leaf)


def _perm_graph_def(
    permutations: list[list[int]],
    central_state: list[int],
    name: str = "test_perm",
) -> SimpleNamespace:
    generators_type = SimpleNamespace(name="PERMUTATION")
    return SimpleNamespace(
        generators_type=generators_type,
        generators_permutations=np.array(permutations, dtype=np.int32),
        generators_matrices=None,
        central_state=central_state,
        name=name,
    )


def _small_cayley_puzzle() -> CayleyPuzzle:
    n = 4
    rot_left = list(range(1, n)) + [0]
    rot_right = [n - 1] + list(range(n - 1))
    graph_def = _perm_graph_def([rot_left, rot_right], list(range(n)), name="cyclic4")
    return CayleyPuzzle(graph_def, ensure_inverse_closed=True, num_shuffle=2)


def test_random_trajectory_fast_path():
    rc = RubiksCube(size=3)
    key = jax.random.PRNGKey(42)

    # Fast path: reversible + non_backtracking_steps=1
    traj = rc.batched_get_random_trajectory(
        k_max=5, shuffle_parallel=2, key=key, non_backtracking_steps=1
    )

    assert traj.states.faces.shape == (6, 2, 21), "State shape mismatch"
    assert traj.actions.shape == (5, 2), "Action shape mismatch"
    assert traj.move_costs.shape == (6, 2), "Move costs shape mismatch"
    assert traj.move_costs_tm1.shape == (6, 2), "Move costs tm1 shape mismatch"
    assert traj.action_costs.shape == (5, 2), "Action costs shape mismatch"


def test_random_trajectory_legacy_path():
    sp = SlidePuzzle(size=4)
    key = jax.random.PRNGKey(42)

    # Legacy path: non_backtracking_steps=3
    traj = sp.batched_get_random_trajectory(
        k_max=5, shuffle_parallel=2, key=key, non_backtracking_steps=3
    )

    assert traj.states.board_unpacked.shape == (6, 2, 16), "State shape mismatch"
    assert traj.actions.shape == (5, 2), "Action shape mismatch"
    assert traj.move_costs.shape == (6, 2), "Move costs shape mismatch"


def test_random_trajectory_transitions_are_finite_and_aligned():
    sp = SlidePuzzle(size=3)
    traj = sp.batched_get_random_trajectory(
        k_max=20,
        shuffle_parallel=64,
        key=jax.random.PRNGKey(0),
        non_backtracking_steps=3,
    )

    replayed_states, selected_costs = jax.vmap(
        lambda states, actions: sp.batched_get_actions(
            traj.solve_configs,
            states,
            actions,
            filleds=jnp.ones(64, dtype=jnp.bool_),
            multi_solve_config=True,
        )
    )(traj.states[:-1], traj.actions)

    assert jnp.all(jnp.isfinite(selected_costs))
    assert jnp.array_equal(selected_costs, traj.action_costs)
    assert jnp.array_equal(
        traj.move_costs[1:] - traj.move_costs[:-1], traj.action_costs
    )
    _assert_tree_equal(replayed_states, traj.states[1:])
    for lag in range(1, 5):
        assert not jnp.any(
            jnp.all(
                traj.states.board_unpacked[lag:] == traj.states.board_unpacked[:-lag],
                axis=-1,
            )
        )


def test_random_inverse_trajectory_fast_path():
    rc = RubiksCube(size=3)
    key = jax.random.PRNGKey(42)

    traj = rc.batched_get_random_inverse_trajectory(
        k_max=4, shuffle_parallel=3, key=key, non_backtracking_steps=1
    )

    assert traj.states.faces.shape == (5, 3, 21), "State shape mismatch"
    assert traj.actions.shape == (4, 3), "Action shape mismatch"


def test_random_inverse_trajectory_legacy_path():
    sp = SlidePuzzle(size=3)
    key = jax.random.PRNGKey(42)

    traj = sp.batched_get_random_inverse_trajectory(
        k_max=4, shuffle_parallel=3, key=key, non_backtracking_steps=2
    )

    assert traj.states.board_unpacked.shape == (5, 3, 9), "State shape mismatch"
    assert traj.actions.shape == (4, 3), "Action shape mismatch"


def test_scramble_module_imports_without_puzzle_base_cycle():
    assert scramble._get_shuffled_state
    assert scramble._batched_get_random_trajectory
    assert scramble._batched_get_random_inverse_trajectory


def test_puzzle_scramble_methods_delegate_to_module(monkeypatch):
    rc = RubiksCube(size=3)
    key = jax.random.PRNGKey(42)
    solve_config, initial_state = rc.get_inits(key)
    calls = []
    sentinel = object()

    def fake_get_shuffled_state(puzzle, *args):
        calls.append(("shuffled", puzzle, args))
        return sentinel

    def fake_random_trajectory(puzzle, *args):
        calls.append(("trajectory", puzzle, args))
        return sentinel

    def fake_random_inverse_trajectory(puzzle, *args):
        calls.append(("inverse", puzzle, args))
        return sentinel

    monkeypatch.setattr(scramble, "_get_shuffled_state", fake_get_shuffled_state)
    monkeypatch.setattr(
        scramble, "_batched_get_random_trajectory", fake_random_trajectory
    )
    monkeypatch.setattr(
        scramble,
        "_batched_get_random_inverse_trajectory",
        fake_random_inverse_trajectory,
    )

    assert rc._get_shuffled_state(solve_config, initial_state, key, 2) is sentinel
    assert rc.batched_get_random_trajectory(2, 1, key, 1) is sentinel
    assert rc.batched_get_random_inverse_trajectory(2, 1, key, 1) is sentinel
    assert [call[0] for call in calls] == ["shuffled", "trajectory", "inverse"]
    assert all(call[1] is rc for call in calls)


def test_scramble_free_functions_match_puzzle_delegates():
    rc = RubiksCube(size=3)
    key = jax.random.PRNGKey(42)
    solve_config, initial_state = rc.get_inits(key)

    _assert_tree_equal(
        rc._get_shuffled_state(solve_config, initial_state, key, 3),
        scramble._get_shuffled_state(rc, solve_config, initial_state, key, 3),
    )
    _assert_tree_equal(
        rc.batched_get_random_trajectory(3, 2, key, non_backtracking_steps=1),
        scramble._batched_get_random_trajectory(
            rc, 3, 2, key, non_backtracking_steps=1
        ),
    )
    _assert_tree_equal(
        rc.batched_get_random_inverse_trajectory(3, 2, key, non_backtracking_steps=1),
        scramble._batched_get_random_inverse_trajectory(
            rc, 3, 2, key, non_backtracking_steps=1
        ),
    )


@pytest.mark.parametrize(
    "puzzle",
    [
        RubiksCube(size=3),
        LightsOut(size=3, initial_shuffle=2),
        _small_cayley_puzzle(),
        TopSpin(size=6, turnstile_size=3),
    ],
)
def test_get_shuffled_state_call_sites_smoke(puzzle):
    key = jax.random.PRNGKey(42)
    solve_config = puzzle.get_solve_config(key)
    state = puzzle._get_shuffled_state(
        solve_config, solve_config.GoalSpec, key, num_shuffle=2
    )
    assert state is not None


def test_create_target_shuffled_path():
    rc = RubiksCube(size=3)
    key = jax.random.PRNGKey(42)

    wrapped = create_target_shuffled_path(
        rc,
        k_max=5,
        shuffle_parallel=2,
        include_solved_states=True,
        key=key,
        non_backtracking_steps=1,
    )

    assert wrapped.states.faces.shape == (10, 21), "Wrapped state shape mismatch"
    assert wrapped.actions.shape == (10,), "Wrapped actions shape mismatch"
    assert wrapped.parent_indices.shape == (10,), (
        "Wrapped parent indices shape mismatch"
    )
    assert wrapped.trajectory_indices.shape == (10,), (
        "Wrapped trajectory indices shape mismatch"
    )
    assert wrapped.step_indices.shape == (10,), "Wrapped step indices shape mismatch"


def test_create_hindsight_target_shuffled_path():
    sk = Sokoban()
    key = jax.random.PRNGKey(42)

    wrapped = create_hindsight_target_shuffled_path(
        sk,
        k_max=5,
        shuffle_parallel=2,
        include_solved_states=True,
        key=key,
        non_backtracking_steps=1,
    )

    assert wrapped.states.board_unpacked.shape == (10, 100), (
        "Wrapper output shape mismatch"
    )


def test_create_hindsight_target_triangular_shuffled_path():
    sk = Sokoban()
    key = jax.random.PRNGKey(42)

    wrapped = create_hindsight_target_triangular_shuffled_path(
        sk,
        k_max=5,
        shuffle_parallel=2,
        include_solved_states=False,
        key=key,
        non_backtracking_steps=1,
    )

    assert wrapped.states.board_unpacked.shape == (
        10,
        100,
    ), "Triangular Wrapper output shape mismatch"


def test_chain_trajectory_indices_are_canonical():
    sk = Sokoban()
    key = jax.random.PRNGKey(42)

    wrapped = create_hindsight_target_shuffled_path(
        sk,
        k_max=5,
        shuffle_parallel=2,
        include_solved_states=True,
        key=key,
        non_backtracking_steps=1,
    )

    assert jnp.array_equal(
        wrapped.parent_indices,
        jnp.array([-1, 0, 1, 2, 3, -1, 5, 6, 7, 8], dtype=jnp.int32),
    )
    assert jnp.array_equal(
        wrapped.trajectory_indices,
        jnp.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=jnp.int32),
    )
    assert jnp.array_equal(
        wrapped.step_indices,
        jnp.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=jnp.int32),
    )


def test_triangular_trajectory_indices_are_independent_samples():
    sk = Sokoban()
    key = jax.random.PRNGKey(42)

    wrapped = create_hindsight_target_triangular_shuffled_path(
        sk,
        k_max=5,
        shuffle_parallel=2,
        include_solved_states=True,
        key=key,
        non_backtracking_steps=1,
    )

    assert jnp.array_equal(wrapped.parent_indices, jnp.full((10,), -1, dtype=jnp.int32))
    assert jnp.array_equal(
        wrapped.trajectory_indices,
        jnp.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=jnp.int32),
    )
    assert jnp.array_equal(
        wrapped.step_indices.reshape(2, 5),
        jnp.sort(wrapped.step_indices.reshape(2, 5), axis=1),
    )


def test_puzzle_does_not_expose_trajectory_methods():
    """Lock the **Puzzle Trajectory Module** seam: target-shuffled-path
    generation must live in ``puxle.core.trajectory``, not as methods on the
    ``Puzzle`` base. Any new puzzle inheriting from ``Puzzle`` must access the
    three creators through the Module, never through ``self.create_*``.
    """
    from puxle.core.puzzle_base import Puzzle

    for method_name in (
        "create_target_shuffled_path",
        "create_hindsight_target_shuffled_path",
        "create_hindsight_target_triangular_shuffled_path",
    ):
        assert not hasattr(Puzzle, method_name), (
            f"`Puzzle.{method_name}` must NOT exist — it lives in "
            "`puxle.core.trajectory` per CONTEXT.md 'Puzzle Trajectory Module'."
        )


def test_hindsight_assertion_on_fixed_target():
    rc = RubiksCube(size=3)
    key = jax.random.PRNGKey(42)
    with pytest.raises(
        AssertionError, match="Fixed target is not supported for hindsight target"
    ):
        create_hindsight_target_shuffled_path(
            rc,
            k_max=5,
            shuffle_parallel=2,
            include_solved_states=True,
            key=key,
            non_backtracking_steps=1,
        )
