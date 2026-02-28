import jax
import pytest

from puxle.puzzles.rubikscube import RubiksCube
from puxle.puzzles.slidepuzzle import SlidePuzzle
from puxle.puzzles.sokoban import Sokoban


def test_random_trajectory_fast_path():
    rc = RubiksCube(size=3)
    key = jax.random.PRNGKey(42)

    # Fast path: reversible + non_backtracking_steps=1
    traj = rc.batched_get_random_trajectory(
        k_max=5, shuffle_parallel=2, key=key, non_backtracking_steps=1
    )

    assert traj["states"].faces.shape == (6, 2, 21), "State shape mismatch"
    assert traj["actions"].shape == (5, 2), "Action shape mismatch"
    assert traj["move_costs"].shape == (6, 2), "Move costs shape mismatch"
    assert traj["move_costs_tm1"].shape == (6, 2), "Move costs tm1 shape mismatch"
    assert traj["action_costs"].shape == (5, 2), "Action costs shape mismatch"


def test_random_trajectory_legacy_path():
    sp = SlidePuzzle(size=4)
    key = jax.random.PRNGKey(42)

    # Legacy path: non_backtracking_steps=3
    traj = sp.batched_get_random_trajectory(
        k_max=5, shuffle_parallel=2, key=key, non_backtracking_steps=3
    )

    assert traj["states"].board_unpacked.shape == (6, 2, 16), "State shape mismatch"
    assert traj["actions"].shape == (5, 2), "Action shape mismatch"
    assert traj["move_costs"].shape == (6, 2), "Move costs shape mismatch"


def test_random_inverse_trajectory_fast_path():
    rc = RubiksCube(size=3)
    key = jax.random.PRNGKey(42)

    traj = rc.batched_get_random_inverse_trajectory(
        k_max=4, shuffle_parallel=3, key=key, non_backtracking_steps=1
    )

    assert traj["states"].faces.shape == (5, 3, 21), "State shape mismatch"
    assert traj["actions"].shape == (4, 3), "Action shape mismatch"


def test_random_inverse_trajectory_legacy_path():
    sp = SlidePuzzle(size=3)
    key = jax.random.PRNGKey(42)

    traj = sp.batched_get_random_inverse_trajectory(
        k_max=4, shuffle_parallel=3, key=key, non_backtracking_steps=2
    )

    assert traj["states"].board_unpacked.shape == (5, 3, 9), "State shape mismatch"
    assert traj["actions"].shape == (4, 3), "Action shape mismatch"


def test_create_target_shuffled_path():
    rc = RubiksCube(size=3)
    key = jax.random.PRNGKey(42)

    wrapped = rc.create_target_shuffled_path(
        k_max=5,
        shuffle_parallel=2,
        include_solved_states=True,
        key=key,
        non_backtracking_steps=1,
    )

    assert wrapped["states"].faces.shape == (10, 21), "Wrapped state shape mismatch"
    assert wrapped["actions"].shape == (10,), "Wrapped actions shape mismatch"
    assert wrapped["parent_indices"].shape == (10,), (
        "Wrapped parent indices shape mismatch"
    )
    assert wrapped["trajectory_indices"].shape == (10,), (
        "Wrapped trajectory indices shape mismatch"
    )
    assert wrapped["step_indices"].shape == (10,), "Wrapped step indices shape mismatch"


def test_create_hindsight_target_shuffled_path():
    sk = Sokoban()
    key = jax.random.PRNGKey(42)

    wrapped = sk.create_hindsight_target_shuffled_path(
        k_max=5,
        shuffle_parallel=2,
        include_solved_states=True,
        key=key,
        non_backtracking_steps=1,
    )

    assert wrapped["states"].board_unpacked.shape == (10, 100), (
        "Wrapper output shape mismatch"
    )


def test_create_hindsight_target_triangular_shuffled_path():
    sk = Sokoban()
    key = jax.random.PRNGKey(42)

    wrapped = sk.create_hindsight_target_triangular_shuffled_path(
        k_max=5,
        shuffle_parallel=2,
        include_solved_states=False,
        key=key,
        non_backtracking_steps=1,
    )

    assert wrapped["states"].board_unpacked.shape == (10, 100), (
        "Triangular Wrapper output shape mismatch"
    )


def test_hindsight_assertion_on_fixed_target():
    rc = RubiksCube(size=3)
    key = jax.random.PRNGKey(42)
    with pytest.raises(
        AssertionError, match="Fixed target is not supported for hindsight target"
    ):
        rc.create_hindsight_target_shuffled_path(
            k_max=5,
            shuffle_parallel=2,
            include_solved_states=True,
            key=key,
            non_backtracking_steps=1,
        )
