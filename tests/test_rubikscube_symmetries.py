import jax
import jax.numpy as jnp
import numpy as np
import pytest

from puxle.puzzles.rubikscube import RubiksCube


@pytest.mark.parametrize(
    ("size", "metric", "color_embedding"),
    [
        (2, "QTM", False),
        (3, "QTM", True),
        (3, "UQTM", False),
        (4, "QTM", False),
    ],
)
def test_rubikscube_actions_match_slice_rotation_reference(
    size: int, metric: str, color_embedding: bool
) -> None:
    cube = RubiksCube(
        size=size,
        initial_shuffle=0,
        color_embedding=color_embedding,
        metric=metric,
    )
    solve_config = cube.get_solve_config()
    state = cube.get_target_state()

    neighbours, costs = cube.get_neighbours(solve_config, state)

    for action in range(cube.action_size):
        axis = (action // 2) % 3
        index = int(cube.index_grid[action // 6])
        clockwise = bool(action % 2)
        reference = cube._rotate(state, axis, index, clockwise)
        np.testing.assert_array_equal(
            np.asarray(neighbours.faces_unpacked[action]),
            np.asarray(reference.faces_unpacked),
        )

    np.testing.assert_array_equal(
        np.asarray(costs), np.ones(cube.action_size, dtype=np.float32)
    )


def _ref_state_symmetries(cube: RubiksCube, state: RubiksCube.State) -> jnp.ndarray:
    """Reference implementation using repeated slice rotations (matches old fori_loop semantics)."""

    def rotate_whole(axis: int) -> RubiksCube.State:
        s = state
        for i in range(cube.size):
            s = cube._rotate(s, axis, i, True)
        return s

    def repeat(s: RubiksCube.State, axis: int, k: int) -> RubiksCube.State:
        out = s
        for _ in range(k % 4):
            tmp = out
            for i in range(cube.size):
                tmp = cube._rotate(tmp, axis, i, True)
            out = tmp
        return out

    bases = [[], [(0, 1)], [(0, 2)], [(0, 3)], [(2, 1)], [(2, 3)]]
    out = []
    for base in bases:
        s_base = state
        for ax, kk in base:
            s_base = repeat(s_base, ax, kk)
        s = s_base
        for _ in range(4):
            out.append(s.faces)
            s = repeat(s, 1, 1)
    return jnp.stack(out, axis=0)


def test_rubikscube_state_symmetries_returns_24_and_identity_first() -> None:
    cube = RubiksCube(size=3, initial_shuffle=0, color_embedding=True, metric="UQTM")
    state = cube.get_target_state()
    syms = cube.state_symmetries(state)

    assert syms.faces.shape[0] == 24
    assert bool(jnp.all(syms.faces[0] == state.faces))


def test_rubikscube_state_symmetries_solved_are_all_unique() -> None:
    """
    For a standard solved cube (solid-color faces), global rotations permute faces,
    producing 24 distinct packed states.
    """
    cube = RubiksCube(size=3, initial_shuffle=0, color_embedding=True, metric="UQTM")
    state = cube.get_target_state()
    syms = cube.state_symmetries(state)

    faces_np = np.asarray(syms.faces)
    assert faces_np.shape[0] == 24
    assert np.unique(faces_np, axis=0).shape[0] == 24


def test_rubikscube_state_symmetries_works_for_scrambled_state() -> None:
    cube = RubiksCube(size=3, initial_shuffle=5, color_embedding=True, metric="UQTM")
    key = jax.random.PRNGKey(0)
    solve_config = cube.get_solve_config(key=key)
    state = cube.get_initial_state(solve_config, key=key)
    syms = cube.state_symmetries(state)

    assert syms.faces.shape[0] == 24


def test_rubikscube_state_symmetries_match_reference_for_sizes() -> None:
    key = jax.random.PRNGKey(123)
    for size in (3, 4, 5):
        cube = RubiksCube(
            size=size, initial_shuffle=3, color_embedding=True, metric="UQTM"
        )
        k = jax.random.fold_in(key, size)
        solve_config = cube.get_solve_config(key=k)
        state = cube.get_initial_state(solve_config, key=k)

        got = cube.state_symmetries(state).faces
        ref = _ref_state_symmetries(cube, state)
        assert bool(jnp.all(got == ref))
