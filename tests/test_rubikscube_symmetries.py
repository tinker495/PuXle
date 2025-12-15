import numpy as np
import jax
import jax.numpy as jnp

from puxle.puzzles.rubikscube import RubiksCube


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

