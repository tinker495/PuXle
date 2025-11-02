"""Compare neighbour dynamics between puxle RubiksCube and legacy Cube3 implementations.

The script starts from the solved (target) state for both environments, enumerates
all single-move neighbours, and reports a per-action comparison so that the current
implementations can be checked one-to-one.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
import numpy as np

# Ensure repository root and scripts directory are importable
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

for path in {str(REPO_ROOT), str(SCRIPTS_DIR)}:
    if path not in sys.path:
        sys.path.append(path)

from puxle.puzzles.rubikscube import RubiksCube
from puxle.benchmark.rubikscube_deepcubea import RubiksCubeDeepCubeABenchmark

from cube3 import Cube3, Cube3State


def flatten_rubiks_state(state: RubiksCube.State) -> np.ndarray:
    """Return the unpacked RubiksCube state as a flat numpy array."""

    faces = np.array(state.unpacked.faces, dtype=np.int32)
    return faces.reshape(-1)


def flatten_cube3_state(state: Cube3State) -> np.ndarray:
    """Return the Cube3 state as a flat numpy array."""

    return np.array(state.colors, dtype=np.int32).reshape(-1)


def convert_cube3_to_rubiks_flat(
    state: Cube3State,
    converter: RubiksCubeDeepCubeABenchmark,
    puzzle: RubiksCube,
) -> np.ndarray:
    """Convert a Cube3 state into RubiksCube tile-id ordering."""

    tiles = jnp.asarray(state.colors, dtype=jnp.uint8)
    converted_tiles = converter._convert_deepcubea_to_puxle(tiles, puzzle.size)
    converted_state = puzzle.State(faces=converted_tiles).packed
    return flatten_rubiks_state(converted_state)


def cube3_action_to_notation(action: int, moves: Iterable[str]) -> str:
    """Convert Cube3 action index to standard face notation (e.g. "U", "U'")."""

    move = moves[action]
    face = move[0]
    direction = int(move[1:])
    if direction == 1:
        return face
    if direction == -1:
        return f"{face}'"
    raise ValueError(f"Unexpected Cube3 move encoding: {move}")


def main() -> None:
    # --- RubiksCube setup -------------------------------------------------
    rubiks = RubiksCube(size=3, initial_shuffle=0, color_embedding=False, metric="QTM")
    solve_config = rubiks.get_solve_config(jax.random.PRNGKey(0))
    target_state = rubiks.get_target_state()

    rubiks_goal = flatten_rubiks_state(target_state)

    neighbours, _ = rubiks.get_neighbours(solve_config, target_state)
    rubiks_neighbour_states = [flatten_rubiks_state(neighbours[i]) for i in range(neighbours.faces.shape[0])]
    rubiks_actions = dict([(rubiks.action_to_string(i), i) for i in range(len(rubiks_neighbour_states))])

    converter = RubiksCubeDeepCubeABenchmark()

    # --- Cube3 setup ------------------------------------------------------
    cube3 = Cube3()
    cube3_goal_state = Cube3State(cube3.goal_colors.copy())
    cube3_goal = flatten_cube3_state(cube3_goal_state)
    conv_goal = converter._convert_deepcubea_to_puxle(cube3_goal, rubiks.size)
    cube3_str_actions = cube3.str_moves

    print("Initial solved-states")
    print(xnp.stack([rubiks.State(faces=conv_goal).packed, target_state]))

    for action in range(cube3.get_num_moves()):
        str_action = cube3_str_actions[action]
        rc_action = rubiks_actions[str_action]
        next_states, _ = cube3.next_state([cube3_goal_state], action)
        next_state = next_states[0]
        flattened_next_state = flatten_cube3_state(next_state)
        converted_next_state = converter._convert_deepcubea_to_puxle(flattened_next_state, rubiks.size)
        to_rubiks = rubiks.State(faces=converted_next_state).packed
        rc_state = neighbours[rc_action]
        if to_rubiks == rc_state:
            continue
        print(str_action)
        print(xnp.stack([to_rubiks, rc_state]))
        break


if __name__ == "__main__":
    main()


