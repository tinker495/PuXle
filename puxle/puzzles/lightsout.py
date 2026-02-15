from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np
from termcolor import colored

from puxle.core.puzzle_base import Puzzle
from puxle.core.puzzle_state import FieldDescriptor, PuzzleState, state_dataclass
from puxle.utils.annotate import IMG_SIZE

TYPE = jnp.uint8
_MOVE_MATRIX_CACHE: dict[int, np.ndarray] = {}


def action_to_char(action: int) -> str:
    """
    This function should return a string representation of the action.
    0~9 -> 0~9
    10~35 -> a~z
    36~61 -> A~Z
    """
    if action < 10:
        return colored(str(action), "light_yellow")
    elif action < 36:
        return colored(chr(action + 87), "light_yellow")
    else:
        return colored(chr(action + 29), "light_yellow")


class LightsOut(Puzzle):
    """Lights Out puzzle on an N×N grid.

    Pressing a button toggles it and its four orthogonal neighbours.
    The goal is to turn all lights **off**.  Each action is its own inverse,
    so ``inverse_action_map`` is the identity.

    The board is stored as 1-bit-per-cell via xtructure bitpacking.
    A GF(2) solvability check is available via :meth:`board_is_solvable`.

    Args:
        size: Edge length of the grid (default ``7``).
        initial_shuffle: Number of random presses for scrambling (default ``8``).
    """

    size: int

    def define_state_class(self) -> PuzzleState:
        """Defines the state class for LightsOut using xtructure."""
        str_parser = self.get_string_parser()
        size = self.size

        @state_dataclass
        class State:
            board: FieldDescriptor.packed_tensor(shape=(size * size,), packed_bits=1)

            def __str__(self, **kwargs):
                return str_parser(self, **kwargs)

        return State

    def __init__(self, size: int = 7, initial_shuffle: int = 8, **kwargs):
        self.size = size
        self.initial_shuffle = initial_shuffle
        self.action_size = self.size * self.size
        super().__init__(**kwargs)

    def get_string_parser(self):
        form = self._get_visualize_format()

        def to_char(x):
            return "□" if x == 0 else "■"

        def parser(state: "LightsOut.State", **kwargs):
            return form.format(*map(to_char, state.board_unpacked))

        return parser

    def get_initial_state(
        self, solve_config: Puzzle.SolveConfig, key=None, data=None
    ) -> "LightsOut.State":
        return self._get_suffled_state(
            solve_config, solve_config.TargetState, key, num_shuffle=self.initial_shuffle
        )

    def get_target_state(self, key=None) -> "LightsOut.State":
        board = jnp.zeros(self.size**2, dtype=jnp.bool_)
        return self.State.from_unpacked(board=board)

    def get_solve_config(self, key=None, data=None) -> Puzzle.SolveConfig:
        return self.SolveConfig(TargetState=self.get_target_state(key))

    def get_actions(
        self, solve_config: Puzzle.SolveConfig, state: "LightsOut.State", action: chex.Array, filled: bool = True
    ) -> tuple["LightsOut.State", chex.Array]:
        """
        This function returns the next state and cost for a given action.
        """
        board = state.board_unpacked
        
        # Decode action to (x, y)
        # action is in range [0, size*size - 1]
        # x = action // size
        # y = action % size
        # Or better: use unravel_index but that works on shapes.
        # Since size is scalar, we can compute directly.
        
        x = action // self.size
        y = action % self.size
        
        def flip(board, x, y):
            # Create coordinate grids
            i, j = jnp.meshgrid(jnp.arange(self.size), jnp.arange(self.size), indexing="ij")
            
            # Manhattan distance from center (x,y)
            dist = jnp.abs(i - x) + jnp.abs(j - y)
            
            # Mask includes center (dist=0) and immediate neighbors (dist=1)
            mask = (dist <= 1).reshape(-1)
            
            # XOR flip where mask is true
            return jnp.where(mask, jnp.logical_not(board), board)

        next_board, cost = jax.lax.cond(
            filled, lambda: (flip(board, x, y), 1.0), lambda: (board, jnp.inf)
        )
        next_state = state.set_unpacked(board=next_board)
        return next_state, cost

    def is_solved(self, solve_config: Puzzle.SolveConfig, state: "LightsOut.State") -> bool:
        return state == solve_config.TargetState

    def action_to_string(self, action: int) -> str:
        """
        This function should return a string representation of the action.
        """
        return action_to_char(action)

    @property
    def inverse_action_map(self) -> jnp.ndarray | None:
        """
        Defines the inverse action mapping for LightsOut.
        Each action (flipping a tile) is its own inverse.
        """
        return jnp.arange(self.action_size)

    @classmethod
    def _move_matrix(cls, size: int) -> np.ndarray:
        matrix = _MOVE_MATRIX_CACHE.get(size)
        if matrix is not None:
            return matrix

        total = size * size
        matrix = np.zeros((total, total), dtype=np.uint8)
        for idx in range(total):
            x, y = divmod(idx, size)
            affected = (
                (x, y),
                (x, y + 1),
                (x, y - 1),
                (x + 1, y),
                (x - 1, y),
            )
            for ax, ay in affected:
                if 0 <= ax < size and 0 <= ay < size:
                    matrix[idx, ax * size + ay] = 1
        _MOVE_MATRIX_CACHE[size] = matrix
        return matrix

    @classmethod
    def board_is_solvable(cls, board: np.ndarray, size: int) -> bool:
        board = np.asarray(board, dtype=np.uint8).reshape(size * size)
        matrix = cls._move_matrix(size)
        augmented = np.concatenate([matrix.copy(), board[:, None]], axis=1)
        rows, cols = augmented.shape
        num_vars = cols - 1
        rank = 0
        for col in range(num_vars):
            pivot = None
            for r in range(rank, rows):
                if augmented[r, col]:
                    pivot = r
                    break
            if pivot is None:
                continue
            if pivot != rank:
                augmented[[rank, pivot]] = augmented[[pivot, rank]]
            for r in range(rows):
                if r != rank and augmented[r, col]:
                    augmented[r] ^= augmented[rank]
            rank += 1
        inconsistent = np.logical_and(
            np.all(augmented[:, :-1] == 0, axis=1), augmented[:, -1] == 1
        )
        return not bool(np.any(inconsistent))

    def is_state_solvable(self, state: "LightsOut.State") -> bool:
        board = np.array(state.board_unpacked, dtype=np.uint8)
        return self.board_is_solvable(board, self.size)

    def _get_visualize_format(self):
        size = self.size
        action_idx = 0
        form = "┏━"
        form += "━Board".center((size - 1) * 2, "━")
        form += "━━┳━"
        form += "━Actions".center((size - 1) * 2, "━")
        form += "━━┓"
        form += "\n"
        for i in range(size):
            form += "┃ "
            for j in range(size):
                form += "{:s} "
            form += "┃ "
            for j in range(size):
                form += action_to_char(action_idx) + " "
                action_idx += 1
            form += "┃"
            form += "\n"
        form += "┗━"
        form += "━━" * (size - 1)
        form += "━━┻━"
        form += "━━" * (size - 1)
        form += "━━┛"
        return form

    def get_img_parser(self):
        """
        This function is a decorator that adds an img_parser to the class.
        """
        import cv2
        import numpy as np

        def img_func(state: "LightsOut.State", **kwargs):
            imgsize = IMG_SIZE[0]
            # Create a background image with a dark gray base
            img = np.full((imgsize, imgsize, 3), fill_value=30, dtype=np.uint8)
            # Calculate the size of each cell in the grid
            cell_size = imgsize // self.size
            # Reshape the flat board state into a 2D array
            board = np.array(state.board_unpacked).reshape(self.size, self.size)
            # Define colors in BGR: light on → bright yellow, light off → black, and grid lines → gray
            on_color = (255, 255, 0)  # Yellow
            off_color = (0, 0, 0)  # Black
            grid_color = (50, 50, 50)  # Gray for grid lines
            # Draw each cell of the puzzle
            for i in range(self.size):
                for j in range(self.size):
                    top_left = (j * cell_size, i * cell_size)
                    bottom_right = ((j + 1) * cell_size, (i + 1) * cell_size)
                    cell_color = on_color if board[i, j] else off_color
                    img = cv2.rectangle(img, top_left, bottom_right, cell_color, thickness=-1)
                    img = cv2.rectangle(img, top_left, bottom_right, grid_color, thickness=1)
            return img

        return img_func


class LightsOutRandom(LightsOut):
    """
    This class is a extension of LightsOut, it will generate the random state for the puzzle.
    """

    @property
    def fixed_target(self) -> bool:
        return False

    def get_solve_config(self, key=None, data=None) -> Puzzle.SolveConfig:
        solve_config = super().get_solve_config(key, data)
        solve_config.TargetState = self._get_suffled_state(
            solve_config, solve_config.TargetState, key, num_shuffle=self.initial_shuffle
        )
        return solve_config

    def get_initial_state(
        self, solve_config: Puzzle.SolveConfig, key=None, data=None
    ) -> LightsOut.State:
        return self._get_suffled_state(
            solve_config, solve_config.TargetState, key, num_shuffle=self.initial_shuffle
        )
