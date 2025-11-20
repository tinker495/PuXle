import chex
import jax
import jax.numpy as jnp
from termcolor import colored

from puxle.core.puzzle_base import Puzzle
from puxle.core.puzzle_state import FieldDescriptor, PuzzleState, state_dataclass
from puxle.utils.annotate import IMG_SIZE
from puxle.utils.util import from_uint8, to_uint8

TYPE = jnp.uint8


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

    size: int

    def define_state_class(self) -> PuzzleState:
        """Defines the state class for LightsOut using xtructure."""
        str_parser = self.get_string_parser()
        board = jnp.zeros((self.size * self.size), dtype=bool)
        packed_board = to_uint8(board)
        size = self.size

        @state_dataclass
        class State:
            board: FieldDescriptor.tensor(dtype=TYPE, shape=packed_board.shape)

            def __str__(self, **kwargs):
                return str_parser(self, **kwargs)

            @property
            def packed(self) -> "LightsOut.State":
                board = to_uint8(self.board)
                return State(board=board)

            @property
            def unpacked(self) -> "LightsOut.State":
                board = from_uint8(self.board, (size * size,))
                return State(board=board)

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
            return form.format(*map(to_char, state.unpacked.board))

        return parser

    def get_initial_state(
        self, solve_config: Puzzle.SolveConfig, key=None, data=None
    ) -> "LightsOut.State":
        return self._get_suffled_state(
            solve_config, solve_config.TargetState, key, num_shuffle=self.initial_shuffle
        )

    def get_target_state(self, key=None) -> "LightsOut.State":
        return self.State(board=jnp.zeros(self.size**2, dtype=bool)).packed

    def get_solve_config(self, key=None, data=None) -> Puzzle.SolveConfig:
        return self.SolveConfig(TargetState=self.get_target_state(key))

    def get_actions(
        self, solve_config: Puzzle.SolveConfig, state: "LightsOut.State", action: chex.Array, filled: bool = True
    ) -> tuple["LightsOut.State", chex.Array]:
        """
        This function returns the next state and cost for a given action.
        """
        board = state.unpacked.board
        
        # Decode action to (x, y)
        # action is in range [0, size*size - 1]
        # x = action // size
        # y = action % size
        # Or better: use unravel_index but that works on shapes.
        # Since size is scalar, we can compute directly.
        
        x = action // self.size
        y = action % self.size
        
        def flip(board, x, y):
            xs = jnp.clip(jnp.array([x, x, x + 1, x - 1, x]), 0, self.size - 1)
            ys = jnp.clip(jnp.array([y, y + 1, y, y, y - 1]), 0, self.size - 1)
            idxs = xs * self.size + ys
            return board.at[idxs].set(jnp.logical_not(board[idxs]))

        next_board, cost = jax.lax.cond(
            filled, lambda: (flip(board, x, y), 1.0), lambda: (board, jnp.inf)
        )
        next_state = self.State(board=next_board).packed
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
            board = np.array(state.unpacked.board).reshape(self.size, self.size)
            # Define colors in BGR: light on → bright yellow, light off → black, and grid lines → gray
            on_color = (255, 255, 0)  # Yellow
            off_color = (0, 0, 0)  # Black
            grid_color = (50, 50, 50)  # Gray for grid lines
            # Draw each cell of the puzzle
            for i in range(self.size):
                for j in range(self.size):
                    top_left = (j * cell_size, i * cell_size)
                    bottom_right = ((j + 1) * cell_size, (i + 1) * cell_size)
                    # Use lit color if the cell is "on", otherwise use off color
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
