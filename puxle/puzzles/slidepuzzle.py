from collections.abc import Callable

import chex
import cv2
import jax
import jax.numpy as jnp
import numpy as np

from puxle.core.puzzle_base import Puzzle
from puxle.core.puzzle_state import FieldDescriptor, PuzzleState, state_dataclass
from puxle.utils.util import IMG_SIZE

TYPE = jnp.uint8


class SlidePuzzle(Puzzle):
    """N×N sliding tile puzzle (15-puzzle generalisation).

    The board is a flat array of ``size²`` values where ``0`` represents the
    blank tile.  Actions move the blank in four directions (←, →, ↑, ↓).
    Only solvable permutations are generated.

    State packing uses ``ceil(log₂(size²))`` bits per tile via xtructure.

    Args:
        size: Edge length of the grid (default ``4`` → 15-puzzle).
    """

    size: int

    def define_state_class(self) -> PuzzleState:
        str_parser = self.get_string_parser()
        size = self.size
        max_value = self.size**2 - 1
        packed_bits = max_value.bit_length()

        @state_dataclass
        class State:
            board: FieldDescriptor.packed_tensor(
                shape=(size**2,), packed_bits=packed_bits
            )

            def __str__(self, **kwargs):
                return str_parser(self, **kwargs)

        return State

    def __init__(self, size: int = 4, **kwargs):
        self.size = size
        self.action_size = 4
        super().__init__(**kwargs)

    def get_string_parser(self) -> Callable:
        form = self._get_visualize_format()

        def to_char(x):
            if x == 0:
                return " "
            if x > 9:
                return chr(x + 55)
            return str(x)

        def parser(state: "SlidePuzzle.State", **kwargs):
            return form.format(*map(to_char, state.board_unpacked))

        return parser

    def get_initial_state(
        self, solve_config: Puzzle.SolveConfig, key=None, data=None
    ) -> "SlidePuzzle.State":
        return self._get_random_state(key)

    def get_solve_config(self, key=None, data=None) -> Puzzle.SolveConfig:
        target = jnp.array([*range(1, self.size**2), 0], dtype=TYPE)
        target_state = self.State.from_unpacked(board=target)
        return self.SolveConfig(TargetState=target_state)

    def get_actions(
        self,
        solve_config: Puzzle.SolveConfig,
        state: "SlidePuzzle.State",
        action: chex.Array,
        filled: bool = True,
    ) -> tuple["SlidePuzzle.State", chex.Array]:
        """
        This function should return a state and the cost of the move.
        """
        board = state.board_unpacked
        x, y = self._get_blank_position(board)
        pos = jnp.asarray((x, y))

        # Action mapping: 0: Left, 1: Right, 2: Up, 3: Down
        moves = jnp.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
        move_delta = moves[action]

        next_pos = pos + move_delta

        def is_valid(x, y):
            return jnp.logical_and(
                x >= 0,
                jnp.logical_and(x < self.size, jnp.logical_and(y >= 0, y < self.size)),
            )

        def swap(board, x, y, next_x, next_y):
            flat_index = x * self.size + y
            next_flat_index = next_x * self.size + next_y
            old_board = board
            board = board.at[next_flat_index].set(board[flat_index])
            return board.at[flat_index].set(old_board[next_flat_index])

        next_x, next_y = next_pos
        next_board, cost = jax.lax.cond(
            jnp.logical_and(is_valid(next_x, next_y), filled),
            lambda: (swap(board, x, y, next_x, next_y), 1.0),
            lambda: (board, jnp.inf),
        )
        return state.set_unpacked(board=next_board), cost

    def is_solved(
        self, solve_config: Puzzle.SolveConfig, state: "SlidePuzzle.State"
    ) -> bool:
        return state == solve_config.TargetState

    def action_to_string(self, action: int) -> str:
        return self._directional_action_to_string(action)

    @property
    def inverse_action_map(self) -> jnp.ndarray | None:
        """
        Defines the inverse action mapping for the Slide Puzzle.
        The actions are moving the blank tile [R, L, D, U].
        The inverse is therefore [L, R, U, D].
        """
        return jnp.array([1, 0, 3, 2])

    def _get_visualize_format(self):
        size = self.size
        form = "┏━"
        for i in range(size):
            form += "━━┳━" if i != size - 1 else "━━┓"
        form += "\n"
        for i in range(size):
            form += "┃ "
            for j in range(size):
                form += "{:s}"
                form += " ┃ " if j != size - 1 else " ┃"
            form += "\n"
            if i != size - 1:
                form += "┣━"
                for j in range(size):
                    form += "━━╋━" if j != size - 1 else "━━┫"
                form += "\n"
        form += "┗━"
        for i in range(size):
            form += "━━┻━" if i != size - 1 else "━━┛"
        return form

    def _get_random_state(self, key):
        """
        This function should return a random state.
        """

        def get_random_state(key):
            board = jax.random.permutation(key, jnp.arange(0, self.size**2, dtype=TYPE))
            return self.State.from_unpacked(board=board)

        def not_solverable(x):
            state = x[0]
            return ~self._solvable(state)

        def while_loop(x):
            state, key = x
            next_key, key = jax.random.split(key)
            state = get_random_state(key)
            return state, next_key

        next_key, key = jax.random.split(key)
        state = get_random_state(key)
        state, _ = jax.lax.while_loop(not_solverable, while_loop, (state, next_key))
        return state

    def _solvable(self, state: "SlidePuzzle.State"):
        """Check if the state is solvable"""
        board = state.board_unpacked
        N = self.size
        inv_count = self._get_inv_count(board)
        return jax.lax.cond(
            N % 2 == 1,
            lambda inv_count: inv_count % 2 == 0,
            lambda inv_count: jnp.logical_xor(
                self._get_blank_row(board) % 2 == 0, inv_count % 2 == 0
            ),
            inv_count,
        )

    def _get_blank_position(self, board: chex.Array):
        flat_index = jnp.argmax(board == 0)
        return jnp.unravel_index(flat_index, (self.size, self.size))

    def _get_blank_row(self, board: chex.Array):
        return self._get_blank_position(board)[0]

    def _get_inv_count(self, board: chex.Array):
        # Count ordered pairs (i < j) where both tiles are non-blank and out of
        # order. Vectorized equivalent of the O(n^4) double loop.
        flat = self.size * self.size
        a = board
        out_of_order = a[:, None] > a[None, :]
        both_nonzero = (a[:, None] != 0) & (a[None, :] != 0)
        upper = jnp.triu(jnp.ones((flat, flat), dtype=bool), k=1)
        return jnp.sum(out_of_order & both_nonzero & upper)

    def get_img_parser(self) -> Callable:
        """
        This function is a decorator that adds an img_parser to the class.
        """
        size = self.size

        def img_func(state: "SlidePuzzle.State", **kwargs):
            imgsize = IMG_SIZE[0]
            img = np.full((IMG_SIZE[1], IMG_SIZE[0], 3), (144, 96, 8), dtype=np.uint8)
            img = cv2.rectangle(
                img,
                (int(imgsize * 0.03), int(imgsize * 0.03)),
                (
                    int(imgsize - imgsize * 0.02),
                    int(imgsize - imgsize * 0.02),
                ),
                (104, 56, 8),
                -1,
            )
            fontsize = 2.5
            cell_size = int(imgsize * 0.87 / size)
            board_flat = state.board_unpacked
            for idx, val in enumerate(board_flat):
                if val == 0:
                    continue
                stx = int(imgsize * 0.04 + (imgsize * 0.95 / size) * (idx % size))
                sty = int(imgsize * 0.04 + (imgsize * 0.95 / size) * (idx // size))
                img = cv2.rectangle(
                    img,
                    (stx, sty),
                    (stx + cell_size, sty + cell_size),
                    (240, 240, 232),
                    -1,
                )
                text = str(val)
                (text_w, text_h), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, fontsize, 5
                )
                img = cv2.putText(
                    img,
                    text,
                    (
                        int(stx + (cell_size - text_w) / 2),
                        int(sty + (cell_size + text_h) / 2),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontsize,
                    (10, 10, 10),
                    5,
                )
            return img

        return img_func


class SlidePuzzleHard(SlidePuzzle):
    """
    This class is a extension of SlidePuzzle, it will generate the hardest state for the puzzle.
    """

    def __init__(self, size: int = 4, **kwargs):
        super().__init__(size, **kwargs)
        if size not in [3, 4]:
            raise ValueError(f"Size of the puzzle must be 3 or 4, got {size}")

        if size == 3:
            board = jnp.array([3, 1, 2, 0, 4, 5, 6, 7, 8], dtype=TYPE)
            self.hardest_state = self.State.from_unpacked(board=board)
        elif size == 4:
            board = jnp.array(
                [0, 12, 9, 13, 15, 11, 10, 14, 3, 7, 2, 5, 4, 8, 6, 1], dtype=TYPE
            )
            self.hardest_state = self.State.from_unpacked(board=board)

    def get_initial_state(
        self, solve_config: Puzzle.SolveConfig, key=None, data=None
    ) -> SlidePuzzle.State:
        return self.hardest_state


class SlidePuzzleRandom(SlidePuzzle):
    """
    This class is a extension of SlidePuzzle, it will generate the random state for the puzzle.
    """

    @property
    def fixed_target(self) -> bool:
        return False

    def get_solve_config(self, key=None, data=None) -> Puzzle.SolveConfig:
        solve_config = super().get_solve_config(key, data)
        solve_config.TargetState = self._get_random_state(key)
        return solve_config

    def get_initial_state(
        self, solve_config: Puzzle.SolveConfig, key=None, data=None
    ) -> SlidePuzzle.State:
        return self._get_random_state(key)
