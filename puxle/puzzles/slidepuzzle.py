from collections.abc import Callable
from math import gcd

import chex
import cv2
import jax
import jax.numpy as jnp
import numpy as np
from xtructure import FieldDescriptor, Xtructurable, xtructure_dataclass

from puxle.core.puzzle_base import Puzzle
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

    def define_state_class(self) -> type[Xtructurable]:
        str_parser = self.get_string_parser()
        size = self.size
        max_value = self.size**2 - 1
        packed_bits = max_value.bit_length()

        @xtructure_dataclass
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
            # Host copy first; per-cell reads off a device array sync per character.
            return form.format(*map(to_char, jax.device_get(state).board_unpacked))

        return parser

    def get_initial_state(
        self, solve_config: Puzzle.SolveConfig, key=None, data=None
    ) -> "SlidePuzzle.State":
        return self._get_random_state(key)

    def get_solve_config(self, key=None, data=None) -> Puzzle.SolveConfig:
        target = jnp.array([*range(1, self.size**2), 0], dtype=TYPE)
        target_state = self.State.from_unpacked(board=target)
        return self.SolveConfig(
            InstanceContext=self.InstanceContext(), GoalSpec=target_state
        )

    def _apply(
        self,
        solve_config: Puzzle.SolveConfig,
        state: "SlidePuzzle.State",
        action: jax.Array,
    ) -> tuple["SlidePuzzle.State", jax.Array]:
        if self.size == 2:
            # A one-byte board needs no unpacked array or blank reduction.
            byte = state.board[0]
            blank = jnp.where((byte & 192) == 0, 3, 0)
            blank = jnp.where((byte & 48) == 0, 2, blank)
            blank = jnp.where((byte & 12) == 0, 1, blank)
            blank = jnp.where((byte & 3) == 0, 0, blank)
            target = blank + jnp.array([1, -1, 2, -2])[action]
            valid = ((blank >> (action // 2)) & 1) == (action & 1)
            start_shift = (2 * blank).astype(jnp.uint8)
            target_shift = (2 * target).astype(jnp.uint8)
            difference = ((byte >> start_shift) ^ (byte >> target_shift)) & jnp.uint8(3)
            moved = byte ^ (difference << start_shift) ^ (difference << target_shift)
            return self.State(board=moved[None]), jnp.where(valid, 1.0, jnp.inf)
        board = state.board_unpacked
        blank = jnp.argmax(board == 0)
        position = jnp.array([blank // self.size, blank % self.size])
        position += jnp.array([[0, 1], [0, -1], [1, 0], [-1, 0]])[action]
        target = position[0] * self.size + position[1]
        valid = (position >= 0).all() & (position < self.size).all()
        cost = jnp.where(valid, 1.0, jnp.inf)
        packed_bits = (self.size * self.size - 1).bit_length()
        if packed_bits == 8:
            indices = jnp.arange(self.size * self.size)
            source = jnp.where(
                indices == blank, target, jnp.where(indices == target, blank, indices)
            )
            return state.set_unpacked(board=board[source]), cost
        # Six-bit blocks favor select/pack over byte-fragment masks.
        if packed_bits == 6:
            indices = jnp.arange(self.size * self.size)
            board = jnp.where(
                indices == blank,
                board[target],
                jnp.where(indices == target, board[blank], board),
            )
            return state.set_unpacked(board=board), cost
        delta = board[blank] ^ board[target]
        bytes_ = jnp.arange(state.board.size)
        toggle = jnp.zeros_like(state.board)
        # A field can start at offsets up to 8 - gcd(packed_bits, 8).
        for cell in (blank, target):
            bit = cell * packed_bits
            shift = (bit % 8).astype(delta.dtype)
            toggle ^= jnp.where(
                bytes_ == bit // 8, (delta << shift).astype(jnp.uint8), jnp.uint8(0)
            )
            for fragment in range(1, (packed_bits + 15 - gcd(packed_bits, 8)) // 8):
                toggle ^= jnp.where(
                    bytes_ == bit // 8 + fragment,
                    (
                        delta >> (jnp.asarray(fragment * 8, dtype=delta.dtype) - shift)
                    ).astype(jnp.uint8),
                    jnp.uint8(0),
                )
        return self.State(board=state.board ^ toggle), cost

    def _get_neighbours(
        self, solve_config: Puzzle.SolveConfig, state: "SlidePuzzle.State"
    ) -> tuple["SlidePuzzle.State", jax.Array]:
        board = state.board_unpacked
        blank = jnp.argmax(board == 0)
        positions = jnp.array([blank // self.size, blank % self.size]) + jnp.array(
            [[0, 1], [0, -1], [1, 0], [-1, 0]]
        )
        targets = positions[:, 0] * self.size + positions[:, 1]
        valid = (positions >= 0).all(axis=1) & (positions < self.size).all(axis=1)
        packed_bits = (self.size * self.size - 1).bit_length()
        if packed_bits == 8:
            indices = jnp.arange(self.size * self.size)[None, :]
            boards = jnp.where(
                indices == blank,
                board[targets, None],
                jnp.where(indices == targets[:, None], board[blank], board[None, :]),
            )
            states = jax.vmap(lambda values: state.set_unpacked(board=values))(boards)
            return states, jnp.where(valid, 1.0, jnp.inf)
        deltas = (board[blank] ^ board[targets])[:, None]
        bytes_ = jnp.arange(state.board.size)[None, :]
        toggles = jnp.zeros((self.action_size, state.board.size), dtype=jnp.uint8)
        for cells in (blank, targets[:, None]):
            bits = cells * packed_bits
            shifts = (bits % 8).astype(deltas.dtype)
            toggles ^= jnp.where(
                bytes_ == bits // 8, (deltas << shifts).astype(jnp.uint8), jnp.uint8(0)
            )
            for fragment in range(1, (packed_bits + 15 - gcd(packed_bits, 8)) // 8):
                toggles ^= jnp.where(
                    bytes_ == bits // 8 + fragment,
                    (
                        deltas
                        >> (jnp.asarray(fragment * 8, dtype=deltas.dtype) - shifts)
                    ).astype(jnp.uint8),
                    jnp.uint8(0),
                )
        return self.State(board=state.board[None, :] ^ toggles), jnp.where(
            valid, 1.0, jnp.inf
        )

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
        return solve_config.replace(GoalSpec=self._get_random_state(key))

    def get_initial_state(
        self, solve_config: Puzzle.SolveConfig, key=None, data=None
    ) -> SlidePuzzle.State:
        return self._get_random_state(key)
