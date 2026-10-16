from collections.abc import Callable

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from xtructure import FieldDescriptor, Xtructurable, xtructure_dataclass

from puxle.core.puzzle_base import Puzzle
from puxle.utils.util import IMG_SIZE, coloring_str

TYPE = jnp.uint8

COLOR_MAP = {
    0: (255, 0, 0),  # Red
    1: (0, 255, 0),  # Green
    2: (255, 255, 0),  # Yellow
    3: (0, 0, 255),  # Blue
}


class DotKnot(Puzzle):
    """Dot-and-Knot path-connection puzzle.

    On an ``size × size`` grid, pairs of same-coloured dots must be
    connected by moving them toward each other.  When two dots of the
    same colour meet they merge into a path segment.  The puzzle is
    solved when no unmerged dots remain (and the board is non-empty).

    Cell encoding (4 bits per cell via xtructure bitpacking):

    * ``0``: empty.
    * ``1 .. 2·color_num``: dot endpoints (two per colour).
    * ``> 2·color_num``: path segments.

    Four directional actions move the lowest-indexed available dot.

    This puzzle is **not reversible**.

    Args:
        size: Edge length of the grid (default ``10``; must be ≥ 4).
        color_num: Number of dot colours (default ``4``).
    """

    size: int

    def define_goal_spec_class(self) -> type[Xtructurable]:
        @xtructure_dataclass
        class GoalSpec:
            pass

        return GoalSpec

    def define_state_class(self) -> type[Xtructurable]:
        str_parser = self.get_string_parser()
        size = self.size

        @xtructure_dataclass
        class State:
            board: FieldDescriptor.packed_tensor(shape=(size * size,), packed_bits=4)

            def __str__(self, **kwargs):
                return str_parser(self, **kwargs)

        return State

    def __init__(self, size: int = 10, color_num: int = 4, **kwargs):
        assert size >= 4, "Size must be at least 4 for packing"
        self.size = size
        self.color_num = color_num
        self.action_size = 4
        super().__init__(**kwargs)

    def get_solve_config_string_parser(self) -> Callable:
        def parser(solve_config: "DotKnot.SolveConfig"):
            return ""

        return parser

    def get_string_parser(self) -> Callable:
        form = self._get_visualize_format()

        def to_char(x):
            if x == 0:
                return " "
            elif x <= 2 * self.color_num:
                color_idx = int((x - 1) % self.color_num)
                return coloring_str("●", COLOR_MAP[color_idx])
            elif x <= 3 * self.color_num:
                color_idx = int((x - 1) % self.color_num)
                return coloring_str("■", COLOR_MAP[color_idx])
            else:
                return "?"  # for debug and target

        def parser(state, **kwargs):
            # Host copy first; per-cell reads off a device array sync per character.
            unpacked = jax.device_get(state).board_unpacked
            return form.format(*map(to_char, unpacked))

        return parser

    def get_initial_state(
        self,
        solve_config: "DotKnot.SolveConfig",
        key=jax.random.PRNGKey(128),
        data=None,
    ) -> "DotKnot.State":
        return self._get_random_state(key)

    def get_solve_config(
        self, key=jax.random.PRNGKey(128), data=None
    ) -> "DotKnot.SolveConfig":
        return self.SolveConfig(
            InstanceContext=self.InstanceContext(), GoalSpec=self.GoalSpec()
        )

    def _apply(
        self,
        solve_config: "DotKnot.SolveConfig",
        state: "DotKnot.State",
        action: jax.Array,
    ) -> tuple["DotKnot.State", jax.Array]:
        board = state.board_unpacked
        selected = jnp.min(
            jnp.where(
                (board > 0) & (board <= self.color_num), board, self.color_num + 1
            )
        )
        endpoint = board == selected
        start = jnp.argmax(endpoint)
        position = jnp.array([start // self.size, start % self.size])
        position += jnp.array([[0, -1], [0, 1], [-1, 0], [1, 0]])[action]
        target = position[0] * self.size + position[1]
        color = (selected - 1) % self.color_num
        merge = (
            (board[target] > 0)
            & (board[target] <= 2 * self.color_num)
            & ((board[target] - 1) % self.color_num == color)
        )
        valid = endpoint.any() & (position >= 0).all() & (position < self.size).all()
        valid &= (board[target] == 0) | merge
        path = (color + 2 * self.color_num + 1).astype(jnp.uint8)
        next_value = jnp.where(merge, path, selected).astype(jnp.uint8)
        bytes_ = jnp.arange(state.board.size)
        start_shift = (4 * (start % 2)).astype(jnp.uint8)
        target_shift = (4 * (target % 2)).astype(jnp.uint8)
        clear = jnp.where(
            bytes_ == start // 2, jnp.uint8(15) << start_shift, jnp.uint8(0)
        )
        clear |= jnp.where(
            bytes_ == target // 2, jnp.uint8(15) << target_shift, jnp.uint8(0)
        )
        values = jnp.where(bytes_ == start // 2, path << start_shift, jnp.uint8(0))
        values |= jnp.where(
            bytes_ == target // 2, next_value << target_shift, jnp.uint8(0)
        )
        return self.State(board=(state.board & ~clear) | values), jnp.where(
            valid, 1.0, jnp.inf
        )

    def _get_neighbours(
        self, solve_config: "DotKnot.SolveConfig", state: "DotKnot.State"
    ) -> tuple["DotKnot.State", jax.Array]:
        board = state.board_unpacked
        selected = jnp.min(
            jnp.where(
                (board > 0) & (board <= self.color_num), board, self.color_num + 1
            )
        )
        endpoint = board == selected
        start = jnp.argmax(endpoint)
        positions = jnp.array([start // self.size, start % self.size]) + jnp.array(
            [[0, -1], [0, 1], [-1, 0], [1, 0]]
        )
        targets = positions[:, 0] * self.size + positions[:, 1]
        color = (selected - 1) % self.color_num
        merge = (
            (board[targets] > 0)
            & (board[targets] <= 2 * self.color_num)
            & ((board[targets] - 1) % self.color_num == color)
        )
        valid = (
            endpoint.any()
            & (positions >= 0).all(axis=1)
            & (positions < self.size).all(axis=1)
        )
        valid &= (board[targets] == 0) | merge
        path = (color + 2 * self.color_num + 1).astype(jnp.uint8)
        values = jnp.where(merge, path, selected).astype(jnp.uint8)[:, None]
        bytes_ = jnp.arange(state.board.size)[None, :]
        start_shift = (4 * (start % 2)).astype(jnp.uint8)
        target_shifts = (4 * (targets[:, None] % 2)).astype(jnp.uint8)
        clear = jnp.where(
            bytes_ == start // 2, jnp.uint8(15) << start_shift, jnp.uint8(0)
        )
        clear = clear | jnp.where(
            bytes_ == targets[:, None] // 2,
            jnp.uint8(15) << target_shifts,
            jnp.uint8(0),
        )
        packed = jnp.where(bytes_ == start // 2, path << start_shift, jnp.uint8(0))
        packed = packed | jnp.where(
            bytes_ == targets[:, None] // 2, values << target_shifts, jnp.uint8(0)
        )
        return self.State(board=(state.board[None, :] & ~clear) | packed), jnp.where(
            valid, 1.0, jnp.inf
        )

    def is_solved(
        self, solve_config: "DotKnot.SolveConfig", state: "DotKnot.State"
    ) -> jax.Array:
        unpacked = state.board_unpacked
        empty = jnp.all(unpacked == 0)  # ALL empty is not solved condition
        gr = jnp.greater_equal(unpacked, 1)  # ALL point a is solved condition
        le = jnp.less_equal(
            unpacked, self.color_num * 2
        )  # ALL point b is solved condition
        points = gr & le
        no_point = ~jnp.any(points)  # if there is no point, it is solved
        return no_point & ~empty

    def action_to_string(self, action: int) -> str:
        return self._directional_action_to_string(action)

    def _get_visualize_format(self):
        return self._grid_visualize_format(self.size)

    def _get_random_state(self, key, num_shuffle=30):
        """
        This function should return a random state.
        """
        init_board = jnp.zeros((self.size * self.size), dtype=TYPE)

        def _while_loop(val):
            board, key, idx = val
            key, subkey = jax.random.split(key)
            pos = jax.random.randint(
                subkey, minval=0, maxval=self.size - 2, shape=(2,)
            ) + jnp.array([1, 1])
            random_index = pos[0] * self.size + pos[1]
            is_already_filled = board[random_index] != 0
            board = jax.lax.cond(
                is_already_filled,
                lambda: board,
                lambda: board.at[random_index].set(idx),
            )
            next_idx = jnp.where(is_already_filled, idx, idx + 1)
            return board, key, next_idx

        board, _, _ = jax.lax.while_loop(
            lambda val: val[2] < self.color_num * 2 + 1,
            _while_loop,
            (init_board, key, 1),
        )
        return self.State.from_unpacked(board=board)

    def get_solve_config_img_parser(self) -> Callable:
        def parser(solve_config: "DotKnot.SolveConfig"):
            pass

        return parser

    def get_img_parser(self) -> Callable:
        """
        This function is a decorator that adds an img_parser to the class.
        """

        def img_func(state: "DotKnot.State", **kwargs):
            imgsize = IMG_SIZE[0]
            img = np.full(
                (IMG_SIZE[1], IMG_SIZE[0], 3), (190, 190, 190), dtype=np.uint8
            )
            img = cv2.rectangle(
                img,
                (int(imgsize * 0.03), int(imgsize * 0.03)),
                (
                    int(imgsize - imgsize * 0.02),
                    int(imgsize - imgsize * 0.02),
                ),
                (50, 50, 50),
                -1,
            )
            board_flat = state.board_unpacked
            knot_max = 2 * self.color_num  # Values <= knot_max represent knots
            for idx, val in enumerate(board_flat):
                if val == 0:
                    continue
                stx = int(
                    imgsize * 0.04 + (imgsize * 0.95 / self.size) * (idx % self.size)
                )
                sty = int(
                    imgsize * 0.04 + (imgsize * 0.95 / self.size) * (idx // self.size)
                )
                bs = int(imgsize * 0.87 / self.size)
                center = (stx + bs // 2, sty + bs // 2)
                color = COLOR_MAP[(int(val) - 1) % len(COLOR_MAP)]
                if val <= knot_max:
                    # Draw knot as a filled circle
                    radius = int(bs * 0.6)
                    img = cv2.circle(img, center, radius, color, -1)
                else:
                    # Draw path as a filled rectangle
                    img = cv2.rectangle(
                        img, (stx, sty), (stx + bs, sty + bs), color, -1
                    )
            return img

        return img_func
