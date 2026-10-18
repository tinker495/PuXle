import os
from collections.abc import Callable
from enum import Enum
from importlib.resources import files
from typing import Annotated

import chex
import jax
import jax.numpy as jnp
import numpy as np
from xtructure import FieldDescriptor, Xtructurable, xtructure_dataclass

from puxle.core.puzzle_base import Puzzle
from puxle.utils.util import IMG_SIZE, colored_str

TYPE = jnp.uint8


class Sokoban(Puzzle):
    """Sokoban (box-pushing) puzzle using the Boxoban dataset.

    Each cell is one of four base types (empty / wall / player / box) packed
    in 2 bits.  The board is fixed at 10×10 and levels are loaded from
    pre-packed ``.npy`` files shipped with the ``puxle.data`` subpackage.

    This puzzle is **not reversible** in the standard sense — inverse
    neighbours are computed via a dedicated pull-move implementation in
    :meth:`get_inverse_neighbours`.

    Two solve conditions are supported:

    * ``ALL_BOXES_ON_TARGET`` (default): only box positions must match.
    * ``ALL_BOXES_ON_TARGET_AND_PLAYER_ON_TARGET``: both box **and** player
      positions must match the goal.

    Args:
        size: Board edge length (must be ``10``).
        solve_condition: Which condition defines a solved state.
    """

    size: int = 10
    solve_condition: "Sokoban.SolveCondition" = None

    class Object(Enum):
        EMPTY = 0
        WALL = 1
        PLAYER = 2
        BOX = 3
        TARGET = 4
        PLAYER_ON_TARGET = 5
        BOX_ON_TARGET = 6
        TARGET_PLAYER = 7

    class SolveCondition(Enum):
        ALL_BOXES_ON_TARGET = 0
        ALL_BOXES_ON_TARGET_AND_PLAYER_ON_TARGET = 1

    def define_state_class(self) -> type[Xtructurable]:
        """Defines the state class for Sokoban using xtructure."""
        str_parser = self.get_string_parser()
        size = self.size

        @xtructure_dataclass
        class State:
            board: Annotated[
                chex.Array,
                FieldDescriptor.packed_tensor(shape=(size * size,), packed_bits=2),
            ]

            def __str__(self, **kwargs):
                return str_parser(self, **kwargs)

        return State

    def __init__(
        self,
        size: int = 10,
        solve_condition: SolveCondition = SolveCondition.ALL_BOXES_ON_TARGET,
        **kwargs,
    ):
        self.size = size
        self.solve_condition = solve_condition
        assert size == 10, "Boxoban dataset only supports size 10"
        self.action_size = 4
        super().__init__(**kwargs)

    @property
    def is_reversible(self) -> bool:
        return False

    @property
    def fixed_target(self) -> bool:
        return False

    # Filename suffix selecting the bundled dataset (overridden by SokobanHard).
    _data_suffix = ""

    def data_init(self):
        init_name = f"init{self._data_suffix}.npy"
        target_name = f"target{self._data_suffix}.npy"
        try:
            # Try to load as package resources first (for installed packages)
            data_pkg = files("puxle.data.sokoban")
            with (data_pkg / init_name).open("rb") as init_file:
                self.init_puzzles = jnp.load(init_file)
            with (data_pkg / target_name).open("rb") as target_file:
                self.target_puzzles = jnp.load(target_file)
        except (FileNotFoundError, ModuleNotFoundError):
            # Fallback to relative paths (for development/source directory)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(current_dir, "..", "data", "sokoban")

            self.init_puzzles = jnp.load(os.path.join(data_dir, init_name))
            self.target_puzzles = jnp.load(os.path.join(data_dir, target_name))

        self.num_puzzles = self.init_puzzles.shape[0]

    def get_data(self, key: jax.random.PRNGKey) -> tuple[chex.Array, chex.Array]:
        idx = jax.random.randint(key, (), 0, self.num_puzzles)
        return self.target_puzzles[idx, ...], self.init_puzzles[idx, ...]

    def get_initial_state(
        self, solve_config: Puzzle.SolveConfig, key=None, data=None
    ) -> "Sokoban.State":
        # NOTE: The Boxoban dataset (init.npy, target.npy) contains already bit-packed
        # board data (2 bits per cell, 10x10 -> 25 bytes).
        # We pass it directly to the State constructor as the underlying packed buffer.
        if data is None:
            # When no data is provided, generate data using get_data method
            data = self.get_data(key)
        _, init_data = data
        return self.State(board=init_data)

    def get_solve_config(self, key=None, data=None) -> Puzzle.SolveConfig:
        if data is None:
            # When no data is provided, generate data using get_data method
            data = self.get_data(key)
        target_data, _ = data
        if (
            self.solve_condition
            == Sokoban.SolveCondition.ALL_BOXES_ON_TARGET_AND_PLAYER_ON_TARGET
        ):
            # If we need to modify the board (e.g. place agent randomly),
            # we first unpack the data to work with logical values.
            board = self.State(board=target_data).board_unpacked
            board = self._place_agent_randomly(board, key)
            target_state = self.State.from_unpacked(board=board)
        else:
            # Otherwise, use the pre-packed target data directly.
            target_state = self.State(board=target_data)
        return self.SolveConfig(
            InstanceContext=self.InstanceContext(), GoalSpec=target_state
        )

    def is_solved(
        self, solve_config: Puzzle.SolveConfig, state: "Sokoban.State"
    ) -> jax.Array:
        # Unpack boards for comparison.
        board = state.board_unpacked
        t_board = solve_config.GoalSpec.board_unpacked
        # Remove the player from the current board.
        if (
            self.solve_condition
            == Sokoban.SolveCondition.ALL_BOXES_ON_TARGET_AND_PLAYER_ON_TARGET
        ):
            return jnp.all(board == t_board)
        else:
            rm_player = jnp.where(
                board == Sokoban.Object.PLAYER.value, Sokoban.Object.EMPTY.value, board
            )
            return jnp.all(rm_player == t_board)

    def action_to_string(self, action: int) -> str:
        return self._directional_action_to_string(action)

    def get_solve_config_string_parser(self) -> Callable:
        def parser(solve_config: "Sokoban.SolveConfig", **kwargs):
            return solve_config.GoalSpec.str(solve_config=solve_config)

        return parser

    def get_string_parser(self) -> Callable:
        form = self._get_visualize_format()

        def to_char(x):
            match x:
                case Sokoban.Object.EMPTY.value:
                    return " "
                case Sokoban.Object.WALL.value:
                    return colored_str("■", "white")
                case Sokoban.Object.PLAYER.value:
                    return colored_str("●", "green")
                case Sokoban.Object.BOX.value:
                    return colored_str("■", "yellow")
                case Sokoban.Object.TARGET.value:
                    return colored_str("x", "red")
                case Sokoban.Object.PLAYER_ON_TARGET.value:
                    return colored_str("ⓧ", "red")
                case Sokoban.Object.BOX_ON_TARGET.value:
                    return colored_str("■", "green")
                case Sokoban.Object.TARGET_PLAYER.value:
                    return colored_str("●", "red")
                case _:
                    return "?"

        def parser(
            state: "Sokoban.State", solve_config: "Sokoban.SolveConfig" = None, **kwargs
        ):
            # Unpack the board before visualization. The overlay below walks every
            # cell in Python, so it has to run on a host copy — indexing a device
            # array per cell costs a synchronisation each time.
            board = np.array(jax.device_get(state).board_unpacked)
            if solve_config is not None:
                goal = jax.device_get(solve_config).GoalSpec.board_unpacked
                for idx in range(self.size * self.size):
                    if goal[idx] == Sokoban.Object.BOX.value:
                        match board[idx]:
                            case Sokoban.Object.PLAYER.value:
                                board[idx] = Sokoban.Object.PLAYER_ON_TARGET.value
                            case Sokoban.Object.BOX.value:
                                board[idx] = Sokoban.Object.BOX_ON_TARGET.value
                            case Sokoban.Object.EMPTY.value:
                                board[idx] = Sokoban.Object.TARGET.value
                    if goal[idx] == Sokoban.Object.PLAYER.value:
                        match board[idx]:
                            case Sokoban.Object.EMPTY.value:
                                board[idx] = Sokoban.Object.TARGET_PLAYER.value

            return form.format(*map(to_char, board))

        return parser

    def _apply(
        self,
        solve_config: Puzzle.SolveConfig,
        state: "Sokoban.State",
        action: jax.Array,
    ) -> tuple["Sokoban.State", jax.Array]:
        board = state.board_unpacked
        player = jnp.argmax(board == Sokoban.Object.PLAYER.value)
        direction = jnp.array([[0, -1], [0, 1], [-1, 0], [1, 0]])[action]
        position = jnp.array([player // self.size, player % self.size]) + direction
        target = position[0] * self.size + position[1]
        push_position = position + direction
        pushed = push_position[0] * self.size + push_position[1]
        box = board[target] == Sokoban.Object.BOX.value
        push_valid = (push_position >= 0).all() & (push_position < self.size).all()
        push_valid &= board[pushed] == Sokoban.Object.EMPTY.value
        valid = (position >= 0).all() & (position < self.size).all()
        valid &= (board[target] == Sokoban.Object.EMPTY.value) | (box & push_valid)
        bytes_ = jnp.arange(state.board.size)
        clear = jnp.zeros_like(state.board)
        values = jnp.zeros_like(state.board)
        for cell, value, enabled in (
            (player, Sokoban.Object.EMPTY.value, True),
            (target, Sokoban.Object.PLAYER.value, True),
            (pushed, Sokoban.Object.BOX.value, box),
        ):
            shift = (2 * (cell % 4)).astype(jnp.uint8)
            affected = enabled & (bytes_ == cell // 4)
            clear |= jnp.where(affected, jnp.uint8(3) << shift, jnp.uint8(0))
            values |= jnp.where(affected, jnp.uint8(value) << shift, jnp.uint8(0))
        return self.State(board=(state.board & ~clear) | values), jnp.where(
            valid, 1.0, jnp.inf
        )

    def _get_neighbours(
        self, solve_config: Puzzle.SolveConfig, state: "Sokoban.State"
    ) -> tuple["Sokoban.State", jax.Array]:
        board = state.board_unpacked
        player = jnp.argmax(board == Sokoban.Object.PLAYER.value)
        directions = jnp.array([[0, -1], [0, 1], [-1, 0], [1, 0]])
        positions = jnp.array([player // self.size, player % self.size]) + directions
        targets = positions[:, 0] * self.size + positions[:, 1]
        push_positions = positions + directions
        pushed = push_positions[:, 0] * self.size + push_positions[:, 1]
        boxes = board[targets] == Sokoban.Object.BOX.value
        push_valid = (push_positions >= 0).all(axis=1) & (
            push_positions < self.size
        ).all(axis=1)
        push_valid &= board[pushed] == Sokoban.Object.EMPTY.value
        valid = (positions >= 0).all(axis=1) & (positions < self.size).all(axis=1)
        valid &= (board[targets] == Sokoban.Object.EMPTY.value) | (boxes & push_valid)
        bytes_ = jnp.arange(state.board.size)[None, :]
        clear = jnp.zeros((self.action_size, state.board.size), dtype=jnp.uint8)
        values = jnp.zeros_like(clear)
        for cells, value, enabled in (
            (player, Sokoban.Object.EMPTY.value, True),
            (targets[:, None], Sokoban.Object.PLAYER.value, True),
            (pushed[:, None], Sokoban.Object.BOX.value, boxes[:, None]),
        ):
            shifts = (2 * (cells % 4)).astype(jnp.uint8)
            affected = enabled & (bytes_ == cells // 4)
            clear |= jnp.where(affected, jnp.uint8(3) << shifts, jnp.uint8(0))
            values |= jnp.where(affected, jnp.uint8(value) << shifts, jnp.uint8(0))
        return self.State(board=(state.board[None, :] & ~clear) | values), jnp.where(
            valid, 1.0, jnp.inf
        )

    def _get_visualize_format(self):
        size = self.size
        top_border = "┏━" + "━━" * size + "┓\n"
        middle = ""
        for _ in range(size):
            middle += "┃ " + " ".join(["{:s}"] * size) + " ┃\n"
        bottom_border = "┗━" + "━━" * size + "┛"
        return top_border + middle + bottom_border

    def _get_player_position(self, state: "Sokoban.State"):
        board = state.board_unpacked
        flat_index = jnp.argmax(board == Sokoban.Object.PLAYER.value)
        return jnp.unravel_index(flat_index, (self.size, self.size))

    def get_img_parser(self) -> Callable:
        """
        This function is a decorator that adds an img_parser to the class.
        """
        import cv2
        import numpy as np

        cell_w = IMG_SIZE[0] // self.size
        cell_h = IMG_SIZE[1] // self.size

        try:
            # Try to load as package resources first (for installed packages)
            imgs_pkg = files("puxle.data.sokoban.imgs")
            image_dir = imgs_pkg
        except (FileNotFoundError, ModuleNotFoundError):
            # Fallback to relative paths (for development/source directory)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            image_dir = os.path.join(current_dir, "..", "data", "sokoban", "imgs")

        tile_filenames = {
            0: "floor.png",
            1: "wall.png",
            2: "agent.png",
            3: "box.png",
            4: "box_target.png",
            5: "agent_on_target.png",
            6: "box_on_target.png",
        }
        assets = {
            code: cv2.resize(
                cv2.cvtColor(
                    cv2.imread(os.path.join(image_dir, filename), cv2.IMREAD_COLOR),
                    cv2.COLOR_BGR2RGB,
                ),
                (cell_w, cell_h),
                interpolation=cv2.INTER_AREA,
            )
            for code, filename in tile_filenames.items()
        }
        # Entry 7 is the "player pushing" sprite: same agent art with the green
        # channel rolled into red instead of a BGR->RGB conversion.
        assets[7] = cv2.resize(
            np.roll(
                cv2.imread(os.path.join(image_dir, "agent.png"), cv2.IMREAD_COLOR),
                -1,
                axis=2,  # color channel G -> R
            ),
            (cell_w, cell_h),
            interpolation=cv2.INTER_AREA,
        )

        def img_func(
            state: "Sokoban.State", solve_config: "Sokoban.SolveConfig" = None, **kwargs
        ):
            img = np.zeros(IMG_SIZE + (3,), np.uint8)

            board = np.array(state.board_unpacked)
            if solve_config is not None:
                goal = np.array(solve_config.GoalSpec.board_unpacked)
            else:
                goal = None
            for i in range(self.size):
                for j in range(self.size):
                    cell_val = int(board[i * self.size + j])
                    if (
                        goal is not None
                        and goal[i * self.size + j] == Sokoban.Object.BOX.value
                    ):  # If this cell is marked as a target
                        match cell_val:
                            case Sokoban.Object.PLAYER.value:
                                asset = assets.get(
                                    Sokoban.Object.PLAYER_ON_TARGET.value
                                )  # agent on target
                            case Sokoban.Object.BOX.value:
                                asset = assets.get(
                                    Sokoban.Object.BOX_ON_TARGET.value
                                )  # box on target
                            case Sokoban.Object.EMPTY.value:
                                asset = assets.get(
                                    Sokoban.Object.TARGET.value
                                )  # target floor (box target)
                            case _:
                                asset = assets.get(cell_val)
                    elif (
                        goal is not None
                        and goal[i * self.size + j] == Sokoban.Object.PLAYER.value
                    ):
                        match board[i * self.size + j]:
                            case Sokoban.Object.BOX.value:
                                asset = assets.get(Sokoban.Object.BOX.value)
                            case Sokoban.Object.PLAYER.value:
                                asset = assets.get(Sokoban.Object.PLAYER.value)
                            case Sokoban.Object.EMPTY.value:
                                asset = assets.get(Sokoban.Object.TARGET_PLAYER.value)
                            case _:
                                pass
                    else:
                        asset = assets.get(cell_val)
                    if asset is not None:
                        img[
                            i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w
                        ] = asset
                    else:
                        # Fallback: fill with a gray square if image not found
                        img[
                            i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w
                        ] = (
                            127,
                            127,
                            127,
                        )
            return img

        return img_func

    def _get_box_positions(
        self, board: jax.Array, number_of_boxes: int = 4
    ) -> tuple[jax.Array, ...]:
        """
        Get the positions of all boxes on the board.
        """
        flat_index = jnp.argsort(board == Sokoban.Object.BOX.value)[-number_of_boxes:]
        return jnp.unravel_index(flat_index, (self.size, self.size))

    def _place_agent_randomly(
        self, board: jnp.ndarray, key: jax.random.PRNGKey
    ) -> jnp.ndarray:
        """
        Place the agent randomly on the board.
        """
        box_xs, box_ys = self._get_box_positions(board)
        box_positions = jnp.expand_dims(
            jnp.stack([box_xs, box_ys], axis=1), 0
        )  # shape: (1, 4, 2)
        _near = jnp.expand_dims(
            jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]]), 1
        )  # shape: (4, 1, 2)
        near_positions = box_positions + _near  # shape: (4, 4, 2)
        near_positions = near_positions.reshape((-1, 2))  # shape: (16, 2)
        mask = jnp.ones(near_positions.shape[0])
        mask = jnp.where(near_positions[:, 0] < 0, 0, mask)
        mask = jnp.where(near_positions[:, 0] >= self.size, 0, mask)
        mask = jnp.where(near_positions[:, 1] < 0, 0, mask)
        mask = jnp.where(near_positions[:, 1] >= self.size, 0, mask)
        mask = jnp.where(
            board[near_positions[:, 0] * self.size + near_positions[:, 1]]
            != Sokoban.Object.EMPTY.value,
            0,
            mask,
        )
        prob = mask / jnp.sum(mask)
        idx = jax.random.choice(key, jnp.arange(near_positions.shape[0]), p=prob)
        new_board = board.at[
            near_positions[idx, 0] * self.size + near_positions[idx, 1]
        ].set(Sokoban.Object.PLAYER.value)
        return new_board

    def solve_config_to_state_transform(
        self, solve_config: "Sokoban.SolveConfig", key: jax.random.PRNGKey = None
    ) -> "Sokoban.State":
        """
        This function shoulde transformt the solve config to the state.
        """
        board = solve_config.GoalSpec.board_unpacked
        if self.solve_condition == Sokoban.SolveCondition.ALL_BOXES_ON_TARGET:
            board = self._place_agent_randomly(board, key)

        return self.State.from_unpacked(board=board)

    def hindsight_transform(
        self, solve_config: "Sokoban.SolveConfig", state: "Sokoban.State"
    ) -> "Sokoban.SolveConfig":
        """
        This function shoulde transformt the state to the solve config.
        """
        board = state.board_unpacked
        if self.solve_condition == Sokoban.SolveCondition.ALL_BOXES_ON_TARGET:
            rm_player = jnp.where(
                board == Sokoban.Object.PLAYER.value, Sokoban.Object.EMPTY.value, board
            )
            solve_config = solve_config.replace(
                GoalSpec=self.State.from_unpacked(board=rm_player)
            )
        else:
            solve_config = solve_config.replace(
                GoalSpec=self.State.from_unpacked(board=board)
            )
        return solve_config

    def get_inverse_neighbours(
        self,
        solve_config: "Sokoban.SolveConfig",
        state: "Sokoban.State",
        filled: bool = True,
    ) -> tuple["Sokoban.State", jax.Array]:
        """Generate pull moves; blocked directions remain unit-cost self-loops.

        This scrambling API preserves its cost-only ``filled`` mask.
        """
        board = state.board_unpacked
        player = jnp.argmax(board == Sokoban.Object.PLAYER.value)
        position = jnp.array([player // self.size, player % self.size])
        directions = jnp.array([[0, -1], [0, 1], [-1, 0], [1, 0]])
        previous = position - directions
        fronts = position + directions
        previous_cells = previous[:, 0] * self.size + previous[:, 1]
        front_cells = fronts[:, 0] * self.size + fronts[:, 1]
        valid = (previous >= 0).all(axis=1) & (previous < self.size).all(axis=1)
        valid &= board[previous_cells] == Sokoban.Object.EMPTY.value
        pull = (fronts >= 0).all(axis=1) & (fronts < self.size).all(axis=1)
        pull &= board[front_cells] == Sokoban.Object.BOX.value
        bytes_ = jnp.arange(state.board.size)[None, :]
        clear = jnp.zeros((self.action_size, state.board.size), dtype=jnp.uint8)
        values = jnp.zeros_like(clear)
        for cells, value, enabled in (
            (
                player,
                jnp.where(pull, Sokoban.Object.BOX.value, Sokoban.Object.EMPTY.value)[
                    :, None
                ],
                True,
            ),
            (previous_cells[:, None], Sokoban.Object.PLAYER.value, True),
            (front_cells[:, None], Sokoban.Object.EMPTY.value, pull[:, None]),
        ):
            shifts = (2 * (cells % 4)).astype(jnp.uint8)
            affected = enabled & (bytes_ == cells // 4)
            clear |= jnp.where(affected, jnp.uint8(3) << shifts, jnp.uint8(0))
            values |= jnp.where(
                affected, jnp.asarray(value, dtype=jnp.uint8) << shifts, jnp.uint8(0)
            )
        boards = (state.board[None, :] & ~clear) | values
        states = self.State(
            board=jnp.where(valid[:, None], boards, state.board[None, :])
        )
        return states, jnp.full((self.action_size,), jnp.where(filled, 1.0, jnp.inf))


class SokobanHard(Sokoban):
    _data_suffix = "_hard"
