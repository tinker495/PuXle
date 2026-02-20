import os
from enum import Enum
from importlib.resources import files

import chex
import jax
import jax.numpy as jnp
from termcolor import colored

from puxle.core.puzzle_base import Puzzle
from puxle.core.puzzle_state import FieldDescriptor, PuzzleState, state_dataclass
from puxle.utils.annotate import IMG_SIZE

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

    def define_state_class(self) -> PuzzleState:
        """Defines the state class for Sokoban using xtructure."""
        str_parser = self.get_string_parser()
        size = self.size

        @state_dataclass
        class State:
            board: FieldDescriptor.packed_tensor(shape=(size * size,), packed_bits=2)

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

    def data_init(self):
        try:
            # Try to load as package resources first (for installed packages)
            data_pkg = files("puxle.data.sokoban")
            self.init_puzzles = jnp.load(data_pkg / "init.npy")
            self.target_puzzles = jnp.load(data_pkg / "target.npy")
        except (FileNotFoundError, ModuleNotFoundError):
            # Fallback to relative paths (for development/source directory)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(current_dir, "..", "data", "sokoban")

            self.init_puzzles = jnp.load(os.path.join(data_dir, "init.npy"))
            self.target_puzzles = jnp.load(os.path.join(data_dir, "target.npy"))

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
        if self.solve_condition == Sokoban.SolveCondition.ALL_BOXES_ON_TARGET_AND_PLAYER_ON_TARGET:
            # If we need to modify the board (e.g. place agent randomly),
            # we first unpack the data to work with logical values.
            board = self.State(board=target_data).board_unpacked
            board = self._place_agent_randomly(board, key)
            target_state = self.State.from_unpacked(board=board)
        else:
            # Otherwise, use the pre-packed target data directly.
            target_state = self.State(board=target_data)
        return self.SolveConfig(TargetState=target_state)

    def is_solved(self, solve_config: Puzzle.SolveConfig, state: "Sokoban.State") -> bool:
        # Unpack boards for comparison.
        board = state.board_unpacked
        t_board = solve_config.TargetState.board_unpacked
        # Remove the player from the current board.
        if self.solve_condition == Sokoban.SolveCondition.ALL_BOXES_ON_TARGET_AND_PLAYER_ON_TARGET:
            return jnp.all(board == t_board)
        else:
            rm_player = jnp.where(
                board == Sokoban.Object.PLAYER.value, Sokoban.Object.EMPTY.value, board
            )
            return jnp.all(rm_player == t_board)

    def action_to_string(self, action: int) -> str:
        return self._directional_action_to_string(action)

    def get_solve_config_string_parser(self):
        def parser(solve_config: "Sokoban.SolveConfig", **kwargs):
            return solve_config.TargetState.str(solve_config=solve_config)

        return parser

    def get_string_parser(self):
        form = self._get_visualize_format()

        def to_char(x):
            match x:
                case Sokoban.Object.EMPTY.value:
                    return " "
                case Sokoban.Object.WALL.value:
                    return colored("■", "white")
                case Sokoban.Object.PLAYER.value:
                    return colored("●", "green")
                case Sokoban.Object.BOX.value:
                    return colored("■", "yellow")
                case Sokoban.Object.TARGET.value:
                    return colored("x", "red")
                case Sokoban.Object.PLAYER_ON_TARGET.value:
                    return colored("ⓧ", "red")
                case Sokoban.Object.BOX_ON_TARGET.value:
                    return colored("■", "green")
                case Sokoban.Object.TARGET_PLAYER.value:
                    return colored("●", "red")
                case _:
                    return "?"

        def parser(state: "Sokoban.State", solve_config: "Sokoban.SolveConfig" = None, **kwargs):
            # Unpack the board before visualization.
            board = state.board_unpacked
            if solve_config is not None:
                goal = solve_config.TargetState.board_unpacked
                for i in range(self.size):
                    for j in range(self.size):
                        if goal[i * self.size + j] == Sokoban.Object.BOX.value:
                            match board[i * self.size + j]:
                                case Sokoban.Object.PLAYER.value:
                                    board = board.at[i * self.size + j].set(
                                        Sokoban.Object.PLAYER_ON_TARGET.value
                                    )
                                case Sokoban.Object.BOX.value:
                                    board = board.at[i * self.size + j].set(
                                        Sokoban.Object.BOX_ON_TARGET.value
                                    )
                                case Sokoban.Object.EMPTY.value:
                                    board = board.at[i * self.size + j].set(
                                        Sokoban.Object.TARGET.value
                                    )
                        if goal[i * self.size + j] == Sokoban.Object.PLAYER.value:
                            match board[i * self.size + j]:
                                case Sokoban.Object.PLAYER.value:
                                    board = board.at[i * self.size + j].set(
                                        Sokoban.Object.PLAYER.value
                                    )
                                case Sokoban.Object.BOX.value:
                                    board = board.at[i * self.size + j].set(
                                        Sokoban.Object.BOX.value
                                    )
                                case Sokoban.Object.EMPTY.value:
                                    board = board.at[i * self.size + j].set(
                                        Sokoban.Object.TARGET_PLAYER.value
                                    )
                                case _:
                                    pass

            return form.format(*map(to_char, board))

        return parser

    def get_actions(
        self, solve_config: Puzzle.SolveConfig, state: "Sokoban.State", action: chex.Array, filled: bool = True
    ) -> tuple["Sokoban.State", chex.Array]:
        """
        Returns the next state and cost for a given action.
        """
        # Unpack the board so that we work on a flat representation.
        board = state.board_unpacked
        x, y = self._get_player_position(state)
        current_pos = jnp.array([x, y])
        moves = jnp.array([[0, -1], [0, 1], [-1, 0], [1, 0]])
        
        direction = moves[action]

        # Helper: convert (row, col) to flat index
        def flat_idx(i, j):
            return i * self.size + j

        def is_empty(i, j):
            return board[flat_idx(i, j)] == Sokoban.Object.EMPTY.value

        def is_valid_pos(i, j):
            return jnp.logical_and(
                jnp.logical_and(i >= 0, i < self.size), jnp.logical_and(j >= 0, j < self.size)
            )

        new_pos = (current_pos + direction).astype(current_pos.dtype)
        new_x, new_y = new_pos[0], new_pos[1]
        valid_move = is_valid_pos(new_x, new_y)

        def invalid_case():
            return state, jnp.inf

        def process_move():
            target = board[flat_idx(new_x, new_y)]
            # Case when target cell is empty: simply move the player.

            def move_empty():
                new_board = board.at[flat_idx(current_pos[0], current_pos[1])].set(
                    Sokoban.Object.EMPTY.value
                )
                new_board = new_board.at[flat_idx(new_x, new_y)].set(
                    Sokoban.Object.PLAYER.value
                )
                return state.set_unpacked(board=new_board), 1.0

            # Case when target cell contains a box: attempt to push it.
            def push_box():
                push_pos = (new_pos + direction).astype(current_pos.dtype)
                push_x, push_y = push_pos[0], push_pos[1]
                valid_push = jnp.logical_and(
                    is_valid_pos(push_x, push_y), is_empty(push_x, push_y)
                )
                valid_push = jnp.logical_and(valid_push, filled)

                def do_push():
                    new_board = board.at[flat_idx(current_pos[0], current_pos[1])].set(
                        Sokoban.Object.EMPTY.value
                    )
                    new_board = new_board.at[flat_idx(new_x, new_y)].set(
                        Sokoban.Object.PLAYER.value
                    )
                    new_board = new_board.at[flat_idx(push_x, push_y)].set(
                        Sokoban.Object.BOX.value
                    )
                    return state.set_unpacked(board=new_board), 1.0

                return jax.lax.cond(valid_push, do_push, invalid_case)

            return jax.lax.cond(
                jnp.equal(target, Sokoban.Object.EMPTY.value),
                move_empty,
                lambda: jax.lax.cond(
                    jnp.equal(target, Sokoban.Object.BOX.value), push_box, invalid_case
                ),
            )

        next_state, cost = jax.lax.cond(valid_move, process_move, invalid_case)
        cost = jnp.where(filled, cost, jnp.inf)
        return next_state, cost

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

    def get_img_parser(self):
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

        assets = {
            0: cv2.resize(
                cv2.cvtColor(
                    cv2.imread(os.path.join(image_dir, "floor.png"), cv2.IMREAD_COLOR),
                    cv2.COLOR_BGR2RGB,
                ),
                (cell_w, cell_h),
                interpolation=cv2.INTER_AREA,
            ),
            1: cv2.resize(
                cv2.cvtColor(
                    cv2.imread(os.path.join(image_dir, "wall.png"), cv2.IMREAD_COLOR),
                    cv2.COLOR_BGR2RGB,
                ),
                (cell_w, cell_h),
                interpolation=cv2.INTER_AREA,
            ),
            2: cv2.resize(
                cv2.cvtColor(
                    cv2.imread(os.path.join(image_dir, "agent.png"), cv2.IMREAD_COLOR),
                    cv2.COLOR_BGR2RGB,
                ),
                (cell_w, cell_h),
                interpolation=cv2.INTER_AREA,
            ),
            3: cv2.resize(
                cv2.cvtColor(
                    cv2.imread(os.path.join(image_dir, "box.png"), cv2.IMREAD_COLOR),
                    cv2.COLOR_BGR2RGB,
                ),
                (cell_w, cell_h),
                interpolation=cv2.INTER_AREA,
            ),
            4: cv2.resize(
                cv2.cvtColor(
                    cv2.imread(os.path.join(image_dir, "box_target.png"), cv2.IMREAD_COLOR),
                    cv2.COLOR_BGR2RGB,
                ),
                (cell_w, cell_h),
                interpolation=cv2.INTER_AREA,
            ),
            5: cv2.resize(
                cv2.cvtColor(
                    cv2.imread(os.path.join(image_dir, "agent_on_target.png"), cv2.IMREAD_COLOR),
                    cv2.COLOR_BGR2RGB,
                ),
                (cell_w, cell_h),
                interpolation=cv2.INTER_AREA,
            ),
            6: cv2.resize(
                cv2.cvtColor(
                    cv2.imread(os.path.join(image_dir, "box_on_target.png"), cv2.IMREAD_COLOR),
                    cv2.COLOR_BGR2RGB,
                ),
                (cell_w, cell_h),
                interpolation=cv2.INTER_AREA,
            ),
            7: cv2.resize(
                np.roll(
                    cv2.imread(os.path.join(image_dir, "agent.png"), cv2.IMREAD_COLOR),
                    -1,
                    axis=2,  # color channel G -> R
                ),
                (cell_w, cell_h),
                interpolation=cv2.INTER_AREA,
            ),
        }

        def img_func(state: "Sokoban.State", solve_config: "Sokoban.SolveConfig" = None, **kwargs):
            img = np.zeros(IMG_SIZE + (3,), np.uint8)

            board = np.array(state.board_unpacked)
            if solve_config is not None:
                goal = np.array(solve_config.TargetState.board_unpacked)
            else:
                goal = None
            for i in range(self.size):
                for j in range(self.size):
                    cell_val = int(board[i * self.size + j])
                    if (
                        goal is not None and goal[i * self.size + j] == Sokoban.Object.BOX.value
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
                        goal is not None and goal[i * self.size + j] == Sokoban.Object.PLAYER.value
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
                        img[i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w] = asset
                    else:
                        # Fallback: fill with a gray square if image not found
                        img[i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w] = (
                            127,
                            127,
                            127,
                        )
            return img

        return img_func

    def _get_box_positions(self, board: jnp.ndarray, number_of_boxes: int = 4) -> jnp.ndarray:
        """
        Get the positions of all boxes on the board.
        """
        flat_index = jnp.argsort(board == Sokoban.Object.BOX.value)[-number_of_boxes:]
        return jnp.unravel_index(flat_index, (self.size, self.size))

    def _place_agent_randomly(self, board: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Place the agent randomly on the board.
        """
        box_xs, box_ys = self._get_box_positions(board)
        box_positions = jnp.expand_dims(jnp.stack([box_xs, box_ys], axis=1), 0)  # shape: (1, 4, 2)
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
        new_board = board.at[near_positions[idx, 0] * self.size + near_positions[idx, 1]].set(
            Sokoban.Object.PLAYER.value
        )
        return new_board

    def solve_config_to_state_transform(
        self, solve_config: "Sokoban.SolveConfig", key: jax.random.PRNGKey = None
    ) -> "Sokoban.State":
        """
        This function shoulde transformt the solve config to the state.
        """
        board = solve_config.TargetState.board_unpacked
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
            solve_config = solve_config.replace(TargetState=self.State.from_unpacked(board=rm_player))
        else:
            solve_config = solve_config.replace(TargetState=self.State.from_unpacked(board=board))
        return solve_config

    def get_inverse_neighbours(
        self, solve_config: "Sokoban.SolveConfig", state: "Sokoban.State", filled: bool = True
    ) -> tuple["Sokoban.State", chex.Array]:
        """
        Returns possible previous states and their associated costs.
        In Sokoban, inverse moves correspond to 'pulling' a box or simply moving back
        to the previous position if no box is involved.
        If an inverse move is not possible, it returns the original state with an infinite cost.
        """

        board = state.board_unpacked
        x, y = self._get_player_position(state)
        current_pos = jnp.array([x, y])
        moves = jnp.array([[0, -1], [0, 1], [-1, 0], [1, 0]])

        def flat_idx(i, j):
            return i * self.size + j

        def is_empty(i, j):
            return board[flat_idx(i, j)] == Sokoban.Object.EMPTY.value

        def is_valid_pos(i, j):
            return jnp.logical_and(
                jnp.logical_and(i >= 0, i < self.size), jnp.logical_and(j >= 0, j < self.size)
            )

        def inv_move(direction):
            direction = direction.astype(current_pos.dtype)
            # Candidate previous player's position (i.e. where the player came from)
            prev_pos = current_pos - direction
            valid_prev = is_valid_pos(prev_pos[0], prev_pos[1])
            # For a pull move, consider the cell in front of the current player.
            front_pos = current_pos + direction
            valid_front = is_valid_pos(front_pos[0], front_pos[1])
            box_at_front = jnp.logical_and(
                valid_front, board[flat_idx(front_pos[0], front_pos[1])] == Sokoban.Object.BOX.value
            )
            empty_prev = jnp.logical_and(valid_prev, is_empty(prev_pos[0], prev_pos[1]))

            def invalid():
                return state, 1.0

            # Inverse pull: the forward push is inverted by pulling the box from ahead.
            # In a forward push, the player moved from prev_pos
            # to current_pos and pushed the box from current_pos to front_pos.
            # Here, we "pull" the box from front_pos back into current_pos while moving the player to prev_pos.
            def do_pull():
                new_board = board
                new_board = new_board.at[flat_idx(front_pos[0], front_pos[1])].set(
                    Sokoban.Object.EMPTY.value
                )
                new_board = new_board.at[flat_idx(current_pos[0], current_pos[1])].set(
                    Sokoban.Object.BOX.value
                )
                new_board = new_board.at[flat_idx(prev_pos[0], prev_pos[1])].set(
                    Sokoban.Object.PLAYER.value
                )
                return state.set_unpacked(board=new_board), 1.0

            # Inverse simple move: simply move the player back if no box is involved.
            def do_simple():
                new_board = board
                new_board = new_board.at[flat_idx(current_pos[0], current_pos[1])].set(
                    Sokoban.Object.EMPTY.value
                )
                new_board = new_board.at[flat_idx(prev_pos[0], prev_pos[1])].set(
                    Sokoban.Object.PLAYER.value
                )
                return state.set_unpacked(board=new_board), 1.0

            def branch_fn():
                # Only allow a move if the previous cell is empty.
                return jax.lax.cond(
                    empty_prev,
                    lambda: jax.lax.cond(
                        box_at_front,
                        do_pull,
                        do_simple,
                    ),
                    invalid,
                )

            return jax.lax.cond(valid_prev, branch_fn, invalid)

        new_states, costs = jax.vmap(inv_move)(moves)
        costs = jnp.where(filled, costs, jnp.inf)
        return new_states, costs


class SokobanHard(Sokoban):
    def data_init(self):
        try:
            # Try to load as package resources first (for installed packages)
            data_pkg = files("puxle.data.sokoban")
            self.init_puzzles = jnp.load(data_pkg / "init_hard.npy")
            self.target_puzzles = jnp.load(data_pkg / "target_hard.npy")
        except (FileNotFoundError, ModuleNotFoundError):
            # Fallback to relative paths (for development/source directory)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(current_dir, "..", "data", "sokoban")

            self.init_puzzles = jnp.load(os.path.join(data_dir, "init_hard.npy"))
            self.target_puzzles = jnp.load(os.path.join(data_dir, "target_hard.npy"))

        self.num_puzzles = self.init_puzzles.shape[0]
