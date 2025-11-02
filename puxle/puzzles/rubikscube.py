from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np
from tabulate import tabulate

from puxle.core.puzzle_base import Puzzle
from puxle.core.puzzle_state import FieldDescriptor, PuzzleState, state_dataclass
from puxle.utils.annotate import IMG_SIZE
from puxle.utils.util import coloring_str, from_uint8, to_uint8

TYPE = jnp.uint8
LINE_THICKNESS = 3

UP = 0
FRONT = 1
RIGHT = 2
BACK = 3
LEFT = 4
DOWN = 5

rotate_face_map = {UP: "u", FRONT: "f", RIGHT: "r", BACK: "b", LEFT: "l", DOWN: "d"}
face_map_legend = {
    UP: "up",
    FRONT: "front",
    RIGHT: "right",
    BACK: "back",
    LEFT: "left",
    DOWN: "down",
}
face_map = {
    UP: "up━",
    FRONT: "front",
    RIGHT: "right",
    BACK: "back━",
    LEFT: "left━",
    DOWN: "down━",
}
rgb_map = {
    UP: (255, 255, 255),  # white
    FRONT: (0, 255, 0),  # green
    RIGHT: (255, 0, 0),  # red
    BACK: (0, 0, 255),  # blue
    LEFT: (255, 128, 0),  # orange
    DOWN: (255, 255, 0),  # yellow
}


def rot90_traceable(m, k=1, axes=(0, 1)):
    k %= 4
    return jax.lax.switch(k, [partial(jnp.rot90, m, k=i, axes=axes) for i in range(4)])


# (rolled_faces, rotate_axis_for_rolled_faces)
# 0: x-axis(left), 1: y-axis(up), 2: z-axis(front)


class RubiksCube(Puzzle):
    size: int
    index_grid: chex.Array

    @property
    def _active_bits(self) -> int:
        return 3 if self.color_embedding else 8

    @property
    def _token_width(self) -> int:
        return 1 if self.color_embedding else len(str(self._num_tiles - 1))

    def define_state_class(self) -> PuzzleState:
        str_parser = self.get_string_parser()
        raw_shape = (6, self.size * self.size)
        raw = jnp.zeros(raw_shape, dtype=TYPE)
        active_bits = self._active_bits
        packed_faces = to_uint8(raw, active_bits)

        @state_dataclass
        class State:
            faces: FieldDescriptor[TYPE, packed_faces.shape]

            def __str__(self, **kwargs):
                return str_parser(self, **kwargs)

            @property
            def packed(self):
                return State(faces=to_uint8(self.faces, active_bits))

            @property
            def unpacked(self):
                return State(faces=from_uint8(self.faces, raw_shape, active_bits))

        return State

    def __init__(
        self,
        size: int = 3,
        initial_shuffle: int = 10,
        color_embedding: bool = True,
        metric: str = "QTM",
        **kwargs,
    ):
        self.size = size
        self.initial_shuffle = initial_shuffle
        self.color_embedding = color_embedding
        metric_upper = metric.upper()
        supported_metrics = {"QTM", "UQTM"}
        if metric_upper not in supported_metrics:
            raise ValueError(
                f"Unsupported metric '{metric}'. Supported metrics: {', '.join(sorted(supported_metrics))}."
            )
        self.metric = metric_upper
        self._tile_count = self.size * self.size
        self._num_tiles = 6 * self._tile_count
        self._validate_tile_capacity()
        is_even = size % 2 == 0
        center_index = size // 2
        include_center = is_even or self.metric == "UQTM"
        if include_center:
            indices = list(range(size))
        else:
            indices = [i for i in range(size) if i != center_index]
        self.index_grid = jnp.asarray(indices, dtype=jnp.uint8)
        super().__init__(**kwargs)

    def _validate_tile_capacity(self):
        if not self.color_embedding and self._num_tiles > 256:
            raise ValueError(
                "Tile-ID mode requires unique 8-bit identifiers; decrease cube size to keep tile count ≤ 256."
            )

    def _solved_faces(self) -> chex.Array:
        if self.color_embedding:
            return jnp.repeat(jnp.arange(6, dtype=TYPE)[:, None], self._tile_count, axis=1)
        return jnp.arange(self._num_tiles, dtype=TYPE).reshape((6, self._tile_count))

    def _color_index_from_value(self, value: int) -> int:
        value_int = int(value)
        if self.color_embedding:
            return value_int
        return value_int // self._tile_count

    def _color_indices(self, stickers: np.ndarray | chex.Array) -> np.ndarray:
        stickers_np = np.array(stickers)
        if self.color_embedding:
            return stickers_np
        return stickers_np // self._tile_count

    def convert_tile_to_color_embedding(
        self, tile_faces: np.ndarray | chex.Array
    ) -> jnp.ndarray:
        """
        Convert faces expressed with tile identifiers (0..6*tile_count-1) into
        color embedding (0..5). Accepts shapes (6, tile_count), (6, size, size) or flat.
        """
        faces = jnp.asarray(tile_faces)
        tile_count = self._tile_count
        if faces.size != 6 * tile_count:
            raise ValueError(
                f"Expected {6 * tile_count} elements for tile faces, got {faces.size}"
            )
        color_faces = (faces.reshape(6, tile_count) // tile_count).astype(jnp.uint8)
        return color_faces.reshape(faces.shape)

    def _format_tile(self, value: int, *, as_color: bool) -> str:
        color_idx = self._color_index_from_value(value)
        if as_color:
            token = "■"
        else:
            token = str(int(value)).rjust(self._token_width)
        return coloring_str(token, rgb_map[color_idx])

    def get_string_parser(self):
        def parser(state: "RubiksCube.State", *, use_color_overlay: bool = False, **_):
            # Unpack the state faces before printing
            unpacked_faces = state.unpacked.faces
            as_color = self.color_embedding or use_color_overlay

            # Helper function to get face string
            def get_empty_face_string():
                return "\n".join(["  " * (self.size + 2) for _ in range(self.size + 2)])

            def color_legend():
                return "\n".join(
                    [f"{face_map_legend[i]:<6}:{coloring_str('■', rgb_map[i])}" for i in range(6)]
                )

            def get_face_string(face):
                face_str = face_map[face]
                display_tile_width = 1 if as_color else self._token_width
                row_display_width = self.size * display_tile_width + (self.size - 1)
                inner_width = row_display_width
                string = f"┏━{face_str.center(inner_width, '━')}━┓\n"
                for j in range(self.size):
                    tokens = []
                    for i in range(self.size):
                        value = unpacked_faces[face, j * self.size + i]
                        tokens.append(self._format_tile(value, as_color=as_color))
                    row = " ".join(tokens)
                    string += f"┃ {row.ljust(row_display_width)} ┃\n"
                string += f"┗━{'━' * inner_width}━┛\n"
                return string

            # Create the cube string representation
            cube_str = tabulate(
                [
                    [color_legend(), (".\n" + get_face_string(UP))],
                    [
                        get_face_string(LEFT),
                        get_face_string(FRONT),
                        get_face_string(RIGHT),
                        get_face_string(BACK),
                    ],
                    [get_empty_face_string(), get_face_string(DOWN)],
                ],
                tablefmt="plain",
                rowalign="center",
            )
            return cube_str

        return parser

    def get_initial_state(
        self, solve_config: Puzzle.SolveConfig, key=None, data=None
    ) -> "RubiksCube.State":
        return self._get_suffled_state(
            solve_config, solve_config.TargetState, key, num_shuffle=self.initial_shuffle
        )

    def get_target_state(self, key=None) -> "RubiksCube.State":
        faces = self._solved_faces()
        return self.State(faces=faces).packed

    def get_solve_config(self, key=None, data=None) -> Puzzle.SolveConfig:
        return self.SolveConfig(TargetState=self.get_target_state(key))

    def get_neighbours(
        self, solve_config: Puzzle.SolveConfig, state: "RubiksCube.State", filled: bool = True
    ) -> tuple["RubiksCube.State", chex.Array]:
        def map_fn(state, axis, index, clockwise):
            return jax.lax.cond(
                filled,
                lambda: (self._rotate(state, axis, index, clockwise), 1.0),
                lambda: (state, jnp.inf),
            )

        axis_grid, index_grid, clockwise_grid = jnp.meshgrid(
            jnp.arange(3), self.index_grid, jnp.arange(2)
        )
        axis_grid = axis_grid.reshape(-1)
        index_grid = index_grid.reshape(-1)
        clockwise_grid = clockwise_grid.reshape(-1)
        new_states, costs = jax.vmap(map_fn, in_axes=(None, 0, 0, 0))(
            state, axis_grid, index_grid, clockwise_grid
        )
        return new_states, costs

    def is_solved(self, solve_config: Puzzle.SolveConfig, state: "RubiksCube.State") -> bool:
        return state == solve_config.TargetState

    @property
    def inverse_action_map(self) -> jnp.ndarray | None:
        """
        Defines the inverse action mapping for Rubik's Cube.
        A rotation in one direction (e.g., clockwise) is inverted by a rotation
        in the opposite direction (counter-clockwise) on the same axis and slice.

        Actions are generated from a meshgrid of (axis, index, clockwise), with
        clockwise being the fastest-changing dimension. This means actions are
        interleaved as [cw, ccw, cw, ccw, ...]. The inverse of action `2k` (cw)
        is `2k+1` (ccw), and vice versa.
        """
        num_actions = 3 * len(self.index_grid) * 2
        actions = jnp.arange(num_actions)

        # Reshape to pair up cw/ccw actions, flip them, and flatten back
        inv_map = jnp.reshape(actions, (-1, 2))
        inv_map = jnp.flip(inv_map, axis=1)
        inv_map = jnp.reshape(inv_map, (-1,))

        return inv_map

    def action_to_string(self, action: int) -> str:
        """
        This function should return a string representation of the action.
        Actions are encoded as (axis, index, clockwise) where:
        - axis: 0=x-axis, 1=y-axis, 2=z-axis
        - index: slice index (0 or 2 for 3x3 cube)
        - clockwise: 0=counterclockwise, 1=clockwise

        For cubes larger than 3x3x3, internal slice rotations are named
        with layer numbers (e.g., L2, R2 for 4x4x4 cube).
        """
        # Decode action into components. The meshgrid in `get_neighbours` yields
        # actions ordered as:
        #   counterclockwise/clockwise (fastest) × axis (next) × slice index (slowest).
        num_axes = 3
        num_indices = len(self.index_grid)
        action_limit = num_axes * num_indices * 2
        if action < 0 or action >= action_limit:
            raise ValueError(f"Action {action} is out of bounds for action space size {action_limit}.")

        clockwise = bool(action % 2)
        axis = int((action // 2) % num_axes)
        index_idx = int(action // (2 * num_axes))

        # Map (axis, index) to face using the same logic as _rotate method
        actual_index = int(self.index_grid[index_idx])

        is_center_slice = (
            self.metric == "UQTM" and self.size % 2 == 1 and actual_index == (self.size // 2)
        )
        if is_center_slice:
            center_labels = {0: "M", 1: "E", 2: "S"}
            try:
                face_str = center_labels[axis]
            except KeyError as exc:
                raise ValueError(f"Invalid center rotation (axis={axis})") from exc
        elif self.size <= 3:
            edge_labels = {
                (0, 0): "L",
                (0, self.size - 1): "R",
                (1, 0): "D",
                (1, self.size - 1): "U",
                (2, 0): "F",
                (2, self.size - 1): "B",
            }
            try:
                face_str = edge_labels[(axis, actual_index)]
            except KeyError as exc:
                raise ValueError(f"Invalid edge rotation (axis={axis}, index={actual_index})") from exc
        else:
            # For cubes larger than 3x3x3, use layer-based naming
            # Define face name pairs for each axis: (negative_direction, positive_direction)
            face_pairs = [("L", "R"), ("D", "U"), ("F", "B")]
            negative_face, positive_face = face_pairs[axis]

            # Determine which face this slice is closer to and calculate layer number
            mid_point = (self.size - 1) / 2
            if actual_index < mid_point:
                # Closer to negative direction face (L, D, F)
                face_name = negative_face
                layer_num = actual_index + 1
            else:
                # Closer to positive direction face (R, U, B)
                face_name = positive_face
                layer_num = self.size - actual_index

            # For layer 1, don't include the number
            face_str = face_name if layer_num == 1 else f"{face_name}{layer_num}"

        if face_str in {"U", "R", "F"}:
            suffix = "" if not clockwise else "'"
        else:
            suffix = "" if clockwise else "'"
        return f"{face_str}{suffix}"

    @staticmethod
    def _rotate_face(shaped_faces: chex.Array, clockwise: bool, mul: int):
        return rot90_traceable(shaped_faces, jnp.where(clockwise, mul, -mul))

    def _rotate(self, state: "RubiksCube.State", axis: int, index: int, clockwise: bool = True):
        # rotate the edge clockwise or counterclockwise
        # axis is the axis of the rotation, 0 for x, 1 for y, 2 for z
        # index is the index of the edge to rotate
        # clockwise is a boolean, True for clockwise, False for counterclockwise
        faces = state.unpacked.faces
        shaped_faces = faces.reshape((6, self.size, self.size))

        rotate_edge_map = jnp.array(
            [
                [UP, FRONT, DOWN, BACK],  # x-axis (rotates around columns)
                [LEFT, FRONT, RIGHT, BACK],  # y-axis (rotates around rows)
                [UP, LEFT, DOWN, RIGHT],  # z-axis (rotates around depth)
            ]
        )
        rotate_edge_rot = jnp.array(
            [
                [-1, -1, -1, -1],  # x-axis
                [2, 2, 2, 0],  # y-axis
                [2, 1, 0, 3],  # z-axis
            ]
        )
        edge_faces = rotate_edge_map[axis]
        edge_rot = rotate_edge_rot[axis]
        shaped_faces = shaped_faces.at[BACK].set(
            jnp.flip(jnp.flip(shaped_faces[BACK], axis=0), axis=1)
        )
        rolled_faces = shaped_faces[edge_faces]
        rolled_faces = jax.vmap(lambda face, rot: rot90_traceable(face, k=rot))(
            rolled_faces, edge_rot
        )
        rolled_faces = rolled_faces.at[:, index, :].set(
            jnp.roll(rolled_faces[:, index, :], jnp.where(clockwise, 1, -1), axis=0)
        )
        rolled_faces = jax.vmap(lambda face, rot: rot90_traceable(face, k=-rot))(
            rolled_faces, edge_rot
        )
        shaped_faces = shaped_faces.at[edge_faces].set(rolled_faces)
        shaped_faces = shaped_faces.at[BACK].set(
            jnp.flip(jnp.flip(shaped_faces[BACK], axis=1), axis=0)
        )
        is_edge = jnp.isin(index, jnp.array([0, self.size - 1]))
        switch_num = jnp.where(
            is_edge, 1 + 2 * axis + index // (self.size - 1), 0
        )  # 0: None, 1: left, 2: right, 3: down, 4: up, 5: front, 6: back
        shaped_faces = jax.lax.switch(
            switch_num,
            [
                lambda: shaped_faces,  # 0: None
                lambda: shaped_faces.at[LEFT].set(
                    self._rotate_face(shaped_faces[LEFT], clockwise, -1)
                ),  # 1: left
                lambda: shaped_faces.at[RIGHT].set(
                    self._rotate_face(shaped_faces[RIGHT], clockwise, 1)
                ),  # 2: right
                lambda: shaped_faces.at[DOWN].set(
                    self._rotate_face(shaped_faces[DOWN], clockwise, -1)
                ),  # 3: down
                lambda: shaped_faces.at[UP].set(
                    self._rotate_face(shaped_faces[UP], clockwise, 1)
                ),  # 4: up
                lambda: shaped_faces.at[FRONT].set(
                    self._rotate_face(shaped_faces[FRONT], clockwise, 1)
                ),  # 5: front
                lambda: shaped_faces.at[BACK].set(
                    self._rotate_face(shaped_faces[BACK], clockwise, -1)
                ),  # 6: back
            ],
        )
        faces = jnp.reshape(shaped_faces, (6, self.size * self.size))
        return self.State(faces=faces).packed

    def get_img_parser(self):
        """
        This function is a decorator that adds an img_parser to the class.
        """
        import math

        import cv2
        import numpy as np

        def img_func(state: "RubiksCube.State", another_faces: bool = True, **kwargs):
            imgsize = IMG_SIZE[0]
            # Create a blank image with a neutral background
            img = np.zeros((imgsize, imgsize, 3), dtype=np.uint8)
            img[:] = (190, 190, 190)

            # Set up projection parameters for a 45° view from above
            cos45 = math.cos(math.pi / 4)
            sin45 = math.sin(math.pi / 4)

            # Orthographic projection after a rotation: first around y then around x
            def project(x, y, z):
                u = cos45 * x - sin45 * z  # Changed sign for z component
                v = cos45 * y + 0.5 * (x + z)  # Modified formula for correct orientation
                return u, v

            # Determine the cube's bounding box in projection to scale and center it on the image
            vertices = []
            # Top face (UP): shifted down by adjusting y coordinates
            vertices += [(0, 0, 0), (self.size, 0, 0), (self.size, 0, self.size), (0, 0, self.size)]
            # Front face (FRONT): shifted down
            vertices += [
                (0, 0, self.size),
                (self.size, 0, self.size),
                (self.size, -self.size, self.size),
                (0, -self.size, self.size),
            ]
            # Right face (RIGHT): shifted down
            vertices += [
                (self.size, 0, self.size),
                (self.size, -self.size, self.size),
                (self.size, -self.size, 0),
                (self.size, 0, 0),
            ]

            proj_pts = [project(x, y, z) for (x, y, z) in vertices]
            us = [pt[0] for pt in proj_pts]
            vs = [pt[1] for pt in proj_pts]
            min_u, max_u = min(us), max(us)
            min_v, max_v = min(vs), max(vs)
            margin = imgsize * 0.05  # Increased margin to 15% to move image down
            available_width = imgsize - 2 * margin
            available_height = imgsize - 2 * margin
            scale = min(available_width / (max_u - min_u), available_height / (max_v - min_v))
            offset_x = margin - min_u * scale
            offset_y = (
                margin - min_v * scale - 0.25 * available_width
            )  # Increased y offset by 50% to move image down

            def transform(x, y, z):
                u, v = project(x, y, z)
                return int(u * scale + offset_x), int(v * scale + offset_y)

            # Obtain sticker data, colour mapping, and helper for drawing numbered tiles
            stickers = np.array(state.unpacked.faces, dtype=np.int32).reshape((6, self.size, self.size))
            color_faces = self._color_indices(stickers).reshape((6, self.size, self.size))

            def draw_tile(img_target, pts, face_id, row, col):
                color_idx = int(color_faces[face_id, row, col])
                value = int(stickers[face_id, row, col])
                color = rgb_map[color_idx]
                cv2.fillPoly(img_target, [pts], color)
                cv2.polylines(
                    img_target,
                    [pts],
                    isClosed=True,
                    color=(0, 0, 0),
                    thickness=LINE_THICKNESS,
                )

                if not self.color_embedding:
                    center = np.mean(pts[:, 0, :], axis=0)
                    edge = np.linalg.norm(pts[0, 0, :] - pts[1, 0, :])
                    font_scale = max(0.3, min(1.2, edge / 32.0))
                    thickness = max(1, int(round(LINE_THICKNESS / 2)))
                    text = str(value)
                    (text_width, text_height), baseline = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                    )
                    text_x = int(center[0] - text_width / 2)
                    text_y = int(center[1] + text_height / 2)

                    cv2.putText(
                        img_target,
                        text,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (0, 0, 0),
                        thickness,
                        lineType=cv2.LINE_AA,
                    )

            # Draw faces in correct order for proper depth.
            # 1. Draw the front face (FRONT)
            for i in range(self.size):
                for j in range(self.size):
                    # Modified coordinates for correct orientation
                    p0 = (j, i, self.size)
                    p1 = (j + 1, i, self.size)
                    p2 = (j + 1, i + 1, self.size)
                    p3 = (j, i + 1, self.size)
                    pts = np.array(
                        [transform(*p0), transform(*p1), transform(*p2), transform(*p3)], np.int32
                    ).reshape((-1, 1, 2))
                    draw_tile(img, pts, FRONT, i, j)

            # 2. Draw the right face (RIGHT)
            for i in range(self.size):
                for j in range(self.size):
                    # Modified coordinates for correct orientation
                    p0 = (self.size, i, self.size - j)
                    p1 = (self.size, i, self.size - (j + 1))
                    p2 = (self.size, i + 1, self.size - (j + 1))
                    p3 = (self.size, i + 1, self.size - j)
                    pts = np.array(
                        [transform(*p0), transform(*p1), transform(*p2), transform(*p3)], np.int32
                    ).reshape((-1, 1, 2))
                    draw_tile(img, pts, RIGHT, i, j)

            # 3. Draw the top face (UP) last so that it appears above the other faces
            for i in range(self.size):
                for j in range(self.size):
                    p0 = (j, 0, self.size - i)
                    p1 = (j + 1, 0, self.size - i)
                    p2 = (j + 1, 0, self.size - (i + 1))
                    p3 = (j, 0, self.size - (i + 1))
                    pts = np.array(
                        [transform(*p0), transform(*p1), transform(*p2), transform(*p3)], np.int32
                    ).reshape((-1, 1, 2))
                    # Note: for UP, flip the row order to match orientation
                    draw_tile(img, pts, UP, self.size - i - 1, j)

            # If another_faces is True, draw additional faces (DOWN, BACK, LEFT) as flat squares
            if another_faces:
                img2 = np.zeros((imgsize, imgsize, 3), dtype=np.uint8)
                img2[:] = (190, 190, 190)

                # 4. Draw the back face (BACK)
                for i in range(self.size):
                    for j in range(self.size):
                        # Modified coordinates for correct orientation
                        p0 = (self.size - j - 1, i, 0)
                        p1 = (self.size - j, i, 0)
                        p2 = (self.size - j, i + 1, 0)
                        p3 = (self.size - j - 1, i + 1, 0)
                        pts = np.array(
                            [transform(*p0), transform(*p1), transform(*p2), transform(*p3)],
                            np.int32,
                        ).reshape((-1, 1, 2))
                        draw_tile(img2, pts, BACK, i, j)

                # 2. Draw the down face (DOWN)
                for i in range(self.size):
                    for j in range(self.size):
                        # Modified coordinates for correct orientation
                        p0 = (i, self.size, j)
                        p1 = (i, self.size, j + 1)
                        p2 = (i + 1, self.size, j + 1)
                        p3 = (i + 1, self.size, j)
                        pts = np.array(
                            [transform(*p0), transform(*p1), transform(*p2), transform(*p3)],
                            np.int32,
                        ).reshape((-1, 1, 2))
                        draw_tile(img2, pts, DOWN, self.size - j - 1, i)

                # 3. Draw the left face (LEFT) last so that it appears above the other faces
                for i in range(self.size):
                    for j in range(self.size):
                        # Modified coordinates for correct orientation
                        p0 = (0, i, j)
                        p1 = (0, i, j + 1)
                        p2 = (0, i + 1, j + 1)
                        p3 = (0, i + 1, j)
                        pts = np.array(
                            [transform(*p0), transform(*p1), transform(*p2), transform(*p3)],
                            np.int32,
                        ).reshape((-1, 1, 2))
                        draw_tile(img2, pts, LEFT, i, j)

                img = np.concatenate([img, img2], axis=1)

            return img

        return img_func


class RubiksCubeRandom(RubiksCube):
    """
    This class is a extension of RubiksCube, it will generate the state with random moves.
    """

    @property
    def fixed_target(self) -> bool:
        return False

    def __init__(self, size: int = 3, initial_shuffle: int = 100, **kwargs):
        super().__init__(size=size, initial_shuffle=initial_shuffle, **kwargs)

    def get_solve_config(self, key=None, data=None) -> Puzzle.SolveConfig:
        solve_config = super().get_solve_config(key, data)
        solve_config.TargetState = self._get_suffled_state(
            solve_config, solve_config.TargetState, key, num_shuffle=100
        )
        return solve_config
