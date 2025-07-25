from functools import partial

import chex
import jax
import jax.numpy as jnp
from tabulate import tabulate

from puxle.utils.annotate import IMG_SIZE
from puxle.core.puzzle_base import Puzzle
from puxle.core.puzzle_state import FieldDescriptor, PuzzleState, state_dataclass
from puxle.utils.util import coloring_str, to_uint8, from_uint8

TYPE = jnp.uint8
LINE_THICKNESS = 3

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
FRONT = 4
BACK = 5
rotate_face_map = {0: "l", 1: "d", 2: "f", 3: "r", 4: "b", 5: "u"}
face_map_legend = {0: "up", 1: "down", 2: "left", 3: "right", 4: "front", 5: "back"}
face_map = {0: "up━", 1: "down━", 2: "left━", 3: "right", 4: "front", 5: "back━"}
rgb_map = {
    0: (255, 255, 255),  # white
    1: (255, 255, 0),  # yellow
    2: (255, 128, 0),  # orange
    3: (255, 0, 0),  # red
    4: (0, 255, 0),  # green
    5: (0, 0, 255),  # blue
}


def rot90_traceable(m, k=1, axes=(0, 1)):
    k %= 4
    return jax.lax.switch(k, [partial(jnp.rot90, m, k=i, axes=axes) for i in range(4)])


# (rolled_faces, rotate_axis_for_rolled_faces)
# 0: x-axis(left), 1: y-axis(up), 2: z-axis(front)


class RubiksCube(Puzzle):
    size: int
    index_grid: chex.Array

    def define_state_class(self) -> PuzzleState:
        str_parser = self.get_string_parser()
        raw_shape = (6, self.size * self.size)
        raw = jnp.full(raw_shape, -1, dtype=TYPE)
        packed_faces = to_uint8(raw, 3)

        @state_dataclass
        class State:
            faces: FieldDescriptor[TYPE, packed_faces.shape]

            def __str__(self, **kwargs):
                return str_parser(self, **kwargs)
            
            @property
            def packed(self):
                return State(faces=to_uint8(self.faces, 3))
            
            @property
            def unpacked(self):
                return State(faces=from_uint8(self.faces, raw_shape, 3))

        return State

    def __init__(self, size: int = 3, initial_shuffle: int = 10, **kwargs):
        self.size = size
        self.initial_shuffle = initial_shuffle
        is_even = size % 2 == 0
        self.index_grid = jnp.asarray(
            [i for i in range(size) if is_even or not i == (size // 2)], dtype=jnp.uint8
        )
        super().__init__(**kwargs)

    def get_string_parser(self):
        def parser(state: "RubiksCube.State", **kwargs):
            # Unpack the state faces before printing
            unpacked_faces = state.unpacked.faces

            # Helper function to get face string
            def get_empty_face_string():
                return "\n".join(["  " * (self.size + 2) for _ in range(self.size + 2)])

            def color_legend():
                return "\n".join(
                    [f"{face_map_legend[i]:<6}:{coloring_str('■', rgb_map[i])}" for i in range(6)]
                )

            def get_face_string(face):
                face_str = face_map[face]
                string = f"┏━{face_str.center(self.size * 2 - 1, '━')}━┓\n"
                for j in range(self.size):
                    string += (
                        "┃ "
                        + " ".join(
                            [
                                coloring_str(
                                    "■", rgb_map[int(unpacked_faces[face, j * self.size + i])]
                                )
                                for i in range(self.size)
                            ]
                        )
                        + " ┃\n"
                    )
                string += "┗━" + "━━" * (self.size - 1) + "━━┛\n"
                return string

            # Create the cube string representation
            cube_str = tabulate(
                [
                    [color_legend(), (".\n" + get_face_string(0))],
                    [
                        get_face_string(2),
                        get_face_string(4),
                        get_face_string(3),
                        get_face_string(5),
                    ],
                    [get_empty_face_string(), get_face_string(1)],
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
            solve_config,
            solve_config.TargetState,
            key,
            num_shuffle=self.initial_shuffle
        )

    def get_target_state(self, key=None) -> "RubiksCube.State":
        faces = jnp.repeat(jnp.arange(6)[:, None], self.size * self.size, axis=1).astype(TYPE)  # 6 faces, 3x3 each
        return self.State(faces=faces).packed

    def get_solve_config(self, key=None, data=None) -> Puzzle.SolveConfig:
        return self.SolveConfig(TargetState=self.get_target_state(key))

    def get_neighbours(
        self, solve_config: Puzzle.SolveConfig, state: "RubiksCube.State", filled: bool = True
    ) -> tuple["RubiksCube.State", chex.Array]:
        def map_fn(state, axis, index, clockwise):
            return jax.lax.cond(
                filled,
                lambda : (self._rotate(state, axis, index, clockwise), 1.0),
                lambda : (state, jnp.inf),
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
        """
        return f"{rotate_face_map[int(action // 2)]}_{'cw' if action % 2 == 0 else 'ccw'}"

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
                [UP, FRONT, DOWN, BACK],  # x-axis
                [LEFT, FRONT, RIGHT, BACK],  # y-axis
                [UP, LEFT, DOWN, RIGHT],  # z-axis
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
        )  # 0: None, 1: left, 2: right, 3: up, 4: down, 5: front, 6: back
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

            # Obtain the color data for each face and reshape them into grids
            board = state.unpacked.faces
            board = np.array(board)
            face_colors = {}
            face_colors[UP] = np.array(board[UP].reshape((self.size, self.size)))
            face_colors[FRONT] = np.array(board[FRONT].reshape((self.size, self.size)))
            face_colors[RIGHT] = np.array(board[RIGHT].reshape((self.size, self.size)))

            # If another_faces is True, get additional faces: DOWN, BACK, LEFT
            if another_faces:
                face_colors[DOWN] = np.array(board[DOWN].reshape((self.size, self.size)))
                face_colors[BACK] = np.array(board[BACK].reshape((self.size, self.size)))
                face_colors[LEFT] = np.array(board[LEFT].reshape((self.size, self.size)))

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
                    color_idx = int(face_colors[FRONT][i, j])
                    color = rgb_map[color_idx]
                    cv2.fillPoly(img, [pts], color)
                    cv2.polylines(
                        img, [pts], isClosed=True, color=(0, 0, 0), thickness=LINE_THICKNESS
                    )

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
                    color_idx = int(face_colors[RIGHT][i, j])
                    color = rgb_map[color_idx]
                    cv2.fillPoly(img, [pts], color)
                    cv2.polylines(
                        img, [pts], isClosed=True, color=(0, 0, 0), thickness=LINE_THICKNESS
                    )

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
                    color_idx = int(face_colors[UP][self.size - i - 1, j])
                    color = rgb_map[color_idx]
                    cv2.fillPoly(img, [pts], color)
                    cv2.polylines(
                        img, [pts], isClosed=True, color=(0, 0, 0), thickness=LINE_THICKNESS
                    )

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
                        color_idx = int(face_colors[BACK][i, j])
                        color = rgb_map[color_idx]
                        cv2.fillPoly(img2, [pts], color)
                        cv2.polylines(
                            img2, [pts], isClosed=True, color=(0, 0, 0), thickness=LINE_THICKNESS
                        )

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
                        color_idx = int(face_colors[DOWN][self.size - j - 1, i])
                        color = rgb_map[color_idx]
                        cv2.fillPoly(img2, [pts], color)
                        cv2.polylines(
                            img2, [pts], isClosed=True, color=(0, 0, 0), thickness=LINE_THICKNESS
                        )

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
                        color_idx = int(face_colors[LEFT][i, j])
                        color = rgb_map[color_idx]
                        cv2.fillPoly(img2, [pts], color)
                        cv2.polylines(
                            img2, [pts], isClosed=True, color=(0, 0, 0), thickness=LINE_THICKNESS
                        )

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
            solve_config, solve_config.TargetState, key,
            num_shuffle=100
        )
        return solve_config
