from collections.abc import Callable

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from xtructure import FieldDescriptor, Xtructurable, xtructure_dataclass

from puxle.core.puzzle_base import Puzzle
from puxle.utils.util import IMG_SIZE

# Use 16-bit unsigned integers so that problem sizes >255 are handled without overflow.
TYPE = jnp.uint16


class TSP(Puzzle):
    """Travelling Salesman Problem (TSP) as a sequential-visit puzzle.

    ``size`` cities are uniformly sampled in the unit square.  The agent
    starts at a random city and must visit every remaining city exactly
    once, minimising total Euclidean distance (including the return to
    the start city).

    State consists of a packed visited-mask (1 bit / city) and the
    index of the current city.  The action space equals the number of
    cities; visiting an already-visited city yields infinite cost.

    This puzzle is **not reversible**.

    Args:
        size: Number of cities (default ``16``).
    """

    size: int

    def define_state_class(self) -> type[Xtructurable]:
        """Defines the state class for TSP using xtructure."""
        str_parser = self.get_string_parser()
        size = self.size

        @xtructure_dataclass
        class State:
            mask: FieldDescriptor.packed_tensor(shape=(size,), packed_bits=1)
            point: FieldDescriptor.scalar(dtype=self._point_dtype)

            def __str__(self, **kwargs):
                return str_parser(self, **kwargs)

        return State

    def define_instance_context_class(self) -> type[Xtructurable]:
        @xtructure_dataclass
        class InstanceContext:
            points: FieldDescriptor.tensor(dtype=jnp.float32, shape=(self.size, 2))
            distance_matrix: FieldDescriptor.tensor(
                dtype=jnp.float32, shape=(self.size, self.size)
            )
            start: FieldDescriptor.scalar(dtype=self._point_dtype)

        return InstanceContext

    def define_goal_spec_class(self) -> type[Xtructurable]:
        @xtructure_dataclass
        class GoalSpec:
            pass

        return GoalSpec

    def __init__(self, size: int = 16, **kwargs):
        self.size = size
        self._point_dtype = jnp.uint8 if size <= 256 else jnp.uint16
        self.action_size = self.size
        super().__init__(**kwargs)

    def get_solve_config_string_parser(self) -> Callable:
        def parser(solve_config: "TSP.SolveConfig", **kwargs):
            return (
                "TSP SolveConfig: "
                f"{self.size} points, start at {solve_config.InstanceContext.start}"
            )

        return parser

    def get_string_parser(self):
        form = self._get_visualize_format()

        def to_char(x, true_char="■", false_char="☐"):
            return true_char if x else false_char

        def parser(state: "TSP.State", **kwargs):
            # Host copy first; per-cell reads off a device array sync per character.
            state = jax.device_get(state)
            mask = state.mask_unpacked
            point_mask = np.zeros_like(mask)
            point_mask[state.point] = True
            maps = [to_char(x, true_char="↓", false_char=" ") for x in point_mask]
            maps += [to_char(x, true_char="■", false_char="☐") for x in mask]
            return form.format(*maps)

        return parser

    def get_initial_state(
        self, solve_config: Puzzle.SolveConfig, key=jax.random.PRNGKey(0), data=None
    ) -> Puzzle.State:
        mask = jnp.zeros(self.size, dtype=jnp.bool_)
        point = solve_config.InstanceContext.start
        mask = mask.at[point].set(True)
        return self.State.from_unpacked(mask=mask, point=point)

    def get_solve_config(self, key=None, data=None) -> Puzzle.SolveConfig:
        # Split PRNG key so that the start index is independent of point positions.
        key_points, key_start = jax.random.split(key)

        points = jax.random.uniform(
            key_points, shape=(self.size, 2), minval=0, maxval=1, dtype=jnp.float32
        )
        distance_matrix = jnp.linalg.norm(
            points[:, None] - points[None, :], axis=-1
        ).astype(jnp.float32)
        start = jax.random.randint(
            key_start, shape=(), minval=0, maxval=self.size, dtype=TYPE
        ).astype(self._point_dtype)
        return self.SolveConfig(
            InstanceContext=self.InstanceContext(
                points=points, distance_matrix=distance_matrix, start=start
            ),
            GoalSpec=self.GoalSpec(),
        )

    def _apply(
        self,
        solve_config: Puzzle.SolveConfig,
        state: Puzzle.State,
        action: jax.Array,
    ) -> tuple[Puzzle.State, jax.Array]:
        """Pure transition: revisiting an already-visited point costs infinity.

        The closing leg back to the start is added once every point is visited.
        """
        context = solve_config.InstanceContext
        bit = jnp.uint8(1) << jnp.uint8(action % 8)
        visited = (state.mask[action // 8] & bit) != 0
        new_mask = state.mask | jnp.where(
            jnp.arange(state.mask.size) == action // 8, bit, jnp.uint8(0)
        )
        all_visited = jnp.all(new_mask[:-1] == 255) & (
            new_mask[-1] == (1 << (self.size % 8 or 8)) - 1
        )
        cost = jnp.where(
            visited, jnp.inf, context.distance_matrix[state.point, action]
        ) + jnp.where(
            all_visited,
            jnp.linalg.norm(context.points[context.start] - context.points[action]),
            0,
        )
        return self.State(mask=new_mask, point=action.astype(state.point.dtype)), cost

    def _get_neighbours(
        self, solve_config: Puzzle.SolveConfig, state: "TSP.State"
    ) -> tuple["TSP.State", jax.Array]:
        actions = jnp.arange(self.size)
        bits = jnp.uint8(1) << (actions % 8).astype(jnp.uint8)
        visited = (state.mask[actions // 8] & bits) != 0
        masks = state.mask[None, :] | jnp.where(
            actions[:, None] // 8 == jnp.arange(state.mask.size)[None, :],
            bits[:, None],
            jnp.uint8(0),
        )
        context = solve_config.InstanceContext
        costs = context.distance_matrix[state.point] + jnp.where(
            jnp.sum(visited) == self.size - 1,
            jnp.linalg.norm(context.points[context.start] - context.points, axis=1),
            0.0,
        )
        return self.State(
            mask=masks, point=actions.astype(state.point.dtype)
        ), jnp.where(visited, jnp.inf, costs)

    def is_solved(
        self, solve_config: Puzzle.SolveConfig, state: Puzzle.State
    ) -> jax.Array:
        """
        TSP is solved when all points have been visited.
        """
        return jnp.all(state.mask[:-1] == 255) & (
            state.mask[-1] == (1 << (self.size % 8 or 8)) - 1
        )

    def action_to_string(self, action: int) -> str:
        """
        This function should return a string representation of the action.
        """
        return f"{action:02d}"

    def _get_visualize_format(self):
        size = self.size
        form = " " + "{:s} " * size + "\n"
        form += "[" + "{:s} " * size + "]"
        return form

    def get_solve_config_img_parser(self) -> Callable:
        def parser(solve_config: "TSP.SolveConfig", **kwargs):
            raise NotImplementedError("TSP does not support image visualization")

        return parser

    def get_img_parser(self):
        """
        This function returns an img_parser that visualizes the TSP problem.
        It draws all the points scaled to fit into the image, highlights the start point in green,
        marks visited points in blue and unvisited in red, and outlines the current point with a black border.
        If all points are visited, it draws a line from the current point back to the start point.
        """

        def img_func(
            state: "TSP.State",
            solve_config: "TSP.SolveConfig",
            **kwargs,
        ):
            imgsize = IMG_SIZE[0]
            # Create a white background image
            img = np.full(
                (IMG_SIZE[1], IMG_SIZE[0], 3), (255, 255, 255), dtype=np.uint8
            )
            path = kwargs.get("path", [])
            idx = kwargs.get("idx", 0)

            # Get the visited mask as booleans
            visited = state.mask_unpacked
            # Convert the TSP points (assumed to be an array of shape [number_of_points, 2]) to a numpy array
            context = solve_config.InstanceContext
            points_np = np.array(context.points)

            # Compute scaling parameters to fit all points within the image with a margin
            margin = 20
            if points_np.size > 0:
                xmin, xmax = points_np[:, 0].min(), points_np[:, 0].max()
                ymin, ymax = points_np[:, 1].min(), points_np[:, 1].max()
            else:
                xmin, xmax, ymin, ymax = 0, 1, 0, 1

            # Scale points to image coordinates
            scaled_points = []
            for pt in points_np:
                if xmax > xmin:
                    x_coord = margin + int(
                        (pt[0] - xmin) / (xmax - xmin) * (imgsize - 2 * margin)
                    )
                else:
                    x_coord = imgsize // 2
                if ymax > ymin:
                    y_coord = margin + int(
                        (pt[1] - ymin) / (ymax - ymin) * (imgsize - 2 * margin)
                    )
                else:
                    y_coord = imgsize // 2
                scaled_points.append((x_coord, y_coord))

            # Visualize the given path by drawing lines connecting the successive points from 'paths'
            # up to the current index 'idx'
            if path and idx > 0 and len(path) > idx:
                route_points = [scaled_points[path[i].point] for i in range(idx + 1)]
                img = cv2.polylines(
                    img, [np.array(route_points, dtype=np.int32)], False, (0, 0, 0), 2
                )

            # Draw each point with different colors based on status
            for i, (x, y) in enumerate(scaled_points):  # Renamed idx to i for clarity
                # Color: green for start, blue for visited, red for unvisited
                if i == context.start:
                    color = (0, 255, 0)
                elif visited[i]:
                    color = (255, 0, 0)
                else:
                    color = (0, 0, 255)

                img = cv2.circle(img, (x, y), 5, color, -1)

                # Highlight the current point with an outer black circle
                if i == state.point:
                    img = cv2.circle(img, (x, y), 8, (0, 0, 0), 2)

                # Optionally, label the point with its index
                img = cv2.putText(
                    img,
                    str(i),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (50, 50, 50),
                    1,
                )

            # If all points are visited, draw a line from the current point to the start point to close the tour
            if np.all(visited):
                img = cv2.line(
                    img,
                    scaled_points[state.point],
                    scaled_points[context.start],
                    (0, 0, 0),
                    2,
                )

            return img

        return img_func
