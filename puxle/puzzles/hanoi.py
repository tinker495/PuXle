import colorsys
from collections.abc import Callable

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from xtructure import FieldDescriptor, Xtructurable, xtructure_dataclass

from puxle.core.puzzle_base import Puzzle
from puxle.utils.util import IMG_SIZE, colored_str

TYPE = jnp.uint8


class TowerOfHanoi(Puzzle):
    """Tower of Hanoi puzzle with packed disk-to-peg positions.

    Move all disks from the first peg to the third peg, obeying three rules:

    1. Only one disk may be moved at a time.
    2. A move takes the topmost disk from one peg and places it on another.
    3. No disk may be placed on top of a smaller disk.

    Each disk stores its peg index, using two bits for the default three pegs.
    ``state.pegs`` is a derived ``(num_pegs, num_disks + 1)`` view: each row
    starts with its disk count, followed by disk sizes from smallest to largest.

    Actions encode ordered ``(from_peg, to_peg)`` pairs, giving
    ``num_pegs × (num_pegs − 1)`` possible moves (invalid moves yield
    infinite cost).

    Args:
        size: Number of disks (default ``5``).
    """

    num_disks: int
    num_pegs: int = 3  # Classic Tower of Hanoi has 3 pegs
    max_disk_value: int

    def define_state_class(self) -> type[Xtructurable]:
        """Defines the state class for Tower of Hanoi using xtructure."""
        str_parser = self.get_string_parser()
        count, peg_count = self.num_disks, self.num_pegs

        @xtructure_dataclass
        class State:
            disk_pegs: FieldDescriptor.packed_tensor(
                shape=(count,), packed_bits=(peg_count - 1).bit_length()
            )

            @property
            def pegs(self: "TowerOfHanoi.State") -> jax.Array | np.ndarray:
                """Materialize ordered peg rows, retaining host arrays on the host."""
                positions = self.disk_pegs_unpacked
                xp = np if isinstance(positions, np.ndarray) else jnp
                present = positions[..., None, :] == xp.arange(peg_count)[:, None]
                ordered = xp.sort(
                    xp.where(present, xp.arange(1, count + 1), count + 1), axis=-1
                )
                return xp.concatenate(
                    (
                        xp.sum(present, axis=-1)[..., None],
                        xp.where(ordered <= count, ordered, 0),
                    ),
                    axis=-1,
                ).astype(TYPE)

            def __str__(self, **kwargs):
                return str_parser(self, **kwargs)

        return State

    def __init__(self, size: int = 5, **kwargs):
        """
        Initialize the Tower of Hanoi puzzle

        Args:
            num_disks: The number of disks in the puzzle
        """
        self.num_disks = size
        self.max_disk_value = size
        self.action_size = self.num_pegs * (self.num_pegs - 1)
        # (from_peg, to_peg) pairs indexed by action; built once and shared by
        # get_actions (as a jnp array) and action_to_string.
        self._possible_moves = jnp.array(
            [
                [from_peg, to_peg]
                for from_peg in range(self.num_pegs)
                for to_peg in range(self.num_pegs)
                if from_peg != to_peg
            ]
        )
        super().__init__(**kwargs)

    def get_string_parser(self):
        """Returns a function to convert a state to a string representation"""

        def parser(state: "TowerOfHanoi.State", **kwargs):
            result = []

            # Get the pegs array - has shape (num_pegs, num_disks + 1)
            # Host copy first; reading a device array per disk syncs each time.
            pegs = jax.device_get(state).pegs

            # Find the maximum height
            max_height = self.num_disks

            # For each level from top to bottom
            for level in range(max_height):
                row = []

                # For each peg
                for peg_idx in range(self.num_pegs):
                    peg = pegs[peg_idx]
                    num_disks_on_peg = int(peg[0])

                    # Calculate position from the top
                    pos_from_top = level

                    # If there's a disk at this position
                    if pos_from_top < num_disks_on_peg:
                        # Get the disk at this position (index 1 + pos_from_top has the disk size)
                        disk_size = int(peg[1 + pos_from_top])
                        disk_str = "=" * (2 * disk_size - 1)
                        colored_disk = colored_str(
                            disk_str.center(2 * self.num_disks + 1),
                            get_color(disk_size),
                        )
                        row.append(colored_disk)
                    else:
                        # No disk, just show the peg
                        row.append("|".center(2 * self.num_disks + 1))

                result.append("   ".join(row))

            # Add base
            base_row = []
            for _ in range(self.num_pegs):
                base = "-" * (2 * self.num_disks + 1)
                base_row.append(base)

            result.append("   ".join(base_row))

            # Add peg numbers
            label_row = []
            for i in range(self.num_pegs):
                label = f"Peg {i + 1}".center(2 * self.num_disks + 1)
                label_row.append(label)

            result.append("   ".join(label_row))

            return "\n".join(result)

        return parser

    def get_img_parser(self) -> Callable:
        """Returns a function to convert a state to an image representation"""

        def img_func(state: "TowerOfHanoi.State", **kwargs):
            # Get dimensions
            width, height = IMG_SIZE

            # Create blank image with correct dimensions
            image = np.full((height, width, 3), (240, 240, 240), dtype=np.uint8)

            # Parameters for visualization
            peg_width = 10
            peg_height = height * 0.6
            base_height = 20
            base_width = width * 0.8

            # Bottom of pegs (y-coordinate)
            base_y = height - 80

            # Draw base
            base_x = (width - base_width) / 2
            image = cv2.rectangle(
                image,
                (int(base_x), int(base_y)),
                (
                    int(base_x + base_width),
                    int(base_y + base_height),
                ),
                (120, 80, 40),  # Brown color
                -1,
            )

            # Calculate peg positions
            peg_xs = [
                base_x + base_width * (i + 1) / (self.num_pegs + 1)
                for i in range(self.num_pegs)
            ]

            # Draw pegs
            for peg_x in peg_xs:
                image = cv2.rectangle(
                    image,
                    (
                        int(peg_x - peg_width / 2),
                        int(base_y - peg_height),
                    ),
                    (int(peg_x + peg_width / 2), int(base_y)),
                    (120, 80, 40),  # Brown color
                    -1,
                )

            # Draw disks on pegs
            max_disk_width = base_width / (self.num_pegs + 1) * 0.9
            disk_height = 20

            # Get the pegs array
            pegs = jax.device_get(state).pegs

            # For each peg
            for peg_idx, peg_x in enumerate(peg_xs):
                peg = pegs[peg_idx]
                num_disks_on_peg = int(peg[0])

                # For each disk on this peg (from bottom to top)
                for disk_idx in range(num_disks_on_peg):
                    disk_size = int(peg[1 + disk_idx])
                    disk_width = max_disk_width * disk_size / self.max_disk_value

                    # Position from bottom
                    pos_from_bottom = num_disks_on_peg - disk_idx - 1
                    disk_y = base_y - (pos_from_bottom + 1) * disk_height

                    # Generate color based on disk size
                    color = get_disk_color(disk_size, self.max_disk_value)

                    # Draw disk
                    image = cv2.rectangle(
                        image,
                        (
                            int(peg_x - disk_width / 2),
                            int(disk_y),
                        ),
                        (
                            int(peg_x + disk_width / 2),
                            int(disk_y + disk_height),
                        ),
                        color,
                        -1,
                    )

                    # Add disk size text
                    text = str(disk_size)
                    (text_w, _text_h), _ = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    text_x = int(peg_x - text_w / 2)
                    text_y = int(disk_y + disk_height - 5)
                    image = cv2.putText(
                        image,
                        text,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

            return image

        return img_func

    def get_initial_state(
        self, solve_config: "TowerOfHanoi.SolveConfig", key=None, data=None
    ) -> "TowerOfHanoi.State":
        """Generate the initial state with all disks on the first peg."""
        return self.State.from_unpacked(disk_pegs=jnp.zeros(self.num_disks, dtype=TYPE))

    def get_solve_config(self, key=None, data=None) -> "TowerOfHanoi.SolveConfig":
        """Create the solving configuration (target) with all disks on the third peg."""
        return self.SolveConfig(
            InstanceContext=self.InstanceContext(),
            GoalSpec=self.State.from_unpacked(
                disk_pegs=jnp.full(self.num_disks, 2, dtype=TYPE)
            ),
        )

    def _apply(
        self,
        solve_config: "TowerOfHanoi.SolveConfig",
        state: "TowerOfHanoi.State",
        action: jax.Array,
    ) -> tuple["TowerOfHanoi.State", jax.Array]:
        source, target = self._possible_moves[action]
        positions = state.disk_pegs_unpacked
        disks = jnp.arange(self.num_disks)
        moving = jnp.min(
            jnp.where(positions == source, disks, self.num_disks),
            initial=self.num_disks,
        )
        destination = jnp.min(
            jnp.where(positions == target, disks, self.num_disks),
            initial=self.num_disks,
        )
        positions = jnp.where(
            disks == moving, target.astype(positions.dtype), positions
        )
        return state.set_unpacked(disk_pegs=positions), jnp.where(
            moving < destination, 1.0, jnp.inf
        )

    def _get_neighbours(
        self, solve_config: "TowerOfHanoi.SolveConfig", state: "TowerOfHanoi.State"
    ) -> tuple["TowerOfHanoi.State", jax.Array]:
        positions = state.disk_pegs_unpacked
        disks = jnp.arange(self.num_disks)
        top = jnp.min(
            jnp.where(
                positions[None, :] == jnp.arange(self.num_pegs)[:, None],
                disks,
                self.num_disks,
            ),
            axis=1,
            initial=self.num_disks,
        )
        moving = top[self._possible_moves[:, 0]]
        targets = self._possible_moves[:, 1]
        moved = jnp.where(
            disks[None, :] == moving[:, None],
            targets[:, None].astype(positions.dtype),
            positions[None, :],
        )
        states = jax.vmap(lambda values: state.set_unpacked(disk_pegs=values))(moved)
        return states, jnp.where(moving < top[targets], 1.0, jnp.inf)

    def action_to_string(self, action: int) -> str:
        """Return a string representation of the action"""
        from_peg, to_peg = self._possible_moves[action]
        return f"Move disk from peg {int(from_peg) + 1} to peg {int(to_peg) + 1}"


def get_color(size):
    """Get color based on disk size"""
    colors = ["red", "green", "yellow", "blue", "magenta", "cyan"]
    return colors[(size - 1) % len(colors)]


def get_disk_color(size, max_size):
    """Get disk color as RGB based on size"""
    # Rainbow gradient from blue (hue 240) for small disks to red (hue 0).
    hue = 240 * (1 - size / max_size)
    r, g, b = colorsys.hsv_to_rgb(hue / 360, 0.8, 0.9)
    return int(r * 255), int(g * 255), int(b * 255)
