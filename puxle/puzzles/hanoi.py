import colorsys
from collections.abc import Callable

import chex
import cv2
import jax
import jax.numpy as jnp
import numpy as np

from puxle.core.puzzle_base import Puzzle
from puxle.core.puzzle_state import FieldDescriptor, PuzzleState, state_dataclass
from puxle.utils.util import IMG_SIZE, colored_str

TYPE = jnp.uint8


class TowerOfHanoi(Puzzle):
    """Tower of Hanoi puzzle with variable pegs.

    Move all disks from the first peg to the last peg, obeying three rules:

    1. Only one disk may be moved at a time.
    2. A move takes the topmost disk from one peg and places it on another.
    3. No disk may be placed on top of a smaller disk.

    Each peg is stored as a fixed-length array of shape
    ``(num_disks + 1,)`` whose first element is the current disk count
    and subsequent elements hold disk sizes (smallest at index 1 = top).

    Actions encode ordered ``(from_peg, to_peg)`` pairs, giving
    ``num_pegs × (num_pegs − 1)`` possible moves (invalid moves yield
    infinite cost).

    Args:
        size: Number of disks (default ``5``).
        num_pegs: Number of pegs (default ``3``).
    """

    num_disks: int
    num_pegs: int = 3  # Classic Tower of Hanoi has 3 pegs
    max_disk_value: int

    def define_state_class(self) -> PuzzleState:
        """Defines the state class for Tower of Hanoi using xtructure."""
        str_parser = self.get_string_parser()
        # Default pegs value for FieldDescriptor, initialized when class is defined
        # self.num_pegs and self.num_disks are available from TowerOfHanoi.__init__
        default_pegs_val = jnp.zeros((self.num_pegs, self.num_disks + 1), dtype=TYPE)

        @state_dataclass
        class State:
            pegs: FieldDescriptor.tensor(dtype=TYPE, shape=default_pegs_val.shape)

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
            pegs = state.pegs

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
            pegs = state.pegs

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

    def _stacked_pegs(self, peg_idx: int) -> chex.Array:
        """Build a pegs array with every disk stacked on ``peg_idx``.

        Disks are placed smallest-on-top: index 1 holds the smallest disk and
        index ``num_disks`` the largest (e.g. with 3 disks the peg column is
        ``[3, 1, 2, 3]``).
        """
        pegs = jnp.zeros((self.num_pegs, self.num_disks + 1), dtype=TYPE)
        pegs = pegs.at[peg_idx, 0].set(self.num_disks)
        for i in range(self.num_disks):
            pegs = pegs.at[peg_idx, i + 1].set(i + 1)
        return pegs

    def get_initial_state(
        self, solve_config: "TowerOfHanoi.SolveConfig", key=None, data=None
    ) -> "TowerOfHanoi.State":
        """Generate the initial state with all disks on the first peg."""
        return self.State(pegs=self._stacked_pegs(0))

    def get_solve_config(self, key=None, data=None) -> "TowerOfHanoi.SolveConfig":
        """Create the solving configuration (target) with all disks on the third peg."""
        return self.SolveConfig(TargetState=self.State(pegs=self._stacked_pegs(2)))

    def _apply(
        self,
        solve_config: "TowerOfHanoi.SolveConfig",
        state: "TowerOfHanoi.State",
        action: chex.Array,
    ) -> tuple["TowerOfHanoi.State", chex.Array]:
        """Pure transition: an illegal disk move costs infinity."""
        pegs = state.pegs

        move = self._possible_moves[action]
        from_peg, to_peg = move[0], move[1]

        def is_valid_move(pegs, from_peg, to_peg):
            # Check if the from_peg has disks
            disks_on_from = pegs[from_peg, 0]
            valid_from = disks_on_from > 0

            # Get the top disk size from from_peg (if there are disks)
            # Top disk is at index 1 (smallest disk)
            from_top_disk = jax.lax.cond(
                disks_on_from > 0,
                lambda: pegs[from_peg, 1],
                lambda: jnp.array(0, dtype=TYPE),
            )

            # Check if the to_peg has space and the top disk on to_peg is larger
            disks_on_to = pegs[to_peg, 0]

            # If to_peg is empty, it's valid. Otherwise, compare disk sizes:
            # Only allow placing a smaller disk on top of a larger disk
            valid_to = jax.lax.cond(
                disks_on_to == 0,
                lambda: jnp.array(True, dtype=bool),
                lambda: from_top_disk < pegs[to_peg, 1],
            )

            return jnp.logical_and(valid_from, valid_to)

        def make_move(pegs, from_peg, to_peg):
            # Get the number of disks on the from_peg
            disks_on_from = pegs[from_peg, 0]

            # Get the top disk size from from_peg (smallest disk at top = index 1)
            from_top_disk = pegs[from_peg, 1]

            # Create a copy of the pegs array
            new_pegs = pegs  # JAX arrays are immutable, ops return new array

            # Remove the top disk from from_peg
            # Shift all disks up (disk at position n moves to position n-1)
            new_pegs = new_pegs.at[from_peg, 1:-1].set(new_pegs[from_peg, 2:])
            new_pegs = new_pegs.at[from_peg, -1].set(0)  # Clear the last position

            # Decrement the disk count on from_peg
            new_pegs = new_pegs.at[from_peg, 0].set(disks_on_from - 1)

            # Get the number of disks on the to_peg
            disks_on_to = new_pegs[to_peg, 0]

            # Add the disk to to_peg (at the top position = index 1)
            # Shift all disks down to make room at index 1
            new_pegs = new_pegs.at[to_peg, 2:].set(new_pegs[to_peg, 1:-1])
            new_pegs = new_pegs.at[to_peg, 1].set(from_top_disk)

            # Increment the disk count on to_peg
            new_pegs = new_pegs.at[to_peg, 0].set(disks_on_to + 1)

            return new_pegs

        valid = is_valid_move(pegs, from_peg, to_peg)

        # If valid, make the move; otherwise, keep the original pegs
        new_pegs = jax.lax.cond(
            valid, lambda: make_move(pegs, from_peg, to_peg), lambda: pegs
        )

        # Cost is 1 if valid, infinity if invalid
        cost = jax.lax.cond(valid, lambda: jnp.array(1.0), lambda: jnp.array(jnp.inf))

        return self.State(pegs=new_pegs), cost

    def is_solved(
        self, solve_config: "TowerOfHanoi.SolveConfig", state: "TowerOfHanoi.State"
    ) -> bool:
        """Check if the current state matches the target state"""
        return state == solve_config.TargetState

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
