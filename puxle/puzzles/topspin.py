from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp

from puxle.core.puzzle_base import Puzzle
from puxle.core.puzzle_state import FieldDescriptor, PuzzleState, state_dataclass
from puxle.utils.annotate import IMG_SIZE  # Assuming IMG_SIZE is defined

TYPE = jnp.uint8


class TopSpin(Puzzle):
    """Top Spin puzzle on a circular track.

    ``n_discs`` numbered tokens sit on a ring.  Three actions are
    available:

    * **Shift left** (action 0): rotate the entire ring one position
      counter-clockwise.
    * **Shift right** (action 1): rotate the ring one position clockwise.
    * **Reverse turnstile** (action 2): reverse the first
      ``turnstile_size`` tokens in the array.

    The goal is the sorted permutation ``[1, 2, …, n_discs]``.

    Inverse action map: left ↔ right; reverse is self-inverse.

    Args:
        size: Number of tokens on the ring (default ``20``).
        turnstile_size: Number of tokens covered by the turnstile
            (default ``4``).
    """

    n_discs: int
    turnstile_size: int

    def define_state_class(self) -> PuzzleState:
        """Defines the state class for TopSpin using xtructure."""
        str_parser = self.get_string_parser()

        @state_dataclass
        class State:
            permutation: FieldDescriptor.tensor(dtype=TYPE, shape=(self.n_discs,))

            def __str__(self, **kwargs):
                return str_parser(self, **kwargs)

        return State

    def __init__(self, size: int = 20, turnstile_size: int = 4, **kwargs):
        if turnstile_size > size:
            raise ValueError(
                "Turnstile size cannot be larger than the number of discs."
            )
        self.n_discs = size
        self.turnstile_size = turnstile_size
        self.action_size = 3
        super().__init__(**kwargs)

    def get_string_parser(self) -> Callable:
        def parser(state: "TopSpin.State", **kwargs):
            # Highlight the turnstile
            turnstile_str = " ".join(
                map(lambda x: f"{x:2d}", state.permutation[: self.turnstile_size])
            )
            rest_str = " ".join(
                map(lambda x: f"{x:2d}", state.permutation[self.turnstile_size :])
            )
            return f"[{turnstile_str}] {rest_str}"

        return parser

    def get_solve_config(self, key=None, data=None) -> Puzzle.SolveConfig:
        # The target state is the sorted permutation
        target_state = self.State(
            permutation=jnp.arange(1, self.n_discs + 1, dtype=TYPE)
        )
        return self.SolveConfig(TargetState=target_state)

    def get_initial_state(
        self, solve_config: Puzzle.SolveConfig, key=None, data=None
    ) -> "TopSpin.State":
        # Start from solved state and apply random moves
        return self._get_shuffled_state(solve_config, solve_config.TargetState, key, 18)

    def get_actions(
        self,
        solve_config: Puzzle.SolveConfig,
        state: "TopSpin.State",
        action: chex.Array,
        filled: bool = True,
    ) -> tuple["TopSpin.State", chex.Array]:
        """
        Returns the next state and cost for a given action.
        """
        p = state.permutation

        def get_next_state(action):
            return jax.lax.switch(
                action,
                [
                    lambda: self.State(permutation=jnp.roll(p, -1)),
                    lambda: self.State(permutation=jnp.roll(p, 1)),
                    lambda: self.State(
                        permutation=p.at[: self.turnstile_size].set(
                            jnp.flip(p[: self.turnstile_size])
                        )
                    ),
                ],
            )

        next_state = get_next_state(action)
        cost = jnp.where(filled, 1.0, jnp.inf)

        return next_state, cost

    def is_solved(
        self, solve_config: Puzzle.SolveConfig, state: "TopSpin.State"
    ) -> bool:
        return state == solve_config.TargetState

    def action_to_string(self, action: int) -> str:
        match action:
            case 0:
                return "Shift Left (<<)"
            case 1:
                return "Shift Right (>>)"
            case 2:
                return f"Reverse Turnstile (R{self.turnstile_size})"
            case _:
                raise ValueError(f"Invalid action: {action}")

    @property
    def inverse_action_map(self) -> jnp.ndarray | None:
        """
        Defines the inverse action mapping for TopSpin.
        - Shift Left (0) <-> Shift Right (1)
        - Reverse Turnstile (2) is its own inverse.
        """
        return jnp.array([1, 0, 2])

    def get_img_parser(self):
        import numpy as np

        from puxle.render import Cv2Backend

        backend = Cv2Backend()

        def img_func(state: "TopSpin.State", **kwargs):
            imgsize = IMG_SIZE[0]
            img = backend.canvas(size=IMG_SIZE, fill_bgr=(240, 240, 240))

            n = self.n_discs
            ts = self.turnstile_size
            center_x, center_y = imgsize // 2, imgsize // 2
            radius = int(imgsize * 0.4)
            font_scale = 1.0
            font_thickness = 2
            disc_radius = int(imgsize * 0.04)

            # Find the position of the first turnstile element to align it at the top
            # This ensures the turnstile is always at the top (12 o'clock position)
            offset = -(
                self.turnstile_size // 2
            )  # No offset needed as we'll place the first ts elements at the top

            # Draw the ring and discs
            for i, val in enumerate(state.permutation):
                # Calculate angle to place turnstile at the top (12 o'clock position)
                # First ts elements will be in the turnstile area
                angle = (2 * np.pi * ((i + offset + 0.5) / n)) - (
                    np.pi / 2
                )  # Start from top (12 o'clock)
                x = int(center_x + radius * np.cos(angle))
                y = int(center_y + radius * np.sin(angle))

                # Determine if this position is part of the turnstile
                is_turnstile = i < ts
                color = (
                    (0, 0, 200) if is_turnstile else (50, 50, 50)
                )  # Blue for turnstile, gray otherwise
                img = backend.circle(
                    img,
                    center=(x, y),
                    radius=disc_radius,
                    color_bgr=color,
                )

                text = str(val)
                text_w, text_h = backend.text_size(
                    text, font_scale=font_scale, thickness=font_thickness
                )
                text_x = x - text_w // 2
                text_y = y + text_h // 2
                img = backend.text(
                    img,
                    text=text,
                    position=(text_x, text_y),
                    color_bgr=(255, 255, 255),
                    font_scale=font_scale,
                    thickness=font_thickness,
                )

            # Draw the turnstile area indicator at the top
            start_angle_rad = -np.pi / 2 - (
                np.pi * ts / n
            )  # Start angle for turnstile area
            end_angle_rad = -np.pi / 2 + (
                np.pi * ts / n
            )  # End angle for turnstile area
            img = backend.ellipse(
                img,
                center=(center_x, center_y),
                axes=(radius + disc_radius + 5, radius + disc_radius + 5),
                angle=0,
                start_angle=float(np.degrees(start_angle_rad)),
                end_angle=float(np.degrees(end_angle_rad)),
                color_bgr=(200, 0, 0),
                thickness=2,
            )

            return img

        return img_func
