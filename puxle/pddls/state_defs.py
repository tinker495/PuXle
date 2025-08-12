from typing import Callable

import jax.numpy as jnp

from puxle.core.puzzle_state import FieldDescriptor, PuzzleState, state_dataclass
from puxle.utils.util import from_uint8, to_uint8


def build_state_class(env, num_atoms: int, init_state: jnp.ndarray, string_parser: Callable) -> PuzzleState:
    packed_size = (num_atoms + 7) // 8
    packed_atoms = to_uint8(init_state, 1)

    @state_dataclass
    class State:
        atoms: FieldDescriptor[jnp.uint8, (packed_size,), packed_atoms]

        def __str__(self, **kwargs):
            return string_parser(self, **kwargs)

        @property
        def packed(self):
            return State(atoms=to_uint8(self.unpacked_atoms, 1))

        @property
        def unpacked(self):
            return self

        @property
        def unpacked_atoms(self):
            return from_uint8(self.atoms, (num_atoms,), 1)

    return State


def build_solve_config_class(env, goal_mask: jnp.ndarray, string_parser: Callable) -> PuzzleState:
    @state_dataclass
    class SolveConfig:
        GoalMask: FieldDescriptor[jnp.bool_, (env.num_atoms,), goal_mask]

        def __str__(self, **kwargs):
            return string_parser(self, **kwargs)

    return SolveConfig
