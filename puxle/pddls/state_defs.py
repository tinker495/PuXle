from typing import Callable

import jax.numpy as jnp

from puxle.core.puzzle_state import FieldDescriptor, PuzzleState, state_dataclass


def build_state_class(env, num_atoms: int, init_state: jnp.ndarray, string_parser: Callable) -> PuzzleState:
    @state_dataclass
    class State:
        atoms: FieldDescriptor.packed_tensor(shape=(num_atoms,), packed_bits=1)

        def __str__(self, **kwargs):
            return string_parser(self, **kwargs)

        @property
        def unpacked_atoms(self):
            return self.atoms_unpacked

    return State


def build_solve_config_class(env, goal_mask: jnp.ndarray, string_parser: Callable) -> PuzzleState:
    @state_dataclass
    class SolveConfig:
        GoalMask: FieldDescriptor.tensor(dtype=jnp.bool_, shape=(env.num_atoms,))

        def __str__(self, **kwargs):
            return string_parser(self, **kwargs)

    return SolveConfig
