"""Dynamic state, context, and goal class builders for PDDL environments.

Constructs xtructure-backed ``State`` (packed boolean atom vector) and
``InstanceContext`` / ``GoalSpec`` dataclasses tailored to a grounded problem.
"""

from typing import Callable

import jax.numpy as jnp
from xtructure import FieldDescriptor, Xtructurable, xtructure_dataclass


def build_state_class(
    env, num_atoms: int, init_state: jnp.ndarray, string_parser: Callable
) -> type[Xtructurable]:
    @xtructure_dataclass
    class State:
        atoms: FieldDescriptor.packed_tensor(shape=(num_atoms,), packed_bits=1)

        def __str__(self, **kwargs):
            return string_parser(self, **kwargs)

        @property
        def unpacked_atoms(self):
            return self.atoms_unpacked

    return State


def build_instance_context_class(env) -> type[Xtructurable]:
    @xtructure_dataclass
    class InstanceContext:
        pre_mask: FieldDescriptor.tensor(
            dtype=jnp.bool_, shape=(env.num_actions, env.num_atoms)
        )
        pre_neg_mask: FieldDescriptor.tensor(
            dtype=jnp.bool_, shape=(env.num_actions, env.num_atoms)
        )
        add_mask: FieldDescriptor.tensor(
            dtype=jnp.bool_, shape=(env.num_actions, env.num_atoms)
        )
        del_mask: FieldDescriptor.tensor(
            dtype=jnp.bool_, shape=(env.num_actions, env.num_atoms)
        )

    return InstanceContext


def build_goal_spec_class(env) -> type[Xtructurable]:
    @xtructure_dataclass
    class GoalSpec:
        GoalMask: FieldDescriptor.tensor(dtype=jnp.bool_, shape=(env.num_atoms,))

    return GoalSpec
