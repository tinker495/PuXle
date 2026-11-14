"""Dynamic state, context, and goal class builders for PDDL environments.

Constructs xtructure-backed ``State`` (packed boolean atom vector) and
``InstanceContext`` / ``GoalSpec`` dataclasses tailored to a grounded problem.
"""

from typing import TYPE_CHECKING, Annotated, Callable

import chex
import jax.numpy as jnp
from xtructure import FieldDescriptor, Xtructurable


def build_state_class(
    env, num_atoms: int, init_state: jnp.ndarray, string_parser: Callable
) -> type[Xtructurable]:
    class State(Xtructurable):
        atoms: Annotated[
            chex.Array, FieldDescriptor.packed_tensor(shape=(num_atoms,), packed_bits=1)
        ]

        if TYPE_CHECKING:

            @property
            def atoms_unpacked(self) -> chex.Array: ...

        def __str__(self, **kwargs):
            return string_parser(self, **kwargs)

        @property
        def unpacked_atoms(self):
            return self.atoms_unpacked

    return State


def build_instance_context_class(env) -> type[Xtructurable]:
    class InstanceContext(Xtructurable):
        pre_mask: Annotated[
            chex.Array,
            FieldDescriptor.tensor(
                dtype=jnp.bool_, shape=(env.num_actions, env.num_atoms)
            ),
        ]
        pre_neg_mask: Annotated[
            chex.Array,
            FieldDescriptor.tensor(
                dtype=jnp.bool_, shape=(env.num_actions, env.num_atoms)
            ),
        ]
        add_mask: Annotated[
            chex.Array,
            FieldDescriptor.tensor(
                dtype=jnp.bool_, shape=(env.num_actions, env.num_atoms)
            ),
        ]
        del_mask: Annotated[
            chex.Array,
            FieldDescriptor.tensor(
                dtype=jnp.bool_, shape=(env.num_actions, env.num_atoms)
            ),
        ]

    return InstanceContext


def build_goal_spec_class(env) -> type[Xtructurable]:
    class GoalSpec(Xtructurable):
        GoalMask: Annotated[
            chex.Array, FieldDescriptor.tensor(dtype=jnp.bool_, shape=(env.num_atoms,))
        ]

    return GoalSpec
