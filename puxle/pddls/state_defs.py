"""Dynamic state, context, and goal class builders for PDDL environments.

Constructs xtructure-backed ``State`` (packed boolean atom vector) and
``InstanceContext`` / ``GoalSpec`` dataclasses tailored to a grounded problem.
"""

from typing import TYPE_CHECKING, Annotated, Callable, Self

import jax
import jax.numpy as jnp
from xtructure import FieldDescriptor, Xtructurable

from puxle.core.puzzle_base import RenderedStateView


class PDDLState(RenderedStateView, abstract=True):
    """Typed view of PDDL states (packed atom vector)."""

    atoms: jax.Array

    if TYPE_CHECKING:

        @property
        def atoms_unpacked(self) -> jax.Array: ...

        @classmethod
        def from_unpacked(
            cls, *, shape: tuple[int, ...] = (), atoms: jax.Array = ...
        ) -> Self: ...

        def set_unpacked(self, *, atoms: jax.Array = ...) -> Self: ...

        @property
        def unpacked_atoms(self) -> jax.Array: ...


class PDDLInstanceContext(Xtructurable, abstract=True):
    """Typed view of PDDL action masks."""

    pre_mask: jax.Array
    pre_neg_mask: jax.Array
    add_mask: jax.Array
    del_mask: jax.Array


class PDDLGoalSpec(Xtructurable, abstract=True):
    """Typed view of PDDL goal masks."""

    GoalMask: jax.Array


def build_state_class(
    env, num_atoms: int, init_state: jnp.ndarray, string_parser: Callable
) -> type[PDDLState]:
    class State(PDDLState):
        atoms: Annotated[
            jax.Array, FieldDescriptor.packed_tensor(shape=(num_atoms,), packed_bits=1)
        ]

        def __str__(self, **kwargs):
            return string_parser(self, **kwargs)

        @property
        def unpacked_atoms(self):
            return self.atoms_unpacked

    return State


def build_instance_context_class(env) -> type[PDDLInstanceContext]:
    class InstanceContext(PDDLInstanceContext):
        pre_mask: Annotated[
            jax.Array,
            FieldDescriptor.tensor(
                dtype=jnp.bool_, shape=(env.num_actions, env.num_atoms)
            ),
        ]
        pre_neg_mask: Annotated[
            jax.Array,
            FieldDescriptor.tensor(
                dtype=jnp.bool_, shape=(env.num_actions, env.num_atoms)
            ),
        ]
        add_mask: Annotated[
            jax.Array,
            FieldDescriptor.tensor(
                dtype=jnp.bool_, shape=(env.num_actions, env.num_atoms)
            ),
        ]
        del_mask: Annotated[
            jax.Array,
            FieldDescriptor.tensor(
                dtype=jnp.bool_, shape=(env.num_actions, env.num_atoms)
            ),
        ]

    return InstanceContext


def build_goal_spec_class(env) -> type[PDDLGoalSpec]:
    class GoalSpec(PDDLGoalSpec):
        GoalMask: Annotated[
            jax.Array, FieldDescriptor.tensor(dtype=jnp.bool_, shape=(env.num_atoms,))
        ]

    return GoalSpec
