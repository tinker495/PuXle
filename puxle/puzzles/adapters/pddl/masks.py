"""JAX boolean-mask construction for PDDL preconditions, effects, and goals.

Each grounded action is encoded as four boolean vectors over the atom
universe: positive preconditions, negative preconditions, add effects,
and delete effects.  Initial-state and goal masks are also built here.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import jax.numpy as jnp


def extract_goal_conditions(goal) -> List[str]:
    """Extract atomic conditions from a goal formula object."""
    if goal is None:
        return []

    goal_type = type(goal).__name__
    if goal_type == "Not":
        raise ValueError("Negative goals are not supported in STRIPS mode.")
    if goal_type == "Or":
        parts = []
        if hasattr(goal, "parts"):
            parts = list(goal.parts)
        elif hasattr(goal, "operands"):
            parts = list(goal.operands)
        if not parts:
            # pddl parser may represent empty `()` as `Or()`; treat as empty goal.
            return []
        raise ValueError(
            "Disjunctive goals `(or ...)` are not supported in STRIPS mode."
        )
    if goal_type == "EqualTo":
        raise ValueError("Equality goals are not supported in STRIPS mode.")

    if hasattr(goal, "name"):
        # Handle atomic predicate
        args = " ".join([getattr(arg, "name", str(arg)) for arg in goal.terms])
        return [f"({goal.name} {args})" if args else f"({goal.name})"]

    if hasattr(goal, "parts"):
        conditions: List[str] = []
        for part in goal.parts:
            conditions.extend(extract_goal_conditions(part))
        return conditions

    if hasattr(goal, "operands"):
        conditions = []
        for operand in goal.operands:
            conditions.extend(extract_goal_conditions(operand))
        return conditions

    raise ValueError(f"Unsupported goal node `{goal_type}` in STRIPS mode.")


def build_masks(
    grounded_actions: List[Dict], atom_to_idx: Dict[str, int], num_atoms: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Builds JAX masks for preconditions (pos/neg) and effects."""
    num_actions = len(grounded_actions)
    pre_mask = jnp.zeros((num_actions, num_atoms), dtype=bool)
    pre_neg_mask = jnp.zeros((num_actions, num_atoms), dtype=bool)
    add_mask = jnp.zeros((num_actions, num_atoms), dtype=bool)
    del_mask = jnp.zeros((num_actions, num_atoms), dtype=bool)

    for i, action in enumerate(grounded_actions):
        # Positive Preconditions
        for precondition in action.get("preconditions", []):
            if precondition in atom_to_idx:
                pre_mask = pre_mask.at[i, atom_to_idx[precondition]].set(True)

        # Negative Preconditions
        for neg_precondition in action.get("preconditions_neg", []):
            if neg_precondition in atom_to_idx:
                pre_neg_mask = pre_neg_mask.at[i, atom_to_idx[neg_precondition]].set(
                    True
                )

        effects = action["effects"]
        for add_effect in effects["add"]:
            if add_effect in atom_to_idx:
                add_mask = add_mask.at[i, atom_to_idx[add_effect]].set(True)
        for del_effect in effects["delete"]:
            if del_effect in atom_to_idx:
                del_mask = del_mask.at[i, atom_to_idx[del_effect]].set(True)

    return pre_mask, pre_neg_mask, add_mask, del_mask


def build_initial_state(
    problem, atom_to_idx: Dict[str, int], num_atoms: int
) -> jnp.ndarray:
    """Build initial state as boolean array from PDDL problem init facts."""
    init_state = jnp.zeros(num_atoms, dtype=jnp.bool_)

    for fact in getattr(problem, "init", []) or []:
        if not hasattr(fact, "name") or not hasattr(fact, "terms"):
            raise ValueError(
                f"Unsupported initial-state element `{type(fact).__name__}` in STRIPS mode."
            )
        args = " ".join([getattr(arg, "name", str(arg)) for arg in fact.terms])
        fact_str = f"({fact.name} {args})" if args else f"({fact.name})"
        if fact_str not in atom_to_idx:
            raise ValueError(
                f"Initial fact `{fact_str}` is not in grounded atom universe."
            )
        atom_idx = atom_to_idx[fact_str]
        init_state = init_state.at[atom_idx].set(True)

    return init_state


def build_goal_mask(
    problem, atom_to_idx: Dict[str, int], num_atoms: int
) -> jnp.ndarray:
    """Build goal mask for conjunctive positive goals."""
    goal_mask = jnp.zeros(num_atoms, dtype=jnp.bool_)

    goal_conditions = extract_goal_conditions(getattr(problem, "goal", None))

    for condition in goal_conditions:
        if condition not in atom_to_idx:
            raise ValueError(
                f"Goal atom `{condition}` is not in grounded atom universe."
            )
        atom_idx = atom_to_idx[condition]
        goal_mask = goal_mask.at[atom_idx].set(True)

    return goal_mask
