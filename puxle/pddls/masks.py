from __future__ import annotations

from typing import Dict, List, Tuple

import jax.numpy as jnp


def build_masks(
    grounded_actions: List[Dict], atom_to_idx: Dict[str, int], num_atoms: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build JAX arrays for precondition, add, and delete masks."""
    pre_mask = jnp.zeros((len(grounded_actions), num_atoms), dtype=jnp.bool_)
    add_mask = jnp.zeros((len(grounded_actions), num_atoms), dtype=jnp.bool_)
    del_mask = jnp.zeros((len(grounded_actions), num_atoms), dtype=jnp.bool_)

    for action_idx, action in enumerate(grounded_actions):
        for precondition in action["preconditions"]:
            if precondition in atom_to_idx:
                atom_idx = atom_to_idx[precondition]
                pre_mask = pre_mask.at[action_idx, atom_idx].set(True)
        for add_effect in action["effects"][0]:
            if add_effect in atom_to_idx:
                atom_idx = atom_to_idx[add_effect]
                add_mask = add_mask.at[action_idx, atom_idx].set(True)
        for del_effect in action["effects"][1]:
            if del_effect in atom_to_idx:
                atom_idx = atom_to_idx[del_effect]
                del_mask = del_mask.at[action_idx, atom_idx].set(True)

    return pre_mask, add_mask, del_mask


def build_initial_state(problem, atom_to_idx: Dict[str, int], num_atoms: int) -> jnp.ndarray:
    """Build initial state as boolean array from PDDL problem init facts."""
    init_state = jnp.zeros(num_atoms, dtype=jnp.bool_)

    for fact in getattr(problem, "init", []) or []:
        fact_str = f"({fact.name} {' '.join([getattr(arg, 'name', str(arg)) for arg in fact.terms])})"
        if fact_str in atom_to_idx:
            atom_idx = atom_to_idx[fact_str]
            init_state = init_state.at[atom_idx].set(True)

    return init_state


def extract_goal_conditions(goal) -> List[str]:
    """Extract atomic conditions from a goal formula object."""
    if goal is None:
        return []

    if hasattr(goal, "name"):
        return [f"({goal.name} {' '.join([getattr(arg, 'name', str(arg)) for arg in goal.terms])})"]

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

    return []


def build_goal_mask(problem, atom_to_idx: Dict[str, int], num_atoms: int) -> jnp.ndarray:
    """Build goal mask for conjunctive positive goals."""
    goal_mask = jnp.zeros(num_atoms, dtype=jnp.bool_)

    goal_conditions = extract_goal_conditions(getattr(problem, "goal", None))

    for condition in goal_conditions:
        if condition in atom_to_idx:
            atom_idx = atom_to_idx[condition]
            goal_mask = goal_mask.at[atom_idx].set(True)

    return goal_mask
