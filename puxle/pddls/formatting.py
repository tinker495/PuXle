"""Pretty-printing utilities for PDDL states, actions, and solve configs.

Provides colour-coded terminal output for debugging and visualisation of grounded atoms and actions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Dict, List, Tuple

import jax.numpy as jnp

from puxle.utils.util import colored_str


def split_atom(atom_str: str) -> tuple[str, list[str]]:
    content = atom_str
    if content.startswith("(") and content.endswith(")"):
        content = content[1:-1]
    parts = content.split()
    if not parts:
        return "", []
    return parts[0], parts[1:]


def build_label_color_maps(domain) -> Tuple[Dict[str, str], Dict[str, str]]:
    labels = set()
    try:
        for action in getattr(domain, "actions", []) or []:
            if hasattr(action, "name"):
                labels.add(action.name)
        for predicate in getattr(domain, "predicates", []) or []:
            if hasattr(predicate, "name"):
                labels.add(predicate.name)
    except (AttributeError, TypeError):
        labels = set()

    ansi_palette = ["cyan", "magenta", "green", "yellow", "blue", "white", "red"]
    label_text_color_map: Dict[str, str] = {}
    for idx, label in enumerate(sorted(labels)):
        label_text_color_map[label] = ansi_palette[idx % len(ansi_palette)]
    label_text_color_map.setdefault("default", "white")

    return dict(label_text_color_map), label_text_color_map


def action_to_string(
    grounded_actions: List[Dict],
    index: int,
    label_text_color_map: Dict[str, str],
    colored: bool = True,
) -> str:
    if 0 <= index < len(grounded_actions):
        action_data = grounded_actions[index]
        name = action_data["name"]
        params = action_data["parameters"]
        if colored:
            color = label_text_color_map.get(name, "white")
            colored_name = colored_str(name, color)
            params_str = " ".join(params)
            return f"{colored_name} {params_str}" if params_str else colored_name
        return f"({name} {' '.join(params)})"
    return f"action_{index}"


def build_state_string_parser(env) -> Callable:
    def parser(state, solve_config=None, **kwargs):
        atoms = state.unpacked_atoms
        true_indices = [i for i in range(env.num_atoms) if bool(atoms[i])]
        true_count = len(true_indices)
        density = (true_count / max(1, env.num_atoms)) * 100.0

        goal_mask = None
        goal_count = 0
        goals_satisfied = 0
        if solve_config is not None:
            goal_mask = solve_config.GoalSpec.GoalMask
            try:
                goal_count = int(jnp.sum(goal_mask))
                goals_satisfied = int(jnp.sum(jnp.logical_and(goal_mask, atoms)))
            except (TypeError, IndexError, ValueError):
                gm = [bool(goal_mask[i]) for i in range(env.num_atoms)]
                goal_count = sum(gm)
                goals_satisfied = sum(
                    1 for i in range(env.num_atoms) if gm[i] and bool(atoms[i])
                )

        max_show = int(kwargs.get("max_show", 12))
        if goal_mask is not None:
            try:
                goal_true = [i for i in true_indices if bool(goal_mask[i])]
                non_goal_true = [i for i in true_indices if not bool(goal_mask[i])]
            except (TypeError, IndexError, ValueError):
                gm = [bool(goal_mask[i]) for i in range(env.num_atoms)]
                goal_true = [i for i in true_indices if gm[i]]
                non_goal_true = [i for i in true_indices if not gm[i]]
            ordered_true_indices = goal_true + non_goal_true
        else:
            ordered_true_indices = true_indices

        sample_indices = ordered_true_indices[:max_show]
        sample_atoms = [env.grounded_atoms[i] for i in sample_indices]
        truncated = true_count > len(sample_atoms)
        raw_sample_line = (
            "Raw sample atoms: " + ", ".join(sample_atoms)
            if sample_atoms
            else "Raw sample atoms: <none>"
        )

        show_summary = bool(kwargs.get("show_summary", False))
        show_more = bool(kwargs.get("show_more", False))

        pieces: list[str] = []
        if kwargs.get("header", False):
            pieces.append(
                f"State: {true_count}/{env.num_atoms} true atoms ({density:.2f}%)"
            )
        if kwargs.get("raw", False):
            pieces.append(raw_sample_line)
        if show_summary:
            pieces.append(f"Summary: true={true_count}, total={env.num_atoms}")
            if goal_mask is not None:
                pieces.append(f"Goals satisfied: {goals_satisfied}/{goal_count}")
        annotated_atoms: list[str] = []
        for idx, atom_str in zip(sample_indices, sample_atoms):
            if goal_mask is not None and bool(goal_mask[idx]):
                mark = "✓" if bool(atoms[idx]) else "✗"
                annotated_atoms.append(f"{atom_str} - {mark}")
            else:
                annotated_atoms.append(atom_str)
        pieces.append(", ".join(annotated_atoms) if annotated_atoms else "")
        if truncated and not show_more:
            pieces.append("...")
        if show_more:
            remaining = max(0, true_count - len(sample_atoms))
            if remaining > 0:
                pieces.append(f"... and {remaining} more true atoms")
        return "\n".join([p for p in pieces if p])

    return parser


def build_solve_config_string_parser(env) -> Callable:
    def parser(solve_config, **kwargs):
        goal_mask = solve_config.GoalSpec.GoalMask
        goal_indices = [i for i in range(env.num_atoms) if bool(goal_mask[i])]
        goal_count = len(goal_indices)
        max_show = int(kwargs.get("max_show", 12))
        sample_indices = goal_indices[:max_show]
        sample_atoms = [env.grounded_atoms[i] for i in sample_indices]
        raw_sample_line = (
            "Raw sample goals: " + ", ".join(sample_atoms)
            if sample_atoms
            else "Raw sample goals: <none>"
        )

        show_summary = bool(kwargs.get("show_summary", False))
        show_more = bool(kwargs.get("show_more", False))

        pieces: list[str] = []
        if kwargs.get("header", False):
            pieces.append(f"Goal: {goal_count} atoms")
        if kwargs.get("raw", False):
            pieces.append(raw_sample_line)
        if show_summary:
            pieces.append(f"Goals: {goal_count}/{env.num_atoms}")
        sample_text = ", ".join(sample_atoms) if sample_atoms else "<none>"
        pieces.append(f"Goal atoms: {sample_text}")
        if show_more:
            remaining = max(0, goal_count - len(sample_atoms))
            if remaining > 0:
                pieces.append(f"... and {remaining} more goal atoms")
        return "\n".join([p for p in pieces if p])

    return parser
