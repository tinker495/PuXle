"""Pretty-printing and domain-diagram utilities for PDDL states, actions, and solve configs."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any, Dict, List, Tuple

import jax
import numpy as np

from puxle.render.text import text_frame, visible_width, wrap_items
from puxle.utils.util import colored_str

_BLOCK_COLORS = ("red", "green", "yellow", "blue", "magenta", "cyan")
_FLOW_PREDICATES = frozenset(
    {
        "at",
        "at-robby",
        "at-veh",
        "at-pkg",
        "at-sample",
        "pointing",
        "in",
        "carry",
        "on-board",
        "calibration-target",
    }
)


def split_atom(atom_str: str) -> tuple[str, list[str]]:
    content = atom_str
    if content.startswith("(") and content.endswith(")"):
        content = content[1:-1]
    parts = content.split()
    if not parts:
        return "", []
    return parts[0], parts[1:]


def build_label_color_maps(domain: Any) -> Tuple[Dict[str, str], Dict[str, str]]:
    labels: set[str] = set()
    try:
        actions = domain.actions or []
        for action in actions:
            try:
                labels.add(action.name)
            except AttributeError:
                continue
    except (AttributeError, TypeError):
        pass
    try:
        predicates = domain.predicates or []
        for predicate in predicates:
            try:
                labels.add(predicate.name)
            except AttributeError:
                continue
    except (AttributeError, TypeError):
        pass

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
            color = (
                label_text_color_map[name] if name in label_text_color_map else "white"
            )
            colored_name = colored_str(name, color)
            params_str = " ".join(params)
            return f"{colored_name} {params_str}" if params_str else colored_name
        return f"({name} {' '.join(params)})"
    return f"action_{index}"


def _format_atom_args(predicate: str, args: list[str]) -> str:
    if not args:
        return "true"
    if len(args) == 1:
        return args[0]
    if len(args) == 2 and predicate in _FLOW_PREDICATES:
        return f"{args[0]} → {args[1]}"
    return f"({', '.join(args)})"


def _atom_panel(
    title: str,
    sample: list[str],
    colors: dict[str, str],
    *,
    satisfied: set[str] | None = None,
    omitted: int,
    noun: str,
    show_more: bool,
    footer_rows: list[str] | None = None,
    priority_predicates: set[str] | None = None,
) -> str:
    groups: dict[str, list[str]] = {}
    for atom in sorted(sample):
        predicate, args = split_atom(atom)
        value = _format_atom_args(predicate, args)
        if satisfied is not None:
            mark = (
                colored_str("✓", "green")
                if atom in satisfied
                else colored_str("✗", "red")
            )
            value = f"{mark} {value}".rstrip()
        groups.setdefault(predicate, []).append(value)

    ordered_predicates = sorted(
        groups.keys(),
        key=lambda pred: (
            0 if (priority_predicates is None or pred in priority_predicates) else 1,
            pred,
        ),
    )
    extra_labels = ["Static"] if footer_rows else []
    label_width = max(
        map(visible_width, [*ordered_predicates, *extra_labels]), default=0
    )
    rows: list[str] = []
    for predicate in ordered_predicates:
        values = groups[predicate]
        color = colors[predicate] if predicate in colors else colors["default"]
        label = colored_str(predicate, color)
        label += " " * (label_width - visible_width(predicate) + 2)
        first, *rest = wrap_items(values, width=max(24, 56 - label_width), gap="  ·  ")
        rows.append(label + first)
        rows.extend(" " * (label_width + 2) + line for line in rest)
    if footer_rows:
        prefix = colored_str("Static", "white") + " " * (
            label_width - visible_width("Static") + 2
        )
        rows.append(prefix + footer_rows[0])
        rows.extend(" " * (label_width + 2) + line for line in footer_rows[1:])
    if omitted > 0:
        rows.append(f"… and {omitted} more {noun}" if show_more else f"… +{omitted}")
    return text_frame(rows or ["None"], title=title)


def _reconstruct_towers(
    atoms: Iterable[str],
) -> tuple[list[list[str]], list[str], bool]:
    parsed = [split_atom(atom) for atom in sorted(atoms)]
    ontable = [args[0] for pred, args in parsed if pred == "ontable" and len(args) == 1]
    holding = [args[0] for pred, args in parsed if pred == "holding" and len(args) == 1]
    handempty = any(pred == "handempty" for pred, _ in parsed)
    support_to_above: dict[str, str] = {}
    above_to_support: dict[str, str] = {}
    all_blocks: set[str] = set(ontable) | set(holding)
    for pred, args in parsed:
        if pred == "clear" and len(args) == 1:
            all_blocks.add(args[0])
        elif pred == "on" and len(args) == 2:
            above, below = args
            all_blocks.update((above, below))
            if below not in support_to_above:
                support_to_above[below] = above
            if above not in above_to_support:
                above_to_support[above] = below

    visited: set[str] = set(holding)
    towers: list[list[str]] = []
    roots = list(ontable) + [
        b
        for b in sorted(all_blocks)
        if b not in ontable and b not in holding and b not in above_to_support
    ]
    for root in roots:
        if root in visited:
            continue
        tower = [root]
        visited.add(root)
        curr = root
        while curr in support_to_above and support_to_above[curr] not in visited:
            nxt = support_to_above[curr]
            tower.append(nxt)
            visited.add(nxt)
            curr = nxt
        towers.append(tower)

    for leftover in sorted(all_blocks - visited):
        if leftover not in visited:
            towers.append([leftover])
            visited.add(leftover)
    return towers, holding, handempty


def _render_blocksworld_panel(
    title: str,
    facts: set[str],
    *,
    goals: set[str] | None = None,
    show_hand: bool = True,
) -> str:
    towers, holding, handempty = _reconstruct_towers(facts)
    all_names = [b for tower in towers for b in tower] + holding
    inner_w = max((len(name) for name in all_names), default=1)
    col_w = max(3, inner_w + 2)
    goal_on: dict[str, str] = {}
    if goals is not None:
        for atom in goals:
            pred, args = split_atom(atom)
            if pred == "on" and len(args) == 2:
                goal_on[args[0]] = args[1]

    def format_block(block: str, below: str | None) -> str:
        token = f"[{block.center(inner_w)}]"
        if goals is not None and block in goal_on:
            color = "green" if below == goal_on[block] else "yellow"
            return colored_str(token, color)
        color = _BLOCK_COLORS[sum(ord(c) for c in block) % len(_BLOCK_COLORS)]
        return colored_str(token, color)

    rows: list[str] = []
    if show_hand:
        if holding:
            held = "  ".join(format_block(b, None) for b in holding)
            rows.append(f"Hand   {held}")
        else:
            status = "empty" if handempty else "─"
            rows.append(f"Hand   [{status}]")

    if towers:
        max_h = max(len(t) for t in towers)
        if show_hand:
            rows.append("")
        for level in range(max_h - 1, -1, -1):
            cells: list[str] = []
            for tower in towers:
                if level < len(tower):
                    below = tower[level - 1] if level > 0 else None
                    cells.append(format_block(tower[level], below))
                else:
                    cells.append("│".center(col_w))
            rows.append(" " + "   ".join(cells))
        bases = ["┗" + "━" * (col_w - 2) + "┛" for _ in towers]
        rows.append(" " + "━━━".join(bases) + "  Table")
    else:
        rows.append(" ┗━━━┛  Table")
    return text_frame(rows, title=title)


def _render_gripper_panel(
    title: str,
    facts: set[str],
    all_atoms: Iterable[str],
    *,
    goals: set[str] | None = None,
) -> str:
    parsed = [split_atom(atom) for atom in sorted(facts)]
    robby_rooms = {args[0] for pred, args in parsed if pred == "at-robby" and args}
    free_grippers = [args[0] for pred, args in parsed if pred == "free" and args]
    carried = {
        args[1]: args[0] for pred, args in parsed if pred == "carry" and len(args) == 2
    }
    rooms: set[str] = set(robby_rooms)
    grippers: set[str] = set(free_grippers) | set(carried.keys())
    balls_in_room: dict[str, list[str]] = {}
    for pred, args in parsed:
        if pred == "at" and len(args) == 2:
            ball, room = args
            rooms.add(room)
            balls_in_room.setdefault(room, []).append(ball)

    for atom in all_atoms:
        pred, args = split_atom(atom)
        if pred == "at-robby" and args:
            rooms.add(args[0])
        elif pred == "free" and args:
            grippers.add(args[0])

    goal_room: dict[str, str] = {}
    if goals is not None:
        for atom in goals:
            pred, args = split_atom(atom)
            if pred == "at" and len(args) == 2:
                goal_room[args[0]] = args[1]
                rooms.add(args[1])

    gripper_tokens: list[str] = []
    for g in sorted(grippers):
        if g in carried:
            ball = carried[g]
            gripper_tokens.append(f"[{g}: {colored_str(ball, 'yellow')}]")
        else:
            gripper_tokens.append(f"[{g}: empty]")
    rows: list[str] = []
    if gripper_tokens:
        rows.append("Grippers  " + "  ".join(gripper_tokens))
    room_w = max((len(r) for r in rooms), default=4)
    for room in sorted(rooms):
        occupants: list[str] = []
        if room in robby_rooms:
            occupants.append(colored_str("● robby", "green"))
        for ball in sorted(balls_in_room[room] if room in balls_in_room else ()):
            if ball in goal_room:
                if goal_room[ball] == room:
                    occupants.append(colored_str(f"● {ball} ✓", "green"))
                else:
                    occupants.append(
                        colored_str(f"● {ball} → {goal_room[ball]}", "yellow")
                    )
            else:
                occupants.append(colored_str(f"● {ball}", "cyan"))
        if occupants:
            first, *rest = wrap_items(occupants, width=48, gap="   ")
            rows.append(f"{room:<{room_w}s}  {first}")
            rows.extend(" " * (room_w + 2) + line for line in rest)
        else:
            rows.append(f"{room:<{room_w}s}  ·")
    return text_frame(rows or ["None"], title=title)


def _domain_kind(grounded_atoms: Sequence[str]) -> str:
    predicates = {split_atom(atom)[0] for atom in grounded_atoms}
    if {"ontable", "on"}.issubset(predicates):
        return "blocksworld"
    if {"at-robby", "at", "free", "carry"}.issubset(predicates):
        return "gripper"
    return "general"


def build_state_string_parser(env: Any) -> Callable[..., str]:
    mutable = {
        atom
        for action in env.grounded_actions
        for effect in ("add", "delete")
        for atom in action["effects"][effect]
    }
    mutable_predicates = {split_atom(atom)[0] for atom in mutable}
    kind = _domain_kind(env.grounded_atoms)

    def parser(
        state: Any,
        solve_config: Any = None,
        *,
        max_show: int = 12,
        header: bool = False,
        raw: bool = False,
        show_summary: bool = False,
        show_more: bool = False,
        **kwargs: Any,
    ) -> str:
        limit = max(0, int(max_show))
        atoms = np.asarray(jax.device_get(state).unpacked_atoms, dtype=bool)
        facts = {env.grounded_atoms[i] for i in np.flatnonzero(atoms)}
        goals: list[str] = []
        if solve_config is not None:
            mask = jax.device_get(solve_config.GoalSpec.GoalMask)
            goals = [env.grounded_atoms[i] for i in np.flatnonzero(mask)]
        goal_set = set(goals)
        met = facts.intersection(goal_set)
        ordered_facts = sorted(
            facts, key=lambda atom: (atom not in met, atom not in mutable, atom)
        )
        raw_sample = ordered_facts[:limit]
        count = len(facts)
        lines: list[str] = []
        if header:
            lines.append(
                f"State: {count}/{env.num_atoms} true atoms ({100 * count / max(1, env.num_atoms):.2f}%)"
            )
        if raw:
            lines.append("Raw sample atoms: " + (", ".join(raw_sample) or "<none>"))
        if show_summary:
            lines.append(f"Summary: true={count}, total={env.num_atoms}")
            if solve_config is not None:
                lines.append(f"Goals satisfied: {len(met)}/{len(goals)}")

        if kind == "blocksworld":
            lines.append(
                _render_blocksworld_panel(
                    f"Blocksworld · {count} facts",
                    facts,
                    goals=goal_set if solve_config is not None else None,
                    show_hand=True,
                )
            )
        elif kind == "gripper":
            lines.append(
                _render_gripper_panel(
                    f"Gripper · {count} facts",
                    facts,
                    env.grounded_atoms,
                    goals=goal_set if solve_config is not None else None,
                )
            )
        else:
            dynamic_facts = [atom for atom in ordered_facts if atom in mutable]
            static_facts = [atom for atom in ordered_facts if atom not in mutable]
            if count <= limit or not dynamic_facts or not static_facts:
                panel_sample = raw_sample
                footer_rows = None
                omitted = count - len(panel_sample)
            else:
                panel_sample = list(dynamic_facts[:limit])
                remaining_slots = max(0, limit - len(panel_sample))
                static_by_pred: dict[str, list[str]] = {}
                for atom in static_facts:
                    pred, _ = split_atom(atom)
                    static_by_pred.setdefault(pred, []).append(atom)
                summarized_static: list[tuple[str, int]] = []
                for pred, group in sorted(
                    static_by_pred.items(), key=lambda item: (len(item[1]), item[0])
                ):
                    if len(group) <= remaining_slots:
                        panel_sample.extend(group)
                        remaining_slots -= len(group)
                    else:
                        summarized_static.append((pred, len(group)))
                if summarized_static:
                    static_items = [
                        f"{pred} ×{cnt}" for pred, cnt in sorted(summarized_static)
                    ]
                    footer_rows = wrap_items(static_items, width=46, gap="  ·  ")
                else:
                    footer_rows = None
                omitted = max(0, len(dynamic_facts) - limit)
            lines.append(
                _atom_panel(
                    f"Facts · {count}",
                    panel_sample,
                    env._label_text_color_map,
                    omitted=omitted,
                    noun="true atoms",
                    show_more=show_more,
                    footer_rows=footer_rows,
                    priority_predicates=mutable_predicates or None,
                )
            )

        if solve_config is not None:
            lines.append(
                _atom_panel(
                    f"Goals · {len(met)}/{len(goals)}",
                    sorted(goals, key=lambda atom: (atom in met, atom))[:limit],
                    env._label_text_color_map,
                    satisfied=met,
                    omitted=max(0, len(goals) - limit),
                    noun="goal atoms",
                    show_more=show_more,
                )
            )
        return "\n".join(lines)

    return parser


def build_solve_config_string_parser(env: Any) -> Callable[..., str]:
    kind = _domain_kind(env.grounded_atoms)

    def parser(
        solve_config: Any,
        *,
        max_show: int = 12,
        header: bool = False,
        raw: bool = False,
        show_summary: bool = False,
        show_more: bool = False,
        **kwargs: Any,
    ) -> str:
        mask = jax.device_get(solve_config.GoalSpec.GoalMask)
        goals = sorted(env.grounded_atoms[i] for i in np.flatnonzero(mask))
        sample = goals[: max(0, int(max_show))]
        lines: list[str] = []
        if header:
            lines.append(f"Goal: {len(goals)} atoms")
        if raw:
            lines.append("Raw sample goals: " + (", ".join(sample) or "<none>"))
        if show_summary:
            lines.append(f"Goals: {len(goals)}/{env.num_atoms}")
        if kind == "blocksworld" and goals:
            lines.append(
                _render_blocksworld_panel(
                    f"Goal Towers · {len(goals)}",
                    set(goals),
                    show_hand=False,
                )
            )
        lines.append(
            _atom_panel(
                f"Goals · {len(goals)}",
                sample,
                env._label_text_color_map,
                omitted=len(goals) - len(sample),
                noun="goal atoms",
                show_more=show_more,
            )
        )
        return "\n".join(lines)

    return parser
