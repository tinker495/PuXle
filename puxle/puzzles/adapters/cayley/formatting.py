"""Generic host-side text rendering and action labels for CayleyPuzzle."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import numpy as np

from puxle.render.text import text_frame
from puxle.utils.util import colored_str


def build_action_names(
    graph_name: str,
    raw_names: Sequence[str],
    deduped_indices: Sequence[int],
    synthesized_from: Sequence[int],
) -> tuple[str, ...]:
    names = [
        raw_names[idx]
        if idx < len(raw_names) and raw_names[idx]
        else f"{graph_name}.gen[{idx}]"
        for idx in deduped_indices
    ]
    for source_pos in synthesized_from:
        base = names[source_pos]
        if base.endswith("'"):
            names.append(base[:-1])
        elif base.endswith("_inv"):
            names.append(base[:-4])
        else:
            names.append(f"{base}'")
    return tuple(names)


def _format_colored_cells(
    values: np.ndarray, target: np.ndarray | None, width: int
) -> tuple[list[str], str | None]:
    labels = [f"{int(v):>{width}d}" for v in values]
    if target is not None:
        matches = values == target
        cells = [
            colored_str(label, "green" if bool(match) else "yellow")
            for label, match in zip(labels, matches, strict=True)
        ]
        return cells, f"{int(np.count_nonzero(matches))}/{len(cells)} in place"
    palette = ("red", "green", "yellow", "blue", "magenta", "cyan")
    cells = [
        colored_str(label, palette[int(v) % len(palette)])
        for label, v in zip(labels, values, strict=True)
    ]
    return cells, None


def _render_boxed_permutation(
    graph_name: str, values: np.ndarray, target: np.ndarray | None
) -> str:
    n = int(values.size)
    if n == 0:
        return text_frame(["(empty)"], title=graph_name)
    width = max(
        len(str(n - 1)),
        max((len(str(int(v))) for v in values), default=1),
    )
    cells, footer = _format_colored_cells(values, target, width)
    cols = min(8, n)
    n_rows = (n + cols - 1) // cols
    cell_bar = "━" * (width + 2)
    lines: list[str] = [
        graph_name,
        "┏" + "┳".join([cell_bar] * cols) + "┓",
    ]
    for r in range(n_rows):
        chunk = cells[r * cols : (r + 1) * cols]
        if len(chunk) < cols:
            chunk = [*chunk, *[" " * width] * (cols - len(chunk))]
        lines.append("┃ " + " ┃ ".join(chunk) + " ┃")
        if r != n_rows - 1:
            lines.append("┣" + "╋".join([cell_bar] * cols) + "┫")
    lines.append("┗" + "┻".join([cell_bar] * cols) + "┛")
    if footer is not None:
        lines.append(footer)
    return "\n".join(lines)


def _render_matrix(
    graph_name: str,
    values: np.ndarray,
    target: np.ndarray | None,
    m: int,
    n: int,
    modulo: int,
) -> str:
    width = max(
        len(str(modulo - 1)),
        max((len(str(int(v))) for v in values), default=1),
    )
    cells, footer = _format_colored_cells(values, target, width)
    grid = np.asarray(cells).reshape(m, n)
    rows = ["  ".join(row) for row in grid]
    title = f"{graph_name} · mod {modulo}"
    lines = [text_frame(rows, title=title)]
    if footer is not None:
        lines.append(footer)
    return "\n".join(lines)


def build_cayley_string_parser(
    *,
    graph_name: str,
    generators_type: str,
    m: int = 0,
    n: int = 0,
    modulo: int = 0,
) -> Callable[..., str]:
    is_perm = generators_type == "PERMUTATION"

    def parser(state: Any, solve_config: Any = None, **kwargs: Any) -> str:
        host, goal = jax.device_get(
            (state, solve_config.GoalSpec if solve_config is not None else None)
        )
        (raw_values,) = jax.tree.leaves(host)
        values = np.asarray(raw_values)
        target: np.ndarray | None = None
        if goal is not None:
            (raw_target,) = jax.tree.leaves(goal)
            target = np.asarray(raw_target)

        if is_perm:
            return _render_boxed_permutation(graph_name, values, target)
        return _render_matrix(graph_name, values, target, m, n, modulo)

    return parser
