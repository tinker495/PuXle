"""Host-side terminal diagram renderers for CayleyPuzzle families."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from typing import Any

import jax
import numpy as np

from puxle.puzzles.pancake import get_color as _pancake_color
from puxle.puzzles.rubikscube import (
    BACK,
    DOWN,
    FRONT,
    LEFT,
    RIGHT,
    UP,
    _plain_table,
    face_map,
    face_map_legend,
    rgb_map,
)
from puxle.render.text import text_frame
from puxle.utils.util import colored_str, coloring_str

_HUNGARIAN_NAME = re.compile(r"^hungarian_rings-(\d+)-(\d+)-(\d+)-(\d+)$")
_GLOBE_NAME = re.compile(r"^globe_puzzle-(\d+)-(\d+)$")
_RAW_PERM_NAME = re.compile(r"^\d+(?:,\d+)+$")


def _is_prefix_reversal(perm: np.ndarray) -> int | None:
    n = int(perm.size)
    moved = np.flatnonzero(perm != np.arange(n))
    if moved.size == 0:
        return None
    k = int(moved[-1]) + 1
    expected = np.concatenate((np.arange(k - 1, -1, -1), np.arange(k, n)))
    if np.array_equal(perm, expected):
        return k
    return None


def _is_pancake_graph(graph_name: str, perms: np.ndarray) -> bool:
    lowered = graph_name.lower()
    if lowered.startswith(("pancake", "cubic_pancake")):
        return True
    if perms.ndim != 2 or perms.shape[0] == 0 or perms.shape[1] < 2:
        return False
    return all(_is_prefix_reversal(row) is not None for row in perms)


def _is_burnt_pancake_graph(graph_name: str, perms: np.ndarray) -> bool:
    lowered = graph_name.lower()
    if lowered.startswith(("burnt_pancake", "signed_reversals")):
        return True
    if (
        perms.ndim != 2
        or perms.shape[0] == 0
        or perms.shape[1] < 2
        or perms.shape[1] % 2 != 0
    ):
        return False
    half = perms.shape[1] // 2
    for row in perms:
        if not np.all(np.abs(row[:half] - row[half:]) == half):
            return False
    return True


def _detect_ring_window(graph_name: str, perms: np.ndarray) -> int | None:
    if perms.ndim != 2 or perms.shape[1] < 3:
        return None
    n = int(perms.shape[1])
    shift_left = np.roll(np.arange(n), -1)
    shift_right = np.roll(np.arange(n), 1)
    has_shift = False
    window_sizes: list[int] = []
    for row in perms:
        if np.array_equal(row, shift_left) or np.array_equal(row, shift_right):
            has_shift = True
            continue
        moved = np.flatnonzero(row != np.arange(n))
        if moved.size >= 2 and int(moved[0]) == 0 and int(moved[-1]) < n - 1:
            window_sizes.append(int(moved[-1]) + 1)
        else:
            return None
    if has_shift and window_sizes:
        return max(window_sizes)
    lowered = graph_name.lower()
    if lowered.startswith("top_spin-"):
        parts = lowered.split("-")
        if len(parts) >= 3 and parts[-1].isdigit():
            return int(parts[-1])
    return None


def _detect_rubiks_cube_size(
    graph_name: str, central_state: np.ndarray, perms: np.ndarray
) -> int | None:
    n = int(central_state.size)
    if n < 24 or n % 6 != 0:
        return None
    face_tiles = n // 6
    size = int(math.isqrt(face_tiles))
    if size * size != face_tiles:
        return None
    faces = central_state.reshape(6, face_tiles)
    is_six_color = all(int(np.unique(faces[f]).size) == 1 for f in range(6)) and set(
        int(faces[f, 0]) for f in range(6)
    ) == {0, 1, 2, 3, 4, 5}
    lowered = graph_name.lower()
    if is_six_color or "rubik" in lowered or lowered.startswith("cube"):
        return size
    return None


def infer_cayley_graph_name(
    raw_name: str,
    generators_type: str,
    central_state: np.ndarray,
    perms: np.ndarray | None,
    generator_names: Sequence[str],
) -> str:
    if raw_name:
        return raw_name
    if generators_type == "PERMUTATION" and perms is not None:
        cube_size = _detect_rubiks_cube_size("", central_state, perms)
        if cube_size is not None:
            return f"rubik_cube-{cube_size}"
        if central_state.size == 24 and any(
            name.startswith("M_DF") for name in generator_names
        ):
            return "mini_pyramorphix"
        if central_state.size == 120 and any(
            name.startswith("M_") for name in generator_names
        ):
            return "megaminx"
    return "cayley"


def build_action_names(
    graph_name: str,
    raw_names: Sequence[str],
    deduped_indices: Sequence[int],
    synthesized_from: Sequence[int],
    perms: np.ndarray | None,
) -> tuple[str, ...]:
    window = _detect_ring_window(graph_name, perms) if perms is not None else None
    n = int(perms.shape[1]) if perms is not None and perms.ndim == 2 else 0
    shift_left = np.roll(np.arange(n), -1) if n > 0 else None
    shift_right = np.roll(np.arange(n), 1) if n > 0 else None

    def format_single(idx: int, perm: np.ndarray | None) -> str:
        raw = raw_names[idx] if idx < len(raw_names) else ""
        if (
            window is not None
            and perm is not None
            and (not raw or _RAW_PERM_NAME.match(raw))
        ):
            if shift_left is not None and np.array_equal(perm, shift_left):
                return "Shift Left (<<)"
            if shift_right is not None and np.array_equal(perm, shift_right):
                return "Shift Right (>>)"
            rev_k = _is_prefix_reversal(perm)
            if rev_k is not None:
                return f"Reverse Turnstile (R{rev_k})"
        if raw:
            return raw
        return f"{graph_name}.gen[{idx}]"

    names: list[str] = []
    for pos, orig_idx in enumerate(deduped_indices):
        perm = perms[pos] if perms is not None and pos < len(perms) else None
        names.append(format_single(orig_idx, perm))
    for source_pos in synthesized_from:
        base = names[source_pos]
        if base == "Shift Left (<<)":
            names.append("Shift Right (>>)")
        elif base == "Shift Right (>>)":
            names.append("Shift Left (<<)")
        elif base.endswith("'"):
            names.append(base[:-1])
        elif base.endswith("_inv"):
            names.append(base[:-4])
        else:
            names.append(f"{base}'")
    return tuple(names)


def _render_pancake(values: np.ndarray, target: np.ndarray | None) -> str:
    n = int(values.size)
    unique_sorted = np.unique(values)
    rank_map = {int(v): idx + 1 for idx, v in enumerate(unique_sorted)}
    max_size = max(len(unique_sorted), 1)
    idx_w = max(2, len(str(n)))
    val_w = max(2, max((len(str(int(v))) for v in values), default=2))
    rows: list[str] = []
    for i, raw_val in enumerate(values):
        val = int(raw_val)
        size = rank_map[val]
        bar = "=" * (2 * size - 1)
        colored_bar = colored_str(bar.center(max_size * 2), _pancake_color(size))
        rows.append(f"{i + 1:0{idx_w}d}:{val:0{val_w}d} - {colored_bar}")
    rows.append("Plate " + "┗━" + "━━" * max_size + "┛")
    if target is not None:
        matches = int(np.count_nonzero(values == target))
        rows.append(f"{matches}/{n} in place")
    return "\n".join(rows)


def _render_burnt_pancake(values: np.ndarray, target: np.ndarray | None) -> str:
    half = int(values.size) // 2
    top_stack = values[:half]
    idx_w = max(2, len(str(half)))
    val_w = max(2, len(str(half)))
    rows: list[str] = []
    for i, raw_val in enumerate(top_stack):
        val = int(raw_val)
        burnt = val >= half
        pancake_id = (val - half if burnt else val) + 1
        glyph = "~" if burnt else "="
        sign = "-" if burnt else "+"
        bar = (glyph * (2 * pancake_id - 1)).center(half * 2)
        rows.append(
            f"{i + 1:0{idx_w}d}:{sign}{pancake_id:0{val_w}d} - "
            + colored_str(bar, _pancake_color(pancake_id))
        )
    rows.append("Plate " + "┗━" + "━━" * half + "┛")
    if target is not None:
        matches = int(np.count_nonzero(top_stack == target[:half]))
        rows.append(f"{matches}/{half} in place")
    return "\n".join(rows)


def _render_ring_window(
    values: np.ndarray, target: np.ndarray | None, window: int
) -> str:
    width = max(2, max((len(str(int(v))) for v in values), default=1))
    labels = [f"{int(v):>{width}d}" for v in values]
    if target is not None:
        matches = values == target
        tokens = [
            colored_str(label, "green" if bool(match) else "yellow")
            for label, match in zip(labels, matches, strict=True)
        ]
    else:
        tokens = labels
    turnstile = " ".join(tokens[:window])
    rest = " ".join(tokens[window:])
    return f"[{turnstile}] {rest}".rstrip()


def _render_rubiks_cube(values: np.ndarray, size: int) -> str:
    tile_count = size * size
    faces = values.reshape(6, tile_count)
    unique_vals = np.unique(values)
    as_color = (
        int(unique_vals.size) <= 6
        and int(unique_vals.min()) >= 0
        and int(unique_vals.max()) <= 5
    )
    token_width = 1 if as_color else len(str(int(values.max(initial=0))))

    def format_tile(val: int) -> str:
        color_idx = val if as_color else (val // tile_count) % 6
        token = "■" if as_color else f"{val:>{token_width}d}"
        return coloring_str(token, rgb_map[color_idx])

    def empty_face() -> str:
        return "\n".join(["  " * (size + 2) for _ in range(size + 2)])

    def color_legend() -> str:
        return "\n".join(
            f"{face_map_legend[i]:<6}:{coloring_str('■', rgb_map[i])}" for i in range(6)
        )

    def face_box(face_id: int) -> str:
        title = face_map[face_id]
        row_w = size * token_width + (size - 1)
        lines = [f"┏━{title.center(row_w, '━')}━┓"]
        for r in range(size):
            row_tokens = [
                format_tile(int(faces[face_id, r * size + c])) for c in range(size)
            ]
            lines.append(f"┃ {' '.join(row_tokens)} ┃")
        lines.append(f"┗━{'━' * row_w}━┛")
        return "\n".join(lines)

    return _plain_table(
        [
            [color_legend(), ".\n" + face_box(UP)],
            [face_box(LEFT), face_box(FRONT), face_box(RIGHT), face_box(BACK)],
            [empty_face(), face_box(DOWN)],
        ]
    )


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


def _render_tetrahedral_panels(
    graph_name: str, values: np.ndarray, target: np.ndarray | None
) -> str:
    n = int(values.size)
    width = max(2, max((len(str(int(v))) for v in values), default=1))
    cells, footer = _format_colored_cells(values, target, width)
    face_names = ("Front", "Left", "Right", "Base")
    panels: list[str] = []
    if n == 24:
        # 4 corner triplets (0..11) + 4 inner facet triplets (12..23).
        for f in range(4):
            c_slice = cells[3 * f : 3 * (f + 1)]
            i_slice = cells[12 + 3 * f : 12 + 3 * (f + 1)]
            rows = [
                f"Corner  {' '.join(c_slice)}",
                f"Facet   {' '.join(i_slice)}",
            ]
            panels.append(text_frame(rows, title=face_names[f]))
    else:
        # Pyraminx (36 stickers): 4 tips (24..35), 4 centers (12..23), 6 edges (0..11).
        tip_names = ("Tip U", "Tip B", "Tip R", "Tip L")
        for f in range(4):
            t_slice = cells[24 + 3 * f : 24 + 3 * (f + 1)]
            c_slice = cells[12 + 3 * f : 12 + 3 * (f + 1)]
            e_slice = cells[3 * f : 3 * (f + 1)]
            rows = [
                f"Tip     {' '.join(t_slice)}",
                f"Center  {' '.join(c_slice)}",
                f"Edge    {' '.join(e_slice)}",
            ]
            panels.append(text_frame(rows, title=tip_names[f]))
    grid = _plain_table([[panels[0], panels[1]], [panels[2], panels[3]]])
    lines = [graph_name, grid]
    if footer is not None:
        lines.append(footer)
    return "\n".join(lines)


def _render_hungarian_rings(
    graph_name: str,
    values: np.ndarray,
    target: np.ndarray | None,
    left_size: int,
    left_index: int,
    right_size: int,
    right_index: int,
) -> str:
    width = max(2, max((len(str(int(v))) for v in values), default=1))
    cells, footer = _format_colored_cells(values, target, width)
    intersections = 1 if (left_index == 0 and right_index == 0) else 2
    full_size = left_size + right_size - intersections
    if intersections == 2:
        second_idx = left_size + right_size - right_index - 1
        right_indices = [0, *range(left_size, second_idx), left_index]
        if right_index > 1:
            right_indices.extend(range(second_idx, full_size))
        shared = {0, left_index}
    else:
        right_indices = [0, *range(left_size, full_size)]
        shared = {0}
    left_indices = list(range(left_size))

    def fmt_ring(indices: list[int]) -> str:
        return " ".join(
            f"⟨{cells[i]}⟩" if i in shared else f" {cells[i]} " for i in indices
        )

    rows = [
        f"Left   ({left_size:>2d})  {fmt_ring(left_indices)}",
        f"Right  ({right_size:>2d})  {fmt_ring(right_indices)}",
    ]
    lines = [text_frame(rows, title=graph_name)]
    if footer is not None:
        lines.append(footer)
    return "\n".join(lines)


def _render_globe(
    graph_name: str,
    values: np.ndarray,
    target: np.ndarray | None,
    a: int,
    b: int,
) -> str:
    rows_count = a + 1
    cols_count = 2 * b
    width = max(2, max((len(str(int(v))) for v in values), default=1))
    cells, footer = _format_colored_cells(values, target, width)
    header = "     " + "  ".join(f"{f'f{c}':>{width}s}" for c in range(cols_count))
    rows = [header]
    for r in range(rows_count):
        row_cells = cells[r * cols_count : (r + 1) * cols_count]
        rows.append(f"r{r:<2d}  " + "  ".join(row_cells))
    lines = [text_frame(rows, title=graph_name)]
    if footer is not None:
        lines.append(footer)
    return "\n".join(lines)


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
    central_state: np.ndarray,
    perms: np.ndarray | None,
    m: int = 0,
    n: int = 0,
    modulo: int = 0,
) -> Callable[..., str]:
    is_perm = generators_type == "PERMUTATION"
    cube_size = (
        _detect_rubiks_cube_size(graph_name, central_state, perms)
        if is_perm and perms is not None
        else None
    )
    is_pancake = (
        _is_pancake_graph(graph_name, perms)
        if is_perm and perms is not None and cube_size is None
        else False
    )
    is_burnt_pancake = (
        _is_burnt_pancake_graph(graph_name, perms)
        if is_perm and perms is not None and cube_size is None and not is_pancake
        else False
    )
    ring_window = (
        _detect_ring_window(graph_name, perms)
        if is_perm
        and perms is not None
        and cube_size is None
        and not is_pancake
        and not is_burnt_pancake
        else None
    )
    hungarian_match = _HUNGARIAN_NAME.match(graph_name) if is_perm else None
    globe_match = _GLOBE_NAME.match(graph_name) if is_perm else None
    is_tetrahedral = is_perm and (
        (central_state.size == 24 and graph_name.lower() == "mini_pyramorphix")
        or (central_state.size == 36 and graph_name.lower() == "pyraminx")
    )

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

        if not is_perm:
            return _render_matrix(graph_name, values, target, m, n, modulo)
        if cube_size is not None:
            return _render_rubiks_cube(values, cube_size)
        if is_pancake:
            return _render_pancake(values, target)
        if is_burnt_pancake:
            return _render_burnt_pancake(values, target)
        if ring_window is not None:
            return _render_ring_window(values, target, ring_window)
        if is_tetrahedral:
            return _render_tetrahedral_panels(graph_name, values, target)
        if hungarian_match is not None:
            ls, li, rs, ri = (int(g) for g in hungarian_match.groups())
            return _render_hungarian_rings(graph_name, values, target, ls, li, rs, ri)
        if globe_match is not None:
            ga, gb = (int(g) for g in globe_match.groups())
            return _render_globe(graph_name, values, target, ga, gb)
        return _render_boxed_permutation(graph_name, values, target)

    return parser
