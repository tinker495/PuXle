"""CLI tool to visualize PuXle puzzle states and neighbour transitions.

Example::

    python -m scripts.visualize_puzzle --puzzle RubiksCube --img

The command prints textual descriptions of the initial state and all neighbour
states, and optionally saves image renders in ``images/visualizations/``.
"""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import jax
import numpy as np
import tyro
from jax import tree_util
from PIL import Image

from puxle import Puzzle
from puxle.puzzles import resolve_puzzle_class

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
_ARROW_NAMES = {"←": "left", "→": "right", "↑": "up", "↓": "down"}


def _sanitize_filename(label: str) -> str:
    cleaned = _ANSI_ESCAPE_RE.sub("", label).strip()
    for arrow, name in _ARROW_NAMES.items():
        cleaned = cleaned.replace(arrow, name)
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", cleaned).strip("_")
    return sanitized or "action"


def _save_image(array: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
    return path


def _select_state(puzzle: Puzzle, state_tree: Puzzle.State, index: int) -> Puzzle.State:
    selected = tree_util.tree_map(lambda x: x[index], state_tree)
    if isinstance(selected, puzzle.State):
        return selected
    if isinstance(selected, dict):
        return puzzle.State(**selected)
    return selected


def visualize_puzzle(
    puzzle_name: str,
    seed: int,
    img: bool,
    output_dir: Path,
    puzzle_kwargs: list[str],
) -> None:
    puzzle_class = resolve_puzzle_class(puzzle_name)

    parsed_kwargs: dict[str, object] = {}
    for item in puzzle_kwargs:
        if "=" not in item:
            raise ValueError("--kwarg must be provided as key=value")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("--kwarg key cannot be empty")
        try:
            parsed_value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed_value = value
        parsed_kwargs[key] = parsed_value

    puzzle = puzzle_class(**parsed_kwargs)
    print(f"Loaded puzzle: {puzzle!r}")

    key = jax.random.PRNGKey(seed)
    solve_config, init_state = puzzle.get_inits(key)
    neighbours, costs = puzzle.get_neighbours(solve_config, init_state, filled=True)

    print("\nInitial State:")
    print(puzzle.render_str(init_state, solve_config=solve_config))
    print("\nAction overview:")

    action_labels = [puzzle.action_to_string(i) for i in range(puzzle.action_size)]
    np_costs = np.asarray(costs)
    finite_mask = np.isfinite(np_costs)

    for idx, label in enumerate(action_labels):
        cost_repr = "∞" if not finite_mask[idx] else f"{np_costs[idx]:.3f}"
        print(f"[{idx:02d}] {label:20s} cost={cost_repr}")
        neighbour_state = _select_state(puzzle, neighbours, idx)
        print(puzzle.render_str(neighbour_state, solve_config=solve_config))
        print("-")

    if img:
        img_root = output_dir / puzzle_class.__name__

        init_path = _save_image(
            puzzle.render_img(init_state, solve_config=solve_config),
            img_root / "initial.png",
        )
        print(f"Saved initial state image -> {init_path}")

        for idx in range(puzzle.action_size):
            action_label = action_labels[idx]
            neighbour_state = _select_state(puzzle, neighbours, idx)
            filename = f"{idx:02d}_{_sanitize_filename(action_label)}.png"
            neighbour_path = _save_image(
                puzzle.render_img(neighbour_state, solve_config=solve_config),
                img_root / filename,
            )
            print(f"Saved neighbour {action_label} image -> {neighbour_path}")


@dataclass
class VisualizeOptions:
    """Visualize PuXle puzzle states."""

    puzzle: str
    seed: int = 42
    img: bool = False
    output_dir: Path = Path("images/visualizations")
    puzzle_kwargs: Annotated[
        list[str],
        tyro.conf.arg(name="kwarg"),
        tyro.conf.UseAppendAction,
    ] = field(default_factory=list)


def main(argv: list[str] | None = None) -> None:
    args = tyro.cli(VisualizeOptions, args=argv)

    try:
        visualize_puzzle(
            puzzle_name=args.puzzle,
            seed=args.seed,
            img=args.img,
            output_dir=args.output_dir,
            puzzle_kwargs=args.puzzle_kwargs,
        )
    except ValueError as exc:
        print(f"visualize_puzzle: error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
