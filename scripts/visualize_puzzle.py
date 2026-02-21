"""CLI tool to visualize PuXle puzzle states and neighbour transitions.

Example::

    python -m scripts.visualize_puzzle --puzzle RubiksCube --img

The command prints textual descriptions of the initial state and all neighbour
states, and optionally saves image renders in ``images/visualizations/``.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Dict, Type

import click
import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util
from PIL import Image

from puxle.core.puzzle_base import Puzzle


def discover_puzzles() -> Dict[str, Type[Puzzle]]:
    import puxle.puzzles as puzzles_module

    puzzle_map: Dict[str, Type[Puzzle]] = {}

    for attr_name in dir(puzzles_module):
        attr = getattr(puzzles_module, attr_name)
        if inspect.isclass(attr) and issubclass(attr, Puzzle) and attr is not Puzzle:
            puzzle_map[attr.__name__] = attr

    return puzzle_map


def _ensure_uint8(image: jnp.ndarray | np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype == np.float32 or array.dtype == np.float64:
        array = np.clip(array, 0.0, 1.0)
        array = (array * 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = array.astype(np.uint8)
    return array


def _save_image(array: jnp.ndarray | np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    array_uint8 = _ensure_uint8(array)
    Image.fromarray(array_uint8).save(path)
    return path


def _format_state(puzzle: Puzzle, state: Puzzle.State) -> str:
    parser = puzzle.get_string_parser()
    try:
        return parser(state)
    except Exception:
        return str(state)


def _select_state(puzzle: Puzzle, state_tree: Puzzle.State, index: int) -> Puzzle.State:
    selected = tree_util.tree_map(lambda x: x[index], state_tree)
    if isinstance(selected, puzzle.State):
        return selected
    if isinstance(selected, dict):
        return puzzle.State(**selected)
    return selected


@click.command()
@click.option(
    "--puzzle",
    "puzzle_name",
    type=str,
    required=True,
    help="Puzzle class name to visualize.",
)
@click.option(
    "--seed", default=42, show_default=True, help="PRNG seed for reproducible sampling."
)
@click.option(
    "--img/--no-img", default=False, help="Generate images using puzzle image parsers."
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=Path("images/visualizations"),
    show_default=True,
    help="Directory where generated images are stored when --img is used.",
)
@click.option(
    "--kwarg",
    "puzzle_kwargs",
    multiple=True,
    help="Additional puzzle constructor keyword arguments as key=value pairs.",
)
def visualize_puzzle(
    puzzle_name: str,
    seed: int,
    img: bool,
    output_dir: Path,
    puzzle_kwargs: tuple[str, ...],
) -> None:
    puzzles = discover_puzzles()
    if puzzle_name not in puzzles:
        available = ", ".join(sorted(puzzles.keys()))
        raise click.BadParameter(
            f"Unknown puzzle '{puzzle_name}'. Available puzzles: {available}"
        )

    puzzle_class = puzzles[puzzle_name]
    parsed_kwargs: dict[str, object] = {}
    for item in puzzle_kwargs:
        if "=" not in item:
            raise click.BadParameter("--kwarg must be provided as key=value")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise click.BadParameter("--kwarg key cannot be empty")
        try:
            parsed_value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed_value = value
        parsed_kwargs[key] = parsed_value

    puzzle = puzzle_class(**parsed_kwargs)
    click.echo(f"Loaded puzzle: {puzzle!r}")

    key = jax.random.PRNGKey(seed)
    key, solve_key, state_key = jax.random.split(key, 3)
    solve_config = puzzle.get_solve_config(key=solve_key)
    init_state = puzzle.get_initial_state(solve_config, key=state_key)
    neighbours, costs = puzzle.get_neighbours(solve_config, init_state, filled=True)

    click.echo("\nInitial State:")
    click.echo(_format_state(puzzle, init_state))
    click.echo("\nAction overview:")

    action_labels = [puzzle.action_to_string(i) for i in range(puzzle.action_size)]
    np_costs = np.asarray(costs)
    finite_mask = np.isfinite(np_costs)

    for idx, label in enumerate(action_labels):
        cost_repr = "∞" if not finite_mask[idx] else f"{np_costs[idx]:.3f}"
        click.echo(f"[{idx:02d}] {label:20s} cost={cost_repr}")
        neighbour_state = _select_state(puzzle, neighbours, idx)
        click.echo(_format_state(puzzle, neighbour_state))
        click.echo("-")

    if img:
        img_parser = puzzle.get_img_parser()
        img_root = output_dir / puzzle_name

        init_path = _save_image(img_parser(init_state), img_root / "initial.png")
        click.echo(f"Saved initial state image -> {init_path}")

        for idx in range(puzzle.action_size):
            action_label = action_labels[idx]
            neighbour_state = _select_state(puzzle, neighbours, idx)
            neighbour_path = _save_image(
                img_parser(neighbour_state), img_root / f"{action_label}.png"
            )
            click.echo(f"Saved neighbour {action_label} image -> {neighbour_path}")


if __name__ == "__main__":
    visualize_puzzle()
