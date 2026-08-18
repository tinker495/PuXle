from __future__ import annotations

import argparse

import jax
import numpy as np

from ._puzzles import create_puzzle


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--puzzle", required=True, help="Puzzle class or normalized name."
    )
    parser.add_argument(
        "--puzzle-args", default="{}", help="Puzzle constructor JSON object."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.set_defaults(handler=run)


def _action_aliases(labels: list[str]) -> dict[str, int]:
    aliases = {}
    arrows = {"↑": "w", "↓": "s", "←": "a", "→": "d"}
    for index, label in enumerate(labels):
        for arrow, key in arrows.items():
            if arrow in label:
                aliases[key] = index
    return aliases


def run(args: argparse.Namespace) -> int:
    puzzle = create_puzzle(args.puzzle, args.puzzle_args)
    solve_config, state = puzzle.get_inits(jax.random.PRNGKey(args.seed))
    labels = [puzzle.action_to_string(index) for index in range(puzzle.action_size)]
    aliases = _action_aliases(labels)
    total_cost = 0.0

    print("Start")
    print(state.str(solve_config=solve_config))
    if puzzle.has_goal_data:
        print("Goal")
        print(solve_config)

    while not bool(np.asarray(puzzle.is_solved(solve_config, state))):
        neighbours, costs = puzzle.get_neighbours(solve_config, state)
        cost_values = np.asarray(costs)
        print(f"\nCost: {total_cost:g}")
        print(state.str(solve_config=solve_config))
        for index, (label, cost) in enumerate(
            zip(labels, cost_values, strict=True), start=1
        ):
            if np.isfinite(cost):
                print(f"  {index}: {label}")

        value = input("Action (number/WASD, q to quit): ").strip().lower()
        if value in {"q", "quit", "esc"}:
            return 0
        try:
            action = aliases[value] if value in aliases else int(value) - 1
            if (
                action < 0
                or action >= len(labels)
                or not np.isfinite(cost_values[action])
            ):
                raise ValueError
        except (ValueError, IndexError):
            print("Invalid or impossible action.")
            continue

        state = neighbours[action]
        total_cost += float(cost_values[action])

    print(state.str(solve_config=solve_config))
    print(f"Solved. Total cost: {total_cost:g}")
    return 0
