from __future__ import annotations

import jax
import numpy as np

from ._puzzles import PuzzleName, create_puzzle


def _action_aliases(labels: list[str]) -> dict[str, int]:
    aliases = {}
    arrows = {"↑": "w", "↓": "s", "←": "a", "→": "d"}
    for index, label in enumerate(labels):
        for arrow, key in arrows.items():
            if arrow in label:
                aliases[key] = index
    return aliases


def run(puzzle: PuzzleName, puzzle_args: str = "{}", seed: int = 0) -> int:
    """Play a puzzle interactively.

    Args:
        puzzle: Puzzle class or normalized name.
        puzzle_args: Puzzle constructor JSON object.
        seed: PRNG seed.
    """
    puzzle = create_puzzle(puzzle, puzzle_args)
    solve_config, state = puzzle.get_inits(jax.random.PRNGKey(seed))
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
