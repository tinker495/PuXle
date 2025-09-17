from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import jax
import numpy as np


def _state_key(state) -> bytes:
    return np.asarray(state.atoms).tobytes()

from puxle.pddls.pddl import PDDL


@dataclass
class ValidationResult:
    passed: bool
    explored_states: int
    depth_limit: int


def bfs_validate(domain_path: str | Path, problem_path: str | Path, depth_limit: int = 6) -> ValidationResult:
    """Run a bounded BFS to check reachability of the goal state."""
    env = PDDL(str(domain_path), str(problem_path))
    rng = jax.random.PRNGKey(0)
    solve_config, initial_state = env.get_inits(rng)

    if bool(env.is_solved(solve_config, initial_state)):
        return ValidationResult(True, 1, depth_limit)

    queue = deque([(initial_state, 0)])
    visited = {_state_key(initial_state)}
    explored = 0

    while queue:
        state, depth = queue.popleft()
        explored += 1
        if depth >= depth_limit:
            continue
        neighbours, costs = env.get_neighbours(solve_config, state, filled=True)
        costs_np = np.array(costs)
        applicable = np.where(np.isfinite(costs_np))[0]
        if applicable.size == 0:
            continue
        for idx in applicable:
            next_state = jax.tree_util.tree_map(lambda x: x[idx], neighbours)
            key = _state_key(next_state)
            if key in visited:
                continue
            if bool(env.is_solved(solve_config, next_state)):
                return ValidationResult(True, explored + 1, depth_limit)
            visited.add(key)
            queue.append((next_state, depth + 1))

    return ValidationResult(False, explored, depth_limit)
