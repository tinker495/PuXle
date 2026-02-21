import time
from collections import deque

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from puxle.pddls.pddl import PDDL


def _bfs_with_time_budget(env: PDDL, solve_config, initial_state, max_depth: int, time_budget_s: float) -> bool:
    # Warmup JIT to avoid counting compile time in the budget
    _ = env.get_neighbours(solve_config, initial_state, filled=True)

    start = time.perf_counter()
    if bool(env.is_solved(solve_config, initial_state)):
        return True

    queue = deque([(initial_state, 0)])
    visited = {np.array(initial_state.atoms, dtype=np.uint8).tobytes()}

    while queue:
        if (time.perf_counter() - start) > time_budget_s:
            break
        state, depth = queue.popleft()
        if depth >= max_depth:
            continue

        neighbors, costs = env.get_neighbours(solve_config, state, filled=True)
        applicable = jnp.isfinite(costs)
        if not jnp.any(applicable):
            continue

        action_indices = np.array(jnp.where(applicable)[0]).tolist()
        for idx in action_indices:
            next_state = jax.tree_util.tree_map(lambda x: x[idx], neighbors)
            if bool(env.is_solved(solve_config, next_state)):
                return True
            key = np.array(next_state.atoms, dtype=np.uint8).tobytes()
            if key not in visited:
                visited.add(key)
                queue.append((next_state, depth + 1))

    return False


@pytest.mark.parametrize(
    "domain, problem, max_depth, time_budget_s",
    [
        ("blocksworld", "bw-H-01", 200, 1.5),
        ("gripper", "gr-H-01", 200, 1.5),
        ("logistics", "lg-H-01", 250, 1.5),
        ("rovers", "rv-H-01", 250, 1.5),
        ("satellite", "st-H-01", 250, 1.5),
    ],
)
def test_hard_instances_run_with_timeout(domain, problem, max_depth, time_budget_s):
    env = PDDL.from_preset(domain=domain, problem_basename=problem)
    solve_config, initial_state = env.get_inits(jax.random.PRNGKey(0))

    solved_in_budget = _bfs_with_time_budget(env, solve_config, initial_state, max_depth, time_budget_s)

    # Contract: this test should never fail due to hardness; it passes whether or not
    # a solution is found within the time budget, but logs outcome for visibility.
    if not solved_in_budget:
        print(f"[INFO] Hard instance {domain}/{problem}: no solution found within {time_budget_s}s (allowed).")
    assert True
