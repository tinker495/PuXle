from collections import deque

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from puxle.pddls.pddl import PDDL


def _hash_state(state) -> bytes:
    # Use packed atoms for a compact, hashable representation
    return np.array(state.atoms, dtype=np.uint8).tobytes()


def _bfs_reaches_goal(env: PDDL, solve_config, initial_state, max_depth: int) -> bool:
    if bool(env.is_solved(solve_config, initial_state)):
        return True

    queue = deque([(initial_state, 0)])
    visited = {_hash_state(initial_state)}

    while queue:
        state, depth = queue.popleft()
        if depth >= max_depth:
            continue

        neighbors, costs = env.get_neighbours(solve_config, state, filled=True)
        applicable = jnp.isfinite(costs)
        if not jnp.any(applicable):
            continue

        # Expand all applicable actions
        action_indices = np.array(jnp.where(applicable)[0]).tolist()
        for idx in action_indices:
            next_state = jax.tree_util.tree_map(lambda x: x[idx], neighbors)
            if bool(env.is_solved(solve_config, next_state)):
                return True
            key = _hash_state(next_state)
            if key not in visited:
                visited.add(key)
                queue.append((next_state, depth + 1))

    return False


@pytest.mark.parametrize(
    "domain, problem, max_depth",
    [
        ("blocksworld", "bw-S-01", 6),
        ("blocksworld", "bw-S-02", 8),
        ("blocksworld", "bw-S-03", 12),
        ("gripper", "gr-S-01", 8),
        ("gripper", "gr-S-02", 12),
        ("gripper", "gr-S-03", 16),
    ],
)
def test_presets_bfs_solves_within_bounds(domain, problem, max_depth):
    env = PDDL.from_preset(domain=domain, problem_basename=problem)
    solve_config, initial_state = env.get_inits(jax.random.PRNGKey(0))

    assert _bfs_reaches_goal(
        env, solve_config, initial_state, max_depth
    ), f"Preset {domain}/{problem} should be solvable within {max_depth} steps"


@pytest.mark.parametrize(
    "domain, problem",
    [
        ("blocksworld", "bw-S-01"),
        ("gripper", "gr-S-01"),
    ],
)
def test_presets_jit_and_batch(domain, problem):
    env = PDDL.from_preset(domain=domain, problem_basename=problem)
    rng = jax.random.PRNGKey(42)
    solve_config, initial_state = env.get_inits(rng)

    # JIT single
    jitted_get_neighbours = jax.jit(env.get_neighbours)
    neighbors, costs = jitted_get_neighbours(solve_config, initial_state, filled=True)
    assert neighbors is not None and costs is not None
    assert len(costs) == env.action_size

    jitted_is_solved = jax.jit(env.is_solved)
    solved = jitted_is_solved(solve_config, initial_state)
    assert isinstance(solved, (bool, jnp.bool_)) or (
        hasattr(solved, "dtype") and solved.dtype == jnp.bool_
    )

    # Batched
    keys = jax.random.split(rng, 4)
    solve_configs = jax.vmap(lambda k: env.get_solve_config(k))(keys)
    initial_states = jax.vmap(lambda sc, k: env.get_initial_state(sc, k))(solve_configs, keys)

    solved_mask = env.batched_is_solved(solve_configs, initial_states, multi_solve_config=True)
    assert solved_mask.shape[0] == 4

    filleds = jnp.array([True, True, True, True])
    batched_neighbors, batched_costs = env.batched_get_neighbours(
        solve_configs, initial_states, filleds=filleds, multi_solve_config=True
    )
    assert batched_neighbors is not None and batched_costs is not None
    assert batched_costs.shape[0] == env.action_size
    assert batched_costs.shape[1] == 4
