from collections import deque

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from puxle.pddls.pddl import PDDL
from puxle.utils.util import to_uint8


def _hash_state(state) -> bytes:
    # Use packed atoms for a compact, hashable representation
    return np.array(state.atoms, dtype=np.uint8).tobytes()


def _bfs_reaches_goal(env: PDDL, solve_config, initial_state, max_depth: int) -> bool:
    if bool(env.is_solved(solve_config, initial_state)):
        return True

    queue = deque([(initial_state, 0)])
    visited = {_hash_state(initial_state)}
    
    # Process states in batches to leverage vectorization and reduce JAX dispatch overhead
    BATCH_SIZE = 512

    while queue:
        # Early exit if we reached the depth limit (BFS monotonic property)
        if queue[0][1] >= max_depth:
            return False

        # 1. Collect a batch of states
        batch_states = []
        batch_depths = []
        
        while len(batch_states) < BATCH_SIZE and queue:
            if queue[0][1] >= max_depth:
                break
            s, d = queue.popleft()
            batch_states.append(s)
            batch_depths.append(d)
        
        if not batch_states:
            return False

        # Stack states (List[State] -> State(Batch...))
        stacked_state = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *batch_states)
        
        # Prepare filleds mask (required by batched_get_neighbours)
        # We need to pass a boolean array of shape (batch_size,) because batched_get_neighbours vmaps over it
        filleds_batch = jnp.full((len(batch_states),), True)

        # 2. Get neighbors in batch
        # env.batched_get_neighbours is pre-jitted in Puzzle.__init__
        neighbors, costs = env.batched_get_neighbours(
            solve_config, 
            stacked_state, 
            filleds=filleds_batch
        )
        
        # 3. Check for solutions in batch
        # Flatten neighbors to (Batch*Actions, ...) for solved check
        flat_neighbors = jax.tree_util.tree_map(
            lambda x: x.reshape((-1,) + x.shape[2:]), 
            neighbors
        )
        
        # env.batched_is_solved is pre-jitted
        solved_mask = env.batched_is_solved(solve_config, flat_neighbors)

        if jnp.any(solved_mask):
            return True

        # 4. Filter and process new states on CPU
        # Bulk convert to numpy to avoid repeated device-host transfers
        neighbors_atoms_np = np.array(flat_neighbors.atoms)
        costs_np = np.array(costs).flatten()
        
        # Filter valid transitions (finite cost)
        applicable_indices = np.where(np.isfinite(costs_np))[0]
        
        if len(applicable_indices) == 0:
            continue
            
        # Calculate depths for new states
        parent_indices = applicable_indices // env.action_size
        new_depths = np.array(batch_depths)[parent_indices] + 1
        
        # Filter by max_depth
        valid_mask = new_depths < max_depth
        final_indices = applicable_indices[valid_mask]
        final_depths = new_depths[valid_mask]
        
        if len(final_indices) == 0:
            continue
            
        final_atoms = neighbors_atoms_np[final_indices]
        
        # Add to queue
        for i, atom_row in enumerate(final_atoms):
            key = atom_row.tobytes()
            if key not in visited:
                visited.add(key)
                # Store as State object
                ns = env.State(atoms=atom_row)
                queue.append((ns, int(final_depths[i])))

    return False


@pytest.mark.parametrize(
    "domain, problem, max_depth",
    [
        ("blocksworld", "bw-S-01", 6),
        ("blocksworld", "bw-S-02", 10),
        ("blocksworld", "bw-S-03", 16),
        ("blocksworld", "bw-S-04", 24),
        ("gripper", "gr-S-01", 8),
        ("gripper", "gr-S-02", 12),
        ("gripper", "gr-S-03", 16),
        ("gripper", "gr-S-04", 22),
        ("logistics", "lg-S-01", 12),
        ("logistics", "lg-S-02", 18),
        ("logistics", "lg-S-03", 22),
        ("logistics", "lg-S-04", 30),
        ("rovers", "rv-S-01", 10),
        ("rovers", "rv-S-02", 16),
        ("rovers", "rv-S-03", 22),
        ("rovers", "rv-S-04", 30),
        ("satellite", "st-S-01", 8),
        ("satellite", "st-S-02", 14),
        ("satellite", "st-S-03", 18),
        ("satellite", "st-S-04", 26),
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


@pytest.mark.parametrize(
    "domain, problem",
    [
        ("blocksworld", "bw-S-01"),
        ("blocksworld", "bw-S-02"),
        ("blocksworld", "bw-S-03"),
        ("blocksworld", "bw-S-04"),
        ("gripper", "gr-S-01"),
        ("gripper", "gr-S-02"),
        ("gripper", "gr-S-03"),
        ("gripper", "gr-S-04"),
        ("logistics", "lg-S-01"),
        ("logistics", "lg-S-02"),
        ("logistics", "lg-S-03"),
        ("logistics", "lg-S-04"),
        ("rovers", "rv-S-01"),
        ("rovers", "rv-S-02"),
        ("rovers", "rv-S-03"),
        ("rovers", "rv-S-04"),
        ("satellite", "st-S-01"),
        ("satellite", "st-S-02"),
        ("satellite", "st-S-03"),
        ("satellite", "st-S-04"),
    ],
)
def test_initial_state_not_solved(domain, problem):
    env = PDDL.from_preset(domain=domain, problem_basename=problem)
    solve_config, initial_state = env.get_inits(jax.random.PRNGKey(0))

    assert not bool(
        env.is_solved(solve_config, initial_state)
    ), f"Preset {domain}/{problem} should not be solved at the initial state"


@pytest.mark.parametrize(
    "domain, problem",
    [
        ("blocksworld", "bw-S-01"),
        ("gripper", "gr-S-01"),
        ("logistics", "lg-S-01"),
        ("rovers", "rv-S-01"),
        ("satellite", "st-S-01"),
    ],
)
def test_random_states_not_mostly_solved(domain, problem):
    env = PDDL.from_preset(domain=domain, problem_basename=problem)
    solve_config = env.get_solve_config()

    # Goals should be non-empty; otherwise any state would be trivially solved
    assert (
        int(jnp.sum(solve_config.GoalMask)) > 0
    ), f"Preset {domain}/{problem} has empty goal mask; any state would be solved"

    # Sample random boolean states and ensure the majority are NOT solved
    rng = jax.random.PRNGKey(123)
    batch_size = 64
    rand_bits = jax.random.bernoulli(rng, p=0.5, shape=(batch_size, env.num_atoms))

    def to_state(bits):
        return env.State(atoms=to_uint8(bits, 1))

    states = jax.vmap(to_state)(rand_bits)
    solved_mask = jax.vmap(lambda st: env.is_solved(solve_config, st))(states)
    proportion_solved = jnp.mean(solved_mask.astype(jnp.float32))

    # Guardrail: not more than 95% of random states should be solved
    assert float(proportion_solved) < 0.95, (
        f"Preset {domain}/{problem} appears trivially solved for most random states: "
        f"{float(proportion_solved):.2%} solved"
    )
