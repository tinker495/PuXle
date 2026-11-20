# PDDL Planning Domains in PuXle

PuXle wraps STRIPS-compliant PDDL domains and problems as JAX-native `Puzzle` environments via `puxle.PDDL`. Grounded atoms are bit-packed into `uint8` state arrays (1 bit per atom via Xtructure), and action preconditions and effects are compiled into boolean masks stored in `SolveConfig.InstanceContext`.

> **Note**: PDDL support covers the STRIPS subset described below. Disjunctive preconditions, negative goals, and conditional or quantified effects are rejected at grounding time.

## Supported PDDL Subset

- **Conjunctive preconditions** (positive and negative literals)
- **Add/delete effects** (unconditional)
- **Conjunctive positive goals** (stored as a partial goal mask in `SolveConfig.GoalSpec.GoalMask`)
- **Typed objects and constants** with type-hierarchy and `(either ...)` union resolution
- **Equality / inequality constraints** (`=`, `not (= ...)`) in action preconditions
- **Automatic grounding** of predicates and actions

Bundled preset domains include:

- **Blocksworld** (`blocksworld`): Stacking and unstacking blocks on a table
- **Gripper** (`gripper`): Robot room navigation and dual-gripper ball manipulation
- **Logistics** (`logistics`): Multi-city truck and airplane package delivery
- **Rovers** (`rovers`): Waypoint navigation, rock sampling, and image transmission
- **Satellite** (`satellite`): Instrument calibration, slewing, and observation planning

## Quick Start

Initialize `PDDL` from bundled preset names, `PDDLPresets`, file paths, or parsed `pddl.core.Domain` / `Problem` objects:

```python
import jax
import jax.numpy as jnp
from puxle import PDDL, PDDLPresets

# From a curated preset or bundled domain/problem names
pddl_env = PDDL(**PDDLPresets.BLOCKSWORLD_S01)
# Or from local PDDL files:
# pddl_env = PDDL(domain="path/to/domain.pddl", problem="path/to/problem.pddl")

# Discover all bundled domains and problem names
print(PDDL.list_sources())

# Get solve configuration and initial state
solve_config, initial_state = pddl_env.get_inits(jax.random.PRNGKey(0))

# Get all successor states and transition costs (1.0 for applicable, inf otherwise)
neighbors, costs = pddl_env.get_neighbours(solve_config, initial_state)

# Check if goal is satisfied
is_solved = pddl_env.is_solved(solve_config, initial_state)
print(f"Goal satisfied: {is_solved}")

# Render current state and goal progress
print(initial_state.str(solve_config=solve_config))

# Inspect applicable action names
for i, cost in enumerate(costs):
    if cost < jnp.inf:
        action_name = pddl_env.action_to_string(i)
        print(f"Applicable action {i}: {action_name}")
```

## Inspecting Grounded Atoms and Predicates

```python
# Access grounded atom and action counts
print(f"Number of grounded atoms: {pddl_env.num_atoms}")
print(f"Number of grounded actions: {pddl_env.num_actions}")
print(f"Grounded atoms: {pddl_env.grounded_atoms}")

# Check predicate truth values in a state
predicate_values = pddl_env.static_predicate_profile(initial_state, "on")
print(f"Truth values for 'on' predicate: {predicate_values}")

# Convert state to the set of true atom strings
true_atoms = pddl_env.state_to_atom_set(initial_state)
print(f"True atoms in state: {true_atoms}")
```

## Key Methods

- `PDDL.list_sources()`: Return a dictionary mapping bundled domain names to available problem stems.
- `action_to_string(action_idx, colored=True)`: Format a grounded action as a colored label or `(name params)`.
- `state_to_atom_set(state)`: Convert a packed `PDDL.State` into the set of true grounded atom strings.
- `static_predicate_profile(state, pred_name)`: Return truth values for all grounded atoms of `pred_name`.

## See Also

- Adapter source: `puxle/puzzles/adapters/pddl/pddl.py`
- Presets source: `puxle/puzzles/adapters/pddl/presets.py`
- Formatting and domain diagrams: `puxle/puzzles/adapters/pddl/formatting.py`
