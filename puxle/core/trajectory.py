from typing import Optional

import chex


@chex.dataclass
class PuzzleTrajectory:
    """
    A unified JAX PyTree dataclass representing extracted puzzle environments,
    shuffled states, paths, and their corresponding algorithmic generation costs.
    """

    solve_configs: chex.Array
    states: chex.Array
    move_costs: chex.Array
    move_costs_tm1: chex.Array
    actions: chex.Array
    action_costs: chex.Array

    # Hindsight wrapper auxiliary indices
    parent_indices: Optional[chex.Array] = None
    trajectory_indices: Optional[chex.Array] = None
    step_indices: Optional[chex.Array] = None
