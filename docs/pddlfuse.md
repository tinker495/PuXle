# PDDLFuse: Diverse Planning Domain Generator

**PDDLFuse** is a tool for generating new, complex planning problems by fusing existing PDDL domains and stochastically modifying them. Integrated into `PuXle`, it allows researchers to easily build diverse environments for testing the robustness and generalization capabilities of AI planners.

## Core Features

### 1. N-Way Domain Fusion (Disjoint Union)
Merges multiple source domains (e.g., Blocksworld, Gripper, Logistics) into a single monolithic domain.
- **Disjoint Fusion**: Predicates and actions from each domain are prefixed (e.g., `dom0_on`, `dom1_move`) to ensure no name collisions occur. This guarantees that domains remain structurally independent unless explicitly connected via stochastic modifications.
- **Type Hierarchy Integration**: Analyzes and unions the type structures of each domain (merged by name).
- **Result**: Creates an environment where multiple "worlds" coexist (e.g., blocks and trucks in the same state space), interacting only through stochastically added effects.

### 2. Stochastic Action Modification
Beyond simple merging, PDDLFuse randomly alters the dynamics of the domain to create new challenges.
- **Add Preconditions**: Makes actions harder to execute by requiring more conditions.
- **Add Effects**: Introduces unintended side-effects when actions are executed.
- **Remove Preconditions/Effects**: Simplifies actions by removing constraints or effects, increasing the state space connectivity.
- **Negation Support**: Adds requirements for conditions to be *false*, or effects that delete existing facts.
- **Reversibility Enforcement**: Ensures that if an action deletes a predicate (e.g., `not P`), there exists at least one action that adds it back (`P`), preventing dead-ends in the state space.

### 3. Automated Problem Generation
Automatically generates solvable problem files for the newly created domain.
- **Random Walk Simulation**: Explores the state space by simulating valid random actions starting from an initial state.
- **Goal Setting**: Sets a subset of the reached state as the goal, guaranteeing that a solution exists.

---

## Usage

### 1. Using Python API (`fuse_and_load`)
Load directly into a JAX-based `PuXle` environment for immediate simulation or reinforcement learning.

```python
from puxle.pddls.fusion.api import fuse_and_load, FusionParams

# 1. Prepare list of domain file paths
domains = [
    "puxle/data/pddls/blocksworld/domain.pddl",
    "puxle/data/pddls/gripper/domain.pddl"
]

# 2. Configure parameters
params = FusionParams(
    prob_add_pre=0.1,  # 10% probability to add a precondition
    prob_add_eff=0.05, # 5% probability to add an effect
    prob_neg=0.2,      # 20% probability that added term is negated
    rev_flag=True,     # Ensure reversibility (prevent dead-ends)
    seed=42            # Seed for reproducibility
)

# 3. Fuse and load domain
env = fuse_and_load(domains, params=params, name="blocks-gripper-fused")

# 4. Use PuXle environment
print(f"Fused Actions: {env.num_actions}")
print(f"Fused Atoms: {env.num_atoms}")

# Get initial state
state = env.get_initial_state(env.problem)
```

### 2. Generating Benchmarks (`generate_benchmark`)
Generate a set of `.pddl` files on disk to evaluate external PDDL planners (e.g., Fast Downward).

```python
from puxle.pddls.fusion.api import generate_benchmark, FusionParams

generate_benchmark(
    domain_paths=["domain1.pddl", "domain2.pddl"],
    output_dir="./benchmarks/my_fused_set",
    count=50,                      # Number of problems to generate
    difficulty_depth_range=(10, 50), # Difficulty (solution depth range)
    params=FusionParams(rev_flag=True, seed=123)
)
```

### 3. Iterative Fusion (`iterative_fusion`)
Create highly complex domains by recursively fusing generated domains.

```python
from puxle.pddls.fusion.api import iterative_fusion

complex_domain = iterative_fusion(
    base_domains=domains,
    depth=2, # Fuse base -> result, then result -> result
    params=params
)
```

### 4. Varying Depth Benchmark (`generate_benchmark_with_varying_depth`)
Generate benchmarks organized by fusion depth for systematic complexity studies.

```python
from puxle.pddls.fusion.api import generate_benchmark_with_varying_depth

generate_benchmark_with_varying_depth(
    base_domains=["blocksworld.pddl", "gripper.pddl"],
    output_dir="./benchmarks/depth_study",
    depth_range=(1, 3),        # Generate for depths 1, 2, and 3
    problems_per_depth=10,
    params=params
)
# Output structure:
# ./benchmarks/depth_study/
#   depth_1/domain.pddl, problem_00.pddl, ...
#   depth_2/domain.pddl, problem_00.pddl, ...
#   depth_3/domain.pddl, problem_00.pddl, ...
```

---

## Configuration Parameters (`FusionParams`)

Controlled via `puxle.pddls.fusion.action_modifier.FusionParams`.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `prob_add_pre` | float | 0.1 | Probability of adding a random precondition to each action. |
| `prob_add_eff` | float | 0.1 | Probability of adding a random effect to each action. |
| `prob_rem_pre` | float | 0.05 | Probability of removing an existing precondition. |
| `prob_rem_eff` | float | 0.05 | Probability of removing an existing effect. |
| `prob_neg` | float | 0.1 | Probability that an added precondition/effect is negated (Not). |
| `rev_flag` | bool | True | If True, ensures that for every deleted predicate, there exists an adder action. |
| `seed` | int | 42 | Random seed for generation. |

---

## Architecture

- **`puxle/pddls/fusion/`**
    - `domain_fusion.py`: Logic for merging domain objects (types, constants, predicates, actions).
    - `action_modifier.py`: Logic for stochastically modifying action preconditions/effects.
    - `problem_generator.py`: Generates valid problems via symbolic simulation.
    - `validator.py`: internal validator for checking structural consistency of fused domains.
    - `api.py`: User-facing functions.

## Notes
- Fused domains can become logically complex. Generated problems may range from trivial to very difficult (or contain dead-ends if modification is aggressive, though the generator ensures the *goal* is reachable from start).
- Fixing the `seed` ensures identical domain modifications and problem sets are generated.
