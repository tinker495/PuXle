import os
from typing import Dict, List, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import pddl

from puxle.core.puzzle_base import Puzzle
from puxle.core.puzzle_state import FieldDescriptor, PuzzleState, state_dataclass
from puxle.utils.util import from_uint8, to_uint8


class PDDL(Puzzle):
    """
    PDDL wrapper for PuXle that supports STRIPS subset:
    - Positive preconditions only
    - Add/delete effects (no conditional effects)
    - Conjunctive positive goals
    - Typed objects
    """

    def __init__(self, domain_file: str, problem_file: str, **kwargs):
        """
        Initialize PDDL puzzle from domain and problem files.

        Args:
            domain_file: Path to PDDL domain file
            problem_file: Path to PDDL problem file
        """
        self.domain_file = domain_file
        self.problem_file = problem_file

        # Parse PDDL files
        try:
            self.domain = pddl.parse_domain(domain_file)
            self.problem = pddl.parse_problem(problem_file)
        except Exception as e:
            raise ValueError(f"Failed to parse PDDL files: {e}")

        super().__init__(**kwargs)

    @classmethod
    def from_preset(
        cls,
        domain: str,
        problem: Optional[str] = None,
        *,
        problem_basename: Optional[str] = None,
        **kwargs,
    ) -> "PDDL":
        """Create a PDDL instance by resolving absolute paths to data under `puxle/data/pddls/`.

        This mirrors the absolute-path loading style used by puzzles like Sokoban.

        Args:
            domain: Domain folder name under `puxle/data/pddls/` (e.g., "blocksworld").
            problem: Problem filename within `problems/` (with or without .pddl extension).
            problem_basename: Alternative to `problem`; basename without extension in `problems/`.

        Returns:
            PDDL: Initialized PDDL environment.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.normpath(os.path.join(base_dir, "..", "data", "pddls", domain))

        domain_path = os.path.abspath(os.path.join(data_dir, "domain.pddl"))

        if problem is None and problem_basename is None:
            raise ValueError("Provide `problem` or `problem_basename` to locate a problem file.")

        if problem is None and problem_basename is not None:
            problem = f"{problem_basename}.pddl"

        if not problem.endswith(".pddl"):
            problem = f"{problem}.pddl"

        problem_path = os.path.abspath(os.path.join(data_dir, "problems", problem))

        return cls(domain_file=domain_path, problem_file=problem_path, **kwargs)

    def data_init(self):
        """Initialize PDDL data: ground atoms and actions, build masks."""
        # Extract objects by type
        self.objects_by_type = self._extract_objects_by_type()

        # Ground predicates to atoms
        self.grounded_atoms, self.atom_to_idx = self._ground_predicates()
        self.num_atoms = len(self.grounded_atoms)

        # Ground actions
        self.grounded_actions, self.action_to_idx = self._ground_actions()
        self.num_actions = len(self.grounded_actions)

        # Build masks for JAX operations
        self._build_masks()

        # Set action size for Puzzle base class
        self.action_size = self.num_actions

    def _extract_objects_by_type(self) -> Dict[str, List[str]]:
        """Extract objects grouped by their types."""
        objects_by_type = {}

        # Handle untyped objects
        if not hasattr(self.problem, "objects") or not self.problem.objects:
            objects_by_type["object"] = []
        else:
            # Handle different object container types
            if isinstance(self.problem.objects, dict):
                # If objects is a dict mapping type -> object list
                for obj_type, obj_list in self.problem.objects.items():
                    if obj_type not in objects_by_type:
                        objects_by_type[obj_type] = []
                    for obj in obj_list:
                        obj_name = getattr(obj, "name", str(obj))
                        objects_by_type[obj_type].append(obj_name)
            else:
                # If objects is a set-like container (frozenset, set, list)
                for obj in self.problem.objects:
                    obj_name = getattr(obj, "name", str(obj))
                    # Extract type from type_tags
                    if hasattr(obj, "type_tags") and obj.type_tags:
                        obj_type = list(obj.type_tags)[0]
                    else:
                        obj_type = "object"  # Default type

                    if obj_type not in objects_by_type:
                        objects_by_type[obj_type] = []
                    objects_by_type[obj_type].append(obj_name)

        return objects_by_type

    def _ground_predicates(self) -> Tuple[List[str], Dict[str, int]]:
        """Ground all predicates to create atom universe."""
        grounded_atoms = []
        atom_to_idx = {}

        # Get all predicates from domain
        predicates = self.domain.predicates

        for predicate in predicates:
            pred_name = predicate.name
            # Extract parameter types from terms
            param_types = []
            for term in predicate.terms:
                if hasattr(term, "type_tags") and term.type_tags:
                    param_types.append(list(term.type_tags)[0])
                else:
                    param_types.append("object")  # Default type

            # Generate all type-consistent object combinations
            type_combinations = self._get_type_combinations(param_types)

            for obj_combination in type_combinations:
                # Create grounded atom string
                atom_str = f"({pred_name} {' '.join(obj_combination)})"
                grounded_atoms.append(atom_str)
                atom_to_idx[atom_str] = len(grounded_atoms) - 1

        return grounded_atoms, atom_to_idx

    def _get_type_combinations(self, param_types: List[str]) -> List[List[str]]:
        """Get all valid object combinations for given parameter types."""
        if not param_types:
            return [[]]

        combinations = []
        first_type = param_types[0]
        remaining_types = param_types[1:]

        # Get objects of the first type
        available_objects = self.objects_by_type.get(first_type, [])

        # Fallback to all objects if type not found (matching reference implementation)
        if not available_objects:
            all_objects = []
            for obj_list in self.objects_by_type.values():
                all_objects.extend(obj_list)
            available_objects = all_objects

        # Recursively get combinations for remaining types
        sub_combinations = self._get_type_combinations(remaining_types)

        for obj in available_objects:
            for sub_combo in sub_combinations:
                combinations.append([obj] + sub_combo)

        return combinations

    def _ground_actions(self) -> Tuple[List[Dict], Dict[str, int]]:
        """Ground all actions to create action universe."""
        grounded_actions = []
        action_to_idx = {}

        for action in self.domain.actions:
            action_name = action.name
            # Extract parameter types from parameters
            param_types = []
            for param in action.parameters:
                if hasattr(param, "type_tags") and param.type_tags:
                    param_types.append(list(param.type_tags)[0])
                else:
                    param_types.append("object")  # Default type

            # Generate all type-consistent parameter combinations
            param_combinations = self._get_type_combinations(param_types)

            for param_combo in param_combinations:
                # Get parameter names for this action
                param_names = [param.name for param in action.parameters]

                # Create grounded action
                grounded_action = {
                    "name": action_name,
                    "parameters": param_combo,
                    "preconditions": self._ground_formula(
                        action.precondition, param_combo, param_names
                    ),
                    "effects": self._ground_effects(action.effect, param_combo, param_names),
                }

                action_str = f"({action_name} {' '.join(param_combo)})"
                grounded_actions.append(grounded_action)
                action_to_idx[action_str] = len(grounded_actions) - 1

        return grounded_actions, action_to_idx

    def _ground_formula(
        self, formula, param_substitution: List[str], param_names: List[str]
    ) -> List[str]:
        """Ground a formula (precondition) with parameter substitution."""
        if formula is None:
            return []

        # Handle simple atomic formulas
        if hasattr(formula, "name"):
            pred_name = formula.name
            args = [getattr(arg, "name", str(arg)) for arg in formula.terms]

            # Apply parameter substitution
            substituted_args = []
            for arg in args:
                if arg in param_names:
                    # This is a parameter, substitute
                    param_idx = param_names.index(arg)
                    if param_idx < len(param_substitution):
                        substituted_args.append(param_substitution[param_idx])
                    else:
                        substituted_args.append(arg)  # Keep original if index out of bounds
                else:
                    # This is a constant
                    substituted_args.append(arg)

            return [f"({pred_name} {' '.join(substituted_args)})"]

        # Handle compound formulas (AND, OR, etc.)
        if hasattr(formula, "parts"):
            grounded_parts = []
            for part in formula.parts:
                grounded_parts.extend(self._ground_formula(part, param_substitution, param_names))
            return grounded_parts

        # Handle And/Or objects
        if hasattr(formula, "operands"):
            grounded_parts = []
            for operand in formula.operands:
                grounded_parts.extend(
                    self._ground_formula(operand, param_substitution, param_names)
                )
            return grounded_parts

        return []

    def _ground_effects(
        self, effect, param_substitution: List[str], param_names: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Ground effects with parameter substitution, return (add_effects, delete_effects)."""
        if effect is None:
            return [], []

        add_effects = []
        delete_effects = []

        # Handle simple effects
        if hasattr(effect, "predicate"):
            effect_str = self._ground_formula(effect, param_substitution, param_names)[0]
            if hasattr(effect, "negated") and effect.negated:
                delete_effects.append(effect_str)
            else:
                add_effects.append(effect_str)

        # Handle Not effects
        elif hasattr(effect, "argument"):
            # This is a Not object
            effect_str = self._ground_formula(effect.argument, param_substitution, param_names)[0]
            delete_effects.append(effect_str)

        # Handle compound effects
        elif hasattr(effect, "parts"):
            for part in effect.parts:
                part_add, part_delete = self._ground_effects(part, param_substitution, param_names)
                add_effects.extend(part_add)
                delete_effects.extend(part_delete)

        # Handle And/Or effects
        elif hasattr(effect, "operands"):
            for operand in effect.operands:
                part_add, part_delete = self._ground_effects(
                    operand, param_substitution, param_names
                )
                add_effects.extend(part_add)
                delete_effects.extend(part_delete)

        # Handle Predicate effects (positive literals)
        elif hasattr(effect, "name") and hasattr(effect, "terms"):
            effect_str = self._ground_formula(effect, param_substitution, param_names)[0]
            add_effects.append(effect_str)

        return add_effects, delete_effects

    def _build_masks(self):
        """Build JAX arrays for precondition, add, and delete masks."""
        # Initialize masks
        pre_mask = jnp.zeros((self.num_actions, self.num_atoms), dtype=jnp.bool_)
        add_mask = jnp.zeros((self.num_actions, self.num_atoms), dtype=jnp.bool_)
        del_mask = jnp.zeros((self.num_actions, self.num_atoms), dtype=jnp.bool_)

        # Fill masks based on grounded actions
        for action_idx, action in enumerate(self.grounded_actions):
            # Preconditions
            for precondition in action["preconditions"]:
                if precondition in self.atom_to_idx:
                    atom_idx = self.atom_to_idx[precondition]
                    pre_mask = pre_mask.at[action_idx, atom_idx].set(True)

            # Add effects
            for add_effect in action["effects"][0]:  # add_effects
                if add_effect in self.atom_to_idx:
                    atom_idx = self.atom_to_idx[add_effect]
                    add_mask = add_mask.at[action_idx, atom_idx].set(True)

            # Delete effects
            for del_effect in action["effects"][1]:  # delete_effects
                if del_effect in self.atom_to_idx:
                    atom_idx = self.atom_to_idx[del_effect]
                    del_mask = del_mask.at[action_idx, atom_idx].set(True)

        self.pre_mask = pre_mask
        self.add_mask = add_mask
        self.del_mask = del_mask

        # Build initial state and goal mask
        self._build_initial_state()
        self._build_goal_mask()

    def _build_initial_state(self):
        """Build initial state as boolean array."""
        init_state = jnp.zeros(self.num_atoms, dtype=jnp.bool_)

        # Set initial facts to True
        for fact in self.problem.init:
            fact_str = (
                f"({fact.name} {' '.join([getattr(arg, 'name', str(arg)) for arg in fact.terms])})"
            )
            if fact_str in self.atom_to_idx:
                atom_idx = self.atom_to_idx[fact_str]
                init_state = init_state.at[atom_idx].set(True)

        self.init_state = init_state

    def _build_goal_mask(self):
        """Build goal mask for conjunctive positive goals."""
        goal_mask = jnp.zeros(self.num_atoms, dtype=jnp.bool_)

        # Extract goal conditions
        goal_conditions = self._extract_goal_conditions(self.problem.goal)

        for condition in goal_conditions:
            if condition in self.atom_to_idx:
                atom_idx = self.atom_to_idx[condition]
                goal_mask = goal_mask.at[atom_idx].set(True)

        self.goal_mask = goal_mask

    def _extract_goal_conditions(self, goal) -> List[str]:
        """Extract atomic conditions from goal formula."""
        if goal is None:
            return []

        # Handle simple atomic goals
        if hasattr(goal, "name"):
            return [
                f"({goal.name} {' '.join([getattr(arg, 'name', str(arg)) for arg in goal.terms])})"
            ]

        # Handle compound goals (AND, OR, etc.)
        if hasattr(goal, "parts"):
            conditions = []
            for part in goal.parts:
                conditions.extend(self._extract_goal_conditions(part))
            return conditions

        return []

    def define_state_class(self) -> PuzzleState:
        """Define state class with packed atoms."""
        str_parser = self.get_string_parser()
        num_atoms = self.num_atoms

        # Calculate packed size
        packed_size = (self.num_atoms + 7) // 8  # Round up for bit packing
        packed_atoms = to_uint8(self.init_state, 1)

        @state_dataclass
        class State:
            atoms: FieldDescriptor[jnp.uint8, (packed_size,), packed_atoms]

            def __str__(self, **kwargs):
                return str_parser(self, **kwargs)

            @property
            def packed(self):
                return State(atoms=to_uint8(self.unpacked_atoms, 1))

            @property
            def unpacked(self):
                return State(atoms=from_uint8(self.atoms, (num_atoms,), 1))

            @property
            def unpacked_atoms(self):
                """Get boolean view of atoms for convenience."""
                return from_uint8(self.atoms, (num_atoms,), 1)

        return State

    def define_solve_config_class(self) -> PuzzleState:
        """Define solve config with goal mask instead of target state."""

        @state_dataclass
        class SolveConfig:
            GoalMask: FieldDescriptor[jnp.bool_, (self.num_atoms,), self.goal_mask]

            def __str__(self, **kwargs):
                return f"Goal: {jnp.sum(self.GoalMask)} atoms"

        return SolveConfig

    def get_initial_state(
        self, solve_config: Puzzle.SolveConfig, key=None, data=None
    ) -> "PDDL.State":
        """Return initial state."""
        return self.State(atoms=to_uint8(self.init_state, 1))

    def get_solve_config(self, key=None, data=None) -> Puzzle.SolveConfig:
        """Return solve config with goal mask."""
        return self.SolveConfig(GoalMask=self.goal_mask)

    def get_neighbours(
        self, solve_config: Puzzle.SolveConfig, state: "PDDL.State", filled: bool = True
    ) -> tuple["PDDL.State", chex.Array]:
        """Get all possible next states and costs using JAX vectorization."""
        # Unpack state to boolean array
        s = state.unpacked_atoms

        # Compute applicability: app[i] = True if action i is applicable
        app = jnp.all(jnp.logical_or(~self.pre_mask, s[None, :]), axis=1)

        # Compute next states: s_next[i] = (s & ~del_mask[i]) | add_mask[i]
        s_next = jnp.logical_or(jnp.logical_and(s[None, :], ~self.del_mask), self.add_mask)

        # Pack next states
        def pack_state(atoms_bool):
            return self.State(atoms=to_uint8(atoms_bool, 1))

        next_states = jax.vmap(pack_state)(s_next)

        # Set costs: 1.0 for applicable actions, inf for non-applicable
        costs = jnp.where(app, 1.0, jnp.inf)

        # If filled=False, return all costs as inf (contract with other puzzles)
        # Use jax.lax.cond to handle the conditional in JAX-traceable way
        costs = jax.lax.cond(
            filled,
            lambda: costs,  # If filled=True, keep original costs
            lambda: jnp.full_like(costs, jnp.inf),  # If filled=False, all inf
        )

        return next_states, costs

    def is_solved(self, solve_config: Puzzle.SolveConfig, state: "PDDL.State") -> bool:
        """Check if state satisfies goal conditions."""
        s = state.unpacked_atoms
        goal_mask = solve_config.GoalMask

        # Check if all goal atoms are true: all(~goal_mask | s)
        return jnp.all(jnp.logical_or(~goal_mask, s))

    def get_string_parser(self) -> callable:
        """Return string parser for states."""

        def parser(state: "PDDL.State", **kwargs):
            atoms = state.unpacked_atoms
            true_atoms = [self.grounded_atoms[i] for i in range(self.num_atoms) if atoms[i]]

            if len(true_atoms) <= 10:
                return f"State: {', '.join(true_atoms)}"
            else:
                return f"State: {len(true_atoms)} atoms (showing first 10): {', '.join(true_atoms[:10])}..."

        return parser

    def get_img_parser(self) -> callable:
        """Return image parser for states."""

        def img_parser(state: "PDDL.State", **kwargs):
            # Create a simple visualization: grid showing atom values
            atoms = state.unpacked_atoms

            # Create a square grid
            grid_size = int(jnp.ceil(jnp.sqrt(self.num_atoms)))
            img = jnp.zeros((grid_size, grid_size, 3), dtype=jnp.float32)

            for i in range(self.num_atoms):
                row = i // grid_size
                col = i % grid_size
                if row < grid_size and col < grid_size:
                    # Green for true atoms, red for false
                    color = jnp.array([0.0, 1.0, 0.0]) if atoms[i] else jnp.array([1.0, 0.0, 0.0])
                    img = img.at[row, col].set(color)

            return img

        return img_parser

    def action_to_string(self, action: int) -> str:
        """Return string representation of action."""
        if 0 <= action < len(self.grounded_actions):
            action_data = self.grounded_actions[action]
            return f"({action_data['name']} {' '.join(action_data['parameters'])})"
        return f"action_{action}"

    @property
    def has_target(self) -> bool:
        """Override to handle goal mask instead of target state."""
        return True

    @property
    def only_target(self) -> bool:
        """Override to handle goal mask instead of target state."""
        return False

    @property
    def fixed_target(self) -> bool:
        """Override to handle goal mask instead of target state."""
        return True

    def get_solve_config_string_parser(self) -> callable:
        """Return string parser for solve config with goal mask."""

        def parser(solve_config: "PDDL.SolveConfig", **kwargs):
            goal_count = jnp.sum(solve_config.GoalMask)
            return f"Goal mask with {goal_count} required atoms"

        return parser

    def get_solve_config_img_parser(self) -> callable:
        """Return image parser for solve config with goal mask."""

        def img_parser(solve_config: "PDDL.SolveConfig", **kwargs):
            # Create a simple visualization of goal mask
            goal_mask = solve_config.GoalMask

            # Create a square grid
            grid_size = int(jnp.ceil(jnp.sqrt(self.num_atoms)))
            img = jnp.zeros((grid_size, grid_size, 3), dtype=jnp.float32)

            for i in range(self.num_atoms):
                row = i // grid_size
                col = i % grid_size
                if row < grid_size and col < grid_size:
                    # Blue for goal atoms, gray for non-goal
                    color = (
                        jnp.array([0.0, 0.0, 1.0]) if goal_mask[i] else jnp.array([0.5, 0.5, 0.5])
                    )
                    img = img.at[row, col].set(color)

            return img

        return img_parser

    def state_to_atom_set(self, state: "PDDL.State") -> set[str]:
        """Convert state to set of true atom strings for testing."""
        s = state.unpacked_atoms
        return {self.grounded_atoms[i] for i in range(self.num_atoms) if bool(s[i])}

    def static_predicate_profile(self, state: "PDDL.State", pred_name: str) -> list[bool]:
        """Get truth values of all grounded atoms for a predicate in given state."""
        s = state.unpacked_atoms
        vals = []
        for i, atom in enumerate(self.grounded_atoms):
            # parse predicate name from "(pred arg1 arg2 ...)"
            p = atom[1:].split(" ")[0]
            if p == pred_name:
                vals.append(bool(s[i]))
        return vals
