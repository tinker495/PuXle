import os
from typing import Dict, List, Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import pddl
from pddl.core import Domain, Problem
import termcolor

from puxle.core.puzzle_base import Puzzle
from puxle.core.puzzle_state import FieldDescriptor, PuzzleState, state_dataclass

# Refactored helpers
from .type_system import (
    collect_type_hierarchy,
    extract_objects_by_type as ts_extract_objects_by_type,
    select_most_specific_types as ts_select_most_specific_types,
)
from .grounding import ground_actions as gr_ground_actions, ground_predicates as gr_ground_predicates
from .masks import (
    build_goal_mask as mk_build_goal_mask,
    build_initial_state as mk_build_initial_state,
    build_masks as mk_build_masks,
    extract_goal_conditions as mk_extract_goal_conditions,
)
from .formatting import (
    action_to_string as fmt_action_to_string,
    build_label_color_maps,
    build_solve_config_string_parser,
    build_state_string_parser,
    split_atom as fmt_split_atom,
)
from .state_defs import build_solve_config_class, build_state_class


class PDDL(Puzzle):
    """PuXle wrapper that turns a PDDL domain + problem into a :class:`Puzzle`.

    Supports the **STRIPS** subset of PDDL:

    * Positive *and* negative preconditions (conjunctive).
    * Add / delete effects (no conditional or quantified effects).
    * Conjunctive positive goals.
    * Typed objects with type-hierarchy resolution.

    The state is a packed boolean vector over **grounded atoms**
    (1 bit per atom via xtructure bitpacking).  The solve-config
    stores a goal mask rather than a full target state, enabling
    partial-goal problems.

    The class delegates heavy lifting to helper modules:

    * :mod:`~puxle.pddls.type_system` — type hierarchy extraction.
    * :mod:`~puxle.pddls.grounding` — predicate and action grounding.
    * :mod:`~puxle.pddls.masks` — JAX mask construction.
    * :mod:`~puxle.pddls.formatting` — pretty-printing utilities.
    * :mod:`~puxle.pddls.state_defs` — dynamic state/solve-config classes.

    Args:
        domain: Path to a PDDL domain file **or** a ``pddl.core.Domain``
            object.
        problem: Path to a PDDL problem file **or** a ``pddl.core.Problem``
            object.
    """

    def __init__(
        self,
        domain: Union[str, Domain],
        problem: Union[str, Problem],
        **kwargs,
    ):
        """
        Initialize PDDL puzzle from domain and problem (files or objects).

        Args:
            domain: Path to PDDL domain file OR pddl.core.Domain object.
            problem: Path to PDDL problem file OR pddl.core.Problem object.
        """
        # Parse PDDL files if paths are provided
        if isinstance(domain, str):
            self.domain_file = domain
            try:
                self.domain = pddl.parse_domain(domain)
            except Exception as e:
                raise ValueError(f"Failed to parse PDDL domain file: {e}")
        else:
            self.domain_file = None
            self.domain = domain

        if isinstance(problem, str):
            self.problem_file = problem
            try:
                self.problem = pddl.parse_problem(problem)
            except Exception as e:
                raise ValueError(f"Failed to parse PDDL problem file: {e}")
        else:
            self.problem_file = None
            self.problem = problem

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

        return cls(domain=domain_path, problem=problem_path, **kwargs)

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
        self.pre_mask, self.pre_neg_mask, self.add_mask, self.del_mask = self._build_masks()
        self._build_initial_state()
        self._build_goal_mask()

        # Set action size for Puzzle base class
        self.action_size = self.num_actions

        # Build label->color map for visualization (actions and predicates)
        self._build_label_color_map()

    def _build_label_color_map(self) -> None:
        """Assign deterministic colors to action and predicate names (delegated)."""
        label_color_map, label_termcolor_map = build_label_color_maps(self.domain)
        self._label_color_map = label_color_map
        self._label_termcolor_map = label_termcolor_map

    @staticmethod
    def _split_atom(atom_str: str) -> tuple[str, list[str]]:
        """Split an atom string like "(pred a b)" into ("pred", ["a", "b"])."""
        return fmt_split_atom(atom_str)

    # -------------------------
    # Type hierarchy utilities
    # -------------------------
    def _collect_type_hierarchy(
        self,
    ) -> tuple[dict[str, str], dict[str, set[str]], dict[str, set[str]]]:
        """Extract type hierarchy from the domain (delegated)."""
        return collect_type_hierarchy(self.domain)

    def _select_most_specific_types(self, type_tags: set[str]) -> list[str]:
        """Keep the most specific types from a set of tags using the hierarchy (delegated)."""
        if not hasattr(self, "_type_hierarchy_cache"):
            self._type_hierarchy_cache = self._collect_type_hierarchy()
        return ts_select_most_specific_types(type_tags, self._type_hierarchy_cache)

    def _extract_objects_by_type(self) -> Dict[str, List[str]]:
        """Extract objects grouped by types, respecting hierarchy (delegated)."""
        if not hasattr(self, "_type_hierarchy_cache"):
            self._type_hierarchy_cache = self._collect_type_hierarchy()
        return ts_extract_objects_by_type(self.problem, self._type_hierarchy_cache, domain=self.domain)

    def _ground_predicates(self) -> Tuple[List[str], Dict[str, int]]:
        """Ground all predicates to create atom universe (delegated)."""
        if not hasattr(self, "_type_hierarchy_cache"):
            self._type_hierarchy_cache = self._collect_type_hierarchy()
        return gr_ground_predicates(
            getattr(self.domain, "predicates", []),
            self.objects_by_type,
            self._type_hierarchy_cache,
        )

    def _get_type_combinations(self, param_types: List[str]) -> List[List[str]]:
        """Deprecated: combinations are now handled in the delegated grounding module."""
        # Backward-compatible fallback using local logic (kept for safety if called elsewhere)
        if not param_types:
            return [[]]
        combinations: list[list[str]] = []
        first_type = param_types[0]
        remaining_types = param_types[1:]
        if isinstance(first_type, (list, tuple, set)):
            seen_union: set[str] = set()
            available_objects: list[str] = []
            for t in first_type:
                for o in self.objects_by_type.get(t, []):
                    if o not in seen_union:
                        seen_union.add(o)
                        available_objects.append(o)
        else:
            available_objects = list(self.objects_by_type.get(first_type, []))
        if not available_objects:
            return []
        sub_combinations = self._get_type_combinations(remaining_types)
        for obj in available_objects:
            for sub_combo in sub_combinations:
                combinations.append([obj] + sub_combo)
        return combinations

    def _ground_actions(self) -> Tuple[List[Dict], Dict[str, int]]:
        """Ground all actions to create action universe (delegated)."""
        if not hasattr(self, "_type_hierarchy_cache"):
            self._type_hierarchy_cache = self._collect_type_hierarchy()
        return gr_ground_actions(
            getattr(self.domain, "actions", []),
            self.objects_by_type,
            self._type_hierarchy_cache,
        )

    def _ground_formula(
        self, formula, param_substitution: List[str], param_names: List[str]
    ) -> List[str]:
        """Deprecated: delegated to grounding module; retained for safety."""
        from .grounding import _ground_formula as _gf

        return _gf(formula, param_substitution, param_names)

    def _ground_effects(
        self, effect, param_substitution: List[str], param_names: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Deprecated: delegated to grounding module; retained for safety."""
        from .grounding import _ground_effects as _ge

        return _ge(effect, param_substitution, param_names)

    def _build_masks(
        self,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Builds action masks (delegated)."""
        return mk_build_masks(self.grounded_actions, self.atom_to_idx, self.num_atoms)

    def _build_initial_state(self):
        """Build initial state as boolean array (delegated)."""
        self.init_state = mk_build_initial_state(self.problem, self.atom_to_idx, self.num_atoms)

    def _build_goal_mask(self):
        """Build goal mask for conjunctive positive goals (delegated)."""
        self.goal_mask = mk_build_goal_mask(self.problem, self.atom_to_idx, self.num_atoms)

    def _extract_goal_conditions(self, goal) -> List[str]:
        """Extract atomic conditions from goal formula (delegated)."""
        return mk_extract_goal_conditions(goal)

    def define_state_class(self) -> PuzzleState:
        """Define state class with packed atoms."""
        # Delegate to builder for clarity
        str_parser = self.get_string_parser()
        return build_state_class(self, self.num_atoms, self.init_state, str_parser)

    def define_solve_config_class(self) -> PuzzleState:
        """Define solve config with goal mask instead of target state."""
        # Delegate to builder for clarity
        str_parser = self.get_solve_config_string_parser()
        return build_solve_config_class(self, self.goal_mask, str_parser)

    def get_initial_state(
        self, solve_config: Puzzle.SolveConfig, key=None, data=None
    ) -> "PDDL.State":
        """Return initial state."""
        return self.State.from_unpacked(atoms=self.init_state)

    def get_solve_config(self, key=None, data=None) -> Puzzle.SolveConfig:
        """Return solve config with goal mask."""
        return self.SolveConfig(GoalMask=self.goal_mask)

    def get_actions(
        self,
        solve_config: Puzzle.SolveConfig,
        state: "PDDL.State",
        action: chex.Array,
        filled: bool = True,
    ) -> tuple["PDDL.State", chex.Array]:
        """
        Get the next state and cost for a given action using JAX.
        """
        # Unpack state to boolean array
        s = state.unpacked_atoms

        # Get masks for this action
        pre = self.pre_mask[action]
        pre_neg = self.pre_neg_mask[action] # Added pre_neg
        add = self.add_mask[action]
        dele = self.del_mask[action]

        # Check applicability
        # Positive preconditions: (~pre | s)
        # Negative preconditions: (~pre_neg | ~s)
        applicable = jnp.all(jnp.logical_or(~pre, s)) & jnp.all(
            jnp.logical_or(~pre_neg, ~s)
        )

        # Compute next state: (s & ~del) | add
        s_next = jnp.logical_or(jnp.logical_and(s, ~dele), add)

        # If action is inapplicable, keep original state
        s_next = jnp.where(applicable, s_next, s)

        next_state = state.set_unpacked(atoms=s_next)

        # Cost: 1.0 for applicable, inf otherwise
        cost = jnp.where(applicable, 1.0, jnp.inf)
        cost = jnp.where(filled, cost, jnp.inf)

        return next_state, cost

    def is_solved(self, solve_config: Puzzle.SolveConfig, state: "PDDL.State") -> bool:
        """Check if state satisfies goal conditions."""
        s = state.unpacked_atoms
        goal_mask = solve_config.GoalMask

        # Check if all goal atoms are true: all(~goal_mask | s)
        return jnp.all(jnp.logical_or(~goal_mask, s))

    def get_string_parser(self) -> callable:
        """Return string parser for states. If a solve_config is provided, annotate goal atoms."""

        return build_state_string_parser(self)

    def get_img_parser(self) -> callable:
        """Return image parser for states. If a solve_config is provided, annotate goal atoms."""

        def img_parser(state: "PDDL.State", solve_config: "PDDL.SolveConfig" = None, **kwargs):
            # Create a simple visualization: grid showing atom values
            atoms = state.unpacked_atoms

            # Optional goal context
            goal_mask = None
            if solve_config is not None and hasattr(solve_config, "GoalMask"):
                goal_mask = solve_config.GoalMask

            # Create a square grid
            grid_size = int(jnp.ceil(jnp.sqrt(self.num_atoms)))
            img = jnp.zeros((grid_size, grid_size, 3), dtype=jnp.float32)

            for i in range(self.num_atoms):
                row = i // grid_size
                col = i % grid_size
                if row < grid_size and col < grid_size:
                    if goal_mask is not None and bool(goal_mask[i]):
                        # Goal-aware coloring
                        color = (
                            jnp.array([0.0, 0.0, 1.0])  # blue for goal satisfied
                            if atoms[i]
                            else jnp.array([1.0, 1.0, 0.0])  # yellow for goal not yet satisfied
                        )
                    else:
                        # Green for true atoms, red for false
                        color = (
                            jnp.array([0.0, 1.0, 0.0]) if atoms[i] else jnp.array([1.0, 0.0, 0.0])
                        )
                    img = img.at[row, col].set(color)

            return img

        return img_parser

    def action_to_string(self, action: int, colored: bool = True) -> str:
        """Return string representation of action (delegated)."""
        return fmt_action_to_string(self.grounded_actions, action, getattr(self, "_label_termcolor_map", {}), colored)

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

        return build_solve_config_string_parser(self)

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
