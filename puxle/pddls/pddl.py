import os
from collections.abc import Callable
from typing import Optional, Union

import chex
import jax.numpy as jnp
import pddl
from pddl.core import Domain, Problem

from puxle.core.puzzle_base import Puzzle
from puxle.core.puzzle_state import PuzzleState

from .formatting import (
    action_to_string as fmt_action_to_string,
)
from .formatting import (
    build_label_color_maps,
    build_solve_config_string_parser,
    build_state_string_parser,
)
from .grounding import ground_actions, ground_predicates
from .masks import build_goal_mask, build_initial_state, build_masks
from .state_defs import build_solve_config_class, build_state_class
from .type_system import collect_type_hierarchy, extract_objects_by_type


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
                raise ValueError(f"Failed to parse PDDL domain file: {e}") from e
        else:
            self.domain_file = None
            self.domain = domain

        if isinstance(problem, str):
            self.problem_file = problem
            try:
                self.problem = pddl.parse_problem(problem)
            except Exception as e:
                raise ValueError(f"Failed to parse PDDL problem file: {e}") from e
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
        data_dir = os.path.normpath(
            os.path.join(base_dir, "..", "data", "pddls", domain)
        )

        domain_path = os.path.abspath(os.path.join(data_dir, "domain.pddl"))

        if problem is None and problem_basename is None:
            raise ValueError(
                "Provide `problem` or `problem_basename` to locate a problem file."
            )

        if problem is None and problem_basename is not None:
            problem = f"{problem_basename}.pddl"

        if not problem.endswith(".pddl"):
            problem = f"{problem}.pddl"

        problem_path = os.path.abspath(os.path.join(data_dir, "problems", problem))

        return cls(domain=domain_path, problem=problem_path, **kwargs)

    def data_init(self) -> None:
        """Initialize PDDL data: ground atoms and actions, build masks."""
        type_hierarchy = collect_type_hierarchy(self.domain)
        objects_by_type = extract_objects_by_type(
            self.problem, type_hierarchy, domain=self.domain
        )

        self.grounded_atoms, atom_to_idx = ground_predicates(
            self.domain.predicates, objects_by_type, type_hierarchy
        )
        self.num_atoms = len(self.grounded_atoms)

        self.grounded_actions, _ = ground_actions(
            self.domain.actions, objects_by_type, type_hierarchy
        )
        self.num_actions = len(self.grounded_actions)
        self.action_size = self.num_actions

        self.pre_mask, self.pre_neg_mask, self.add_mask, self.del_mask = build_masks(
            self.grounded_actions, atom_to_idx, self.num_atoms
        )
        self.init_state = build_initial_state(self.problem, atom_to_idx, self.num_atoms)
        self.goal_mask = build_goal_mask(self.problem, atom_to_idx, self.num_atoms)

        _, self._label_text_color_map = build_label_color_maps(self.domain)

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
        pre_neg = self.pre_neg_mask[action]  # Added pre_neg
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

    def get_string_parser(self) -> Callable:
        """Return string parser for states. If a solve_config is provided, annotate goal atoms."""

        return build_state_string_parser(self)

    def get_img_parser(self) -> Callable:
        """Return image parser for states. If a solve_config is provided, annotate goal atoms."""

        def img_parser(
            state: "PDDL.State", solve_config: "PDDL.SolveConfig" = None, **kwargs
        ):
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
                            else jnp.array(
                                [1.0, 1.0, 0.0]
                            )  # yellow for goal not yet satisfied
                        )
                    else:
                        # Green for true atoms, red for false
                        color = (
                            jnp.array([0.0, 1.0, 0.0])
                            if atoms[i]
                            else jnp.array([1.0, 0.0, 0.0])
                        )
                    img = img.at[row, col].set(color)

            return img

        return img_parser

    def action_to_string(self, action: int, colored: bool = True) -> str:
        """Return string representation of action (delegated)."""
        return fmt_action_to_string(
            self.grounded_actions,
            action,
            self._label_text_color_map,
            colored,
        )

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

    def get_solve_config_string_parser(self) -> Callable:
        """Return string parser for solve config with goal mask."""

        return build_solve_config_string_parser(self)

    def get_solve_config_img_parser(self) -> Callable:
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
                        jnp.array([0.0, 0.0, 1.0])
                        if goal_mask[i]
                        else jnp.array([0.5, 0.5, 0.5])
                    )
                    img = img.at[row, col].set(color)

            return img

        return img_parser

    def state_to_atom_set(self, state: "PDDL.State") -> set[str]:
        """Convert state to set of true atom strings for testing."""
        s = state.unpacked_atoms
        return {self.grounded_atoms[i] for i in range(self.num_atoms) if bool(s[i])}

    def static_predicate_profile(
        self, state: "PDDL.State", pred_name: str
    ) -> list[bool]:
        """Get truth values of all grounded atoms for a predicate in given state."""
        s = state.unpacked_atoms
        vals = []
        for i, atom in enumerate(self.grounded_atoms):
            # parse predicate name from "(pred arg1 arg2 ...)"
            p = atom[1:].split(" ")[0]
            if p == pred_name:
                vals.append(bool(s[i]))
        return vals
