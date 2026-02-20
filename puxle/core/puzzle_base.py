from abc import ABC, abstractmethod
from typing import Any, Optional, TypeVar

import chex
import jax
import jax.numpy as jnp

from puxle.core.puzzle_state import FieldDescriptor, PuzzleState, state_dataclass
from puxle.utils.util import add_img_parser

T = TypeVar("T")


class Puzzle(ABC):
    """Abstract base class for all PuXle puzzle and planning environments.

    Every concrete puzzle subclass must:

    1. Set ``action_size`` (number of possible actions).
    2. Implement :meth:`define_state_class` to return a ``@state_dataclass``-decorated class.
    3. Implement :meth:`get_actions`, :meth:`is_solved`, :meth:`get_solve_config`,
       :meth:`get_initial_state`, :meth:`get_string_parser`, and :meth:`get_img_parser`.

    The base class handles JIT compilation of core methods and provides
    default batch and inverse-neighbour logic.

    Attributes:
        action_size: Number of discrete actions available in this puzzle.
        State: The ``@state_dataclass`` class representing states (set during ``__init__``).
        SolveConfig: The ``@state_dataclass`` class representing goal configurations
            (set during ``__init__``).
    """

    action_size: int = None

    @property
    def inverse_action_map(self) -> Optional[jnp.ndarray]:
        """
        Returns an array mapping each action to its inverse, or None if not defined.
        If implemented, this method should return a jnp.ndarray where `map[i]` is the
        inverse of action `i`. This is used by the default `get_inverse_neighbours`
        to automatically calculate inverse transitions for reversible puzzles.

        For example, if action 0 is 'up' and 1 is 'down', then the map
        should contain inverse_action_map[0] = 1 and inverse_action_map[1] = 0.

        If this is not implemented or returns None, `get_inverse_neighbours` will raise
        a NotImplementedError.
        """
        return None

    @property
    def is_reversible(self) -> bool:
        """
        Indicates whether the puzzle is fully reversible through the inverse_action_map.
        This is true if an inverse_action_map is provided.
        Puzzles with custom, non-symmetric inverse logic (like Sokoban)
        should override this to return False.
        """
        return self.inverse_action_map is not None

    class State(PuzzleState):
        pass

    class SolveConfig(PuzzleState):
        pass

    def define_solve_config_class(self) -> PuzzleState:
        """Return the ``@state_dataclass`` class used for goal/solve configuration.

        The default implementation creates a ``SolveConfig`` with a single
        ``TargetState`` field.  Override this when the goal representation
        requires additional fields (e.g., a goal mask for PDDL domains).

        Returns:
            A ``@state_dataclass`` class describing the solve configuration.
        """
        @state_dataclass
        class SolveConfig:
            TargetState: FieldDescriptor.scalar(dtype=self.State)

            def __str__(self, **kwargs):
                return self.TargetState.str(**kwargs)

        return SolveConfig

    @abstractmethod
    def define_state_class(self) -> PuzzleState:
        """Return the ``@state_dataclass`` class used for puzzle states.

        Subclasses **must** implement this method.  The returned class should
        use :class:`FieldDescriptor` to declare its fields.

        Returns:
            A ``@state_dataclass`` class describing the puzzle state.
        """
        pass

    @property
    def has_target(self) -> bool:
        """
        This function should return a boolean that indicates whether the environment has a target state or not.
        """
        return "TargetState" in self.SolveConfig.__annotations__.keys()

    @property
    def only_target(self) -> bool:
        """
        This function should return a boolean that indicates whether the environment has only a target state or not.
        """
        return self.has_target and len(self.SolveConfig.__annotations__.keys()) == 1

    @property
    def fixed_target(self) -> bool:
        """
        This function should return a boolean that indicates whether the target state is fixed and doesn't change.
        default is only_target, but if the target state is not fixed, you should redefine this function.
        """
        return self.only_target

    def __init__(self, **kwargs):
        """Initialise the puzzle.

        Subclass constructors **must** call ``super().__init__(**kwargs)``
        after setting ``action_size`` and any instance attributes needed by
        :meth:`define_state_class` / :meth:`data_init`.

        This method:

        1. Calls :meth:`data_init` for optional dataset loading.
        2. Builds ``State`` and ``SolveConfig`` classes.
        3. JIT-compiles core methods (``get_neighbours``, ``is_solved``, etc.).
        4. Validates ``action_size`` and pre-computes the inverse-action permutation.

        Raises:
            ValueError: If ``action_size`` is still ``None`` after subclass init.
        """
        super().__init__()
        self.data_init()

        self.State = self.define_state_class()
        self.SolveConfig = self.define_solve_config_class()
        self.State = add_img_parser(self.State, self.get_img_parser())
        self.SolveConfig = add_img_parser(self.SolveConfig, self.get_solve_config_img_parser())

        self.get_initial_state = jax.jit(self.get_initial_state)
        self.get_solve_config = jax.jit(self.get_solve_config)
        self.get_inits = jax.jit(self.get_inits)
        self.get_actions = jax.jit(self.get_actions)
        self.batched_get_actions = jax.jit(self.batched_get_actions, static_argnums=(4,))
        self.get_neighbours = jax.jit(self.get_neighbours)
        self.batched_get_neighbours = jax.jit(self.batched_get_neighbours, static_argnums=(3,))
        self.is_solved = jax.jit(self.is_solved)
        self.batched_is_solved = jax.jit(self.batched_is_solved, static_argnums=(2,))

        if self.action_size is None:
            raise ValueError(
                f"{self.__class__.__name__} must define `action_size` before calling Puzzle.__init__"
            )

        inv_map = self.inverse_action_map
        if inv_map is not None:
            # _inverse_action_permutation is an array of indices such that
            # the i-th inverse neighbour is neighbours[_inverse_action_permutation[i]]
            # where neighbours are the forward neighbours from get_neighbours.
            self._inverse_action_permutation = inv_map
        else:
            self._inverse_action_permutation = None

    def data_init(self):
        """Hook for loading datasets or heavy resources during init.

        Called *before* ``define_state_class()``.  Override in puzzles that
        require external data (e.g., Sokoban level files).
        """
        pass

    def get_solve_config_string_parser(self) -> callable:
        """Return a callable that renders a ``SolveConfig`` as a string.

        The default implementation delegates to :meth:`get_string_parser` on
        ``solve_config.TargetState``.  Override when the solve config
        contains fields beyond ``TargetState``.

        Returns:
            A function ``(solve_config: SolveConfig) -> str``.
        """
        assert self.only_target, (
            "You should redefine this function, because this function is only for target state"
            f"has_target: {self.has_target}, only_target: {self.only_target}"
            f"SolveConfig: {self.SolveConfig.__annotations__.keys()}"
        )
        stringparser_state = self.get_string_parser()

        def stringparser(solve_config: "Puzzle.SolveConfig") -> str:
            return stringparser_state(solve_config.TargetState)

        return stringparser

    @abstractmethod
    def get_string_parser(self) -> callable:
        """Return a callable that renders a ``State`` as a human-readable string.

        Returns:
            A function ``(state: State, **kwargs) -> str``.
        """
        pass

    def get_solve_config_img_parser(self) -> callable:
        """Return a callable that renders a ``SolveConfig`` as an image array.

        The default implementation delegates to :meth:`get_img_parser` on
        ``solve_config.TargetState``.  Override when the solve config
        contains fields beyond ``TargetState``.

        Returns:
            A function ``(solve_config: SolveConfig) -> jnp.ndarray``.
        """
        assert self.only_target, (
            "You should redefine this function, because this function is only for target state"
            f"has_target: {self.has_target}, only_target: {self.only_target}"
            f"SolveConfig: {self.SolveConfig.__annotations__.keys()}"
        )
        imgparser_state = self.get_img_parser()

        def imgparser(solve_config: "Puzzle.SolveConfig") -> jnp.ndarray:
            return imgparser_state(solve_config.TargetState)

        return imgparser

    @abstractmethod
    def get_img_parser(self) -> callable:
        """Return a callable that renders a ``State`` as an image (NumPy/JAX array).

        Returns:
            A function ``(state: State, **kwargs) -> jnp.ndarray``
            producing an ``(H, W, 3)`` RGB image.
        """
        pass

    def get_data(self, key=None) -> Any:
        """Optionally sample or return puzzle-specific data used by ``get_inits``.

        Args:
            key: Optional JAX PRNG key for stochastic data selection.

        Returns:
            Puzzle-specific data (e.g., a Sokoban level index) or ``None``.
        """
        return None

    @abstractmethod
    def get_solve_config(self, key=None, data=None) -> SolveConfig:
        """Build and return a goal / solve configuration.

        Args:
            key: Optional JAX PRNG key for stochastic goal generation.
            data: Optional puzzle-specific data from :meth:`get_data`.

        Returns:
            A ``SolveConfig`` instance describing the puzzle objective.
        """
        pass

    @abstractmethod
    def get_initial_state(self, solve_config: SolveConfig, key=None, data=None) -> State:
        """Build and return the initial (scrambled) state for a given goal.

        Args:
            solve_config: The goal configuration for this episode.
            key: Optional JAX PRNG key for random scrambling.
            data: Optional puzzle-specific data from :meth:`get_data`.

        Returns:
            A ``State`` instance representing the starting position.
        """
        pass

    def get_inits(self, key=None) -> tuple[SolveConfig, State]:
        """Convenience method returning ``(solve_config, initial_state)``.

        Splits ``key`` internally to call :meth:`get_data`,
        :meth:`get_solve_config`, and :meth:`get_initial_state`.

        Args:
            key: JAX PRNG key.

        Returns:
            A ``(SolveConfig, State)`` tuple.
        """
        datakey, solveconfigkey, initkey = jax.random.split(key, 3)
        data = self.get_data(datakey)
        solve_config = self.get_solve_config(solveconfigkey, data)
        return solve_config, self.get_initial_state(solve_config, initkey, data)

    def batched_get_actions(self, solve_configs: SolveConfig, states: State, actions: chex.Array, filleds: bool = True, multi_solve_config: bool = False) -> tuple[State, chex.Array]:
        """Vectorised version of :meth:`get_actions`.

        Args:
            solve_configs: Solve configurations — single or batched.
            states: Batch of states with leading batch dimension.
            actions: Batch of action indices.
            filleds: Whether to fill invalid moves (broadcast scalar or batch).
            multi_solve_config: If ``True``, ``solve_configs`` has the same
                batch dimension as ``states``; otherwise a single config is
                broadcast.

        Returns:
            ``(next_states, costs)`` with shapes matching the input batch.
        """
        if multi_solve_config:
            return jax.vmap(self.get_actions, in_axes=(0, 0, 0, 0))(
                solve_configs, states, actions, filleds
            )
        else:
            return jax.vmap(self.get_actions, in_axes=(None, 0, 0, 0))(
                solve_configs, states, actions, filleds
            )

    @abstractmethod
    def get_actions(self, solve_config: SolveConfig, state: State, actions: chex.Array, filled: bool = True) -> tuple[State, chex.Array]:
        """Apply a single action to a state and return the result.

        Args:
            solve_config: Current goal configuration.
            state: Current puzzle state.
            actions: Scalar action index.
            filled: If ``True``, invalid actions return the same state with
                ``jnp.inf`` cost; if ``False``, behaviour is puzzle-specific.

        Returns:
            ``(next_state, cost)`` where ``cost`` is ``jnp.inf`` for invalid moves.
        """
        pass

    def batched_get_neighbours(
        self,
        solve_configs: SolveConfig,
        states: State,
        filleds: bool = True,
        multi_solve_config: bool = False,
    ) -> tuple[State, chex.Array]:
        """Vectorised version of :meth:`get_neighbours`.

        Args:
            solve_configs: Solve configurations — single or batched.
            states: Batch of states with leading batch dimension.
            filleds: Whether to fill invalid moves.
            multi_solve_config: If ``True``, ``solve_configs`` has the same
                batch dimension as ``states``.

        Returns:
            ``(neighbour_states, costs)`` with shapes
            ``(action_size, batch, ...)`` and ``(action_size, batch)``.
        """
        if multi_solve_config:
            return jax.vmap(self.get_neighbours, in_axes=(0, 0, 0), out_axes=(1, 1))(
                solve_configs, states, filleds
            )
        else:
            return jax.vmap(self.get_neighbours, in_axes=(None, 0, 0), out_axes=(1, 1))(
                solve_configs, states, filleds
            )

    def get_neighbours(
        self, solve_config: SolveConfig, state: State, filled: bool = True
    ) -> tuple[State, chex.Array]:
        """Compute all successor states for every action.

        Equivalent to calling :meth:`get_actions` for each action index and
        stacking the results.  Invalid actions produce ``cost = jnp.inf``
        and the original state.

        Args:
            solve_config: Current goal configuration.
            state: Current puzzle state.
            filled: If ``True``, invalid actions are filled with
                ``(state, jnp.inf)``.

        Returns:
            ``(neighbour_states, costs)`` where ``neighbour_states`` has
            shape ``(action_size, ...)`` and ``costs`` has shape
            ``(action_size,)``.
        """
        actions = jnp.arange(self.action_size)
        states, costs = jax.vmap(self.get_actions, in_axes=(None, None, 0, None), out_axes=(0, 0))(
            solve_config, state, actions, filled
        )
        return states, costs

    def batched_is_solved(
        self, solve_configs: SolveConfig, states: State, multi_solve_config: bool = False
    ) -> bool:
        """Vectorised version of :meth:`is_solved`.

        Args:
            solve_configs: Solve configurations — single or batched.
            states: Batch of states.
            multi_solve_config: If ``True``, solve configs are batched
                alongside states.

        Returns:
            Boolean array of shape ``(batch,)``.
        """
        if multi_solve_config:
            return jax.vmap(self.is_solved, in_axes=(0, 0))(solve_configs, states)
        else:
            return jax.vmap(self.is_solved, in_axes=(None, 0))(solve_configs, states)

    @abstractmethod
    def is_solved(self, solve_config: SolveConfig, state: State) -> bool:
        """
        This function should return True if the state is the target state.
        if the puzzle has multiple target states, this function should return
        True if the state is one of the target conditions.
        e.g sokoban puzzle has multiple target states. box's position should
        be the same as the target position but the player's position can be different.
        """
        pass

    def action_to_string(self, action: int) -> str:
        """Return a human-readable name for the given action index.

        Override in subclasses to provide meaningful names
        (e.g., ``"R"`` for right, ``"U'"`` for counter-clockwise).

        Args:
            action: Integer action index in ``[0, action_size)``.

        Returns:
            String representation of the action.
        """
        return f"action {action}"

    _DIRECTIONAL_LABELS = ("←", "→", "↑", "↓")

    @staticmethod
    def _directional_action_to_string(action: int) -> str:
        """Shared helper for puzzles using 4 directional actions (←→↑↓)."""
        if 0 <= action <= 3:
            return Puzzle._DIRECTIONAL_LABELS[action]
        raise ValueError(f"Invalid action: {action}")

    @staticmethod
    def _grid_visualize_format(size: int) -> str:
        """Build a box-drawing grid format string for an ``size × size`` board."""
        form = "┏━"
        for i in range(size):
            form += "━━" if i != size - 1 else "━━┓"
        form += "\n"
        for i in range(size):
            form += "┃ "
            for j in range(size):
                form += "{:s} "
            form += "┃\n"
        form += "┗━"
        for i in range(size):
            form += "━━" if i != size - 1 else "━━┛"
        return form

    def batched_hindsight_transform(self, solve_configs: SolveConfig, states: State) -> SolveConfig:
        """Vectorised version of :meth:`hindsight_transform`.

        Args:
            solve_configs: Batch of solve configurations.
            states: Batch of states to treat as new goals.

        Returns:
            Batch of updated ``SolveConfig`` instances.
        """
        return jax.vmap(self.hindsight_transform)(solve_configs, states)

    def solve_config_to_state_transform(
        self, solve_config: SolveConfig, key: jax.random.PRNGKey = None
    ) -> State:
        """Convert a ``SolveConfig`` into the corresponding target ``State``.

        The default implementation simply extracts ``solve_config.TargetState``.
        Override for puzzles whose goal is not a single target state.

        Args:
            solve_config: The goal configuration.
            key: Optional PRNG key (unused in default implementation).

        Returns:
            The target ``State`` encoded in the configuration.

        Raises:
            AssertionError: If the puzzle does not have a target state or
                the config has additional fields.
        """
        assert self.has_target, "This puzzle does not have target state"
        assert self.only_target, (
            "Default solve config to state transform is for only target state,"
            "you should redefine this function"
        )
        return solve_config.TargetState

    def hindsight_transform(self, solve_config: SolveConfig, states: State) -> SolveConfig:
        """Hindsight experience replay: rewrite the goal to match *states*.

        Creates a new ``SolveConfig`` whose ``TargetState`` equals the given
        state, enabling hindsight relabelling for training neural heuristics.

        Args:
            solve_config: Original solve configuration (used as template).
            states: State to embed as the new target.

        Returns:
            A new ``SolveConfig`` with ``TargetState`` replaced.

        Raises:
            AssertionError: If the puzzle goal is not a simple target state.
        """
        assert self.has_target, "This puzzle does not have target state"
        assert self.only_target, (
            "Default hindsight transform is for only target state,"
            "you should redefine this function"
        )
        solve_config = solve_config.replace(TargetState=states)
        return solve_config

    def get_inverse_neighbours(
        self, solve_config: SolveConfig, state: State, filled: bool = True
    ) -> tuple[State, chex.Array]:
        """
        This function should return inverse neighbours and the cost of the move.
        By default, it uses `inverse_action_map` to calculate inverse transitions
        for reversible puzzles. If `inverse_action_map` is not defined, this function
        will raise a NotImplementedError.

        For puzzles that are not reversible (e.g., Sokoban), this method must be
        overridden with a specific implementation.
        """
        if self._inverse_action_permutation is None:
            raise NotImplementedError(
                "This puzzle does not define an `inverse_action_map`. "
                "To use `get_inverse_neighbours`, you must either implement the map "
                "for a reversible puzzle or override this method for a non-reversible one."
            )

        neighbours, costs = self.get_neighbours(solve_config, state, filled)
        # The i-th inverse neighbour is the state from which applying action i leads to the current state.
        # This is found by permuting the forward neighbours using _inverse_action_permutation.
        permuted_neighbours = neighbours[self._inverse_action_permutation]
        permuted_costs = costs[self._inverse_action_permutation]

        return permuted_neighbours, permuted_costs

    def batched_get_inverse_neighbours(
        self,
        solve_configs: SolveConfig,
        states: State,
        filleds: bool = True,
        multi_solve_config: bool = False,
    ) -> tuple[State, chex.Array]:
        """Vectorised version of :meth:`get_inverse_neighbours`.

        Args:
            solve_configs: Solve configurations — single or batched.
            states: Batch of states.
            filleds: Whether to fill invalid moves.
            multi_solve_config: If ``True``, solve configs share the batch dim.

        Returns:
            ``(inverse_neighbour_states, costs)``.
        """
        if multi_solve_config:
            return jax.vmap(self.get_inverse_neighbours, in_axes=(0, 0, 0), out_axes=(1, 1))(
                solve_configs, states, filleds
            )
        else:
            return jax.vmap(self.get_inverse_neighbours, in_axes=(None, 0, 0), out_axes=(1, 1))(
                solve_configs, states, filleds
            )

    def _get_shuffled_state(
        self, solve_config: "Puzzle.SolveConfig", init_state: "Puzzle.State", key, num_shuffle
    ):
        """Generate a scrambled state by applying random actions.

        Uses a ``while_loop`` to apply ``num_shuffle`` (±1) random actions,
        avoiding immediate backtracking.

        Args:
            solve_config: Goal configuration (passed to ``get_neighbours``).
            init_state: State to start scrambling from (usually the solved state).
            key: JAX PRNG key.
            num_shuffle: Base number of random actions to apply.

        Returns:
            A scrambled ``State``.
        """
        key, subkey = jax.random.split(key)
        num_shuffle += jax.random.randint(
            subkey, (), 0, 2
        )  # add a random 1 to 0 to the initial shuffle.

        def cond_fun(loop_state):
            iteration_count, _, _, _ = loop_state
            return iteration_count < num_shuffle

        def body_fun(loop_state):
            iteration_count, current_state, previous_state, key = loop_state
            neighbor_states, costs = self.get_neighbours(solve_config, current_state, filled=True)
            old_eq = jax.vmap(lambda x, y: x == y, in_axes=(None, 0))(
                previous_state, neighbor_states
            )
            valid_mask = jnp.where(old_eq, 0.0, 1.0)

            valid_mask_sum = jnp.sum(valid_mask)
            probabilities = jax.lax.cond(
                valid_mask_sum > 0,
                lambda: valid_mask / valid_mask_sum,
                lambda: jnp.ones_like(costs) / costs.shape[0],
            )

            key, subkey = jax.random.split(key)
            idx = jax.random.choice(subkey, jnp.arange(costs.shape[0]), p=probabilities)
            next_state = neighbor_states[idx]
            return (iteration_count + 1, next_state, current_state, key)

        _, final_state, _, _ = jax.lax.while_loop(
            cond_fun, body_fun, (0, init_state, init_state, key)
        )
        return final_state

    def __repr__(self):
        state_fields = list(self.State.__annotations__.keys())
        solve_config_fields = list(self.SolveConfig.__annotations__.keys())
        return (
            f"Puzzle({self.__class__.__name__}, "
            f"action_size={self.action_size}, "
            f"state_fields={state_fields}, "
            f"solve_config_fields={solve_config_fields})"
        )
