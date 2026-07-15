import jax
import jax.numpy as jnp
import pytest
import xtructure.numpy as xnp
from xtructure import StructuredType
from xtructure.core.layout import get_type_layout

from puxle.pddls.pddl import PDDL
from puxle.puzzles.dotknot import DotKnot
from puxle.puzzles.maze import Maze
from puxle.puzzles.sokoban import Sokoban
from puxle.puzzles.tsp import TSP


@pytest.fixture(scope="module")
def pddl_env():
    return PDDL.from_preset("blocksworld", problem_basename="bw-S-01")


@pytest.fixture
def zero_mask_pddl_config(pddl_env):
    config = pddl_env.get_solve_config()
    return config.replace(
        InstanceContext=jax.tree_util.tree_map(jnp.zeros_like, config.InstanceContext)
    )


def test_pddl_uses_fixed_outer_shape_and_non_state_goal(pddl_env):
    assert get_type_layout(pddl_env.SolveConfig).field_names == (
        "InstanceContext",
        "GoalSpec",
    )
    assert pddl_env.GoalSpec is not pddl_env.State


def test_special_environment_property_matrix(pddl_env):
    maze = Maze(size=5)
    tsp = TSP(size=3)
    dotknot = DotKnot(size=4)

    assert (maze.has_target, maze.fixed_target) == (True, False)
    assert (tsp.has_target, tsp.fixed_target) == (False, False)
    assert (pddl_env.has_target, pddl_env.fixed_target) == (False, True)
    assert (dotknot.has_target, dotknot.fixed_target) == (False, False)
    assert maze.has_goal_data
    assert pddl_env.has_goal_data
    assert not tsp.has_goal_data
    assert not dotknot.has_goal_data


def test_maze_context_controls_transition_while_goal_controls_solved_predicate():
    maze = Maze(size=5)
    open_layout = jnp.zeros(maze.size**2, dtype=jnp.bool_)
    blocked_layout = open_layout.at[1 * maze.size + 2].set(True)
    state = maze.State(pos=jnp.array([1, 1], dtype=jnp.uint16))
    goal = maze.State(pos=jnp.array([1, 2], dtype=jnp.uint16))
    open_config = maze.SolveConfig(
        InstanceContext=maze.InstanceContext.from_unpacked(Maze=open_layout),
        GoalSpec=goal,
    )
    blocked_config = open_config.replace(
        InstanceContext=maze.InstanceContext.from_unpacked(Maze=blocked_layout)
    )

    open_state, open_cost = maze.get_actions(open_config, state, jnp.int32(1))
    blocked_state, blocked_cost = maze.get_actions(blocked_config, state, jnp.int32(1))

    assert open_state == goal
    assert open_cost == 1
    assert blocked_state == state
    assert jnp.isinf(blocked_cost)
    assert maze.is_solved(open_config, goal)
    assert maze.is_solved(blocked_config, goal)

    other_goal_config = open_config.replace(
        GoalSpec=maze.State(pos=jnp.array([4, 4], dtype=jnp.uint16))
    )
    other_state, other_cost = maze.get_actions(other_goal_config, state, jnp.int32(1))
    assert other_state == open_state
    assert other_cost == open_cost
    assert not maze.is_solved(other_goal_config, goal)


def test_hindsight_preserves_maze_and_sokoban_contexts():
    maze = Maze(size=5)
    maze_config, maze_state = maze.get_inits(jax.random.PRNGKey(0))
    maze_relabelled = maze.hindsight_transform(maze_config, maze_state)

    assert jnp.array_equal(
        maze_relabelled.InstanceContext.Maze,
        maze_config.InstanceContext.Maze,
    )
    assert maze_relabelled.GoalSpec == maze_state

    sokoban = Sokoban()
    sokoban_config, sokoban_state = sokoban.get_inits(jax.random.PRNGKey(0))
    sokoban_relabelled = sokoban.hindsight_transform(sokoban_config, sokoban_state)
    expected_goal = jnp.where(
        sokoban_state.board_unpacked == Sokoban.Object.PLAYER.value,
        Sokoban.Object.EMPTY.value,
        sokoban_state.board_unpacked,
    )

    assert sokoban_relabelled.InstanceContext is sokoban_config.InstanceContext
    assert sokoban_relabelled is not sokoban_config
    assert jnp.array_equal(sokoban_relabelled.GoalSpec.board_unpacked, expected_goal)


def test_tsp_uses_distance_matrix_and_euclidean_closing_leg():
    tsp = TSP(size=3)
    points = jnp.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]], dtype=jnp.float32)
    distance_matrix = jnp.array(
        [[0.0, 7.0, 8.0], [9.0, 0.0, 11.0], [12.0, 13.0, 0.0]],
        dtype=jnp.float32,
    )
    config = tsp.SolveConfig(
        InstanceContext=tsp.InstanceContext(
            points=points,
            distance_matrix=distance_matrix,
            start=jnp.uint16(0),
        ),
        GoalSpec=tsp.GoalSpec(),
    )
    first_state = tsp.State.from_unpacked(
        mask=jnp.array([True, False, False]), point=jnp.uint16(0)
    )
    _, move_cost = tsp.get_actions(config, first_state, jnp.int32(1))

    closing_state = tsp.State.from_unpacked(
        mask=jnp.array([True, True, False]), point=jnp.uint16(1)
    )
    _, closing_cost = tsp.get_actions(config, closing_state, jnp.int32(2))

    assert move_cost == distance_matrix[0, 1]
    assert closing_cost == distance_matrix[1, 2] + jnp.linalg.norm(
        points[0] - points[2]
    )


def test_tsp_empty_goal_spec_is_batch_neutral_under_jit_and_vmap():
    tsp = TSP(size=3)
    goal_spec = tsp.GoalSpec.default((2,))

    assert goal_spec.structured_type == StructuredType.SINGLE
    assert goal_spec.batch_shape == ()
    assert goal_spec[1].structured_type == StructuredType.SINGLE
    assert goal_spec.replace().structured_type == StructuredType.SINGLE
    assert jax.jit(lambda goal: goal.replace())(goal_spec).batch_shape == ()

    config, state = tsp.get_inits(jax.random.PRNGKey(0))
    configs = xnp.stack([config, config], axis=0)
    states = xnp.stack([state, state], axis=0)
    next_states, costs = jax.jit(jax.vmap(tsp.get_actions, in_axes=(0, 0, 0)))(
        configs, states, jnp.ones(2, dtype=jnp.int32)
    )

    assert configs.batch_shape == (2,)
    assert configs.GoalSpec.structured_type == StructuredType.SINGLE
    assert next_states.batch_shape == (2,)
    assert costs.shape == (2,)


def test_dotknot_all_leafless_config_stays_single_and_is_static_under_vmap():
    puzzle = DotKnot(size=4)
    config = puzzle.get_solve_config()
    state = puzzle.State.from_unpacked(board=jnp.zeros(puzzle.size**2, dtype=jnp.uint8))
    default_config = puzzle.SolveConfig.default((2,))

    assert not get_type_layout(puzzle.SolveConfig).leaves
    assert config.structured_type == StructuredType.SINGLE
    assert default_config.structured_type == StructuredType.SINGLE
    assert default_config.batch_shape == ()
    assert default_config[1].structured_type == StructuredType.SINGLE

    states = xnp.stack([state, state], axis=0)
    jitted_solved = jax.jit(puzzle.is_solved)(config, state)
    solved = jax.jit(jax.vmap(puzzle.is_solved, in_axes=(None, 0)))(config, states)

    assert jitted_solved.shape == ()
    assert solved.shape == (2,)


def test_pddl_goal_spec_does_not_affect_transition(pddl_env):
    config = pddl_env.get_solve_config()
    state = pddl_env.State.from_unpacked(
        atoms=jnp.zeros(pddl_env.num_atoms, dtype=jnp.bool_)
    )
    empty_goal = pddl_env.GoalSpec(GoalMask=jnp.zeros_like(config.GoalSpec.GoalMask))
    changed_goal = config.replace(GoalSpec=empty_goal)

    original_state, original_cost = pddl_env.get_actions(config, state, jnp.int32(0))
    changed_state, changed_cost = pddl_env.get_actions(
        changed_goal, state, jnp.int32(0)
    )

    assert changed_goal.InstanceContext is config.InstanceContext
    assert original_state == changed_state
    assert original_cost == changed_cost
    assert jnp.any(config.GoalSpec.GoalMask)
    assert not pddl_env.is_solved(config, state)
    assert pddl_env.is_solved(changed_goal, state)


@pytest.mark.parametrize(
    ("field", "atoms"),
    (("pre_mask", False), ("pre_neg_mask", True)),
)
def test_pddl_precondition_masks_are_read_from_instance_context(
    pddl_env, zero_mask_pddl_config, field, atoms
):
    state = pddl_env.State.from_unpacked(
        atoms=jnp.full(pddl_env.num_atoms, atoms, dtype=jnp.bool_)
    )
    _, base_cost = pddl_env.get_actions(zero_mask_pddl_config, state, jnp.int32(0))
    context = zero_mask_pddl_config.InstanceContext
    mask = getattr(context, field).at[0, 0].set(True)
    changed = zero_mask_pddl_config.replace(
        InstanceContext=context.replace(**{field: mask})
    )
    _, changed_cost = pddl_env.get_actions(changed, state, jnp.int32(0))

    assert base_cost == 1
    assert jnp.isinf(changed_cost)


@pytest.mark.parametrize(
    ("field", "before", "after"),
    (("add_mask", False, True), ("del_mask", True, False)),
)
def test_pddl_effect_masks_are_read_from_instance_context(
    pddl_env, zero_mask_pddl_config, field, before, after
):
    state = pddl_env.State.from_unpacked(
        atoms=jnp.full(pddl_env.num_atoms, before, dtype=jnp.bool_)
    )
    context = zero_mask_pddl_config.InstanceContext
    mask = getattr(context, field).at[0, 0].set(True)
    changed = zero_mask_pddl_config.replace(
        InstanceContext=context.replace(**{field: mask})
    )
    next_state, cost = pddl_env.get_actions(changed, state, jnp.int32(0))

    assert cost == 1
    assert next_state.unpacked_atoms[0] == after
