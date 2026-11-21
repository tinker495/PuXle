"""Focused tests for puzzle-specific APIs and PDDL type-system helpers that complement the generic discovery suite in test_all_puzzles.py."""

import jax
import jax.numpy as jnp
import pytest

from puxle.puzzles.adapters.pddl.type_system import (
    collect_type_hierarchy,
    extract_objects_by_type,
    select_most_specific_types,
)
from puxle.puzzles.dotknot import DotKnot
from puxle.puzzles.hanoi import TowerOfHanoi
from puxle.puzzles.lightsout import LightsOut
from puxle.puzzles.maze import Maze
from puxle.puzzles.pancake import PancakeSorting
from puxle.puzzles.room import Room
from puxle.puzzles.slidepuzzle import SlidePuzzle
from puxle.puzzles.sokoban import Sokoban
from puxle.puzzles.topspin import TopSpin
from puxle.puzzles.tsp import TSP


@pytest.fixture
def rng_key():
    return jax.random.PRNGKey(42)


class TestMaze:
    def test_instantiation(self):
        maze = Maze()
        assert maze.action_size == 4
        assert maze.size == 23

        maze_custom = Maze(size=15)
        assert maze_custom.size == 15
        assert maze_custom.action_size == 4

    def test_state_generation(self, rng_key):
        maze = Maze(size=11)
        sc = maze.get_solve_config(key=rng_key)
        state = maze.get_initial_state(sc, key=rng_key)

        assert hasattr(state, "pos")
        assert state.pos.shape == (2,)
        assert jnp.all(state.pos >= 0)
        assert jnp.all(state.pos < maze.size)

        assert sc.GoalSpec is not None
        assert sc.InstanceContext.Maze_unpacked.shape == (maze.size * maze.size,)

    def test_action_strings(self):
        maze = Maze()
        expected = ["←", "→", "↑", "↓"]
        for i in range(maze.action_size):
            s = maze.action_to_string(i)
            assert isinstance(s, str)
            assert s in expected

    def test_is_solved(self, rng_key):
        maze = Maze(size=11)
        sc = maze.get_solve_config(key=rng_key)
        state = maze.get_initial_state(sc, key=rng_key)

        # Initial state may or may not be solved (random)
        result = maze.is_solved(sc, state)
        assert isinstance(result, (bool, jnp.bool_)) or (
            hasattr(result, "dtype") and result.dtype == jnp.bool_
        )

        assert maze.is_solved(sc, sc.GoalSpec)

    def test_valid_moves(self, rng_key):
        maze = Maze(size=11)
        sc = maze.get_solve_config(key=rng_key)
        state = maze.get_initial_state(sc, key=rng_key)

        for action in range(maze.action_size):
            next_state, cost = maze.get_actions(
                sc, state, jnp.asarray(action), filled=True
            )

            assert jnp.isfinite(cost) or jnp.isinf(cost)

            if jnp.isfinite(cost):
                assert cost == 1.0

    def test_inverse_action_map(self):
        maze = Maze()
        inv_map = maze.inverse_action_map

        assert inv_map is not None
        assert len(inv_map) == 4
        # L <-> R (0 <-> 1), U <-> D (2 <-> 3)
        assert inv_map[0] == 1 and inv_map[1] == 0
        assert inv_map[2] == 3 and inv_map[3] == 2

    def test_maze_generation(self, rng_key):
        maze = Maze(size=11)
        sc = maze.get_solve_config(key=rng_key)

        maze_grid = sc.InstanceContext.Maze_unpacked.reshape((maze.size, maze.size))

        # Should have both walls (True) and paths (False)
        assert jnp.any(maze_grid)
        assert jnp.any(~maze_grid)


class TestPancakeSorting:
    def test_instantiation(self):
        pancake = PancakeSorting()
        assert pancake.size == 35
        assert pancake.action_size == 34

        pancake_custom = PancakeSorting(size=10)
        assert pancake_custom.size == 10
        assert pancake_custom.action_size == 9

    def test_state_generation(self, rng_key):
        pancake = PancakeSorting(size=10)
        sc = pancake.get_solve_config(key=rng_key)
        state = pancake.get_initial_state(sc, key=rng_key)

        assert hasattr(state, "stack")
        assert state.stack.shape == (pancake.size,)

        sorted_stack = jnp.sort(state.stack)
        expected = jnp.arange(1, pancake.size + 1, dtype=jnp.uint8)
        assert jnp.array_equal(sorted_stack, expected)

    def test_target_state(self, rng_key):
        pancake = PancakeSorting(size=10)
        sc = pancake.get_solve_config(key=rng_key)

        target_stack = sc.GoalSpec.stack
        expected = jnp.arange(1, pancake.size + 1, dtype=jnp.uint8)
        assert jnp.array_equal(target_stack, expected)

    def test_action_strings(self):
        pancake = PancakeSorting(size=10)
        for i in range(pancake.action_size):
            s = pancake.action_to_string(i)
            assert isinstance(s, str)
            assert len(s) > 0
            assert "Flip" in s

    def test_is_solved(self, rng_key):
        pancake = PancakeSorting(size=10)
        sc = pancake.get_solve_config(key=rng_key)

        assert pancake.is_solved(sc, sc.GoalSpec)

        state = pancake.get_initial_state(sc, key=rng_key)
        result = pancake.is_solved(sc, state)
        assert isinstance(result, (bool, jnp.bool_)) or (
            hasattr(result, "dtype") and result.dtype == jnp.bool_
        )

    def test_flip_action(self, rng_key):
        pancake = PancakeSorting(size=5)
        sc = pancake.get_solve_config(key=rng_key)

        initial_stack = jnp.array([5, 4, 3, 2, 1], dtype=jnp.uint8)
        state = pancake.State(stack=initial_stack)

        # Flip at action index 2: flip_pos = action + 1 = 3
        # This flips the first 4 elements (indices 0-3): [5,4,3,2] -> [2,3,4,5]
        next_state, cost = pancake.get_actions(sc, state, jnp.asarray(2), filled=True)

        assert cost == 1.0

        expected = jnp.array([2, 3, 4, 5, 1], dtype=jnp.uint8)
        assert jnp.array_equal(next_state.stack, expected)

    def test_inverse_action_map(self):
        pancake = PancakeSorting(size=10)
        inv_map = pancake.inverse_action_map

        assert inv_map is not None
        assert len(inv_map) == pancake.action_size
        assert jnp.array_equal(inv_map, jnp.arange(pancake.action_size))


class TestRoom:
    def test_instantiation(self):
        room = Room()
        assert room.action_size == 4

        # Default size should be valid (3*N+2 formula)
        assert (room.size - 2) % 3 == 0
        assert room.room_dim == (room.size - 2) // 3

    def test_size_adjustment(self):
        room_valid = Room(size=11)
        assert room_valid.size == 11
        assert room_valid.room_dim == 3

        room_invalid = Room(size=12)
        assert (room_invalid.size - 2) % 3 == 0

    def test_state_generation(self, rng_key):
        room = Room(size=11)
        sc = room.get_solve_config(key=rng_key)
        state = room.get_initial_state(sc, key=rng_key)

        assert hasattr(state, "pos")
        assert state.pos.shape == (2,)
        assert jnp.all(state.pos >= 0)
        assert jnp.all(state.pos < room.size)

    def test_action_strings(self):
        room = Room()
        expected = ["←", "→", "↑", "↓"]
        for i in range(room.action_size):
            s = room.action_to_string(i)
            assert isinstance(s, str)
            assert s in expected

    def test_is_solved(self, rng_key):
        room = Room(size=11)
        sc = room.get_solve_config(key=rng_key)

        assert room.is_solved(sc, sc.GoalSpec)

    def test_room_structure(self, rng_key):
        room = Room(size=11, prob_open_extra_door=1.0)
        sc = room.get_solve_config(key=rng_key)

        maze_grid = sc.InstanceContext.Maze_unpacked.reshape((room.size, room.size))

        room_dim = room.room_dim
        for r_idx in range(3):
            for c_idx in range(3):
                room_r_start = (room_dim + 1) * r_idx
                room_c_start = (room_dim + 1) * c_idx
                room_interior = maze_grid[
                    room_r_start : room_r_start + room_dim,
                    room_c_start : room_c_start + room_dim,
                ]
                # Room interior should be paths (False)
                assert jnp.all(~room_interior)

    def test_inverse_action_map(self):
        room = Room()
        inv_map = room.inverse_action_map

        assert inv_map is not None
        assert len(inv_map) == 4
        assert inv_map[0] == 1 and inv_map[1] == 0
        assert inv_map[2] == 3 and inv_map[3] == 2


class TestSokoban:
    def test_instantiation(self):
        sokoban = Sokoban()
        assert sokoban.size == 10
        assert sokoban.action_size == 4

    def test_size_constraint(self):
        with pytest.raises(AssertionError):
            Sokoban(size=8)

    def test_state_generation(self, rng_key):
        sokoban = Sokoban()
        sc = sokoban.get_solve_config(key=rng_key)
        state = sokoban.get_initial_state(sc, key=rng_key)

        assert hasattr(state, "board_unpacked")
        board = state.board_unpacked
        assert board.shape == (sokoban.size * sokoban.size,)

        assert jnp.any(board == Sokoban.Object.PLAYER.value)
        assert jnp.any(board == Sokoban.Object.BOX.value)
        assert jnp.any(board == Sokoban.Object.WALL.value)

    def test_action_strings(self):
        sokoban = Sokoban()
        expected = ["←", "→", "↑", "↓"]
        for i in range(sokoban.action_size):
            s = sokoban.action_to_string(i)
            assert isinstance(s, str)
            assert s in expected

    def test_is_solved_all_boxes(self, rng_key):
        sokoban = Sokoban(solve_condition=Sokoban.SolveCondition.ALL_BOXES_ON_TARGET)
        sc = sokoban.get_solve_config(key=rng_key)
        state = sokoban.get_initial_state(sc, key=rng_key)

        result = sokoban.is_solved(sc, state)
        assert isinstance(result, (bool, jnp.bool_)) or (
            hasattr(result, "dtype") and result.dtype == jnp.bool_
        )

    def test_is_solved_all_boxes_and_player(self, rng_key):
        sokoban = Sokoban(
            solve_condition=Sokoban.SolveCondition.ALL_BOXES_ON_TARGET_AND_PLAYER_ON_TARGET
        )
        sc = sokoban.get_solve_config(key=rng_key)

        assert sokoban.is_solved(sc, sc.GoalSpec)

    def test_not_reversible(self):
        sokoban = Sokoban()
        assert sokoban.is_reversible is False

    def test_custom_inverse_neighbours(self, rng_key):
        sokoban = Sokoban()
        sc = sokoban.get_solve_config(key=rng_key)
        state = sokoban.get_initial_state(sc, key=rng_key)

        inv_states, inv_costs = sokoban.get_inverse_neighbours(sc, state, filled=True)

        assert inv_costs.shape == (sokoban.action_size,)

    def test_player_position_detection(self, rng_key):
        sokoban = Sokoban()
        sc = sokoban.get_solve_config(key=rng_key)
        state = sokoban.get_initial_state(sc, key=rng_key)

        x, y = sokoban._get_player_position(state)

        board = state.board_unpacked.reshape((sokoban.size, sokoban.size))
        assert board[x, y] == Sokoban.Object.PLAYER.value

    def test_valid_moves(self, rng_key):
        sokoban = Sokoban()
        sc = sokoban.get_solve_config(key=rng_key)
        state = sokoban.get_initial_state(sc, key=rng_key)

        for action in range(sokoban.action_size):
            next_state, cost = sokoban.get_actions(
                sc, state, jnp.asarray(action), filled=True
            )

            assert jnp.isfinite(cost) or jnp.isinf(cost)

    def test_filled_parameter(self, rng_key):
        sokoban = Sokoban()
        sc = sokoban.get_solve_config(key=rng_key)
        state = sokoban.get_initial_state(sc, key=rng_key)

        for action in range(sokoban.action_size):
            _, cost = sokoban.get_actions(sc, state, jnp.asarray(action), filled=False)
            assert jnp.isinf(cost)


class TestPuzzleIntegration:
    def test_all_have_action_size(self):
        puzzles = [Maze(), PancakeSorting(size=10), Room(), Sokoban()]
        for puzzle in puzzles:
            assert hasattr(puzzle, "action_size")
            assert puzzle.action_size > 0

    def test_all_generate_valid_states(self, rng_key):
        puzzles = [Maze(size=11), PancakeSorting(size=10), Room(size=11), Sokoban()]

        for puzzle in puzzles:
            sc = puzzle.get_solve_config(key=rng_key)
            state = puzzle.get_initial_state(sc, key=rng_key)

            assert sc is not None
            assert state is not None

    def test_all_have_string_representations(self, rng_key):
        puzzles = [Maze(size=11), PancakeSorting(size=10), Room(size=11), Sokoban()]

        for puzzle in puzzles:
            sc = puzzle.get_solve_config(key=rng_key)
            state = puzzle.get_initial_state(sc, key=rng_key)

            state_str = str(state)
            assert isinstance(state_str, str)
            assert len(state_str) > 0

    def test_all_support_is_solved(self, rng_key):
        puzzles = [Maze(size=11), PancakeSorting(size=10), Room(size=11), Sokoban()]

        for puzzle in puzzles:
            sc = puzzle.get_solve_config(key=rng_key)
            state = puzzle.get_initial_state(sc, key=rng_key)

            result = puzzle.is_solved(sc, state)
            assert isinstance(result, (bool, jnp.bool_)) or (
                hasattr(result, "dtype") and result.dtype == jnp.bool_
            )


class TestDotKnot:
    def test_instantiation(self):
        dotknot = DotKnot()
        assert dotknot.action_size == 4
        assert dotknot.size == 10
        assert dotknot.color_num == 4

        dotknot_custom = DotKnot(size=8, color_num=3)
        assert dotknot_custom.size == 8
        assert dotknot_custom.color_num == 3
        assert dotknot_custom.action_size == 4

    def test_state_generation(self, rng_key):
        dotknot = DotKnot(size=10)
        sc = dotknot.get_solve_config(key=rng_key)
        state = dotknot.get_initial_state(sc, key=rng_key)

        assert hasattr(state, "board_unpacked")
        assert state.board_unpacked.shape == (dotknot.size * dotknot.size,)

        unpacked = state.board_unpacked
        assert jnp.any(unpacked > 0)
        assert jnp.any(unpacked <= 2 * dotknot.color_num)  # Has unmerged dots

    def test_action_strings(self):
        dotknot = DotKnot()
        expected = ["←", "→", "↑", "↓"]
        for i in range(dotknot.action_size):
            s = dotknot.action_to_string(i)
            assert isinstance(s, str)
            assert s in expected

    def test_is_solved(self, rng_key):
        dotknot = DotKnot(size=10)
        sc = dotknot.get_solve_config(key=rng_key)
        state = dotknot.get_initial_state(sc, key=rng_key)

        result = dotknot.is_solved(sc, state)
        assert isinstance(result, (bool, jnp.bool_)) or (
            hasattr(result, "dtype") and result.dtype == jnp.bool_
        )

    def test_valid_moves(self, rng_key):
        dotknot = DotKnot(size=10)
        sc = dotknot.get_solve_config(key=rng_key)
        state = dotknot.get_initial_state(sc, key=rng_key)

        for action in range(dotknot.action_size):
            next_state, cost = dotknot.get_actions(
                sc, state, jnp.asarray(action), filled=True
            )

            assert jnp.isfinite(cost) or jnp.isinf(cost)

    def test_get_string_parser(self, rng_key):
        dotknot = DotKnot()
        parser = dotknot.get_string_parser()
        assert callable(parser)

        sc = dotknot.get_solve_config(key=rng_key)
        state = dotknot.get_initial_state(sc, key=rng_key)
        result = parser(state)
        assert isinstance(result, str)


class TestTowerOfHanoi:
    def test_instantiation(self):
        hanoi = TowerOfHanoi()
        assert hanoi.num_disks == 5
        assert hanoi.num_pegs == 3
        assert hanoi.action_size == 6  # 3 * (3 - 1) = 6 possible moves

        hanoi_custom = TowerOfHanoi(size=7)
        assert hanoi_custom.num_disks == 7
        assert hanoi_custom.action_size == 6

    def test_state_generation(self, rng_key):
        hanoi = TowerOfHanoi(size=5)
        sc = hanoi.get_solve_config(key=rng_key)
        state = hanoi.get_initial_state(sc, key=rng_key)

        assert hasattr(state, "pegs")
        assert state.pegs.shape == (hanoi.num_pegs, hanoi.num_disks + 1)

        assert (
            state.pegs[0, 0] == hanoi.num_disks
        )  # Column 0 holds each peg's disk count
        assert state.pegs[1, 0] == 0
        assert state.pegs[2, 0] == 0

    def test_target_state(self, rng_key):
        hanoi = TowerOfHanoi(size=5)
        sc = hanoi.get_solve_config(key=rng_key)

        assert sc.GoalSpec.pegs[2, 0] == hanoi.num_disks
        assert sc.GoalSpec.pegs[0, 0] == 0
        assert sc.GoalSpec.pegs[1, 0] == 0

    def test_action_strings(self):
        hanoi = TowerOfHanoi(size=5)
        for i in range(hanoi.action_size):
            s = hanoi.action_to_string(i)
            assert isinstance(s, str)
            assert "Move disk from peg" in s
            assert "to peg" in s

    def test_is_solved(self, rng_key):
        hanoi = TowerOfHanoi(size=5)
        sc = hanoi.get_solve_config(key=rng_key)

        assert hanoi.is_solved(sc, sc.GoalSpec)

        state = hanoi.get_initial_state(sc, key=rng_key)
        assert not hanoi.is_solved(sc, state)

    def test_valid_moves(self, rng_key):
        hanoi = TowerOfHanoi(size=3)
        sc = hanoi.get_solve_config(key=rng_key)
        state = hanoi.get_initial_state(sc, key=rng_key)

        for action in range(hanoi.action_size):
            next_state, cost = hanoi.get_actions(
                sc, state, jnp.asarray(action), filled=True
            )

            assert jnp.isfinite(cost) or jnp.isinf(cost)

    def test_get_string_parser(self, rng_key):
        hanoi = TowerOfHanoi()
        parser = hanoi.get_string_parser()
        assert callable(parser)

        sc = hanoi.get_solve_config(key=rng_key)
        state = hanoi.get_initial_state(sc, key=rng_key)
        result = parser(state)
        assert isinstance(result, str)


class TestLightsOut:
    def test_instantiation(self):
        lightsout = LightsOut()
        assert lightsout.size == 7
        assert lightsout.action_size == 49

        lightsout_custom = LightsOut(size=5)
        assert lightsout_custom.size == 5
        assert lightsout_custom.action_size == 25

    def test_state_generation(self, rng_key):
        lightsout = LightsOut(size=5)
        sc = lightsout.get_solve_config(key=rng_key)
        state = lightsout.get_initial_state(sc, key=rng_key)

        assert hasattr(state, "board_unpacked")
        assert state.board_unpacked.shape == (lightsout.size * lightsout.size,)

    def test_target_state(self, rng_key):
        lightsout = LightsOut(size=5)
        sc = lightsout.get_solve_config(key=rng_key)

        assert jnp.all(sc.GoalSpec.board_unpacked == 0)

    def test_action_strings(self):
        lightsout = LightsOut(size=5)
        for i in range(min(10, lightsout.action_size)):
            s = lightsout.action_to_string(i)
            assert isinstance(s, str)
            assert len(s) > 0

    def test_is_solved(self, rng_key):
        lightsout = LightsOut(size=5)
        sc = lightsout.get_solve_config(key=rng_key)

        assert lightsout.is_solved(sc, sc.GoalSpec)

    def test_valid_moves(self, rng_key):
        lightsout = LightsOut(size=5)
        sc = lightsout.get_solve_config(key=rng_key)
        state = lightsout.get_initial_state(sc, key=rng_key)

        for action in range(min(5, lightsout.action_size)):
            next_state, cost = lightsout.get_actions(
                sc, state, jnp.asarray(action), filled=True
            )

            assert cost == 1.0

    def test_inverse_action_map(self):
        lightsout = LightsOut(size=5)
        inv_map = lightsout.inverse_action_map

        assert inv_map is not None
        assert len(inv_map) == lightsout.action_size
        assert jnp.array_equal(inv_map, jnp.arange(lightsout.action_size))

    def test_get_string_parser(self, rng_key):
        lightsout = LightsOut()
        parser = lightsout.get_string_parser()
        assert callable(parser)

        sc = lightsout.get_solve_config(key=rng_key)
        state = lightsout.get_initial_state(sc, key=rng_key)
        result = parser(state)
        assert isinstance(result, str)


class TestSlidePuzzle:
    def test_instantiation(self):
        slide = SlidePuzzle()
        assert slide.size == 4
        assert slide.action_size == 4

        slide_custom = SlidePuzzle(size=3)
        assert slide_custom.size == 3
        assert slide_custom.action_size == 4

    def test_state_generation(self, rng_key):
        slide = SlidePuzzle(size=4)
        sc = slide.get_solve_config(key=rng_key)
        state = slide.get_initial_state(sc, key=rng_key)

        assert hasattr(state, "board_unpacked")
        assert state.board_unpacked.shape == (slide.size * slide.size,)

        sorted_board = jnp.sort(state.board_unpacked)
        expected = jnp.arange(slide.size * slide.size, dtype=jnp.uint8)
        assert jnp.array_equal(sorted_board, expected)

    def test_target_state(self, rng_key):
        slide = SlidePuzzle(size=4)
        sc = slide.get_solve_config(key=rng_key)

        expected = jnp.array([*range(1, slide.size**2), 0], dtype=jnp.uint8)
        assert jnp.array_equal(sc.GoalSpec.board_unpacked, expected)

    def test_action_strings(self):
        slide = SlidePuzzle()
        expected = ["←", "→", "↑", "↓"]
        for i in range(slide.action_size):
            s = slide.action_to_string(i)
            assert isinstance(s, str)
            assert s in expected

    def test_is_solved(self, rng_key):
        slide = SlidePuzzle(size=4)
        sc = slide.get_solve_config(key=rng_key)

        assert slide.is_solved(sc, sc.GoalSpec)

    def test_valid_moves(self, rng_key):
        slide = SlidePuzzle(size=4)
        sc = slide.get_solve_config(key=rng_key)
        state = slide.get_initial_state(sc, key=rng_key)

        for action in range(slide.action_size):
            next_state, cost = slide.get_actions(
                sc, state, jnp.asarray(action), filled=True
            )

            assert jnp.isfinite(cost) or jnp.isinf(cost)

    def test_inverse_action_map(self):
        slide = SlidePuzzle()
        inv_map = slide.inverse_action_map

        assert inv_map is not None
        assert len(inv_map) == 4
        # Inverse: [R, L, D, U] -> [1, 0, 3, 2]
        expected = jnp.array([1, 0, 3, 2])
        assert jnp.array_equal(inv_map, expected)

    def test_get_string_parser(self, rng_key):
        slide = SlidePuzzle()
        parser = slide.get_string_parser()
        assert callable(parser)

        sc = slide.get_solve_config(key=rng_key)
        state = slide.get_initial_state(sc, key=rng_key)
        result = parser(state)
        assert isinstance(result, str)


class TestTopSpin:
    def test_instantiation(self):
        topspin = TopSpin()
        assert topspin.n_discs == 20
        assert topspin.turnstile_size == 4
        assert topspin.action_size == 3

        topspin_custom = TopSpin(size=10, turnstile_size=3)
        assert topspin_custom.n_discs == 10
        assert topspin_custom.turnstile_size == 3
        assert topspin_custom.action_size == 3

    def test_state_generation(self, rng_key):
        topspin = TopSpin(size=10)
        sc = topspin.get_solve_config(key=rng_key)
        state = topspin.get_initial_state(sc, key=rng_key)

        assert hasattr(state, "permutation")
        assert state.permutation.shape == (topspin.n_discs,)

        sorted_perm = jnp.sort(state.permutation)
        expected = jnp.arange(1, topspin.n_discs + 1, dtype=jnp.uint8)
        assert jnp.array_equal(sorted_perm, expected)

    def test_target_state(self, rng_key):
        topspin = TopSpin(size=10)
        sc = topspin.get_solve_config(key=rng_key)

        expected = jnp.arange(1, topspin.n_discs + 1, dtype=jnp.uint8)
        assert jnp.array_equal(sc.GoalSpec.permutation, expected)

    def test_action_strings(self):
        topspin = TopSpin(size=10)
        s0 = topspin.action_to_string(0)
        s1 = topspin.action_to_string(1)
        s2 = topspin.action_to_string(2)

        assert isinstance(s0, str) and "Shift Left" in s0
        assert isinstance(s1, str) and "Shift Right" in s1
        assert isinstance(s2, str) and "Reverse Turnstile" in s2

    def test_is_solved(self, rng_key):
        topspin = TopSpin(size=10)
        sc = topspin.get_solve_config(key=rng_key)

        assert topspin.is_solved(sc, sc.GoalSpec)

    def test_valid_moves(self, rng_key):
        topspin = TopSpin(size=10)
        sc = topspin.get_solve_config(key=rng_key)
        state = topspin.get_initial_state(sc, key=rng_key)

        for action in range(topspin.action_size):
            next_state, cost = topspin.get_actions(
                sc, state, jnp.asarray(action), filled=True
            )

            assert cost == 1.0

    def test_inverse_action_map(self):
        topspin = TopSpin()
        inv_map = topspin.inverse_action_map

        assert inv_map is not None
        assert len(inv_map) == 3
        # Left <-> Right (0 <-> 1), Reverse is self-inverse (2 <-> 2)
        expected = jnp.array([1, 0, 2])
        assert jnp.array_equal(inv_map, expected)

    def test_get_string_parser(self, rng_key):
        topspin = TopSpin()
        parser = topspin.get_string_parser()
        assert callable(parser)

        sc = topspin.get_solve_config(key=rng_key)
        state = topspin.get_initial_state(sc, key=rng_key)
        result = parser(state)
        assert isinstance(result, str)


class TestTSP:
    def test_instantiation(self):
        tsp = TSP()
        assert tsp.size == 16
        assert tsp.action_size == 16

        tsp_custom = TSP(size=10)
        assert tsp_custom.size == 10
        assert tsp_custom.action_size == 10

    def test_state_generation(self, rng_key):
        tsp = TSP(size=10)
        sc = tsp.get_solve_config(key=rng_key)
        state = tsp.get_initial_state(sc, key=rng_key)

        assert hasattr(state, "mask_unpacked")
        assert hasattr(state, "point")
        assert state.mask_unpacked.shape == (tsp.size,)

        assert jnp.sum(state.mask_unpacked) == 1
        assert state.mask_unpacked[state.point]

    def test_solve_config(self, rng_key):
        tsp = TSP(size=10)
        sc = tsp.get_solve_config(key=rng_key)

        context = sc.InstanceContext
        assert context.points.shape == (tsp.size, 2)
        assert context.distance_matrix.shape == (tsp.size, tsp.size)
        assert context.start.shape == ()

    def test_action_strings(self):
        tsp = TSP(size=10)
        for i in range(tsp.action_size):
            s = tsp.action_to_string(i)
            assert isinstance(s, str)
            assert len(s) > 0

    def test_is_solved(self, rng_key):
        tsp = TSP(size=10)
        sc = tsp.get_solve_config(key=rng_key)
        state = tsp.get_initial_state(sc, key=rng_key)

        assert not tsp.is_solved(sc, state)

        solved_mask = jnp.ones(tsp.size, dtype=jnp.bool_)
        solved_state = state.set_unpacked(mask=solved_mask)
        assert tsp.is_solved(sc, solved_state)

    def test_valid_moves(self, rng_key):
        tsp = TSP(size=10)
        sc = tsp.get_solve_config(key=rng_key)
        state = tsp.get_initial_state(sc, key=rng_key)

        for action in range(tsp.action_size):
            next_state, cost = tsp.get_actions(
                sc, state, jnp.asarray(action), filled=True
            )

            if action == state.point:
                assert jnp.isinf(cost)
            else:
                assert jnp.isfinite(cost)

    def test_get_string_parser(self, rng_key):
        tsp = TSP()
        parser = tsp.get_string_parser()
        assert callable(parser)

        sc = tsp.get_solve_config(key=rng_key)
        state = tsp.get_initial_state(sc, key=rng_key)
        result = parser(state)
        assert isinstance(result, str)


class TestTypeSystem:
    def test_collect_type_hierarchy_empty(self):
        class MockDomain:
            pass

        domain = MockDomain()
        parent, ancestors, descendants = collect_type_hierarchy(domain)

        assert isinstance(parent, dict)
        assert isinstance(ancestors, dict)
        assert isinstance(descendants, dict)
        assert len(parent) == 0

    def test_collect_type_hierarchy_simple(self):
        class MockDomain:
            def __init__(self):
                self.types = {"car": "vehicle", "truck": "vehicle"}

        domain = MockDomain()
        parent, ancestors, descendants = collect_type_hierarchy(domain)

        assert parent["car"] == "vehicle"
        assert parent["truck"] == "vehicle"

        assert "vehicle" in ancestors["car"]
        assert "vehicle" in ancestors["truck"]

        assert "car" in descendants["vehicle"]
        assert "truck" in descendants["vehicle"]

    def test_collect_type_hierarchy_multi_level(self):
        class MockDomain:
            def __init__(self):
                self.types = {
                    "sedan": "car",
                    "car": "vehicle",
                    "truck": "vehicle",
                }

        domain = MockDomain()
        parent, ancestors, descendants = collect_type_hierarchy(domain)

        assert "car" in ancestors["sedan"]
        assert "vehicle" in ancestors["sedan"]
        assert "vehicle" in ancestors["car"]

        assert "sedan" in descendants["vehicle"]
        assert "car" in descendants["vehicle"]
        assert "sedan" in descendants["car"]

    def test_select_most_specific_types_empty(self):
        hierarchy = ({}, {}, {})
        result = select_most_specific_types([], hierarchy)

        assert result == ["object"]

    def test_select_most_specific_types_single(self):
        hierarchy = ({}, {}, {})
        result = select_most_specific_types(["car"], hierarchy)

        assert result == ["car"]

    def test_select_most_specific_types_with_object(self):
        hierarchy = ({}, {}, {})
        result = select_most_specific_types(["object", "car"], hierarchy)

        assert "object" not in result
        assert "car" in result

    def test_select_most_specific_types_filters_ancestors(self):
        class MockDomain:
            def __init__(self):
                self.types = {"car": "vehicle"}

        domain = MockDomain()
        hierarchy = collect_type_hierarchy(domain)

        result = select_most_specific_types(["car", "vehicle"], hierarchy)

        assert "car" in result
        assert "vehicle" not in result

    def test_extract_objects_by_type_empty_problem(self):
        class MockProblem:
            def __init__(self):
                self.objects = []

        problem = MockProblem()
        hierarchy = ({}, {}, {})
        result = extract_objects_by_type(problem, hierarchy)

        assert isinstance(result, dict)
        assert "object" in result
        assert result["object"] == []

    def test_extract_objects_by_type_untyped_objects(self):
        class MockObject:
            def __init__(self, name):
                self.name = name

        class MockProblem:
            def __init__(self):
                self.objects = [MockObject("obj1"), MockObject("obj2")]

        problem = MockProblem()
        hierarchy = ({}, {}, {})
        result = extract_objects_by_type(problem, hierarchy)

        assert "object" in result
        assert "obj1" in result["object"]
        assert "obj2" in result["object"]

    def test_extract_objects_by_type_typed_objects(self):
        class MockObject:
            def __init__(self, name, type_tag):
                self.name = name
                self.type_tag = type_tag

        class MockProblem:
            def __init__(self):
                self.objects = [
                    MockObject("car1", "car"),
                    MockObject("car2", "car"),
                    MockObject("truck1", "truck"),
                ]

        class MockDomain:
            def __init__(self):
                self.types = {"car": "vehicle", "truck": "vehicle"}

        domain = MockDomain()
        problem = MockProblem()
        hierarchy = collect_type_hierarchy(domain)
        result = extract_objects_by_type(problem, hierarchy, domain)

        assert "car1" in result["car"]
        assert "car2" in result["car"]
        assert "truck1" in result["truck"]

        assert "car1" in result["vehicle"]
        assert "car2" in result["vehicle"]
        assert "truck1" in result["vehicle"]

    def test_extract_objects_by_type_dict_objects(self):
        class MockObject:
            def __init__(self, name):
                self.name = name

        class MockProblem:
            def __init__(self):
                self.objects = {
                    "car": [MockObject("car1"), MockObject("car2")],
                    "truck": [MockObject("truck1")],
                }

        problem = MockProblem()
        hierarchy = ({}, {}, {})
        result = extract_objects_by_type(problem, hierarchy)

        assert "car1" in result["car"]
        assert "car2" in result["car"]
        assert "truck1" in result["truck"]

    def test_type_hierarchy_cycle_prevention(self):
        class MockDomain:
            def __init__(self):
                self.types = {"car": "vehicle"}

        domain = MockDomain()
        parent, ancestors, descendants = collect_type_hierarchy(domain)

        assert len(ancestors["car"]) == 1
        assert "vehicle" in ancestors["car"]
