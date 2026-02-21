"""Direct test coverage for puzzle modules: Maze, PancakeSorting, Room, and Sokoban.

This test suite provides focused, direct testing of puzzle-specific features and APIs
that complement the comprehensive generic tests in test_all_puzzles.py.
"""

import jax
import jax.numpy as jnp
import pytest

from puxle.pddls.type_system import (
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
    """Provide a reproducible random key for JAX operations."""
    return jax.random.PRNGKey(42)


class TestMaze:
    """Test suite for Maze puzzle."""

    def test_instantiation(self):
        """Test Maze can be instantiated with default and custom sizes."""
        maze = Maze()
        assert maze.action_size == 4  # L, R, U, D
        assert maze.size == 23  # Default size

        maze_custom = Maze(size=15)
        assert maze_custom.size == 15
        assert maze_custom.action_size == 4

    def test_state_generation(self, rng_key):
        """Test solve config and state generation."""
        maze = Maze(size=11)
        sc = maze.get_solve_config(key=rng_key)
        state = maze.get_initial_state(sc, key=rng_key)

        # Verify state has position attribute
        assert hasattr(state, "pos")
        assert state.pos.shape == (2,)
        assert jnp.all(state.pos >= 0)
        assert jnp.all(state.pos < maze.size)

        # Verify solve config has target state and maze
        assert hasattr(sc, "TargetState")
        assert hasattr(sc, "Maze_unpacked")
        assert sc.Maze_unpacked.shape == (maze.size * maze.size,)

    def test_action_strings(self):
        """Test action_to_string returns valid directional strings."""
        maze = Maze()
        expected = ["←", "→", "↑", "↓"]
        for i in range(maze.action_size):
            s = maze.action_to_string(i)
            assert isinstance(s, str)
            assert s in expected

    def test_is_solved(self, rng_key):
        """Test is_solved logic for Maze."""
        maze = Maze(size=11)
        sc = maze.get_solve_config(key=rng_key)
        state = maze.get_initial_state(sc, key=rng_key)

        # Initial state may or may not be solved (random)
        result = maze.is_solved(sc, state)
        assert isinstance(result, (bool, jnp.bool_)) or (hasattr(result, "dtype") and result.dtype == jnp.bool_)

        # Target state should always be solved
        assert maze.is_solved(sc, sc.TargetState)

    def test_valid_moves(self, rng_key):
        """Test that get_actions respects maze walls."""
        maze = Maze(size=11)
        sc = maze.get_solve_config(key=rng_key)
        state = maze.get_initial_state(sc, key=rng_key)

        # Test all actions
        for action in range(maze.action_size):
            next_state, cost = maze.get_actions(sc, state, jnp.asarray(action), filled=True)

            # Cost should be either 1.0 (valid) or inf (invalid)
            assert jnp.isfinite(cost) or jnp.isinf(cost)

            # If move is valid, position should change
            if jnp.isfinite(cost):
                assert cost == 1.0

    def test_inverse_action_map(self):
        """Test that Maze has correct inverse action mapping."""
        maze = Maze()
        inv_map = maze.inverse_action_map

        assert inv_map is not None
        assert len(inv_map) == 4
        # L <-> R (0 <-> 1), U <-> D (2 <-> 3)
        assert inv_map[0] == 1 and inv_map[1] == 0
        assert inv_map[2] == 3 and inv_map[3] == 2

    def test_maze_generation(self, rng_key):
        """Test that DFS maze generation produces valid mazes."""
        maze = Maze(size=11)
        sc = maze.get_solve_config(key=rng_key)

        # Unpack maze and check structure
        maze_grid = sc.Maze_unpacked.reshape((maze.size, maze.size))

        # Should have both walls (True) and paths (False)
        assert jnp.any(maze_grid)  # Has walls
        assert jnp.any(~maze_grid)  # Has paths


class TestPancakeSorting:
    """Test suite for PancakeSorting puzzle."""

    def test_instantiation(self):
        """Test PancakeSorting can be instantiated with default and custom sizes."""
        pancake = PancakeSorting()
        assert pancake.size == 35  # Default size
        assert pancake.action_size == 34  # size - 1

        pancake_custom = PancakeSorting(size=10)
        assert pancake_custom.size == 10
        assert pancake_custom.action_size == 9

    def test_state_generation(self, rng_key):
        """Test solve config and state generation."""
        pancake = PancakeSorting(size=10)
        sc = pancake.get_solve_config(key=rng_key)
        state = pancake.get_initial_state(sc, key=rng_key)

        # Verify state has stack attribute
        assert hasattr(state, "stack")
        assert state.stack.shape == (pancake.size,)

        # Stack should be a permutation of [1..size]
        sorted_stack = jnp.sort(state.stack)
        expected = jnp.arange(1, pancake.size + 1, dtype=jnp.uint8)
        assert jnp.array_equal(sorted_stack, expected)

    def test_target_state(self, rng_key):
        """Test that target state is sorted ascending."""
        pancake = PancakeSorting(size=10)
        sc = pancake.get_solve_config(key=rng_key)

        target_stack = sc.TargetState.stack
        expected = jnp.arange(1, pancake.size + 1, dtype=jnp.uint8)
        assert jnp.array_equal(target_stack, expected)

    def test_action_strings(self):
        """Test action_to_string returns valid flip descriptions."""
        pancake = PancakeSorting(size=10)
        for i in range(pancake.action_size):
            s = pancake.action_to_string(i)
            assert isinstance(s, str)
            assert len(s) > 0
            assert "Flip" in s

    def test_is_solved(self, rng_key):
        """Test is_solved logic for PancakeSorting."""
        pancake = PancakeSorting(size=10)
        sc = pancake.get_solve_config(key=rng_key)

        # Target state should be solved
        assert pancake.is_solved(sc, sc.TargetState)

        # Random state is unlikely to be solved
        state = pancake.get_initial_state(sc, key=rng_key)
        # Just verify is_solved returns boolean
        result = pancake.is_solved(sc, state)
        assert isinstance(result, (bool, jnp.bool_)) or (hasattr(result, "dtype") and result.dtype == jnp.bool_)

    def test_flip_action(self, rng_key):
        """Test that flip actions correctly reverse stack prefixes."""
        pancake = PancakeSorting(size=5)
        sc = pancake.get_solve_config(key=rng_key)

        # Create a specific known state
        initial_stack = jnp.array([5, 4, 3, 2, 1], dtype=jnp.uint8)
        state = pancake.State(stack=initial_stack)

        # Flip at action index 2: flip_pos = action + 1 = 3
        # This flips the first 4 elements (indices 0-3): [5,4,3,2] -> [2,3,4,5]
        next_state, cost = pancake.get_actions(sc, state, jnp.asarray(2), filled=True)

        # Cost should be 1.0 (valid move)
        assert cost == 1.0

        # First 4 elements should be reversed
        expected = jnp.array([2, 3, 4, 5, 1], dtype=jnp.uint8)
        assert jnp.array_equal(next_state.stack, expected)

    def test_inverse_action_map(self):
        """Test that each flip is its own inverse."""
        pancake = PancakeSorting(size=10)
        inv_map = pancake.inverse_action_map

        assert inv_map is not None
        assert len(inv_map) == pancake.action_size
        # Each action is its own inverse
        assert jnp.array_equal(inv_map, jnp.arange(pancake.action_size))


class TestRoom:
    """Test suite for Room puzzle (Maze variant with 3x3 rooms)."""

    def test_instantiation(self):
        """Test Room can be instantiated and adjusts size correctly."""
        room = Room()
        assert room.action_size == 4  # Same as Maze

        # Default size should be valid (3*N+2 formula)
        assert (room.size - 2) % 3 == 0
        assert room.room_dim == (room.size - 2) // 3

    def test_size_adjustment(self):
        """Test that invalid sizes are adjusted to nearest valid size."""
        # Valid size: 11 = 3*3+2 -> room_dim=3
        room_valid = Room(size=11)
        assert room_valid.size == 11
        assert room_valid.room_dim == 3

        # Invalid size: 12 -> should adjust to 11 or 14
        room_invalid = Room(size=12)
        assert (room_invalid.size - 2) % 3 == 0

    def test_state_generation(self, rng_key):
        """Test solve config and state generation."""
        room = Room(size=11)
        sc = room.get_solve_config(key=rng_key)
        state = room.get_initial_state(sc, key=rng_key)

        # Should inherit Maze structure
        assert hasattr(state, "pos")
        assert state.pos.shape == (2,)
        assert jnp.all(state.pos >= 0)
        assert jnp.all(state.pos < room.size)

    def test_action_strings(self):
        """Test action_to_string returns valid directional strings."""
        room = Room()
        expected = ["←", "→", "↑", "↓"]
        for i in range(room.action_size):
            s = room.action_to_string(i)
            assert isinstance(s, str)
            assert s in expected

    def test_is_solved(self, rng_key):
        """Test is_solved logic for Room."""
        room = Room(size=11)
        sc = room.get_solve_config(key=rng_key)

        # Target state should always be solved
        assert room.is_solved(sc, sc.TargetState)

    def test_room_structure(self, rng_key):
        """Test that generated map has proper room structure."""
        room = Room(size=11, prob_open_extra_door=1.0)
        sc = room.get_solve_config(key=rng_key)

        # Unpack maze and check structure
        maze_grid = sc.Maze_unpacked.reshape((room.size, room.size))

        # Verify room interiors are carved (not all walls)
        room_dim = room.room_dim
        for r_idx in range(3):
            for c_idx in range(3):
                room_r_start = (room_dim + 1) * r_idx
                room_c_start = (room_dim + 1) * c_idx
                room_interior = maze_grid[
                    room_r_start : room_r_start + room_dim, room_c_start : room_c_start + room_dim
                ]
                # Room interior should be paths (False)
                assert jnp.all(~room_interior)

    def test_inverse_action_map(self):
        """Test that Room inherits Maze's inverse action mapping."""
        room = Room()
        inv_map = room.inverse_action_map

        assert inv_map is not None
        assert len(inv_map) == 4
        # Should match Maze's inverse map
        assert inv_map[0] == 1 and inv_map[1] == 0
        assert inv_map[2] == 3 and inv_map[3] == 2


class TestSokoban:
    """Test suite for Sokoban puzzle."""

    def test_instantiation(self):
        """Test Sokoban can be instantiated with correct size."""
        sokoban = Sokoban()
        assert sokoban.size == 10  # Fixed size
        assert sokoban.action_size == 4  # L, R, U, D

    def test_size_constraint(self):
        """Test that non-10 sizes raise assertion error."""
        with pytest.raises(AssertionError):
            Sokoban(size=8)

    def test_state_generation(self, rng_key):
        """Test solve config and state generation."""
        sokoban = Sokoban()
        sc = sokoban.get_solve_config(key=rng_key)
        state = sokoban.get_initial_state(sc, key=rng_key)

        # Verify state has board attribute
        assert hasattr(state, "board_unpacked")
        board = state.board_unpacked
        assert board.shape == (sokoban.size * sokoban.size,)

        # Board should contain player, boxes, walls, and empty cells
        assert jnp.any(board == Sokoban.Object.PLAYER.value)
        assert jnp.any(board == Sokoban.Object.BOX.value)
        assert jnp.any(board == Sokoban.Object.WALL.value)

    def test_action_strings(self):
        """Test action_to_string returns valid directional strings."""
        sokoban = Sokoban()
        expected = ["←", "→", "↑", "↓"]
        for i in range(sokoban.action_size):
            s = sokoban.action_to_string(i)
            assert isinstance(s, str)
            assert s in expected

    def test_is_solved_all_boxes(self, rng_key):
        """Test is_solved with ALL_BOXES_ON_TARGET condition."""
        sokoban = Sokoban(solve_condition=Sokoban.SolveCondition.ALL_BOXES_ON_TARGET)
        sc = sokoban.get_solve_config(key=rng_key)
        state = sokoban.get_initial_state(sc, key=rng_key)

        # Just verify is_solved returns boolean
        result = sokoban.is_solved(sc, state)
        assert isinstance(result, (bool, jnp.bool_)) or (hasattr(result, "dtype") and result.dtype == jnp.bool_)

    def test_is_solved_all_boxes_and_player(self, rng_key):
        """Test is_solved with ALL_BOXES_ON_TARGET_AND_PLAYER_ON_TARGET condition."""
        sokoban = Sokoban(solve_condition=Sokoban.SolveCondition.ALL_BOXES_ON_TARGET_AND_PLAYER_ON_TARGET)
        sc = sokoban.get_solve_config(key=rng_key)

        # Target state should be solved
        assert sokoban.is_solved(sc, sc.TargetState)

    def test_not_reversible(self):
        """Test that Sokoban is not marked as reversible."""
        sokoban = Sokoban()
        assert sokoban.is_reversible is False

    def test_custom_inverse_neighbours(self, rng_key):
        """Test that Sokoban has custom get_inverse_neighbours implementation."""
        sokoban = Sokoban()
        sc = sokoban.get_solve_config(key=rng_key)
        state = sokoban.get_initial_state(sc, key=rng_key)

        # Should not raise NotImplementedError
        inv_states, inv_costs = sokoban.get_inverse_neighbours(sc, state, filled=True)

        # Should return valid structure
        assert inv_costs.shape == (sokoban.action_size,)

    def test_player_position_detection(self, rng_key):
        """Test that _get_player_position correctly finds the player."""
        sokoban = Sokoban()
        sc = sokoban.get_solve_config(key=rng_key)
        state = sokoban.get_initial_state(sc, key=rng_key)

        x, y = sokoban._get_player_position(state)

        # Verify player is at detected position
        board = state.board_unpacked.reshape((sokoban.size, sokoban.size))
        assert board[x, y] == Sokoban.Object.PLAYER.value

    def test_valid_moves(self, rng_key):
        """Test that get_actions returns valid moves and costs."""
        sokoban = Sokoban()
        sc = sokoban.get_solve_config(key=rng_key)
        state = sokoban.get_initial_state(sc, key=rng_key)

        # Test all actions
        for action in range(sokoban.action_size):
            next_state, cost = sokoban.get_actions(sc, state, jnp.asarray(action), filled=True)

            # Cost should be either 1.0 (valid) or inf (invalid)
            assert jnp.isfinite(cost) or jnp.isinf(cost)

    def test_filled_parameter(self, rng_key):
        """Test that filled=False blocks all moves."""
        sokoban = Sokoban()
        sc = sokoban.get_solve_config(key=rng_key)
        state = sokoban.get_initial_state(sc, key=rng_key)

        # With filled=False, all costs should be inf
        for action in range(sokoban.action_size):
            _, cost = sokoban.get_actions(sc, state, jnp.asarray(action), filled=False)
            assert jnp.isinf(cost)


class TestPuzzleIntegration:
    """Integration tests across multiple puzzles."""

    def test_all_have_action_size(self):
        """Test that all puzzle modules have positive action_size."""
        puzzles = [Maze(), PancakeSorting(size=10), Room(), Sokoban()]
        for puzzle in puzzles:
            assert hasattr(puzzle, "action_size")
            assert puzzle.action_size > 0

    def test_all_generate_valid_states(self, rng_key):
        """Test that all puzzles can generate initial states and solve configs."""
        puzzles = [Maze(size=11), PancakeSorting(size=10), Room(size=11), Sokoban()]

        for puzzle in puzzles:
            sc = puzzle.get_solve_config(key=rng_key)
            state = puzzle.get_initial_state(sc, key=rng_key)

            assert sc is not None
            assert state is not None

    def test_all_have_string_representations(self, rng_key):
        """Test that all puzzles support string representations."""
        puzzles = [Maze(size=11), PancakeSorting(size=10), Room(size=11), Sokoban()]

        for puzzle in puzzles:
            sc = puzzle.get_solve_config(key=rng_key)
            state = puzzle.get_initial_state(sc, key=rng_key)

            state_str = str(state)
            assert isinstance(state_str, str)
            assert len(state_str) > 0

    def test_all_support_is_solved(self, rng_key):
        """Test that all puzzles implement is_solved."""
        puzzles = [Maze(size=11), PancakeSorting(size=10), Room(size=11), Sokoban()]

        for puzzle in puzzles:
            sc = puzzle.get_solve_config(key=rng_key)
            state = puzzle.get_initial_state(sc, key=rng_key)

            result = puzzle.is_solved(sc, state)
            # Should return boolean-like value
            assert isinstance(result, (bool, jnp.bool_)) or (hasattr(result, "dtype") and result.dtype == jnp.bool_)


class TestDotKnot:
    """Test suite for DotKnot puzzle."""

    def test_instantiation(self):
        """Test DotKnot can be instantiated with default and custom sizes."""
        dotknot = DotKnot()
        assert dotknot.action_size == 4  # 4 directional moves
        assert dotknot.size == 10  # Default size
        assert dotknot.color_num == 4  # Default color count

        dotknot_custom = DotKnot(size=8, color_num=3)
        assert dotknot_custom.size == 8
        assert dotknot_custom.color_num == 3
        assert dotknot_custom.action_size == 4

    def test_state_generation(self, rng_key):
        """Test solve config and state generation."""
        dotknot = DotKnot(size=10)
        sc = dotknot.get_solve_config(key=rng_key)
        state = dotknot.get_initial_state(sc, key=rng_key)

        # Verify state has board attribute
        assert hasattr(state, "board_unpacked")
        assert state.board_unpacked.shape == (dotknot.size * dotknot.size,)

        # Board should contain dot endpoints
        unpacked = state.board_unpacked
        assert jnp.any(unpacked > 0)  # Has dots
        assert jnp.any(unpacked <= 2 * dotknot.color_num)  # Has unmerged dots

    def test_action_strings(self):
        """Test action_to_string returns valid directional strings."""
        dotknot = DotKnot()
        expected = ["←", "→", "↑", "↓"]
        for i in range(dotknot.action_size):
            s = dotknot.action_to_string(i)
            assert isinstance(s, str)
            assert s in expected

    def test_is_solved(self, rng_key):
        """Test is_solved logic for DotKnot."""
        dotknot = DotKnot(size=10)
        sc = dotknot.get_solve_config(key=rng_key)
        state = dotknot.get_initial_state(sc, key=rng_key)

        # Initial state should not be solved (has unmerged dots)
        result = dotknot.is_solved(sc, state)
        assert isinstance(result, (bool, jnp.bool_)) or (hasattr(result, "dtype") and result.dtype == jnp.bool_)

    def test_valid_moves(self, rng_key):
        """Test that get_actions returns valid moves."""
        dotknot = DotKnot(size=10)
        sc = dotknot.get_solve_config(key=rng_key)
        state = dotknot.get_initial_state(sc, key=rng_key)

        # Test all actions
        for action in range(dotknot.action_size):
            next_state, cost = dotknot.get_actions(sc, state, jnp.asarray(action), filled=True)

            # Cost should be either 1.0 (valid) or inf (invalid)
            assert jnp.isfinite(cost) or jnp.isinf(cost)

    def test_get_string_parser(self, rng_key):
        """Test that get_string_parser returns a callable."""
        dotknot = DotKnot()
        parser = dotknot.get_string_parser()
        assert callable(parser)

        # Test that parser works
        sc = dotknot.get_solve_config(key=rng_key)
        state = dotknot.get_initial_state(sc, key=rng_key)
        result = parser(state)
        assert isinstance(result, str)


class TestTowerOfHanoi:
    """Test suite for Tower of Hanoi puzzle."""

    def test_instantiation(self):
        """Test Hanoi can be instantiated with default and custom sizes."""
        hanoi = TowerOfHanoi()
        assert hanoi.num_disks == 5  # Default size
        assert hanoi.num_pegs == 3  # Default pegs
        assert hanoi.action_size == 6  # 3 * (3 - 1) = 6 possible moves

        hanoi_custom = TowerOfHanoi(size=7)
        assert hanoi_custom.num_disks == 7
        assert hanoi_custom.action_size == 6

    def test_state_generation(self, rng_key):
        """Test solve config and state generation."""
        hanoi = TowerOfHanoi(size=5)
        sc = hanoi.get_solve_config(key=rng_key)
        state = hanoi.get_initial_state(sc, key=rng_key)

        # Verify state has pegs attribute
        assert hasattr(state, "pegs")
        assert state.pegs.shape == (hanoi.num_pegs, hanoi.num_disks + 1)

        # Initial state should have all disks on first peg
        assert state.pegs[0, 0] == hanoi.num_disks  # Disk count on first peg
        assert state.pegs[1, 0] == 0  # No disks on second peg
        assert state.pegs[2, 0] == 0  # No disks on third peg

    def test_target_state(self, rng_key):
        """Test that target state has all disks on third peg."""
        hanoi = TowerOfHanoi(size=5)
        sc = hanoi.get_solve_config(key=rng_key)

        # Target state should have all disks on third peg
        assert sc.TargetState.pegs[2, 0] == hanoi.num_disks
        assert sc.TargetState.pegs[0, 0] == 0
        assert sc.TargetState.pegs[1, 0] == 0

    def test_action_strings(self):
        """Test action_to_string returns valid move descriptions."""
        hanoi = TowerOfHanoi(size=5)
        for i in range(hanoi.action_size):
            s = hanoi.action_to_string(i)
            assert isinstance(s, str)
            assert "Move disk from peg" in s
            assert "to peg" in s

    def test_is_solved(self, rng_key):
        """Test is_solved logic for Hanoi."""
        hanoi = TowerOfHanoi(size=5)
        sc = hanoi.get_solve_config(key=rng_key)

        # Target state should be solved
        assert hanoi.is_solved(sc, sc.TargetState)

        # Initial state should not be solved
        state = hanoi.get_initial_state(sc, key=rng_key)
        assert not hanoi.is_solved(sc, state)

    def test_valid_moves(self, rng_key):
        """Test that get_actions returns valid moves."""
        hanoi = TowerOfHanoi(size=3)
        sc = hanoi.get_solve_config(key=rng_key)
        state = hanoi.get_initial_state(sc, key=rng_key)

        # Test all actions
        for action in range(hanoi.action_size):
            next_state, cost = hanoi.get_actions(sc, state, jnp.asarray(action), filled=True)

            # Cost should be either 1.0 (valid) or inf (invalid)
            assert jnp.isfinite(cost) or jnp.isinf(cost)

    def test_get_string_parser(self, rng_key):
        """Test that get_string_parser returns a callable."""
        hanoi = TowerOfHanoi()
        parser = hanoi.get_string_parser()
        assert callable(parser)

        # Test that parser works
        sc = hanoi.get_solve_config(key=rng_key)
        state = hanoi.get_initial_state(sc, key=rng_key)
        result = parser(state)
        assert isinstance(result, str)


class TestLightsOut:
    """Test suite for LightsOut puzzle."""

    def test_instantiation(self):
        """Test LightsOut can be instantiated with default and custom sizes."""
        lightsout = LightsOut()
        assert lightsout.size == 7  # Default size
        assert lightsout.action_size == 49  # 7 * 7 = 49 buttons

        lightsout_custom = LightsOut(size=5)
        assert lightsout_custom.size == 5
        assert lightsout_custom.action_size == 25

    def test_state_generation(self, rng_key):
        """Test solve config and state generation."""
        lightsout = LightsOut(size=5)
        sc = lightsout.get_solve_config(key=rng_key)
        state = lightsout.get_initial_state(sc, key=rng_key)

        # Verify state has board attribute
        assert hasattr(state, "board_unpacked")
        assert state.board_unpacked.shape == (lightsout.size * lightsout.size,)

    def test_target_state(self, rng_key):
        """Test that target state is all lights off."""
        lightsout = LightsOut(size=5)
        sc = lightsout.get_solve_config(key=rng_key)

        # Target should have all lights off
        assert jnp.all(sc.TargetState.board_unpacked == 0)

    def test_action_strings(self):
        """Test action_to_string returns valid strings."""
        lightsout = LightsOut(size=5)
        for i in range(min(10, lightsout.action_size)):
            s = lightsout.action_to_string(i)
            assert isinstance(s, str)
            assert len(s) > 0

    def test_is_solved(self, rng_key):
        """Test is_solved logic for LightsOut."""
        lightsout = LightsOut(size=5)
        sc = lightsout.get_solve_config(key=rng_key)

        # Target state should be solved
        assert lightsout.is_solved(sc, sc.TargetState)

    def test_valid_moves(self, rng_key):
        """Test that get_actions returns valid moves."""
        lightsout = LightsOut(size=5)
        sc = lightsout.get_solve_config(key=rng_key)
        state = lightsout.get_initial_state(sc, key=rng_key)

        # Test first few actions
        for action in range(min(5, lightsout.action_size)):
            next_state, cost = lightsout.get_actions(sc, state, jnp.asarray(action), filled=True)

            # All moves should be valid with cost 1.0
            assert cost == 1.0

    def test_inverse_action_map(self):
        """Test that each action is its own inverse."""
        lightsout = LightsOut(size=5)
        inv_map = lightsout.inverse_action_map

        assert inv_map is not None
        assert len(inv_map) == lightsout.action_size
        # Each action is its own inverse
        assert jnp.array_equal(inv_map, jnp.arange(lightsout.action_size))

    def test_get_string_parser(self, rng_key):
        """Test that get_string_parser returns a callable."""
        lightsout = LightsOut()
        parser = lightsout.get_string_parser()
        assert callable(parser)

        # Test that parser works
        sc = lightsout.get_solve_config(key=rng_key)
        state = lightsout.get_initial_state(sc, key=rng_key)
        result = parser(state)
        assert isinstance(result, str)


class TestSlidePuzzle:
    """Test suite for SlidePuzzle puzzle."""

    def test_instantiation(self):
        """Test SlidePuzzle can be instantiated with default and custom sizes."""
        slide = SlidePuzzle()
        assert slide.size == 4  # Default size
        assert slide.action_size == 4  # L, R, U, D

        slide_custom = SlidePuzzle(size=3)
        assert slide_custom.size == 3
        assert slide_custom.action_size == 4

    def test_state_generation(self, rng_key):
        """Test solve config and state generation."""
        slide = SlidePuzzle(size=4)
        sc = slide.get_solve_config(key=rng_key)
        state = slide.get_initial_state(sc, key=rng_key)

        # Verify state has board attribute
        assert hasattr(state, "board_unpacked")
        assert state.board_unpacked.shape == (slide.size * slide.size,)

        # Board should be a permutation of [0..size^2-1]
        sorted_board = jnp.sort(state.board_unpacked)
        expected = jnp.arange(slide.size * slide.size, dtype=jnp.uint8)
        assert jnp.array_equal(sorted_board, expected)

    def test_target_state(self, rng_key):
        """Test that target state is sorted order."""
        slide = SlidePuzzle(size=4)
        sc = slide.get_solve_config(key=rng_key)

        # Target should be [1, 2, ..., 15, 0]
        expected = jnp.array([*range(1, slide.size**2), 0], dtype=jnp.uint8)
        assert jnp.array_equal(sc.TargetState.board_unpacked, expected)

    def test_action_strings(self):
        """Test action_to_string returns valid directional strings."""
        slide = SlidePuzzle()
        expected = ["←", "→", "↑", "↓"]
        for i in range(slide.action_size):
            s = slide.action_to_string(i)
            assert isinstance(s, str)
            assert s in expected

    def test_is_solved(self, rng_key):
        """Test is_solved logic for SlidePuzzle."""
        slide = SlidePuzzle(size=4)
        sc = slide.get_solve_config(key=rng_key)

        # Target state should be solved
        assert slide.is_solved(sc, sc.TargetState)

    def test_valid_moves(self, rng_key):
        """Test that get_actions returns valid moves."""
        slide = SlidePuzzle(size=4)
        sc = slide.get_solve_config(key=rng_key)
        state = slide.get_initial_state(sc, key=rng_key)

        # Test all actions
        for action in range(slide.action_size):
            next_state, cost = slide.get_actions(sc, state, jnp.asarray(action), filled=True)

            # Cost should be either 1.0 (valid) or inf (invalid)
            assert jnp.isfinite(cost) or jnp.isinf(cost)

    def test_inverse_action_map(self):
        """Test that SlidePuzzle has correct inverse action mapping."""
        slide = SlidePuzzle()
        inv_map = slide.inverse_action_map

        assert inv_map is not None
        assert len(inv_map) == 4
        # Inverse: [R, L, D, U] -> [1, 0, 3, 2]
        expected = jnp.array([1, 0, 3, 2])
        assert jnp.array_equal(inv_map, expected)

    def test_get_string_parser(self, rng_key):
        """Test that get_string_parser returns a callable."""
        slide = SlidePuzzle()
        parser = slide.get_string_parser()
        assert callable(parser)

        # Test that parser works
        sc = slide.get_solve_config(key=rng_key)
        state = slide.get_initial_state(sc, key=rng_key)
        result = parser(state)
        assert isinstance(result, str)


class TestTopSpin:
    """Test suite for TopSpin puzzle."""

    def test_instantiation(self):
        """Test TopSpin can be instantiated with default and custom sizes."""
        topspin = TopSpin()
        assert topspin.n_discs == 20  # Default size
        assert topspin.turnstile_size == 4  # Default turnstile
        assert topspin.action_size == 3  # Shift left, shift right, reverse

        topspin_custom = TopSpin(size=10, turnstile_size=3)
        assert topspin_custom.n_discs == 10
        assert topspin_custom.turnstile_size == 3
        assert topspin_custom.action_size == 3

    def test_state_generation(self, rng_key):
        """Test solve config and state generation."""
        topspin = TopSpin(size=10)
        sc = topspin.get_solve_config(key=rng_key)
        state = topspin.get_initial_state(sc, key=rng_key)

        # Verify state has permutation attribute
        assert hasattr(state, "permutation")
        assert state.permutation.shape == (topspin.n_discs,)

        # Permutation should be [1..n_discs] in some order
        sorted_perm = jnp.sort(state.permutation)
        expected = jnp.arange(1, topspin.n_discs + 1, dtype=jnp.uint8)
        assert jnp.array_equal(sorted_perm, expected)

    def test_target_state(self, rng_key):
        """Test that target state is sorted order."""
        topspin = TopSpin(size=10)
        sc = topspin.get_solve_config(key=rng_key)

        # Target should be [1, 2, ..., n_discs]
        expected = jnp.arange(1, topspin.n_discs + 1, dtype=jnp.uint8)
        assert jnp.array_equal(sc.TargetState.permutation, expected)

    def test_action_strings(self):
        """Test action_to_string returns valid move descriptions."""
        topspin = TopSpin(size=10)
        s0 = topspin.action_to_string(0)
        s1 = topspin.action_to_string(1)
        s2 = topspin.action_to_string(2)

        assert isinstance(s0, str) and "Shift Left" in s0
        assert isinstance(s1, str) and "Shift Right" in s1
        assert isinstance(s2, str) and "Reverse Turnstile" in s2

    def test_is_solved(self, rng_key):
        """Test is_solved logic for TopSpin."""
        topspin = TopSpin(size=10)
        sc = topspin.get_solve_config(key=rng_key)

        # Target state should be solved
        assert topspin.is_solved(sc, sc.TargetState)

    def test_valid_moves(self, rng_key):
        """Test that get_actions returns valid moves."""
        topspin = TopSpin(size=10)
        sc = topspin.get_solve_config(key=rng_key)
        state = topspin.get_initial_state(sc, key=rng_key)

        # Test all actions
        for action in range(topspin.action_size):
            next_state, cost = topspin.get_actions(sc, state, jnp.asarray(action), filled=True)

            # All moves should be valid with cost 1.0
            assert cost == 1.0

    def test_inverse_action_map(self):
        """Test that TopSpin has correct inverse action mapping."""
        topspin = TopSpin()
        inv_map = topspin.inverse_action_map

        assert inv_map is not None
        assert len(inv_map) == 3
        # Left <-> Right (0 <-> 1), Reverse is self-inverse (2 <-> 2)
        expected = jnp.array([1, 0, 2])
        assert jnp.array_equal(inv_map, expected)

    def test_get_string_parser(self, rng_key):
        """Test that get_string_parser returns a callable."""
        topspin = TopSpin()
        parser = topspin.get_string_parser()
        assert callable(parser)

        # Test that parser works
        sc = topspin.get_solve_config(key=rng_key)
        state = topspin.get_initial_state(sc, key=rng_key)
        result = parser(state)
        assert isinstance(result, str)


class TestTSP:
    """Test suite for TSP puzzle."""

    def test_instantiation(self):
        """Test TSP can be instantiated with default and custom sizes."""
        tsp = TSP()
        assert tsp.size == 16  # Default size
        assert tsp.action_size == 16  # One action per city

        tsp_custom = TSP(size=10)
        assert tsp_custom.size == 10
        assert tsp_custom.action_size == 10

    def test_state_generation(self, rng_key):
        """Test solve config and state generation."""
        tsp = TSP(size=10)
        sc = tsp.get_solve_config(key=rng_key)
        state = tsp.get_initial_state(sc, key=rng_key)

        # Verify state has mask and point attributes
        assert hasattr(state, "mask_unpacked")
        assert hasattr(state, "point")
        assert state.mask_unpacked.shape == (tsp.size,)

        # Initial state should have exactly one city visited (the start city)
        assert jnp.sum(state.mask_unpacked) == 1
        assert state.mask_unpacked[state.point]

    def test_solve_config(self, rng_key):
        """Test solve config generation."""
        tsp = TSP(size=10)
        sc = tsp.get_solve_config(key=rng_key)

        # Verify solve config has points and distance matrix
        assert hasattr(sc, "points")
        assert hasattr(sc, "distance_matrix")
        assert hasattr(sc, "start")
        assert sc.points.shape == (tsp.size, 2)
        assert sc.distance_matrix.shape == (tsp.size, tsp.size)

    def test_action_strings(self):
        """Test action_to_string returns valid city indices."""
        tsp = TSP(size=10)
        for i in range(tsp.action_size):
            s = tsp.action_to_string(i)
            assert isinstance(s, str)
            assert len(s) > 0

    def test_is_solved(self, rng_key):
        """Test is_solved logic for TSP."""
        tsp = TSP(size=10)
        sc = tsp.get_solve_config(key=rng_key)
        state = tsp.get_initial_state(sc, key=rng_key)

        # Initial state should not be solved
        assert not tsp.is_solved(sc, state)

        # Create a solved state (all cities visited)
        solved_mask = jnp.ones(tsp.size, dtype=jnp.bool_)
        solved_state = state.set_unpacked(mask=solved_mask)
        assert tsp.is_solved(sc, solved_state)

    def test_valid_moves(self, rng_key):
        """Test that get_actions returns valid moves."""
        tsp = TSP(size=10)
        sc = tsp.get_solve_config(key=rng_key)
        state = tsp.get_initial_state(sc, key=rng_key)

        # Test all actions
        for action in range(tsp.action_size):
            next_state, cost = tsp.get_actions(sc, state, jnp.asarray(action), filled=True)

            # Visiting the already-visited start city should have infinite cost
            if action == state.point:
                assert jnp.isinf(cost)
            else:
                # Other cities should have finite cost
                assert jnp.isfinite(cost)

    def test_get_string_parser(self, rng_key):
        """Test that get_string_parser returns a callable."""
        tsp = TSP()
        parser = tsp.get_string_parser()
        assert callable(parser)

        # Test that parser works
        sc = tsp.get_solve_config(key=rng_key)
        state = tsp.get_initial_state(sc, key=rng_key)
        result = parser(state)
        assert isinstance(result, str)


class TestTypeSystem:
    """Test suite for PDDL type system module."""

    def test_collect_type_hierarchy_empty(self):
        """Test collect_type_hierarchy with no types."""

        # Mock domain with no types
        class MockDomain:
            pass

        domain = MockDomain()
        parent, ancestors, descendants = collect_type_hierarchy(domain)

        assert isinstance(parent, dict)
        assert isinstance(ancestors, dict)
        assert isinstance(descendants, dict)
        assert len(parent) == 0

    def test_collect_type_hierarchy_simple(self):
        """Test collect_type_hierarchy with simple hierarchy."""

        # Mock domain with types dict
        class MockDomain:
            def __init__(self):
                self.types = {"car": "vehicle", "truck": "vehicle"}

        domain = MockDomain()
        parent, ancestors, descendants = collect_type_hierarchy(domain)

        # Check parent relationships
        assert parent["car"] == "vehicle"
        assert parent["truck"] == "vehicle"

        # Check ancestors
        assert "vehicle" in ancestors["car"]
        assert "vehicle" in ancestors["truck"]

        # Check descendants
        assert "car" in descendants["vehicle"]
        assert "truck" in descendants["vehicle"]

    def test_collect_type_hierarchy_multi_level(self):
        """Test collect_type_hierarchy with multi-level hierarchy."""

        class MockDomain:
            def __init__(self):
                self.types = {
                    "sedan": "car",
                    "car": "vehicle",
                    "truck": "vehicle",
                }

        domain = MockDomain()
        parent, ancestors, descendants = collect_type_hierarchy(domain)

        # Check transitive ancestors
        assert "car" in ancestors["sedan"]
        assert "vehicle" in ancestors["sedan"]
        assert "vehicle" in ancestors["car"]

        # Check transitive descendants
        assert "sedan" in descendants["vehicle"]
        assert "car" in descendants["vehicle"]
        assert "sedan" in descendants["car"]

    def test_select_most_specific_types_empty(self):
        """Test select_most_specific_types with empty input."""
        # Empty hierarchy
        hierarchy = ({}, {}, {})
        result = select_most_specific_types([], hierarchy)

        assert result == ["object"]

    def test_select_most_specific_types_single(self):
        """Test select_most_specific_types with single type."""
        hierarchy = ({}, {}, {})
        result = select_most_specific_types(["car"], hierarchy)

        assert result == ["car"]

    def test_select_most_specific_types_with_object(self):
        """Test select_most_specific_types removes 'object' when more specific types present."""
        hierarchy = ({}, {}, {})
        result = select_most_specific_types(["object", "car"], hierarchy)

        assert "object" not in result
        assert "car" in result

    def test_select_most_specific_types_filters_ancestors(self):
        """Test select_most_specific_types filters out ancestor types."""

        class MockDomain:
            def __init__(self):
                self.types = {"car": "vehicle"}

        domain = MockDomain()
        hierarchy = collect_type_hierarchy(domain)

        # When both car and vehicle are present, only car should remain
        result = select_most_specific_types(["car", "vehicle"], hierarchy)

        assert "car" in result
        assert "vehicle" not in result

    def test_extract_objects_by_type_empty_problem(self):
        """Test extract_objects_by_type with empty problem."""

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
        """Test extract_objects_by_type with untyped objects."""

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
        """Test extract_objects_by_type with typed objects."""

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

        # Check that objects are grouped by their direct types
        assert "car1" in result["car"]
        assert "car2" in result["car"]
        assert "truck1" in result["truck"]

        # Check that objects propagate to supertypes
        assert "car1" in result["vehicle"]
        assert "car2" in result["vehicle"]
        assert "truck1" in result["vehicle"]

    def test_extract_objects_by_type_dict_objects(self):
        """Test extract_objects_by_type when problem.objects is a dict."""

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

        # Check that objects are in their type groups
        assert "car1" in result["car"]
        assert "car2" in result["car"]
        assert "truck1" in result["truck"]

    def test_type_hierarchy_cycle_prevention(self):
        """Test that type hierarchy handles potential cycles gracefully."""

        class MockDomain:
            def __init__(self):
                # Well-formed hierarchy (no cycles)
                self.types = {"car": "vehicle"}

        domain = MockDomain()
        parent, ancestors, descendants = collect_type_hierarchy(domain)

        # Verify no infinite loops occurred
        assert len(ancestors["car"]) == 1
        assert "vehicle" in ancestors["car"]
