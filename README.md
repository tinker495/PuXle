<div align="center">
  <img src="images/PuXle.svg" alt="PuXle Logo" width="500">
</div>

# PuXle: Parallelized Puzzles with JAX

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**PuXle** is a high-performance library for parallelized puzzle environments built on JAX. It provides a collection of classic puzzles optimized for AI research, reinforcement learning, and search algorithms. PuXle leverages [Xtructure](https://github.com/tinker495/Xtructure.git) as a backend for JAX-optimized data structures, offering advantages such as efficient batched operations, GPU-accelerated priority queues and hash tables for faster puzzle solving in AI research. PuXle serves as the puzzle implementation backend for [JAxtar](https://github.com/tinker495/JAxtar), enabling JAX-native parallelizable A* and Q* solvers for neural heuristic search research.

## 🚀 Features

- **High Performance**: JAX-powered parallelization for lightning-fast puzzle solving
- **Optimized Data Structures**: Powered by Xtructure for efficient batched operations, GPU-accelerated priority queues, and hash tables
- **Diverse Puzzles**: 11+ classic puzzles including Rubik's Cube, Sokoban, Sliding Puzzle, and more
- **AI Research Ready**: Perfect for reinforcement learning, search algorithms, and AI research
- **Batch Processing**: Efficient batch operations for training and evaluation
- **Extensible**: Easy-to-extend base classes for creating new puzzle environments
- **GPU Acceleration**: Full GPU support through JAX

## 📦 Installation

### Basic Installation
```bash
pip install puxle
pip install git+https://github.com/tinker495/PuXle.git # recommended
```

## 🎯 Quick Start

```python
import jax
import jax.numpy as jnp
from puxle import RubiksCube, SlidePuzzle, Sokoban

# Initialize a puzzle
puzzle = RubiksCube()
key = jax.random.PRNGKey(42)

# Get initial state and solve configuration
solve_config, initial_state = puzzle.get_inits(key)

# Get available actions
neighbors, costs = puzzle.get_neighbours(solve_config, initial_state)

# Check if solved
is_solved = puzzle.is_solved(solve_config, initial_state)
print(f"Initial state solved: {is_solved}")

# Visualize the puzzle (if image parser is available)
try:
    image = puzzle.State.img(initial_state)
    print(f"State visualization shape: {image.shape}")
except:
    print("Image visualization not available for this puzzle")
```

## 🧩 Available Puzzles

| Puzzle | Description | Difficulty |
|--------|-------------|------------|
| **RubiksCube** | Classic 3×3×3 Rubik's Cube | Hard |
| **SlidePuzzle** | N×N sliding tile puzzle | Medium |
| **Sokoban** | Box-pushing warehouse puzzle | Hard |
| **TowerOfHanoi** | Classic disk-moving puzzle | Medium |
| **LightsOut** | Toggle lights to turn all off | Medium |
| **Maze** | Navigate through a maze | Easy-Medium |
| **TSP** | Traveling Salesman Problem | Hard |
| **PancakeSorting** | Sort pancakes by flipping | Medium |
| **TopSpin** | Circular sliding puzzle | Medium |
| **DotKnot** | Untangle knots puzzle | Medium |

## 📚 Advanced Usage

### Batch Processing
```python
# Process multiple puzzle instances simultaneously
batch_size = 1000
keys = jax.random.split(jax.random.PRNGKey(0), batch_size)

# Generate batch of initial states
solve_configs = jax.vmap(lambda k: puzzle.get_solve_config(k))(keys)
initial_states = jax.vmap(lambda sc, k: puzzle.get_initial_state(sc, k))(solve_configs, keys)

# Batch solve checking with multiple solve configs
solved_states = puzzle.batched_is_solved(solve_configs, initial_states, multi_solve_config=True)
print(f"Solved states in batch: {jnp.sum(solved_states)}")

# Alternative: Use single solve config for all states
single_solve_config = puzzle.get_solve_config(jax.random.PRNGKey(42))
solved_states_single = puzzle.batched_is_solved(single_solve_config, initial_states, multi_solve_config=False)
print(f"Solved states with single config: {jnp.sum(solved_states_single)}")
```

### Custom Puzzle Creation
```python
import jax.numpy as jnp
from puxle import Puzzle, state_dataclass, FieldDescriptor

class CustomPuzzle(Puzzle):
    action_size = 4  # Number of possible actions
    
    def define_state_class(self):
        @state_dataclass
        class State:
            position: FieldDescriptor[jnp.ndarray]  # Current position
            
        return State
    
    def get_solve_config(self, key=None, data=None):
        # Define target configuration
        return self.SolveConfig(TargetState=self.State(position=jnp.array([0, 0])))
    
    def get_initial_state(self, solve_config, key=None, data=None):
        # Define initial state
        return self.State(position=jnp.array([5, 5]))
    
    def get_neighbours(self, solve_config, state, filled=True):
        # Define valid moves: up, down, left, right
        moves = jnp.array([[0, 1], [0, -1], [-1, 0], [1, 0]])
        new_positions = state.position[None, :] + moves
        
        # Create next states
        next_states = jax.tree_util.tree_map(
            lambda x: jnp.repeat(x[None, :], 4, axis=0), state
        )
        next_states = next_states.replace(position=new_positions)
        
        # All moves have cost 1
        costs = jnp.ones(4)
        
        return next_states, costs
    
    def is_solved(self, solve_config, state):
        # Check if current state matches target
        return jnp.array_equal(state.position, solve_config.TargetState.position)
    
    def get_string_parser(self):
        # Required: Return function to convert state to string
        def string_parser(state):
            return f"Position: ({state.position[0]}, {state.position[1]})"
        return string_parser
    
    def get_img_parser(self):
        # Required: Return function to convert state to image array
        def img_parser(state):
            # Create a simple 10x10 grid visualization
            grid = jnp.zeros((10, 10, 3))
            x, y = state.position
            # Place a marker at current position (clamp to grid bounds)
            x = jnp.clip(x, 0, 9)
            y = jnp.clip(y, 0, 9)
            grid = grid.at[y, x].set(jnp.array([1.0, 0.0, 0.0]))  # Red dot
            return grid
        return img_parser
```

## 🔧 API Reference

### Core Classes

- **`Puzzle`**: Base class for all puzzles
- **`PuzzleState`**: Base class for puzzle states
- **`SolveConfig`**: Configuration class for puzzle objectives

### Key Methods

- **`get_inits(key)`**: Get initial state and solve configuration
- **`get_neighbours(solve_config, state)`**: Get valid next states and costs
- **`is_solved(solve_config, state)`**: Check if puzzle is solved
- **`batched_*`**: Batch versions of core methods for parallel processing

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=puxle tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## See Also

- [Xtructure](https://github.com/tinker495/Xtructure): JAX-optimized data structures used as a backend for PuXle.
- [JAxtar](https://github.com/tinker495/JAxtar): JAX-native parallelizable A* and Q* solver that uses PuXle as its puzzle implementation backend.
