<div align="center">
  <img src="images/PuXle.png" alt="PuXle Logo" width="500">
</div>

# PuXle: Parallelized Puzzles with JAX

[![PyPI version](https://badge.fury.io/py/puxle.svg)](https://badge.fury.io/py/puxle)
[![Python Version](https://img.shields.io/pypi/pyversions/puxle.svg)](https://pypi.org/project/puxle/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/tinker495/puxle/workflows/tests/badge.svg)](https://github.com/tinker495/puxle/actions)

**PuXle** is a high-performance library for parallelized puzzle environments built on JAX. It provides a collection of classic puzzles optimized for AI research, reinforcement learning, and search algorithms.

## 🚀 Features

- **High Performance**: JAX-powered parallelization for lightning-fast puzzle solving
- **Diverse Puzzles**: 11+ classic puzzles including Rubik's Cube, Sokoban, Sliding Puzzle, and more
- **AI Research Ready**: Perfect for reinforcement learning, search algorithms, and AI research
- **Batch Processing**: Efficient batch operations for training and evaluation
- **Extensible**: Easy-to-extend base classes for creating new puzzle environments
- **GPU Acceleration**: Full GPU support through JAX

## 📦 Installation

### Basic Installation
```bash
pip install puxle
```

### With CUDA Support (Recommended for GPU acceleration)
```bash
pip install "puxle[cuda]"
```

### Development Installation
```bash
git clone https://github.com/tinker495/puxle.git
cd puxle
pip install -e ".[dev]"
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

# Batch solve checking
solved_states = puzzle.batched_is_solved(solve_configs, initial_states)
print(f"Solved states in batch: {jnp.sum(solved_states)}")
```

### Custom Puzzle Creation
```python
from puxle import Puzzle, state_dataclass
from puxle.puzzle_state import FieldDescriptor

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
        # Define valid moves and their costs
        # Return (next_states, costs) 
        pass
    
    def is_solved(self, solve_config, state):
        # Check if current state matches target
        return jnp.array_equal(state.position, solve_config.TargetState.position)
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

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)  
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [JAX](https://github.com/google/jax) for high-performance computing
- Inspired by classic puzzle environments and AI research
- Special thanks to the JAX and Python communities

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/tinker495/puxle/issues)
- **Discussions**: [GitHub Discussions](https://github.com/tinker495/puxle/discussions)
- **Email**: wjdrbtjr495@gmail.com

## 📈 Citation

If you use PuXle in your research, please cite:

```bibtex
@software{puxle2025,
  title={PuXle: Parallelized Puzzles with JAX},
  author={Jung, KyuSeok},
  year={2025},
  url={https://github.com/tinker495/puxle}
}
```

---

<div align="center">
  Made with ❤️ for the AI research community
</div>
