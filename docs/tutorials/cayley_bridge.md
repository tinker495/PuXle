# cayleypy bridge — Cayley/Schreier graphs as PuXle puzzles

PuXle ships an adapter that consumes [cayleypy](https://github.com/cayleypy/cayleypy)'s `CayleyGraphDef` as a JAX-native `Puzzle`. This lets you run JAxtar A\*/Q\*/beam search on any of cayleypy's ~30 permutation-group families and a growing list of matrix groups and Schreier graphs (mini_pyramorphix, pyraminx, globe_puzzle, …) **without dragging PyTorch into the JAX stack**.

The bridge has two layers:

1. The adapter class `CayleyPuzzle(Puzzle)` — accepts a cayleypy factory name (e.g. `"top_spin"`) or a `CayleyGraphDef` instance directly.
2. Curated constructor kwargs in `CayleyPresets` (`puxle.puzzles.adapters.cayley.presets`) and live discovery via `CayleyPuzzle.list_sources()`.

## Installation

```bash
uv add "puxle @ git+https://github.com/tinker495/PuXle.git"
```

`cayleypy` is included in PuXle's default dependencies. If imported in an environment where `cayleypy` is absent, construction raises `ImportError`.

## 1. The adapter class

```python
import jax
from cayleypy.graphs_lib import PermutationGroups
from puxle import CayleyPuzzle

# Any cayleypy CayleyGraphDef factory works.
graph_def = PermutationGroups.pancake(7)
puzzle = CayleyPuzzle(graph_def, num_shuffle=10)

solve_config, initial_state = puzzle.get_inits(jax.random.PRNGKey(0))
neighbours, costs = puzzle.get_neighbours(solve_config, initial_state)
```

### Supported generator types

- `PERMUTATION` — state is a length-`n` int permutation vector. Transition is `jnp.take(state, perm[action])`.
- `MATRIX` — state is a flat length-`n*m` int vector. Transition is left-multiply `(M[action] @ vec_view) % modulo`. The kernel uses int64 intermediates under `JAX_ENABLE_X64=True` (overflow ceiling `m*(modulo-1)² < 2⁶³`) or int32 under the default mode (tighter ceiling `< 2³¹`).

### Construction-time policy

- `num_shuffle: int = 100` — number of random actions applied in `get_initial_state` (the scrambler walks from the goal).
- `ensure_inverse_closed: bool = True` — when `True`, the adapter mirrors cayleypy's `with_inverted_generators` so the puzzle is fully reversible (and JAxtar's `bi_*` algorithms work). Setting `False` keeps the original generator set; if it is not naturally closed under inversion the puzzle becomes non-reversible, `inverse_action_map` returns `None`, and a `RuntimeWarning` is emitted naming `bi_astar`/`bi_qstar` as unusable.

### Direct factory lookup & curated presets

Passing a string as `graph` looks up the name in `PermutationGroups → MatrixGroups → Puzzles` (first-wins) and forwards positional/keyword factory arguments:

```python
from puxle import CayleyPresets, CayleyPuzzle

puzzle = CayleyPuzzle("top_spin", 8, k=4, num_shuffle=20)
puzzle = CayleyPuzzle(**CayleyPresets.TOP_SPIN_8_K4, num_shuffle=20)
sources = CayleyPuzzle.list_sources()
```

## See also

- Adapter source: `puxle/puzzles/adapters/cayley/cayley.py`
- Presets source: `puxle/puzzles/adapters/cayley/presets.py`
- cayleypy upstream: [https://github.com/cayleypy/cayleypy](https://github.com/cayleypy/cayleypy)
