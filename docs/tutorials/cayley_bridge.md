# cayleypy bridge — Cayley/Schreier graphs as PuXle puzzles

PuXle ships an adapter that consumes [cayleypy](https://github.com/cayleypy/cayleypy)'s `CayleyGraphDef` as a JAX-native `Puzzle`. This lets you run JAxtar A\*/Q\*/beam search on any of cayleypy's ~30 permutation-group families and a growing list of matrix groups and Schreier graphs (mini_pyramorphix, pyraminx, globe_puzzle, …) **without dragging PyTorch into the JAX stack**.

The bridge has two layers:

1. The adapter class `CayleyPuzzle(Puzzle)` — wraps a single `CayleyGraphDef` instance.
2. A small `puxle.puzzles.cayley_subclasses` module — exposes no-arg subclasses of `CayleyPuzzle` for JAxtar registry entries.

## Installation

```bash
pip install "puxle[cayley]"
```

The PuXle codebase imports cleanly without cayleypy installed; construction raises `ImportError` naming the extra.

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

### Convenience factory

The classmethod `CayleyPuzzle.from_cayleypy_factory(name, *args, **kwargs)` looks up `name` in `PermutationGroups → MatrixGroups → Puzzles` (first-wins) and forwards args:

```python
puzzle = CayleyPuzzle.from_cayleypy_factory("top_spin", 8, k=4)
```

## 2. Registry subclasses

Some downstream registries (e.g. JAxtar's `config/puzzle_registry.py`) expect no-arg `Puzzle` classes. PuXle exposes only the registry subclasses it actually uses:

```python
from puxle import (
    CayleyPancake7,            # PermutationGroups.pancake(7)
    CayleyPancake8,            # PermutationGroups.pancake(8)
    CayleyLRX8,                # PermutationGroups.lrx(8)
    CayleyTopSpin8K4,          # PermutationGroups.top_spin(8, k=4)
    CayleyCoxeter8,            # PermutationGroups.coxeter(8)
)

puzzle = CayleyTopSpin8K4(num_shuffle=20)
```

Use `CayleyPuzzle.from_cayleypy_factory(...)` directly for any other cayleypy graph.

## Using the bridge with JAxtar

The bridge's primary motivation is enabling JAxtar to search cayleypy graphs. JAxtar ships pre-registered entries for five graphs (`cayley-pancake-7`, `cayley-pancake-8`, `cayley-lrx-8`, `cayley-top-spin-8-k4`, `cayley-coxeter-8`) — invoke them via the JAxtar CLI:

```bash
python main.py astar -p cayley-pancake-7
```

Empirically (autoresearch sweep, py312 CPU, `EmptyHeuristic`):

| Family | Comfortable range | First boundary |
|---|---|---|
| `pancake(n)` | n ≤ 10, scramble 25, ≤ 1.3 M nodes | n = 12 OOMs at 4 M nodes |
| `lrx(n)` | n ≤ 10, scramble 30, ≤ 9 k nodes | n = 12 OOMs at 2 M nodes |
| `top_spin(n, k=4)` | n ≤ 10 | not yet probed beyond n = 10 |
| `heisenberg(n=3, modulo ≤ 7)` | all solved < 1 s | not yet probed |
| `SL_*(n=2, modulo ≤ 5)` | all solved < 1 s | not yet probed |
| `mini_pyramorphix`, `pyraminx`, `globe_puzzle(2,3)` | all solved | `pyraminx` scramble 10 → 222 k nodes |

Beyond the boundaries, an informed heuristic (learned or pattern-database) is needed; the current bridge plus `EmptyHeuristic` is a baseline.

## See also

- Adapter source: `puxle/puzzles/cayley_puzzle.py`
- Meta-module source: `puxle/puzzles/cayley_subclasses.py`
- cayleypy upstream: [https://github.com/cayleypy/cayleypy](https://github.com/cayleypy/cayleypy)
