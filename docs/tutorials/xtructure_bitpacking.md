# Xtructure Bitpacking in PuXle (Packed Runtime States)

PuXle uses [xtructure](https://github.com/tinker495/Xtructure.git) to represent puzzle states as JAX PyTrees.
For many puzzles, the *logical* state (e.g. a boolean board) can be stored more compactly by packing it into a
`uint8` byte-stream while still exposing an easy-to-use logical view for transitions and visualization.

This document summarizes the **recommended bitpacking patterns** in PuXle.

## 1) Field-level in-memory packing: `FieldDescriptor.packed_tensor`

Use `FieldDescriptor.packed_tensor(...)` when you want **a specific field** stored as packed bytes in-memory.

- The stored field (e.g. `board`) is a packed `uint8` array (byte-stream).
- Access `<field>_unpacked` to get the logical array.
- Use `from_unpacked(field=...)` to create a packed instance directly from logical arrays (most efficient).
- Use `set_unpacked(field=...)` to update an existing instance.

### Example: LightsOut (1 bit per cell)

```python
import jax.numpy as jnp
from puxle.core.puzzle_state import FieldDescriptor, state_dataclass


@state_dataclass
class State:
    board: FieldDescriptor.packed_tensor(shape=(49,), packed_bits=1)


# Create directly from logical view (bool[49]):
board = jnp.zeros((49,), dtype=bool)
state = State.from_unpacked(board=board)

# Read logical view:
board_view = state.board_unpacked

# Update logical view (auto-packed into state.board):
state2 = state.set_unpacked(board=jnp.logical_not(board_view))
```

### Example: Rubik's Cube (3 bits per sticker in color mode)

```python
import jax.numpy as jnp
from puxle.core.puzzle_state import FieldDescriptor, state_dataclass


@state_dataclass
class CubeState:
    # 6 faces x (size*size) stickers, each in [0..5] => 3 bits
    faces: FieldDescriptor.packed_tensor(shape=(6, 54), packed_bits=3)


cube = CubeState.from_unpacked(faces=faces)
faces_view = cube.faces_unpacked  # uint8[6,54]
cube2 = cube.set_unpacked(faces=(faces_view + 1) % jnp.uint8(6))
```

## 2) Aggregate bitpacking across fields (single packed stream per instance)

If you want to minimize per-field padding by packing **multiple fields together**, you can use aggregate bitpacking.

Pattern:
- Define fields with `bits=...` (for primitive leaves).
- xtructure can provide a packed representation for the entire dataclass via `.packed / .unpacked`
  (and may generate helper classes like `YourState.Packed` depending on xtructure version/config).

Use this when:
- you have many small fields and want a single compact stream,
- you want partial decode via `unpack_field(...)` (when supported).

## 3) PuXle convention: `@state_dataclass` defaults to `bitpack="auto"`

PuXle's `puxle.core.puzzle_state.state_dataclass` is a thin wrapper around xtructure's `@xtructure_dataclass`.

Behavior:
- Defaults to `bitpack="auto"` (enables packed_tensor helpers, and auto-enables aggregate packing when possible).
- For classes without bitpacking, PuXle provides identity `.packed` / `.unpacked` properties for compatibility.

## 4) Migration from legacy `to_uint8/from_uint8`

Older puzzle implementations sometimes:
- stored packed bytes in a `FieldDescriptor.tensor(...)`, and
- used custom `.packed` / `.unpacked` properties that changed the *shape semantics* of the same `State` class.

Recommended migration:
- Replace the manually-packed field with `FieldDescriptor.packed_tensor(...)`.
- Replace:
  - `state.unpacked.<field>` → `state.<field>_unpacked`
  - `State(<field>=logical).packed` → `State.from_unpacked(<field>=logical)`

The legacy helpers (`puxle.utils.util.to_uint8` / `from_uint8`) are still available for generic packing tasks,
but they should no longer be needed for in-memory puzzle-state representation.
