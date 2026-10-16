# Xtructure Bitpacking in PuXle (Packed Runtime States)

PuXle uses [xtructure](https://github.com/tinker495/Xtructure.git) to represent puzzle states as JAX PyTrees.
For many puzzles, the *logical* state (e.g. a boolean board) can be stored more compactly by packing it into a
`uint8` byte-stream while still exposing an easy-to-use logical view for transitions and visualization.

This document summarizes the **recommended bitpacking patterns** in PuXle.

## 1) Field-level in-memory packing: `FieldDescriptor.packed_tensor`

Use `FieldDescriptor.packed_tensor(...)` when you want **a specific logical field** stored as
packed bytes in-memory.

- `shape=` declares the logical unpacked **Intrinsic Shape** in the **Xtructure Schema**.
- The stored field (e.g. `board`) is a packed `uint8` array (byte-stream) whose storage
  shape is derived by **Type Layout** / **Packed Field Layout**.
- Access `<field>_unpacked` to get the logical array.
- Use `from_unpacked(field=...)` to create a packed instance directly from logical arrays (most efficient).
- Use `set_unpacked(field=...)` to update an existing instance.

### Example: LightsOut (1 bit per cell)

```python
import jax.numpy as jnp
from xtructure import FieldDescriptor, xtructure_dataclass


@xtructure_dataclass
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
from xtructure import FieldDescriptor, xtructure_dataclass


@xtructure_dataclass
class CubeState:
    # 6 faces x (size*size) stickers, each in [0..5] => 3 bits
    faces: FieldDescriptor.packed_tensor(shape=(6, 9), packed_bits=3)


cube = CubeState.from_unpacked(faces=faces)
faces_view = cube.faces_unpacked  # uint8[6,9]
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

## 3) Migration from legacy `to_uint8/from_uint8`

Older puzzle implementations sometimes:
- stored packed bytes in a `FieldDescriptor.tensor(...)`, and
- used custom `.packed` / `.unpacked` properties that changed the *shape semantics* of the same `State` class.

Recommended migration:
- Replace the manually-packed field with `FieldDescriptor.packed_tensor(...)`.
- Replace:
  - `state.unpacked.<field>` → `state.<field>_unpacked`
  - `State(<field>=logical).packed` → `State.from_unpacked(<field>=logical)`

The legacy helpers (`puxle.utils.util.to_uint8` / `from_uint8`) have been removed.
Use the xtructure field or aggregate codec rather than introducing a puzzle-local replacement.

## 4) Measure physical layout and transition cost

For `N` logical elements of width `b`, the field codec uses blocks of `lcm(b, 8)` bits.
Its storage is `ceil(N / (lcm(b, 8) / b)) * (lcm(b, 8) / 8)` bytes, which can exceed
`ceil(N * b / 8)`. Aggregate packing has different word/tail padding and is not always smaller.
Compare actual stored PyTree leaf bytes and GPU transitions; a narrower field can introduce
additional gather/decode work. Packed XOR and single-bit updates can avoid whole-field decoding.

The [native puzzle packing guide](../../puxle/puzzles/AGENTS.md) records per-puzzle value ranges,
padding, shared context, transition invariants, and alternatives that need measurement.
