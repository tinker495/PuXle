<!-- Parent: ../../AGENTS.md -->

# Native Puzzle Packing and Transitions

This directory owns state representations for native puzzles, their Random and Hard variants, Room,
and external puzzle adapters under `adapters/` (`adapters/cayley/`, `adapters/pddl/`). World models
have a separate owner. Optimize storage while preserving logical values, action numbering, costs,
inverse-transition semantics, and DeepCubeA/Santa data. Historical packed bytes and serialized
PyTrees of internal State types are not stable external formats.

## Measuring Storage

- **Information lower bound**: For `S` valid states, `ceil(log2(S))` bits. Achieving this via
  rank/unrank does not imply fast transitions or partial access.
- **Fixed-width element lower bound**: Values in `0..M` require `max(1, M.bit_length())` bits; a
  permutation of `1..N` requires `N.bit_length()` bits per element.
- **Actual State storage**: Sum `nbytes` over stored PyTree leaves, distinct from `_unpacked` views,
  GPU allocator reservations, and search structure memory.
- **Shared context**: Maze walls and TSP coordinates/distance matrices belong in
  `SolveConfig.InstanceContext` rather than per state (batched problems batch context).
- **Temporary memory and traffic**: Account for unpack, gather/scatter, and repack intermediates;
  fewer stored bytes do not guarantee fewer memory transactions or faster kernels.

`FieldDescriptor.packed_tensor(shape=..., packed_bits=b)` stores packed `uint8`. For `N` logical
elements and `L = lcm(b, 8)`, the codec uses `ceil(N / (L / b)) * (L / 8)` bytes (block tail
padding makes e.g. 25×5b = 20 B and 2×5b = 5 B). Changing `bits=b` metadata on `tensor(...)` does
not shrink leaves unless using aggregate `.packed`, whose 32-bit word + tail padding is not always
smaller than field packing (e.g., 35×6b is 27 B field-packed vs 28 B aggregate-packed).

## Representation and Transition Contracts

1. Use `State.from_unpacked(...)` for logical inputs and `set_unpacked(...)` for logical updates;
   raw constructors and `replace` expect storage dtype and shape.
2. Packed 1-bit fields use little-bit order (cell `i` at bit `i % 8` of byte `i // 8`). Keep
   unused final bits zero so identical logical states hash and compare equal.
3. Avoid full unpack/repack when XOR or bit checks/updates work directly on packed bytes; narrow
   bits can raise decode costs for irregular gathers.
4. Compute coordinate differences, negative moves, and matrix products in wider signed dtypes
   before narrowing to storage dtype; validate before narrowing.
5. Keep original PRNG generation dtypes and cast only the result when narrowing generated data so
   seeds preserve identical logical samples.
6. `_apply` and `_get_neighbours` return candidate states and costs without duplicating `filled`
   checks or nonfinite-cost state restoration; `Puzzle.get_actions` / `Puzzle.get_neighbours` own
   those invariants.
7. Validate `State` together with `GoalSpec`, `InstanceContext` indices, backward-search init,
   renderers, JAxtar preprocessing, and benchmark input paths.
8. Batch axes are distinct from intrinsic field axes; batch single-state bit indexing via `vmap`
   and do not infer canonical dtypes/shapes from raw `State` construction.

## Optimizing Single Actions and All Neighbors Independently

Every native base class implements `_apply` and `_get_neighbours` independently without calling each
other (Random/Hard, Room, and Cayley inherit them). Subclasses altering transition semantics must
override both; subclasses altering only goals or initial samples inherit both.

Both paths must produce identical logical states and costs per action (`(action, batch, ...)` for
public batched APIs). `_apply` receives a scalar JAX array `0 <= action < action_size` (the public
boundary normalizes Python ints), and invalid final transitions leave all input leaves unchanged.
Preserve the existing `Puzzle.get_actions` mapping for non-native environments (e.g. PDDL's
cost-only `filled` handling or world-model batching). Batch axes of `gather`, `stack`, and `vmap`
alter XLA fusion and transposes independently—never assume extending one path optimizes the other
(compare TopSpin forward vs inverse kernels separately).

## Per-Puzzle Invariants

### DotKnot

`0` is empty, `1..2*C` are distinct endpoints, and `2*C+1..3*C` are color paths. Required width is
`(3*C).bit_length()` (4 bits for default `C=4`; insufficient for `C>5`). Endpoint identity governs
which dot moves next, so merging endpoints alters transitions. Multiple nibbles in one byte can
change together; combine clear masks and new values with OR and compute coordinates in signed
integers. The puzzle is irreversible.

### Hanoi

Because disk order is fixed, `disk_pegs` (`(num_pegs-1).bit_length()` bits per disk via
`State.from_unpacked(disk_pegs=...)`) encodes the state without redundant counts or slots. `.pegs`
is a read-only `(num_pegs, num_disks+1)` `uint8` view (capping view/renderer support at 255 disks);
compute it on host-resident states with NumPy rather than on the device hot path.

The top disk is the smallest disk index on each peg; using `num_disks` as the empty-peg sentinel
lets `source_top < target_top` enforce both non-empty source and size ordering. Vmap bulk codec
calls over the action axis and never pass an action axis to scalar `from_unpacked`.

Preserve the zero-disk boundary (`num_disks == 0`): storage is 0 B, every move is an infinite-cost
self-loop, and the goal is satisfied. Top-disk reductions need explicit `initial=num_disks`, and for
empty intrinsic shapes the shared field codec must use the known batch element count instead of `-1`
to avoid divide-by-zero reshapes and keep host/device empty arrays consistent.

### LightsOut and Random

`Random` changes goals and initial samples while keeping transitions and `State` layout identical.
General logical inputs can belong to cosets outside a single goal's GF(2) reachable coset, so
never assume single-coset reachability without explicit goal context. The `_move_matrix` XOR table
scales `O(size⁴)` in space and compile constants.

### Maze and Room

Coordinates are `uint8` through size 256 and `uint16` from 257 (5-bit `packed_tensor` coordinates
inflate storage via `lcm(5, 8)` padding); larger ranges are unsupported. Walls live in the shared
1-bit `Maze` bitmap in `InstanceContext`; `Room` adjusts `size`, so compute capacity and storage
from `room.size`.

Compute coordinates, bounds, and negative moves in signed `int32` before narrowing (`uint8(-1)`
wraps to `255` and creates invalid moves at size 256). When `size² > 2³¹`, widen only the flat
index to `uint32` (e.g. `46341×46341` has max index `2147488280`, overflowing signed `int32`);
keep packed-byte shift operands and masks in `uint8` so index dtypes do not widen bit data.

### PancakeSorting and TopSpin

Both store `1..N` permutations in `uint8` (`N <= 255`; 1-based labels use one more bit than 0-based
at powers of 2). Preserve `.stack` / `.permutation` and constructor contracts for downstream
heuristics. In TopSpin, 0- or 1-element reversal segments are the identity and use direct
rotation/identity branches; normalize reversal length like a Python slice to preserve negative
`turnstile_size` semantics.

### RubiksCube and Random

Color mode stores `0..5` (3 bits/sticker); tile-ID mode stores `0..6*size²-1` in `uint8` (max 256
tiles). Preserve general sizes, QTM/UQTM, Random goals, and Santa tile-ID inputs. Public
invalid-state selection affects XLA transpose fusion.

### SlidePuzzle, Random, and Hard

Tiles are a permutation of `0..size²-1` with blank `0`; preserve bundled Hard 3/4 inputs and tile
order. Keep transition support for user-supplied `9..16`-bit `State.from_unpacked(board=...)`
inputs distinct from the default generator's `uint8` range.

Swaps XOR both bit ranges at `cell * b`. Max starting offset is `8 - gcd(b, 8)`, so a field spans
up to `ceil((b + 8 - gcd(b, 8)) / 8)` byte fragments (1 for 2/4b, 2 for 9/10b, 3 for 11b): low
fragment by left shift, later fragments by right shift of `8*k - offset`, cast to `uint8` before
XORing so wide dtypes do not leak into `board`. Dispatch aligns with codec blocks: 6-bit fields
align to 24-bit blocks (`where`+repack fuses cleanly), 8-bit fields align to bytes, and 5/7-bit
(`40`/`56`-bit blocks) and wider fields use packed XOR fragments. For 2×2 single actions on one
byte, preserve `argmax` tie-breaking when bytes contain multiple zeros or none (first zero, or 0).

### Sokoban and Hard

Default and Hard are distinct datasets. Walls stay inside the 10×10 2-bit board so transition
ownership stays self-contained (moving walls to `InstanceContext` requires updating goals, walls,
encoding, renderer, heuristic, and inverse contracts together). Under the inverse scrambling
contract, blocked directions are self-loops with cost `1` and `filled=False` changes only cost to
`jnp.inf` (distinct from the forward public contract). Clear the box cell and place a box at the
player cell only when pulling a front box, and move only when the prior player cell is empty and in
bounds.

### TSP

`Visited` uses little-bit order; the completed final byte is `(1 << (size % 8 or 8)) - 1`, and
earlier bytes are `255` (`is_solved` and final-visit detection share this rule). Revisits cost
`jnp.inf`, and the closing leg back to start is added only on the final visit (`visit_count ==
size - 1`). Ordinary edges read `distance_matrix` while the closing leg uses the coordinates'
Euclidean norm—preserve this difference. `current_point` cannot be recovered from the mask alone
because it is already marked visited. Single-action updates use bitwise OR with a byte-index mask
rather than immutable array scatter to avoid copy/update overhead.

### CayleyPuzzle

`PERMUTATION` preserves the central state's value range (`uint8` for `0..255`, `uint16` up to
`65535`, `int32` for negative or larger values) instead of relabeling to `0..n-1`. `MATRIX` uses
`uint8` when both the central-state range and `modulo-1` fit in `0..255`, otherwise `int32`;
compute `(M @ vec)` in `int32` (or `int64` under `JAX_ENABLE_X64=True`), respecting the
`m*(modulo-1)^2` overflow precondition before narrowing after `% modulo`. Preserve inverse closure,
`inverse_action_map`, and the `ensure_inverse_closed=False` warning/non-reversible contract. Call
`jax.device_get` once before formatting elements to avoid per-element device syncs from `int(...)`.

## GPU Profiling Gotchas

For short kernels, Python dispatch and GPU clock changes can dominate wall time. Separate profiler
GPU-kernel time from call overhead, warm each operation before alternating baseline/candidate order
within one process, and correlate CPU launches to GPU kernels via correlation IDs (trace clocks
differ). Allow a finalization period before trace shutdown and verify every CPU launch has a device
event; never treat missing calls as zero or decide from medians with missing samples. Zero-byte
copy APIs for empty packed fields emit no device work; check explicit `size:0` trace records to
distinguish these no-ops while still checking nonempty copies. If short calls leave the GPU in a
power-saving state, warm it with the same GPU operation immediately before each call group (outside
measurement) rather than altering global clock settings.
