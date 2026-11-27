<!-- Parent: ../../AGENTS.md -->

# Native Puzzle Packing and Transitions

Write documentation in this directory in English, including this file. Follow
the repository's [documentation and communication policy](../../AGENTS.md#language-and-review).

This directory owns the state representations of native puzzles, their Random
and Hard variants, Room, and puzzle adapters under `adapters/`
(`adapters/cayley/`, `adapters/pddl/`, and `adapters/world_model/`). World-model
training kernels, trajectory datasets, and artifact cache paths live in
`puxle/world_model/`. Optimize storage while preserving logical puzzle values,
action numbering, costs, inverse-transition semantics, and DeepCubeA/Santa data.
Historical packed bytes and serialized PyTrees of internal State types are not
stable external formats.

Read [PACKING_REFERENCE.md](PACKING_REFERENCE.md) when changing a specific
puzzle's bit-packing layout, checking per-puzzle storage bounds, or running GPU
transition microbenchmarks.

## Measuring Storage

- **Information lower bound**: For `S` valid states, the bound is
  `ceil(log2(S))` bits. Achieving this bound with rank/unrank does not imply fast
  transitions or partial access.
- **Fixed-width element lower bound**: Values in `0..M` require
  `max(1, M.bit_length())` bits. A permutation of `1..N` requires
  `N.bit_length()` bits per element.
- **Actual State storage**: Sum `nbytes` over stored PyTree leaves, distinct from
  `_unpacked` views, GPU allocator reservations, and search containers.
- **Shared context**: Static problem data (Maze walls, TSP coordinates and
  distance matrices) belongs to `SolveConfig.InstanceContext`.
- **Temporary memory and traffic**: Account for intermediate arrays from unpack,
  gather/scatter, and repack; fewer stored bytes do not guarantee faster kernels.

`FieldDescriptor.packed_tensor(shape=..., packed_bits=b)` stores the field as
packed `uint8`. For `N` logical elements and `L=lcm(b,8)`, the codec uses
`ceil(N / (L/b)) * (L/8)` bytes (including end-of-block padding). Setting `bits=b`
on an ordinary `tensor(...)` shrinks storage only when paired with aggregate
`.packed` storage, which uses 32-bit words and can exceed field packing on
three-byte tails.

## Representation and Transition Contracts

1. Use `State.from_unpacked(...)` for logical inputs and `set_unpacked(...)` for
   logical field updates. Pass storage-dtype/shape values to raw constructors and
   `replace`.
2. Packed one-bit fields use little-bit order: cell `i` is bit `i%8` of byte
   `i//8`, with unused final bits zeroed so byte-based hashing and equality stay
   canonical.
3. Operate directly on packed bytes with XOR or bit masks when practical; measure
   decode overhead before narrowing irregular permutation gathers.
4. Compute coordinate differences, negative moves, and matrix-product
   intermediates in wider signed dtypes and validate before narrowing to storage
   dtypes.
5. Keep random-generation dtypes stable and narrow only the generated result when
   preserving seed-reproducible samples.
6. `_apply` computes a single-action candidate and cost; `_get_neighbours`
   computes all-action candidates and costs. `Puzzle.get_actions` and
   `Puzzle.get_neighbours` own restoring the input state when `filled=False` or
   cost is nonfinite.
7. Validate `State` changes together with `GoalSpec`, `InstanceContext`,
   backward-search initialization, renderers, JAxtar preprocessing, and
   benchmarks.
8. Keep batch axes separate from intrinsic logical-field axes; batch single-state
   bit indexing via `jax.vmap`.

## Optimizing Single Actions and All Neighbors Independently

Every native base class implements independent `_apply` and `_get_neighbours`
kernels; Random/Hard, Room, and Cayley subclasses inherit them unless transition
semantics change. Both paths must return identical logical states and costs in
action order (`(action, batch, ...)` on public batched APIs), with invalid
transitions preserving every input leaf unchanged. Shared `Puzzle` preserves
`get_actions` fallback for adapters (`PDDL`, `WorldModelPuzzle`) that do not
implement `_get_neighbours`.

| Puzzle | Single-action path | All-neighbor path |
| --- | --- | --- |
| DotKnot | Find active endpoint and update source/destination nibbles | Share endpoint selection and build nibble masks for four directions |
| Hanoi | Find smallest disk on source/target pegs and update one peg | Share every peg's top disk and combine the six default moves |
| LightsOut | Look up one packed toggle-mask row and XOR | Prepack state-independent toggle masks and broadcast XOR |
| Maze / Room | Read the packed wall bit for one destination cell | Vectorize four destination coordinates and wall-bit lookups |
| Pancake | Compute prefix-reversal source indices and gather | Gather with a fixed index table for all prefixes |
| Rubik | Gather with the selected sticker permutation | Share unpacking and vmap rotation + packing over permutation rows |
| Slide | Direct byte swap (2×2), select/pack (6-bit), gather (8-bit), packed XOR otherwise | Share blank lookup; select/pack (8-bit) and packed XOR otherwise |
| Sokoban | Update two-bit fields at player, destination, and pushed box | Share player lookup and vectorize four move/push candidates |
| TopSpin | Gather with source indices for the selected operation | Vmap three fixed operations in a separate bulk kernel |
| TSP | Check/set one visited bit and compute closing leg | Share visited count and final-visit check across destinations |
| Cayley | Apply selected permutation or modular matrix | Gather all permutations or run batched modular matmul |

## Adopting Optimizations

Adopt a representation or kernel change when storage or GPU median latency
improves without regressing single actions, bulk neighbors, or end-to-end JAxtar
search (±3% tie band; see [PACKING_REFERENCE.md](PACKING_REFERENCE.md) for GPU
warmup, alternating-order, and trace-correlation rules).
