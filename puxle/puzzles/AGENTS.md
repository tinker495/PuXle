<!-- Parent: ../../AGENTS.md -->

# Native Puzzle Packing and Transitions

Write documentation in this directory in English, including this file. Follow
the repository's [documentation and communication policy](../../AGENTS.md#documentation-language).

This directory owns the state representations of native puzzles, their Random
and Hard variants, Room, and Cayley subclasses. PDDL and world-model
representations have separate owners. Optimize storage while preserving logical
puzzle values, action numbering, costs, and inverse-transition semantics. Preserve
existing logical inputs and DeepCubeA/Santa data. Historical packed bytes and
serialized PyTrees of internal State types are not stable external formats.

## Measuring Storage

- **Information lower bound**: For `S` valid states, the bound is
  `ceil(log2(S))` bits. Achieving this bound with rank/unrank does not imply fast
  transitions or partial access.
- **Fixed-width element lower bound**: Values in `0..M` require
  `max(1, M.bit_length())` bits. Distinguish the number of values from their
  maximum. A permutation of `1..N` requires `N.bit_length()` bits per element.
- **Actual State storage**: Sum `nbytes` over the stored PyTree leaves. Distinguish
  this from the size of an `_unpacked` view, memory reserved by the GPU allocator,
  and the total size of search data structures.
- **Shared context**: Maze walls and TSP coordinates and distance matrices belong
  to `SolveConfig.InstanceContext`. Do not duplicate them per state when
  searching a single problem. Batches of different problems also batch context.
- **Temporary memory and traffic**: Account for intermediate arrays and outputs
  from unpack, gather/scatter, and repack. Fewer stored bytes do not guarantee
  fewer memory transactions or kernels, or shorter execution time.

`FieldDescriptor.packed_tensor(shape=..., packed_bits=b)` stores the field itself
as packed `uint8`. For `N` logical elements and `L=lcm(b,8)`, the current codec
uses `ceil(N / (L/b)) * (L/8)` bytes. For example, 25 elements at 5 bits each use
20 bytes rather than the ideal 16, and two coordinates at 5 bits each use 5 bytes.
Both include padding at the end of a block.

Changing only the bit metadata of an ordinary `tensor(..., bits=b)` does not
shrink the default State leaf. Storage must actually use the aggregate `.packed`
representation. The aggregate codec uses 32-bit words and a small tail; a
three-byte tail consumes a four-byte word. It is therefore not always smaller
than field packing: 35 elements at 6 bits each use 27 bytes with field packing and
28 with aggregate packing.

## Representation and Transition Contracts

1. Use `State.from_unpacked(...)` for logical inputs and `set_unpacked(...)` for
   logical field updates. Pass values matching the storage dtype and shape to
   raw constructors and `replace`.
2. Packed one-bit fields use little-bit order: cell `i` is bit `i%8` of byte
   `i//8`. Keep unused final bits zero. Different padding for the same logical
   state can change equality and hashing based on stored bytes.
3. Avoid full unpack/repack when XOR or individual bit checks and updates work
   directly on packed values. Narrow bits may instead increase decode costs for
   irregular permutation gathers.
4. Narrow dtypes are for storage. Compute coordinate differences, negative moves,
   and matrix-product intermediates in wider signed dtypes, then convert back
   to the storage dtype. Validate before narrowing.
5. When narrowing generated data, keeping the original random-generation dtype
   and converting only the result can preserve logical inputs for the same seed.
   Changing the dtype passed to `random.randint` can change the samples.
6. `_apply` returns a candidate state and cost for one action.
   `_get_neighbours` returns candidate states and a cost vector for all actions.
   Restoring the original state when `filled=False` or the cost is nonfinite
   belongs to `Puzzle.get_actions` and `Puzzle.get_neighbours`, respectively.
   Do not duplicate this logic in private kernels.
7. Check State together with GoalSpec, state indices in InstanceContext,
   backward-search initialization, renderers, JAxtar heuristic/neural
   preprocessing, and benchmark input paths.
8. Batch axes are distinct from intrinsic logical-field axes. Batch single-state
   bit indexing through JAX vmap. Do not infer canonical dtypes or shapes from
   external raw State construction.

## Optimizing Single Actions and All Neighbors Independently

Every native base class implements both `_apply` and `_get_neighbours`.
Random/Hard, Room, and Cayley subclasses inherit the corresponding pair. Do not
generate all neighbors and select one for a single action, or require native
bulk neighbors to use `vmap(self.get_actions)`. The private kernels do not call
each other, so indexing, array layout, and preprocessing can change independently.
A subclass that changes transition semantics must implement both kernels.
Subclasses that change only goals or initial samples can inherit both. An
override of one kernel does not automatically affect the other path.

Both paths must return the same logical state and cost for the same action
number. The leading axis of `_get_neighbours` follows action order; public
batched APIs return `(action, batch, ...)`. The input action to `_apply` is a
scalar JAX array satisfying `0 <= action < action_size`; the public boundary
also normalizes Python integers. The final state of an invalid transition must
retain every leaf of the input state unchanged.

The shared `Puzzle` preserves the existing mapping of public `get_actions`
for environments without a bulk hook. Do not force PDDL's cost-only handling of
filled or world-model-specific batching into the native contract. Native
optimization must preserve the semantics of environments outside its scope.

| Puzzle | Single-action path | All-neighbor path |
| --- | --- | --- |
| DotKnot | Find the active endpoint and update only the source/destination nibbles | Share endpoint selection and generate nibble masks for four directions |
| Hanoi | Find the smallest disk on source/target pegs and change one peg value | Share the top disk of every peg and combine the six default moves |
| LightsOut | Look up one row of packed toggle masks for the selected action and XOR | Prepack all state-independent toggle masks, then XOR |
| Maze / Room | Look up the packed wall bit for one destination cell | Look up four coordinates and four wall bits as vectors |
| Pancake | Compute source indices for the selected prefix and gather | Gather with a fixed index table for all prefixes |
| Rubik | Gather with the selected sticker permutation | Share unpacking and vmap rotation and packing over permutation rows |
| Slide | Swap directly for a one-byte board; use select/pack for 6 bits, index gather for 8 bits, and packed XOR otherwise | Share blank lookup; use select/pack for 8 bits and packed XOR otherwise |
| Sokoban | Update the two-bit fields at the player, destination, and push positions | Share player lookup and vectorize four move/push candidates |
| TopSpin | Gather with source indices for the selected operation | Vmap three fixed operations in a separate bulk kernel |
| TSP | Check/set one visited bit and determine the closing leg | Share visited information and the final-visit check across all destinations |
| Cayley | Apply the selected permutation or modular matrix | Gather all permutations or use batched matrix multiplication |

Even for the same mathematical transformation, the batch axes of `gather`,
`stack`, and `vmap` affect XLA fusion and transposes. Extending an expression
from one path is not sufficient optimization. In particular, compare TopSpin's
full index gather, fixed-output stack, and `vmap(lax.switch)` separately for
forward and inverse transitions.

## Per-Puzzle Analysis

The byte counts below are per-state payloads in the default representation, not
GPU allocations or total search memory.

| Puzzle | Logical representation and actual storage | Optimality assessment |
| --- | --- | --- |
| DotKnot | 4 bits × `size²`; default 10 → 50 B, 16 → 128 B | The default four colors require 4 bits per element. |
| Hanoi | 2 bits for disk→peg; default 5 → 2 B, 15 → 4 B, 31 → 8 B | Minimal fixed width per disk; removes redundant counts, empty slots, and ordering. |
| LightsOut / Random | 1 bit × `size²`; 7 → 7 B, 8 → 8 B | Meets the fixed-width lower bound for independent binary cells. |
| Maze / Room | Two unsigned coordinates; 2 B for `size<=256`, then 4 B | 2 B at the default size without a bit codec. |
| PancakeSorting | `uint8[size]`; 35 → 35 B, 64 → 64 B | Leaves room for permutation compression, but keeps gather simple. |
| RubiksCube / Random | Colors use 3 bits × `6*size²`; 3 → 21 B, 4 → 36 B | Meets the fixed-width color bound; block padding remains. |
| Rubik tile ID | 8 bits × `6*size²`; 2 → 24 B, 3 → 54 B, 4 → 96 B | Wider than the minimum bit width; account for irregular-gather costs. |
| SlidePuzzle / Random / Hard | `(size²-1).bit_length()` bits × `size²`; 3 → 5 B, 4 → 8 B, 5 → 20 B | Minimal width per element; size 5 includes codec padding. |
| Sokoban / Hard | 2 bits × 100 cells → 25 B | Minimal fixed width for the default cell alphabet; static walls add redundancy. |
| TopSpin | `uint8[size]`; 20 → 20 B, 32 → 32 B | Permutation compression is possible, but byte gathers simplify rotations and partial reversals. |
| TSP | Visited mask of `ceil(size/8)` B plus a point of 1 or 2 B; 16 → 3 B, 33 → 6 B, 257 → 35 B | The mask has minimal fixed width; the point uses uint8 for `size<=256`. |
| Cayley | 1/2/4 B per element depending on value range | Determined by the preserved value range, not graph length. |

Interpret information lower bounds within a fixed problem context as follows.
These state counts are starting points for compression design, not permission
to restrict arbitrary logical inputs to reachable states.

| Family | State count or information dimension | Additional compression contract |
| --- | --- | --- |
| DotKnot | The cell alphabet has `3*C+1` values, but endpoint/path constraints reduce valid combinations | Preserve endpoint identity and path progress as well as color |
| Hanoi | `K**N` ordered configurations for `K` pegs and `N` disks | Compare per-disk peg access with decoding a global rank |
| LightsOut | A reachable coset for one goal has `2**r` states, where `r` is the GF(2) move-matrix rank | Handle inputs from other cosets and Random-goal context |
| Maze | `R` cells reachable in the given problem | Provide cell-rank↔coordinate mapping and adjacency in InstanceContext |
| Pancake / TopSpin | `N!` for Pancake; the generated subgroup's orbit size for TopSpin | Rank/unrank must preserve labels and action numbering |
| Rubik | Cubie, orientation, and parity constraints are stronger than the sticker-array representation | Define the state space for each size, color/ID mode, and rotation metric |
| Slide | `N!/2` permutations reachable from one goal (`size>=2`) | Specify in context whether logical inputs of the other parity remain supported |
| Sokoban | With fixed walls, `F` floor cells, and `B` boxes, at most `C(F,B)*(F-B)` arrangements | Account for deadlocks/reachability and ownership of walls, player, and boxes |
| TSP | For a fixed start and `N>=2`, `1+(N-1)*2**(N-2)` reachable mask/point pairs | Distinguish the initial mask from the final point of a partial tour without revisits |
| Cayley | Orbit size under the generators' action on the central state | Provide an orbit codec and a logical view that restores external labels |

For example, Slide sizes 3/4/5 have single-parity information lower bounds of
18/44/83 bits, or 3/6/11 B. Their gap from the current 5/8/20 B comes from
permutation redundancy and codec padding. TSP 16 has 245761 reachable states by
the formula above and requires 18 bits; its current 3 B meets the byte-level
lower bound. This assesses storage in that context, not transition speed or
optimality for every raw input.

### DotKnot

`0` is an empty cell, `1..2*C` are distinct endpoints, and `2*C+1..3*C` are
paths for each color. The required width is `(3*C).bit_length()`: 4 bits for the
default `C=4`. Endpoint identity determines which dot moves next, so merging
endpoints by storing only color changes transition semantics. Narrower
representations are possible for `C=1,2`, but measure codec padding and decode
costs at those widths. Do not assume the current fixed 4 bits represent the
label range for `C>5`. The puzzle is irreversible, and global minimality for
arbitrary state counts has not been proved.

Instead of a `C × size²` intermediate array comparing the entire board for each
color, find the minimum value in the endpoint range once. Replace the source
endpoint with its path label and the destination with an endpoint or merged
path label. Different nibbles in the same byte can change simultaneously, so
combine clear masks and new values with OR. Compute coordinates in signed
integers without repacking the full board.

### Hanoi

Disk size order is fixed, so storing only each disk's peg number is enough to
recover counts, empty slots, and ordering. The canonical field is `disk_pegs`;
its logical values are `disk_pegs_unpacked`. Construct states with
`State.from_unpacked(disk_pegs=...)`. The default three pegs require 2 bits per
disk; use `(num_pegs-1).bit_length()` to match the peg count. This reduces the
default representation from the old matrix's 18 B to 2 B. Removing redundancy
first is more effective than packing the original matrix at 3 bits, which would
use 9 B.

`.pegs` is a read-only logical view of shape `(num_pegs, num_disks+1)`. Each row
contains a count and ascending disk IDs, preserving existing logical access and
rendering. This is not the transition storage format; do not compute it on the
hot path. Compute the view of a host-resident State with NumPy to avoid sending
it back to the device. Historical `State(pegs=...)` construction and matrix
PyTrees are not canonical input paths. Disk IDs and counts in the logical
`pegs` view retain their uint8 contract. A packed field's ability to hold more
disks does not imply that the view or renderer supports more than 255 disks.

The top disk is the smallest disk index on that peg. Using `num_disks` as the
empty-peg sentinel lets `source_top < target_top` enforce both a nonempty source
and the size rule. A single action compares only two pegs; all-neighbor
generation computes every peg's top disk once. Select the disk to change with
`where` and pack with the existing codec to reduce scatter and handle different
peg widths, such as 2/3 bits, consistently. Vmap bulk codec calls over the action
axis. Do not pass an action axis to scalar `from_unpacked` as though it were
part of the intrinsic shape.

Preserve the zero-disk boundary case: storage is 0 B, every move is an
infinite-cost self-loop, and the goal is already satisfied. Give the top-disk
reduction an explicit initial value of `num_disks`. For empty intrinsic shapes,
the shared field codec must use the known batch shape's element count instead
of inferring batch size with `-1`. Avoid reshapes that divide by zero for zero
elements, and keep host/device restoration of empty arrays consistent.

The global information lower bound is `ceil(size*log2(3))` bits. Five default
disks require 8 bits, so even the current 2 B is not globally minimal. A base-3
rank can be denser, but division-based decoding and transition costs need
separate measurement. The current representation provides fixed-width
compression while retaining fast per-disk access.

### LightsOut and Random

Toggling is XOR, so it can operate directly on packed bytes without unpacking.
Packing the existing `_move_matrix` in little-bit order defines the masks for
the center and its four neighbors, excluding out-of-bounds neighbors, with zero
padding. A single action looks up one row of this constant table; all-neighbor
generation broadcasts XOR between the table and the state. Computing five bit
positions at runtime also beat full unpacking, but constant-mask lookup further
reduced arithmetic and temporary buffers for large batches. The table requires
`O(size⁴)` space and compilation constants as the board grows; remeasure when
substantially increasing supported sizes. Improvement in one path does not
justify a regression in the other. Random changes goals and initial samples;
transitions and State representation are identical.

For states reachable from a goal, the GF(2) move-matrix rank determines the
information dimension. A rank below the cell count suggests possible compression
with basis coordinates. General logical inputs can belong to other cosets, and
Random goals also exist, so make reachability assumptions and the goal context
needed for decoding explicit. The fixed-width bound of one bit per cell does
not prove optimality for global compression.

### Maze and Room

The maximum coordinate is `size-1`, so uint8 supports sizes through 256, with
uint16 starting at 257. Packing the two coordinates of the default Maze 23 with
5-bit `packed_tensor` would increase storage to 5 B. Walls are shared as a
one-bit bitmap in SolveConfig; a transition reads only the destination bit.
Check negative-direction moves in int32. Converting to uint8 first maps
`-1→255` and creates invalid wraparound moves at size 256. Room inherits Maze's
representation and transitions and adjusts the requested size to its room
layout, so calculate storage using `room.size`. Coordinate ranges wider than
uint16 are outside the existing representation's support.

Very small grids might fit in 1 B using a flat index or two nibbles. Such a
candidate is smaller than the current two-coordinate byte representation, but
requires changes to coordinate views, constructors, callers, and decoding.
Do not call the 2 B result at the default size the information-theoretic minimum
for every size.

Compute coordinates and bounds in signed int32. When `size² > 2³¹`, widen only
the flat index to uint32 to avoid overflowing wall addresses reached through
uint16 coordinates. For example, the last cell of a 46341×46341 grid has index
2147488280, which signed int32 cannot represent. Keep packed-byte shift operands
and masks in uint8. Propagating index dtypes into bit data can widen integers
unnecessarily. Express the two coordinate bounds directly as row/column
conditions to reduce small reductions and intermediate coordinate arrays.

### PancakeSorting and TopSpin

Both puzzles use permutations of `1..N`. When `N` is a power of two, this
requires one more bit than a zero-based permutation. Existing uint8 labels
support `N<=255`; larger permutations require separate representation changes
in generation, goals, and logical views. Field packing could store Pancake 35
in 27 B at 6 bits and TopSpin 20 in 15 B at 5 bits (13 B with aggregate packing).
A permutation rank representing all `N!` permutations uses
`ceil(log2(N!))` bits, but incurs rank/unrank costs on every transition and
heuristic evaluation. Compute the information bound of a subgroup generated by
specific generators separately from its reachable-state count. Pancake prefix
reversal is one gather with
`source[i] = action+1-i if i<=action+1 else i`; a sequential scan scattering
each element is unnecessary. Consider direct indexing first for TopSpin
rotations and reversals as well. Do not assume additional packing preserves
heuristics using `.stack`/`.permutation` and their constructor contracts.

In particular, Pancake's existing flip specification directly constructs
`State(stack=logical_uint8_array)` and reads `.stack`. Moving to packed fields
requires deciding both the construction contract and the logical view. The
current change keeps byte storage and replaces scan with gather. Additional
packing was not rejected as slow; it remains an unadopted candidate requiring
constructor migration and GPU comparison.

TopSpin's general single-action path computes the selected source indices,
reducing the cost of multiple candidate arrays from batched
`vmap(lax.switch)`. All-neighbor generation separately retains
`vmap(lax.switch)` over three fixed operations. When a reversal segment has
zero or one element and is therefore the identity, the single-action path also
uses direct rotation/identity branches. These existing branches beat general
gather at that boundary. Normalize reversal length like a Python slice to
preserve the semantics of negative `turnstile_size`. Gathering from a full
index table regressed. Directly stacking rolls and partial reversals was faster
for some forward batches but regressed for small inverse batches. Improving
only single-action execution while preserving bulk performance is a valid
choice; a good single-action implementation is not necessarily good for bulk
execution.

Candidates packing TopSpin 20/32 at 5/6 bits reduced storage to 15/24 B, but
decode/gather/repack slowed both single actions and all-neighbor generation,
so they were not adopted. Byte storage can offer a better space/speed tradeoff
for small permutations.

### RubiksCube and Random

Color mode requires 3 bits for values `0..5`. Tile-ID mode uses
`0..6*size²-1`. Applying the minimum bit width to IDs gives payloads of
15/42/84 B for sizes 2/3/4, but requires gathers across bit boundaries and
repacking on every rotation. Measure GPU speed against byte gather before
adopting it; storage alone does not justify replacing the 8-bit tile-ID
representation. Global compression using color counts, fixed centers, and
cubie orientation/parity is a different codec from the current sticker model.
Preserve general sizes, QTM/UQTM, Random goals, and Santa tile-ID inputs
together. The existing uint8 representation cannot hold more than 256 tiles
in ID mode.

Share unpacking across all neighbors and vmap per-permutation-row gather and
packing with the existing codec. Unrolling rotations in Python and stacking
them sped up forward neighbors in large tile-ID mode, but regressed for small
sizes, color mode, and inverse neighbors, so it was not adopted. Public
invalid-state selection also affects XLA transpose fusion; compare more than
the bulk kernel alone.

### SlidePuzzle, Random, and Hard

Tiles are a permutation of `0..size²-1`, including the blank at 0. The nibble
representation for size 4 is dense at 8 B. Size 5 uses 20 B with 5-bit storage
because of block padding; an ideal contiguous bitstream would use 16 B.
Removing this padding is a shared codec-layout change more than a puzzle
change. Measure additional decoding, single-swap cost, and search hashing and
equality. A parity-restricted state space does not automatically shrink a
representation that retains per-element random access. Preserve the supported
3/4 sizes and original tile order of Hard's bundled inputs.

A move swaps two tiles, so apply their XOR difference to both bit ranges. Bit
positions are `cell * b`; the maximum starting offset is `8-gcd(b,8)`. The
number of byte fragments needed is therefore `ceil((b+8-gcd(b,8))/8)`: one
for 2/4 bits, two for 9/10 bits, and three for 11 bits. Obtain the low fragment
with a left shift and later fragments with a right shift of `8*k-offset`.
Convert each fragment to uint8 before XORing it into storage so the wide
logical dtype does not propagate into raw `board`. All-neighbor generation
shares blank lookup and decoding, then independently creates masks for four
directions.

For 2×2, a single action operates directly on one byte. Use scalar selection
to inspect the four two-bit fields for zero and locate the first blank, then
swap the two fields with their XOR difference. This avoids general array
decoding and argmax reduction, also removing increased temporary buffers for
small batches. Preserve the existing argmax behavior for bytes with multiple
zeros or no zero: choose the first zero or the default index 0. All 256 byte
patterns can be compared.

For a 6-bit single action, select the two values with `where`, then pack.
This combines well within a 24-bit codec block and removes the large-batch
regression at 6×6 from using only packed XOR. For an 8-bit single action, swap
the two source indices and gather, reducing temporary buffers and small-batch
time. Keep packed XOR for the 40/56-bit blocks of 5/7-bit fields and for wider
values, where select/pack was slower.

The 8-bit bulk path uses separate select/pack logic. Packed XOR regressed at
large batches for 13×13 and 15×15; this path avoided regressions for other
8-bit sizes and inverse transitions as well. Do not impose a good
single-action implementation on bulk execution or assume repacking is always
slower.

Ordinary generated inputs were checked at sizes 2–16, and wide logical inputs
provided with `State.from_unpacked(board=...)` at widths of 9–16 bits.
Applying only two byte fragments to wide values, or widening the storage dtype
itself, corrupts inputs or breaks types in public conditional branches.
Distinguish the default random generator's uint8 range from the transition
support for user-supplied wide logical inputs.

### Sokoban and Hard

The default 10×10 two-bit board uses 25 B. Separating walls from moving objects
and storing only the player position and box set could reduce it, but moves
information ownership from the board into InstanceContext. First check goals,
walls, box/player encoding, and renderer, heuristic, and inverse-transition
contracts. Default and Hard data are different input sets; do not treat them
as having an identical theoretical state count. Do not benchmark unsupported
large `size` values as evidence of optimality.

Forward transitions locate the player once and vectorize the checks for empty
destinations or pushable boxes. Clear/set only the two-bit fields of the at
most three moving cells, avoiding full-board scatter/repack and nested
conditionals. Do not update the cell behind a box in directions that do not
push one.

Inverse transitions independently vectorize four pull/simple-move candidates.
Under the existing scrambling contract, a blocked direction is a self-loop
with cost 1, and `filled=False` changes only its cost to infinity. Preserve
this distinction from the forward public contract during optimization. Clear
the box cell and place a box at the current player cell only when pulling the
box in front. Move the state only when the prior player cell is empty and
within bounds.

### TSP

The mask uses little-bit order. The completed value of its last valid byte is
`(1 << (size % 8 or 8)) - 1`; every earlier byte must be 255. `is_solved`
and the final-visit check share this rule. Revisits have infinite cost, and
the core restores the state. Add the closing leg back to the start only on
the final visit. The current contract uses `distance_matrix` for ordinary
edges and the coordinates' Euclidean norm for the closing leg; preserve this
difference during packing changes. The current point is already visited,
which is redundant information, but it cannot be recovered from the mask.
Report the distance matrix's `O(size²)` memory separately from the state's
`O(size)` storage.

Single-action mask updates also use OR with a mask derived from byte-index
comparison instead of scatter. Even for a one-byte change, immutable
scatter's copy/update path can add cost. In particular, the scatter candidate
slowed down at 7/8/9 points, while OR both reduced storage and avoided GPU
regressions.

All-neighbor generation does not check whether each destination's entire new
mask is complete. Compute once that, when the current visit count is
`size-1`, the next valid visit is the last. All edges share the current
point's distance-matrix row; add each destination's closing leg only for the
final visit. Revisits still have infinite cost.

### CayleyPuzzle and Its Five Subclasses

PERMUTATION generators only rearrange positions, preserving the central
state's value range. Use uint8 for nonnegative values up to 255, uint16 up to
65535, and int32 for negative or larger values. Even a length-8 state can have
negative labels or labels starting at 100000. Do not relabel values as
`0..n-1` or simply reduce them to `log2(n)` bits. Preserve existing
external logical inputs.

MATRIX uses uint8 when both the central-state range and `modulo-1` fit in
`0..255`; otherwise it retains int32. A uint16 storage candidate was not
adopted because conversion costs regressed single-action GPU performance.
Keep the existing int32/int64 multiplication paths and the
`m*(modulo-1)^2` overflow precondition, then narrow to the storage dtype
after applying modulo. Matrix multiplication in the narrow storage dtype
changes values. Preserve inverse closure, the inverse-action map, and the
cost/warning contract of `ensure_inverse_closed=False`.

CayleyPancake7/8, CayleyLRX8, CayleyTopSpin8K4, and CayleyCoxeter8 share this
adapter. Their default permutations can use 7/8 B respectively; do not copy
codecs into each subclass. Check both external cayleypy factories and
hand-built graphs, and check MATRIX separately.

For string output, call `jax.device_get` once to bring the array to the host
before iterating over elements. Converting each GPU scalar with Python
`int` causes an implicit device synchronization for every element.

## Adopting and Recording Optimizations

Adopt an optimization when storage or GPU speed improves without regressions
in other metrics. Compare repeated medians excluding compilation on the same
GPU, seed, inputs, and batch sizes. Treat ±3% as a tie and remeasure results
outside that range. Cover default sizes, representative large sizes, and bit
width and byte-padding boundaries. Evaluate single transitions, neighbor
generation, and representative JAxtar searches together. CPU time is
supplementary.

For short kernels, Python dispatch and GPU clock changes can dominate wall
time. Separate profiler GPU-kernel time from call overhead, and alternate
baseline/candidate measurement order within the same process. Briefly
interleaving operations with different costs can bias clock changes toward
one sample set, so warm up each operation before alternating measurements.
Connect CPU launches to GPU kernels with correlation IDs; do not assume both
trace clocks use the same timestamps. Kernel time alone cannot establish
faster end-to-end search. GPU activity immediately before trace shutdown
can be missing, so allow a separate collection-finalization period after
measurement and check that every measured CPU launch has a corresponding
device event. Do not count missing calls as zero time or decide adoption
from medians that omit missing samples. Zero-byte copy APIs for empty packed
fields may produce no device work. Use explicit `size:0` trace records to
identify these no-ops while continuing to check for missing kernels and
nonempty copies.

If short calls do not bring the GPU out of its power-saving state, warm it
up with the same GPU operation immediately before each baseline/candidate
call group, excluding that warmup from measurement. This reduces power-state
effects without changing global clock settings. Check measurement conditions
before interpreting large speed differences between mathematically identical
paths as performance gains.

Existing tests may be read and run, but not created, modified, or deleted.
Investigate additional boundary cases with local reproduction code and
propose any needed formal specifications to the user. Keep validation CLIs,
traces, baseline comparison checkouts, diffs, and measurement reports only
in the Git-excluded workspace `runs/` directory. Record representation
contracts and maintenance insights here without incorporating experiment
artifacts.
