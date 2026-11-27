# Per-Puzzle Packing Analysis & GPU Profiling Reference

Detailed per-puzzle representation invariants, information lower bounds, and GPU microbenchmark methodology supporting [AGENTS.md](AGENTS.md).

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
costs at those widths. Keep the label range explicit when `C>5`. The puzzle is
irreversible, and global minimality for arbitrary state counts has not been proved.

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
rendering. Compute the view of a host-resident State with NumPy to avoid sending
it back to the device, and keep `.pegs` off the transition hot path. Disk IDs and
counts in the logical `pegs` view retain their uint8 contract (up to 255 disks).

The top disk is the smallest disk index on that peg. Using `num_disks` as the
empty-peg sentinel lets `source_top < target_top` enforce both a nonempty source
and the size rule. A single action compares only two pegs; all-neighbor
generation computes every peg's top disk once. Select the disk to change with
`where` and pack with the existing codec to reduce scatter and handle different
peg widths, such as 2/3 bits, consistently. Vmap bulk codec calls over the action
axis rather than passing an action axis to scalar `from_unpacked`.

Preserve the zero-disk boundary case: storage is 0 B, every move is an
infinite-cost self-loop, and the goal is already satisfied. Give the top-disk
reduction an explicit initial value of `num_disks`. For empty intrinsic shapes,
the shared field codec must use the known batch shape's element count instead
of inferring batch size with `-1`. Avoid reshapes that divide by zero for zero
elements, and keep host/device restoration of empty arrays consistent.

The global information lower bound is `ceil(size*log2(3))` bits. Five default
disks require 8 bits, so even the current 2 B is not globally minimal. A base-3
rank can be denser, but division-based decoding and transition costs need
separate measurement.

### LightsOut and Random

Toggling is XOR, so it can operate directly on packed bytes without unpacking.
Packing `_move_matrix` in little-bit order defines the masks for the center and
its four neighbors, excluding out-of-bounds neighbors, with zero padding. A
single action looks up one row of this constant table; all-neighbor generation
broadcasts XOR between the table and the state. The table requires `O(size⁴)`
space and compilation constants as the board grows; remeasure when substantially
increasing supported sizes. Random changes goals and initial samples; transitions
and State representation are identical.

For states reachable from a goal, the GF(2) move-matrix rank determines the
information dimension. General logical inputs can belong to other cosets, and
Random goals also exist, so make reachability assumptions and the goal context
needed for decoding explicit.

### Maze and Room

The maximum coordinate is `size-1`, so uint8 supports sizes through 256, with
uint16 starting at 257. Packing the two coordinates of the default Maze 23 with
5-bit `packed_tensor` would increase storage to 5 B. Walls are shared as a
one-bit bitmap in SolveConfig; a transition reads only the destination bit.
Check negative-direction moves in signed int32 so `-1` does not wrap to `255`
at size 256. Room inherits Maze's representation and transitions and adjusts the
requested size to its room layout, so calculate storage using `room.size`.

When `size² > 2³¹`, widen only the flat index to uint32 to avoid overflowing
wall addresses reached through uint16 coordinates (e.g., cell 2147488280 on a
46341×46341 grid). Keep packed-byte shift operands and masks in uint8, and
express the two coordinate bounds directly as row/column conditions.

### PancakeSorting and TopSpin

Both puzzles use permutations of `1..N`. Existing uint8 labels support `N<=255`;
larger permutations require separate representation changes in generation, goals,
and logical views. Pancake prefix reversal is one gather with
`source[i] = action+1-i if i<=action+1 else i`. Pancake's flip specification
constructs `State(stack=logical_uint8_array)` and reads `.stack`, so moving to
packed fields requires migrating both the construction contract and logical view.

TopSpin's single-action path computes selected source indices directly, while
all-neighbor generation retains `vmap(lax.switch)` over three fixed operations.
Normalize reversal length like a Python slice to preserve negative
`turnstile_size` semantics. Candidates packing TopSpin 20/32 at 5/6 bits reduced
storage to 15/24 B, but decode/gather/repack slowed both single actions and
all-neighbor generation.

### RubiksCube and Random

Color mode requires 3 bits for values `0..5`. Tile-ID mode uses `0..6*size²-1`
in uint8 (supporting up to 256 tiles). Measure GPU speed against byte gather
before adopting bit-packed IDs; preserve general sizes, QTM/UQTM, Random goals,
and Santa tile-ID inputs together. Share unpacking across all neighbors and vmap
per-permutation-row gather and packing with the existing codec.

### SlidePuzzle, Random, and Hard

Tiles are a permutation of `0..size²-1`, including the blank at 0. Size 4 uses
8 B (nibbles); size 5 uses 20 B with 5-bit block padding. Preserve the supported
3/4 sizes and original tile order of Hard's bundled inputs.

A move swaps two tiles via their XOR difference across `ceil((b+8-gcd(b,8))/8)`
byte fragments, converting each fragment to uint8 before XORing into storage.
For 2×2, a single action operates directly on one byte, preserving first-zero /
index-0 fallback behavior across all 256 byte patterns. For 6-bit single actions,
select with `where` then pack; for 8-bit single actions, swap indices and gather;
for 8-bit bulk neighbors, use select/pack; keep packed XOR for 5/7-bit and wider
9–16-bit logical inputs supplied via `State.from_unpacked(board=...)`.

### Sokoban and Hard

The default 10×10 two-bit board uses 25 B. Forward transitions locate the player
once, vectorize destination/push checks, and update only the two-bit fields of
the at most three moving cells. Inverse transitions vectorize four pull/move
candidates independently; under the scrambling contract, a blocked direction is
a cost-1 self-loop while `filled=False` sets cost to infinity.

### TSP

The mask uses little-bit order; the completed value of its last valid byte is
`(1 << (size % 8 or 8)) - 1`, and earlier bytes must be 255. Ordinary edges read
`distance_matrix`, while the final closing leg back to start uses the Euclidean
coordinate norm. Single-action mask updates use bitwise OR rather than scatter.
All-neighbor generation checks `visit_count == size - 1` once and adds each
destination's closing leg only on the final visit.

### CayleyPuzzle

`PERMUTATION` uses uint8 up to 255, uint16 up to 65535, and int32 otherwise,
preserving external label values without relabeling to `0..n-1`. `MATRIX` uses
uint8 when both central-state range and `modulo-1` fit in `0..255` and int32
otherwise, keeping int32/int64 multiplication and `m*(modulo-1)^2` overflow
preconditions before narrowing after modulo. Curated presets in `CayleyPresets`
unpack directly into `CayleyPuzzle(**preset)`. Copy arrays to host once via
`jax.device_get` before iterating in string formatting.

## GPU Profiling & Microbenchmark Methodology

Adopt an optimization when storage or GPU speed improves without regressions in
other metrics. Compare repeated medians excluding compilation on the same GPU,
seed, inputs, and batch sizes. Treat ±3% as a tie and remeasure results outside
that range. Cover default sizes, representative large sizes, and bit-width /
byte-padding boundaries across single transitions, neighbor generation, and
representative JAxtar searches.

For short kernels, separate profiler GPU-kernel time from Python dispatch and
clock shifts: warm up each operation, alternate baseline/candidate measurement
order within the same process, correlate CPU launches to GPU kernels via
correlation IDs, and allow trace finalization before shutdown so every CPU
launch has its device event (treating explicit `size:0` records as empty-field
no-ops rather than dropped kernels).
