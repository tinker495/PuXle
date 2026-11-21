# PuXle Puzzle Rendering Design

## Overview

PuXle draws the puzzle itself as a plain physical diagram with a monochrome frame, colored game
pieces, and a goal drawn with the same layout as a state so the two can be compared side by side.

Every puzzle exposes two render surfaces bound centrally by `attach_state_renderer`
(`puxle/render/state_renderer.py`):

| Surface | Contract | Built by |
| --- | --- | --- |
| Text | `state.str(**kw)` / `str(solve_config)` returns a multi-line Python string with ANSI SGR color codes | `Puzzle.get_string_parser`, `Puzzle.get_solve_config_string_parser` |
| Image | `state.img(size=..., **kw)` returns a host `uint8` `(H, W, 3)` RGB `np.ndarray` | `Puzzle.get_img_parser`, normalized by `normalize_rgb_image` |

Text renders are plain strings without a `rich` or `termcolor` dependency so the same string works
both in raw `print(...)` (`puxle human-play`) and inside JAxtar's rich panels wrapped with
`Text.from_ansi`. PuXle renders single states only; Xtructure lays out batches by calling the
single-state parser once per item.

## Colors and ANSI Safety

- Emit color SGR codes only, resetting with `\x1b[0m` after every glyph via `colored_str` or
  `coloring_str`. Never emit bold, italic, underline, background colors, cursor movement, or screen
  clears—they break JAxtar's `Text.from_ansi` layout and log capture.
- Use `colored_str(text, name)` for 16-color semantic roles that should respect the terminal
  palette (`red` pending, `green` done, `yellow` movable, `white` fixed; unknown or `None` leaves
  text uncolored).
- Use `coloring_str(text, (r, g, b))` for 24-bit fixed physical colors (Rubik's Cube stickers,
  DotKnot dots) that terminal themes must not remap.

## Layout and Rendering Invariants

- **2-column cell**: a board cell is one glyph plus one space so monospace fonts (~1:2 aspect
  ratio) render cells roughly square.
- **1-column glyph constraint**: `_plain_table` and grid formats count Python characters, not
  terminal display widths, so emoji and East Asian wide characters break alignment.
- **Terminal-width independence**: render dimensions depend solely on puzzle size, never terminal
  width; centering, wrapping, and side-by-side placement belong to the caller.
- **Heavy square corners**: frames use square heavy box-drawing glyphs (`┏ ┓ ┗ ┛ ━ ┃ ┳ ┻ ┣ ┫ ╋`)
  while JAxtar uses rounded chrome, keeping puzzle diagrams visually distinct from tool chrome. Use
  `Puzzle._grid_visualize_format` for fixed square grids, `text_frame` for rectangular viewports
  or labeled adapter panels, and `_directional_action_to_string` for four-way moves.
- **Host-copy rule**: parsers call `jax.device_get(...)` once before any per-cell NumPy loop;
  indexing a device array cell by cell forces a synchronization per character.
- **Image contract**: `state.img(size=None, interpolation=cv2.INTER_NEAREST, **kwargs)` normalizes
  every frame to host `uint8 (H, W, 3)` RGB via `normalize_rgb_image`, resizes optionally to
  `size=(width, height)`, and stacks batched frames along axis 0. `IMG_SIZE = (640, 640)` is the
  canonical canvas size for native square puzzles.
- **Goal rendering**: the default `get_solve_config_string_parser` delegates to the state parser on
  `GoalSpec`. Puzzles whose goal is a predicate rather than a single state (`DotKnot`, `TSP`,
  `PDDL`) supply their own goal descriptions.

## External Adapter Boundary

External adapters (`CayleyPuzzle`, `PDDL`) render only a generic view of their input model. They
do not detect puzzle families or planning domains and do not reproduce native puzzle diagrams; a
problem that needs a dedicated diagram belongs in a native puzzle class. Catalog presets and
user-supplied inputs render identically, and both `puxle human-play` and
`scripts/visualize_puzzle.py` pass the current `solve_config` to the state renderer.

## Known Gaps

Current code that does not follow the rules above. Don't copy these patterns.

- Green means "player on target" in Maze and Room, but the ordinary Sokoban player `●` is also
  green, while its goal-cell player is red.
- Hanoi uses ASCII pegs and base (`|`, `-`) instead of the heavy frame set.
- The size palettes are offset differently: disk 1 is red in Hanoi, pancake 1 is green in Pancake.
- Maze and Room called without `solve_config` fall back to `Maze State: Player at position ...`
  instead of a board.
- The PDDL image (one pixel per atom) and Cayley image (a `1 × N` strip)
  are not `IMG_SIZE` canvases by default; pass `size=(width, height)` to `state.img(...)` when a
  uniform output size is needed.
