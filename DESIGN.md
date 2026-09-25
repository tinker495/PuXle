---
version: alpha
name: PuXle Puzzle Renders
description: Visual design of PuXle's puzzle renders - ANSI text boards from State.str() and RGB canvases from State.img().
omitted:
  - section: rounded
    reason: Boards use square heavy box-drawing corners; there are no pixel radii (see Shapes).
colors:
  primary: "#000000"
  pending: "#800000"
  done: "#008000"
  movable: "#808000"
  fixed: "#c0c0c0"
  action-key: "#ffff00"
  sticker-up: "#ffffff"
  sticker-front: "#00ff00"
  sticker-right: "#ff0000"
  sticker-back: "#0000ff"
  sticker-left: "#ff8000"
  sticker-down: "#ffff00"
typography:
  glyph:
    fontFamily: monospace
    fontWeight: 400
spacing:
  cell: 2
  frame-padding: 1
  pane-gap: 2
  peg-gap: 3
  image-canvas: 640px
components:
  board-frame:
    textColor: "{colors.primary}"
    typography: "{typography.glyph}"
    padding: "{spacing.frame-padding}"
  board-cell:
    textColor: "{colors.primary}"
    typography: "{typography.glyph}"
    width: "{spacing.cell}"
  player:
    textColor: "{colors.pending}"
  target:
    textColor: "{colors.pending}"
  player-on-target:
    textColor: "{colors.done}"
  sokoban-player:
    textColor: "{colors.done}"
  sokoban-box:
    textColor: "{colors.movable}"
  sokoban-box-on-target:
    textColor: "{colors.done}"
  sokoban-wall:
    textColor: "{colors.fixed}"
  lightsout-action-key:
    textColor: "{colors.action-key}"
  cube-sticker-up:
    textColor: "{colors.sticker-up}"
  cube-sticker-front:
    textColor: "{colors.sticker-front}"
  cube-sticker-right:
    textColor: "{colors.sticker-right}"
  cube-sticker-back:
    textColor: "{colors.sticker-back}"
  cube-sticker-left:
    textColor: "{colors.sticker-left}"
  cube-sticker-down:
    textColor: "{colors.sticker-down}"
  cube-net:
    width: "{spacing.pane-gap}"
  hanoi-peg:
    width: "{spacing.peg-gap}"
  image-canvas:
    size: "{spacing.image-canvas}"
---

# PuXle Puzzle Rendering Design

## Overview

PuXle draws the puzzle itself: the board, the cube, the maze. It should read like a diagram of
the physical puzzle, with a plain monochrome frame, colored game pieces, and a goal drawn exactly
like a state so the two can be compared side by side.

Every puzzle has two render surfaces:

| Surface | Contract | Built by |
| --- | --- | --- |
| Text | `state.str(**kw)` / `str(solve_config)` returns a multi-line Python string with ANSI SGR color codes | `Puzzle.get_string_parser`, `Puzzle.get_solve_config_string_parser` (`puxle/core/puzzle_base.py`) |
| Image | `state.img(**kw)` returns an `(H, W, 3)` RGB array | `Puzzle.get_img_parser`, bound by `attach_state_renderer` (`puxle/render/state_renderer.py`) |

Text renders are plain strings on purpose. PuXle has no rich or termcolor dependency: termcolor
was replaced by a local ANSI helper in `adad4a0`, and the rich tables in PDDL formatting were
removed in `13d7bf3`. The same string has to look right in two places: in `print(...)` (the PuXle
`human_play` CLI) and inside JAxtar's rich panels, which wrap it with `Text.from_ansi`. PuXle
renders only single states. Batches are laid out by Xtructure, which calls the single-state parser
once per item and places the results side by side.

Source of truth:

| Area | Code |
| --- | --- |
| Color helpers, canvas size | `puxle/utils/util.py` (`colored_str`, `coloring_str`, `ANSI_COLOR_CODES`, `IMG_SIZE`) |
| Shared frame and action labels | `puxle/core/puzzle_base.py` (`_grid_visualize_format`, `_directional_action_to_string`, `_DIRECTIONAL_LABELS`) |
| Per-puzzle parsers | `puxle/puzzles/*.py` `get_string_parser`, `get_img_parser`, `action_to_string` |
| PDDL text | `puxle/pddls/formatting.py` |
| World-model text preview | `puxle/world_model/_utils.py` `img_to_colored_str`, `puxle/world_model/world_model_puzzle_base.py` |
| Image binding | `puxle/render/state_renderer.py` `attach_state_renderer` |

## Colors

There are two helpers, and both reset the color after every glyph (`\x1b[0m`):

- `colored_str(text, name)` writes 16-color SGR codes by name (`red`, `light_yellow`, ...;
  `ANSI_COLOR_CODES`). An unknown name or `None` leaves the text uncolored.
- `coloring_str(text, (r, g, b))` writes 24-bit SGR codes. Use it for fixed physical colors
  such as cube stickers and DotKnot dots.

The hex values for 16-color tokens are rich's default export colors for those names (the
terminal remaps them). The 24-bit tokens are the exact values emitted.

| Token | Emitted as | Role | Used by |
| --- | --- | --- | --- |
| `primary` (#000000) | terminal default foreground | Ink | frames, tile digits, Maze/Room walls, labels |
| `pending` (#800000) | `red` (31) | Not yet at goal | Maze/Room player `●` and target `x`; Sokoban empty target `x`, player on a box target `ⓧ`, goal player cell `●` |
| `done` (#008000) | `green` (32) | Goal relation satisfied | Maze/Room player on target `●`, Sokoban box on target `■` |
| `movable` (#808000) | `yellow` (33) | Pushable object | Sokoban box `■` |
| `fixed` (#c0c0c0) | `white` (37) | Immovable object | Sokoban wall `■` |
| `action-key` (#ffff00) | `light_yellow` (93) | Clickable key | Lights Out action keys |
| `sticker-*` | 24-bit | Cube faces | Rubik's Cube stickers, U/F/R/B/L/D = white, green, red, blue, orange (255,128,0), yellow |

Palettes that appear only in prose:

| Palette | Values | Assignment | Code |
| --- | --- | --- | --- |
| DotKnot dots | 24-bit red, green, yellow, blue | color index of the dot pair | `puxle/puzzles/dotknot.py` `COLOR_MAP` |
| Size cycle | `red, green, yellow, blue, magenta, cyan` | Hanoi `(size - 1) % 6`, Pancake `size % 6` | `puxle/puzzles/hanoi.py` / `puxle/puzzles/pancake.py` `get_color` |
| PDDL labels | `cyan, magenta, green, yellow, blue, white, red` | action/predicate names in sorted order, cycling; fallback `white` | `puxle/pddls/formatting.py` `build_label_color_maps` |
| World-model preview | 24-bit per pixel | each image pixel drawn as `██` | `img_to_colored_str` |

## Typography

Monospace text at regular weight only. PuXle emits color SGR codes and nothing else: no bold,
italic, underline or background color.

Glyph vocabulary:

| Glyph | Meaning | Puzzles |
| --- | --- | --- |
| `■` | solid thing: wall, lit light, box, sticker, path segment, visited city | Maze, Room, Sokoban, Lights Out, Rubik's Cube, DotKnot, TSP |
| `□` | unlit light | Lights Out |
| `●` | agent or dot endpoint | Maze, Room, Sokoban, DotKnot |
| `x` | target cell | Maze, Room, Sokoban |
| `ⓧ` | player standing on a box target | Sokoban |
| `☐` / `↓` | unvisited city / current city marker | TSP |
| `=` | disk or pancake bar, width `2·size − 1` | Hanoi, Pancake |
| `\|`, `-` | peg, base | Hanoi |
| `[ ... ]` | turnstile window | TopSpin |
| `✓` / `✗` | goal atom true / false | PDDL |
| `1`–`9`, `A`–`Z` | slide tiles (10 → `A`, `chr(x + 55)`); blank is a space | Slide Puzzle |
| `0`–`9`, `a`–`z`, `A`–`Z` | action keys | Lights Out |

Action labels (`action_to_string`):

- Four-way moves use arrows `←`, `→`, `↑`, `↓` from `Puzzle._DIRECTIONAL_LABELS` (Slide Puzzle,
  Maze and Room, Sokoban, DotKnot, the Sokoban world model).
- The cube uses standard notation (`L`, `L'`, ...).
- Others use short sentence-case phrases: `Move disk from peg 1 to peg 2`, `Flip at position 1`,
  `Shift Left (<<)`, `Reverse Turnstile (R4)`.
- TSP uses two-digit city ids (`03`).
- PDDL uses `name params`, with the name colored; `colored=False` gives `(name params)`.
- The base-class fallback is `action {i}`.

Numbers are fixed-width where they line up: TopSpin `{:2d}`, Pancake rows `NN:SS - `
(1-based index and size, zero-padded).

## Layout

- The unit is a character cell. A board cell is one glyph plus one space (`cell` = 2 columns),
  which looks roughly square in a monospace font with a 1:2 aspect ratio.
- `Puzzle._grid_visualize_format(size)` builds the standard board: `┏━` + `━━` × size + `┓`,
  rows `┃ ` + `{:s} ` × size + `┃`, and a matching bottom edge. The one-space `frame-padding`
  sits inside the left border.
- A render's size depends only on the puzzle size, never on terminal width. Placement (centering,
  side-by-side, wrapping) is the caller's job.
- Multi-part layouts:
  - Lights Out puts `Board` and `Actions` panes side by side, each titled in the top border.
  - Rubik's Cube shows a color legend and a cube net (U above L F R B, D below). The pieces are
    joined by `_plain_table`, which measures ANSI-stripped width, centers cells vertically and
    separates columns with 2 spaces (`pane-gap`).
  - Hanoi pegs are `2n + 1` columns wide and joined by 3 spaces (`peg-gap`), above a dashed base
    and `Peg i` labels.
  - Pancake draws one line per pancake with its bar centered in `2·size` columns, above
    `Plate ┗━…━┛`.
- Goal context: parsers that accept `solve_config=` overlay goal information on the state.
  Sokoban marks targets and the goal player cell, Maze and Room mark the target, PDDL annotates
  goal atoms with `✓`/`✗` and lists them first. Rubik's Cube has `use_color_overlay` for numbered
  (non-color-embedded) cubes.
- PDDL text is a list, not a board: comma-separated true atoms, at most `max_show` (12) of them,
  plus optional `header`, `raw`, `show_summary` and `show_more` lines.
- Parsers render on the host: `jax.device_get(state)` once, then loop over NumPy. Reading a
  device array cell by cell syncs once per character.

Images:

- The canvas is `IMG_SIZE = (640, 640)` (`image-canvas`): RGB, uint8, drawn with OpenCV. Text uses
  `cv2.FONT_HERSHEY_SIMPLEX`.
- Backgrounds: light grey `(240, 240, 240)` for Pancake, Hanoi and TopSpin; white for TSP;
  `(190, 190, 190)` for the cube; `(144, 96, 8)` for the Slide Puzzle board; dark `(30, 30, 30)`
  with a `(50, 50, 50)` grid for Lights Out; black for Sokoban.
- `img()` takes `**kwargs`. JAxtar passes `idx`, `path` and `solve_config` when it renders solution
  frames. A batched `img()` renders each item and stacks the results
  (`attach_state_renderer`).

## Elevation & Depth

Flat. The frame is the only container. Emphasis comes only from the color of the pieces. There
are no background colors or shading in text renders.

## Shapes

- Frames use only the heavy box-drawing set: `┏ ┓ ┗ ┛ ━ ┃ ┳ ┻ ┣ ┫ ╋`. The Slide Puzzle draws
  every cell with `┳ ╋ ┻` separators; other grid puzzles draw only the outer frame.
- Corners are square. JAxtar's chrome is rounded, which keeps puzzle content and tool chrome
  visually distinct.
- Titles sit in the top border, centered and padded with `━`
  (`"Board".center(w, "━")`, `┏━━up━━━┓`).
- Text-only renders have no frame: TopSpin, TSP, PDDL, Cayley puzzles (`<graph>: v0 v1 ...`) and
  the world-model latent line (`latent: 0x...[N bytes]`).

## Components

| Puzzle | State text | Goal text | Actions | Image |
| --- | --- | --- | --- | --- |
| Slide Puzzle | full cell grid, tile characters | same parser (target board) | arrows | `(144,96,8)` board, `(240,240,232)` tiles |
| Lights Out | `Board` / `Actions` panes, `■`/`□` | same parser | bright-yellow keys | dark `(30,30,30)`, yellow lit cells |
| Rubik's Cube | legend + net, 24-bit stickers | same parser | cube notation | grey `(190,190,190)` |
| Maze / Room | outer frame, `■` walls, red `●`/`x` | green `●` at goal | arrows | resized wall mask, `(200,200,200)` grid lines, circle player |
| Sokoban | outer frame, white walls, yellow boxes, green player | goal boxes in green | arrows | tile sprites on black |
| DotKnot | outer frame, 24-bit `●` dots, `■` paths | empty string (goal is not a state) | arrows | circles for dots, rectangles for paths |
| Hanoi | pegs, colored `=` disks, base, labels | same parser | sentence | light grey |
| Pancake | indexed bars, `Plate` base | same parser | sentence | light grey |
| TopSpin | `[turnstile] rest` numbers | same parser | sentence | light grey |
| TSP | `↓` marker over `[■ ☐ ...]` | `TSP SolveConfig: N points, start at i` | two-digit ids | white canvas, polyline tour, status-colored points |
| PDDL | atom list with `✓`/`✗` | `Goal atoms: ...` | colored `name params` | one pixel per atom |
| Cayley | `<graph>: values` | same parser | `<graph>.gen[i]` | `1 × N` grayscale strip |
| World models | optional `██` pixel preview + `latent:` line | image-based | arrows (Sokoban) | decoded image |

Shared parts: `board-frame` / `board-cell` (`_grid_visualize_format`), directional labels
(`_directional_action_to_string`), the color helpers, `_plain_table` (cube only), and
`attach_state_renderer` for `img()`.

## Do's and Don'ts

- Do build grid boards with `Puzzle._grid_visualize_format`, one glyph plus a space per cell.
- Do color only with `colored_str` / `coloring_str`, one glyph at a time, so every glyph resets.
- Don't emit anything but SGR color codes. Cursor movement, clears and background colors break
  JAxtar's `Text.from_ansi` layout and its log capture.
- Don't import rich or termcolor. The text surface stays dependency-free.
- Do render the goal with the state parser (the default `get_solve_config_string_parser` reads
  `GoalSpec`). Override it only when the goal is not a state (DotKnot, TSP, PDDL).
- Do call `jax.device_get(...)` once before any per-cell loop.
- Do use `_directional_action_to_string` for four-way moves.
- Do reuse the roles for a new grid puzzle: red for pending, green for done, yellow for movable,
  white for fixed.
- Do keep glyphs one column wide. `_plain_table` and the grid format count characters, not display
  cells, so emoji and East Asian wide characters break alignment.
- Don't make a render depend on terminal width.

## Known Gaps

Current code that does not follow the rules above. Don't copy these patterns.

- Green means "player on target" in Maze and Room, but the ordinary Sokoban player `●` is also
  green, while its goal-cell player is red.
- Hanoi uses ASCII pegs and base (`|`, `-`) instead of the heavy frame set.
- The size palettes are offset differently: disk 1 is red in Hanoi, pancake 1 is green in Pancake.
- Maze and Room called without `solve_config` fall back to `Maze State: Player at position ...`
  instead of a board.
- The PDDL image (a float32 grid with one pixel per atom) and the Cayley image (a `1 × N` strip)
  are not `IMG_SIZE` canvases.
