"""Puzzle Renderer Module.

Owns the seam that attaches host-side image rendering methods to puzzle
``State`` classes. Mirrors the Path Step Adapter split documented in
``CONTEXT.md`` — display concerns live here, puzzle mechanics
(``get_neighbours`` / ``is_solved`` / ``get_actions``) stay in ``puxle.core`` and
``puxle.puzzles``.

The canonical seam is :func:`attach_state_renderer`; the legacy
``add_img_parser`` alias is kept in :mod:`puxle.utils.util` for backward
compatibility with external puzzle authors but new puzzles should not consume
it directly.
"""

from puxle.render.state_renderer import attach_state_renderer

__all__ = ["attach_state_renderer"]
