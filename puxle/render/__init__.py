"""Puzzle Renderer Module.

Owns the seam that attaches host-side image rendering methods to puzzle
``State`` classes. Mirrors the Xtructure ``BatchedRenderer`` ↔
``RichBackend`` split documented in ``CONTEXT.md``:

- :func:`attach_state_renderer` is the canonical attachment seam — display
  concerns live here while puzzle mechanics (``get_neighbours`` /
  ``is_solved`` / ``get_actions``) stay in ``puxle.core`` and
  ``puxle.puzzles``.
- :class:`Cv2Backend` is the first-party **Puzzle Render Backend**: the
  drawing-primitive Interface consumed *inside* each puzzle's ``imgfunc``
  body. Migrated puzzles construct one ``Cv2Backend()`` per
  ``get_img_parser()`` call and route every cv2 primitive through it instead
  of importing cv2 directly.

The legacy ``puxle.utils.util.add_img_parser`` alias is retained for
backwards compatibility with external puzzle authors but new puzzles should
not consume it directly.
"""

from puxle.render.backends import Cv2Backend
from puxle.render.state_renderer import attach_state_renderer

__all__ = ["attach_state_renderer", "Cv2Backend"]
