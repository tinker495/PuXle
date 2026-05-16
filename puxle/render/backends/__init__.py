"""Puzzle Render Backend implementations.

The Interface is documented in ``CONTEXT.md`` "Puzzle Render Backend"; the
first-party implementation is :class:`Cv2Backend`.
"""

from puxle.render.backends.cv2_backend import BgrColor, Cv2Backend, Point, Size

__all__ = ["Cv2Backend", "BgrColor", "Point", "Size"]
