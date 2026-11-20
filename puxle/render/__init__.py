"""Puzzle Renderer Module.

Owns the single rendering attachment and normalization seam for PuXle.
"""

from puxle.render.state_renderer import (
    attach_state_renderer,
    img_to_colored_str,
    normalize_rgb_image,
)

__all__ = ["attach_state_renderer", "normalize_rgb_image", "img_to_colored_str"]
