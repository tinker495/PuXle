"""cv2-backed **Puzzle Render Backend** implementation.

Mirrors the Xtructure ``BatchedRenderer`` ↔ ``RichBackend`` split:
``puxle.render.attach_state_renderer`` owns the batch-vs-single dispatch seam
that turns a per-state ``imgfunc`` into ``state.img(**kwargs)``; this Module
owns the cv2 + BGR-color + text-centering primitives consumed *inside* each
``imgfunc`` body.

Per ``CONTEXT.md`` "Puzzle Render Backend": the Interface is exactly the
eight methods defined below. Migrated puzzles construct a ``Cv2Backend()``
once per ``get_img_parser()`` call and route every cv2 primitive through it
instead of importing cv2 directly.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

BgrColor = Tuple[int, int, int]
Point = Tuple[int, int]
Size = Tuple[int, int]


class Cv2Backend:
    """First-party Puzzle Render Backend backed by ``cv2``.

    The opaque ``Image`` type is ``np.ndarray`` of shape ``(H, W, 3)`` with
    ``dtype=np.uint8`` and BGR channel order — the native cv2 layout. All
    drawing methods return the (possibly mutated) image so callers can thread
    a single canvas through a sequence of primitives.
    """

    _FONT = cv2.FONT_HERSHEY_SIMPLEX

    def canvas(self, *, size: Size, fill_bgr: BgrColor) -> np.ndarray:
        """Allocate an ``(H, W, 3)`` uint8 image filled with ``fill_bgr``."""
        height, width = size
        img = np.empty((height, width, 3), dtype=np.uint8)
        img[:] = fill_bgr
        return img

    def canvas_from_mono(
        self,
        *,
        size: Size,
        mono_pattern: np.ndarray,
        interpolation: int = cv2.INTER_NEAREST,
    ) -> np.ndarray:
        """Upscale a small ``(h, w)`` uint8 mono mask to ``size`` and convert to BGR.

        Used by puzzles whose background is best drawn at the logical-grid
        resolution and then nearest-neighbour resized (e.g. maze walls). The
        ``mono_pattern`` is interpreted as 0..255 grayscale.
        """
        height, width = size
        resized = cv2.resize(mono_pattern, (width, height), interpolation=interpolation)
        return cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

    def rect(
        self,
        img: np.ndarray,
        *,
        top_left: Point,
        bottom_right: Point,
        color_bgr: BgrColor,
        thickness: int = -1,
    ) -> np.ndarray:
        """Filled (``thickness=-1``) or outlined rectangle."""
        return cv2.rectangle(
            img, tuple(top_left), tuple(bottom_right), color_bgr, thickness
        )

    def circle(
        self,
        img: np.ndarray,
        *,
        center: Point,
        radius: int,
        color_bgr: BgrColor,
        thickness: int = -1,
    ) -> np.ndarray:
        """Filled (``thickness=-1``) or outlined circle."""
        return cv2.circle(img, tuple(center), int(radius), color_bgr, thickness)

    def line(
        self,
        img: np.ndarray,
        *,
        p1: Point,
        p2: Point,
        color_bgr: BgrColor,
        thickness: int = 1,
    ) -> np.ndarray:
        """Single line segment."""
        return cv2.line(img, tuple(p1), tuple(p2), color_bgr, thickness)

    def text(
        self,
        img: np.ndarray,
        *,
        text: str,
        position: Point,
        color_bgr: BgrColor,
        font_scale: float = 1.0,
        thickness: int = 1,
    ) -> np.ndarray:
        """Render ``text`` with ``cv2.FONT_HERSHEY_SIMPLEX`` at ``position``."""
        return cv2.putText(
            img, text, tuple(position), self._FONT, font_scale, color_bgr, thickness
        )

    def text_size(self, text: str, *, font_scale: float, thickness: int) -> Size:
        """Pixel ``(width, height)`` of ``text`` rendered with the same font."""
        ((width, height), _baseline) = cv2.getTextSize(
            text, self._FONT, font_scale, thickness
        )
        return width, height

    def text_centered(
        self,
        img: np.ndarray,
        *,
        text: str,
        top_left: Point,
        cell_size: int,
        color_bgr: BgrColor,
        font_scale: float = 1.0,
        thickness: int = 1,
    ) -> np.ndarray:
        """Draw ``text`` centered inside a square cell.

        Replaces the per-puzzle ``cv2.getTextSize(...)`` +
        ``stx + (bs - tw) / 2`` boilerplate that previously appeared inside
        every ``imgfunc`` that drew labelled tiles.
        """
        text_w, text_h = self.text_size(
            text, font_scale=font_scale, thickness=thickness
        )
        x0, y0 = top_left
        text_x = int(x0 + (cell_size - text_w) / 2)
        text_y = int(y0 + (cell_size + text_h) / 2)
        return self.text(
            img,
            text=text,
            position=(text_x, text_y),
            color_bgr=color_bgr,
            font_scale=font_scale,
            thickness=thickness,
        )

    def ellipse(
        self,
        img: np.ndarray,
        *,
        center: Point,
        axes: Size,
        color_bgr: BgrColor,
        angle: float = 0.0,
        start_angle: float = 0.0,
        end_angle: float = 360.0,
        thickness: int = -1,
    ) -> np.ndarray:
        """Filled (``thickness=-1``) or outlined ellipse / arc segment.

        ``axes`` is the ``(major_semi, minor_semi)`` pair in pixels.
        ``angle``, ``start_angle``, ``end_angle`` are in degrees following the
        cv2 convention; an arc from ``start_angle`` to ``end_angle`` is drawn
        instead of a full ellipse when those differ from ``(0, 360)``.
        """
        return cv2.ellipse(
            img,
            tuple(center),
            tuple(axes),
            angle,
            start_angle,
            end_angle,
            color_bgr,
            thickness,
        )

    def fill_poly(
        self,
        img: np.ndarray,
        *,
        points,
        color_bgr: BgrColor,
    ) -> np.ndarray:
        """Filled polygon. ``points`` is a sequence of ``(N, 2)`` int arrays —
        the same shape ``cv2.fillPoly`` expects so callers can keep their
        existing ``np.array`` / ``reshape`` setup unchanged."""
        return cv2.fillPoly(img, list(points), color_bgr)

    def polylines(
        self,
        img: np.ndarray,
        *,
        points,
        is_closed: bool,
        color_bgr: BgrColor,
        thickness: int = 1,
    ) -> np.ndarray:
        """Open or closed polyline (outline only). ``points`` mirrors the
        ``cv2.polylines`` contract: a sequence of ``(N, 2)`` int arrays."""
        return cv2.polylines(img, list(points), is_closed, color_bgr, thickness)


__all__ = ["Cv2Backend", "BgrColor", "Point", "Size"]
