"""Canonical seam for attaching host-side image and text rendering to ``State`` / ``SolveConfig``.

See ``CONTEXT.md`` "Puzzle Renderer" and ``DESIGN.md``. This module owns:
- ``normalize_rgb_image``: enforces the host ``uint8 (H, W, 3)`` RGB contract
  and optional resizing across every puzzle.
- ``img_to_colored_str``: renders an RGB image as 24-bit ANSI ``██`` rows.
- ``attach_state_renderer``: binds single/batched ``img(**kwargs)`` and
  optional ``str(**kwargs)`` / ``__str__`` onto an ``Xtructurable`` class.
"""

from __future__ import annotations

from collections.abc import Callable
from math import prod
from typing import Any, Type, TypeVar

import cv2
import jax
import numpy as np
from xtructure import StructuredType
from xtructure.core.display import (
    MAX_PRINT_BATCH_SIZE,
    SHOW_BATCH_SIZE,
    BatchedRenderer,
)

T = TypeVar("T")


def normalize_rgb_image(
    image: Any,
    *,
    size: tuple[int, int] | None = None,
    interpolation: int = cv2.INTER_NEAREST,
) -> np.ndarray:
    """Normalize a rendered image to a host ``uint8`` ``(H, W, 3)`` RGB array."""
    array = np.asarray(jax.device_get(image))
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Rendered image must have shape (H, W, 3), got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        array = np.round(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if size is not None:
        width, height = size
        if width <= 0 or height <= 0:
            raise ValueError(f"Target image size must be positive, got {size!r}")
        array = cv2.resize(array, (width, height), interpolation=interpolation)
    return array


def img_to_colored_str(image: Any) -> str:
    """Render an RGB image array as 24-bit ANSI ``██`` terminal rows."""
    rgb = normalize_rgb_image(image)
    return "\n".join(
        "".join(f"\x1b[38;2;{int(r)};{int(g)};{int(b)}m██\x1b[0m" for r, g, b in row)
        for row in rgb
    )


def attach_state_renderer(
    cls: Type[T],
    imgfunc: Callable[..., Any],
    strfunc: Callable[..., str] | None = None,
) -> Type[T]:
    """Attach host-side ``img`` (and optional ``str`` / ``__str__``) to ``cls``.

    ``img`` dispatches on ``self.structured_type``:
    ``StructuredType.SINGLE`` invokes ``imgfunc(self, **kwargs)`` once;
    ``StructuredType.BATCHED`` invokes it once per element and stacks results.
    Every rendered frame is normalized via :func:`normalize_rgb_image`, and an
    optional ``size=(width, height)`` keyword resizes each frame cleanly.

    When ``strfunc`` is provided, ``str`` and ``__str__`` are bound with
    ``StructuredType.SINGLE`` delegating to ``strfunc`` and batched inputs
    formatted via Xtructure's :class:`BatchedRenderer`.
    """

    def get_img(
        self: Any,
        *,
        size: tuple[int, int] | None = None,
        interpolation: int = cv2.INTER_NEAREST,
        **kwargs: Any,
    ) -> np.ndarray:
        structured_type = self.structured_type

        if structured_type == StructuredType.SINGLE:
            return normalize_rgb_image(
                imgfunc(self, **kwargs), size=size, interpolation=interpolation
            )
        if structured_type == StructuredType.BATCHED:
            batch_shape = self.shape
            batch_len = prod(batch_shape)
            results = []
            for i in range(batch_len):
                index = np.unravel_index(i, batch_shape)
                current_state = jax.tree_util.tree_map(lambda x: x[index], self)
                results.append(
                    normalize_rgb_image(
                        imgfunc(current_state, **kwargs),
                        size=size,
                        interpolation=interpolation,
                    )
                )
            return np.stack(results, axis=0)
        raise ValueError(
            f"State is not structured: {self.layout.shape_tuple} != {self.default_shape}"
        )

    setattr(cls, "img", get_img)

    if strfunc is not None:
        batched_renderer = BatchedRenderer(strfunc)

        def render_str(self: Any, **kwargs: Any) -> str:
            if self.structured_type == StructuredType.SINGLE:
                return strfunc(self, **kwargs)
            return batched_renderer.render(
                self,
                max_size=kwargs.pop("max_size", MAX_PRINT_BATCH_SIZE),
                show_size=kwargs.pop("show_size", SHOW_BATCH_SIZE),
                title=f"Batched {type(self).__name__}",
                **kwargs,
            )

        setattr(cls, "str", render_str)
        setattr(cls, "__str__", render_str)

    return cls


__all__ = ["attach_state_renderer", "normalize_rgb_image", "img_to_colored_str"]
