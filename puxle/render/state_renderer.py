"""Canonical seam for attaching image rendering to a puzzle ``State`` class.

See ``CONTEXT.md`` "Puzzle Renderer". The mechanism is the single source of
truth for batching a per-state ``imgfunc`` over ``StructuredType.BATCHED``
inputs and exposing the result as a ``state.img(**kwargs)`` method.
"""

from __future__ import annotations

from math import prod
from typing import Callable, Type, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from xtructure import StructuredType

T = TypeVar("T")


def attach_state_renderer(cls: Type[T], imgfunc: Callable) -> Type[T]:
    """Attach a host-side ``img`` rendering method to a ``State`` class.

    The attached method dispatches on ``self.structured_type``:
    ``StructuredType.SINGLE`` invokes ``imgfunc(self, **kwargs)`` once;
    ``StructuredType.BATCHED`` invokes it once per element via a host-side
    loop and stacks results. Any other ``structured_type`` raises
    ``ValueError`` so that callers see the mismatch immediately rather than
    consuming a silently wrong image.

    Args:
        cls: The xtructure-decorated ``State`` (or ``SolveConfig``) class to
            extend.
        imgfunc: A callable ``(state, **kwargs) -> np.ndarray`` that renders
            one unbatched state as an ``(H, W, 3)`` RGB array.

    Returns:
        ``cls`` with the new ``img`` method bound. The method is attached in
        place; the return is for fluent chaining.
    """

    def get_img(self, **kwargs) -> np.ndarray:
        structured_type = self.structured_type

        if structured_type == StructuredType.SINGLE:
            return imgfunc(self, **kwargs)
        elif structured_type == StructuredType.BATCHED:
            batch_shape = self.batch_shape
            batch_len = prod(batch_shape)
            results = []
            for i in range(batch_len):
                index = jnp.unravel_index(i, batch_shape)
                current_state = jax.tree_util.tree_map(lambda x: x[index], self)
                results.append(imgfunc(current_state, **kwargs))
            results = np.stack(results, axis=0)
            return results
        else:
            raise ValueError(
                f"State is not structured: {self.shape} != {self.default_shape}"
            )

    setattr(cls, "img", get_img)
    return cls


__all__ = ["attach_state_renderer"]
