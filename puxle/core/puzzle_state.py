from __future__ import annotations

from typing import Any, Type, TypeVar

from xtructure import FieldDescriptor, Xtructurable, xtructure_dataclass

T = TypeVar("T")

FieldDescriptor = FieldDescriptor


class PuzzleState(Xtructurable):
    """
    Marker base-class for PuXle states.

    Notes:

    - PuXle state/solve-config classes are typically created via `@state_dataclass`.
    - In-memory bitpacking is handled by xtructure (FieldDescriptor.packed_tensor / aggregate bitpack),
      not by overriding this base class.
    """

    pass


def state_dataclass(cls: Type[T] | None = None, **kwargs: Any):
    """
    Decorator used to define a JAX-compatible xtructure dataclass for PuXle state objects.

    Default behavior:

    - Enables xtructure bitpacking helpers via `bitpack="auto"` by default.
    """

    def wrap(target_cls: Type[T]) -> Type[T]:
        call_kwargs = dict(kwargs)
        call_kwargs.setdefault("bitpack", "auto")
        return xtructure_dataclass(target_cls, **call_kwargs)

    if cls is None:
        return wrap
    return wrap(cls)
