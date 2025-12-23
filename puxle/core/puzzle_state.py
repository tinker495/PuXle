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
    - Enables xtructure bitpacking helpers via `bitpack="auto"` when supported.
    - Preserves backwards compatibility by providing identity `.packed` / `.unpacked`
      properties for non-bitpacked states.
    """

    def wrap(target_cls: Type[T]) -> Type[T]:
        # Prefer enabling xtructure's built-in bitpacking helpers by default.
        call_kwargs = dict(kwargs)
        call_kwargs.setdefault("bitpack", "auto")

        try:
            dc_cls = xtructure_dataclass(target_cls, **call_kwargs)
        except TypeError:
            # Backwards-compatible fallback if the installed xtructure doesn't accept `bitpack=`.
            call_kwargs.pop("bitpack", None)
            dc_cls = xtructure_dataclass(target_cls, **call_kwargs)

        if not hasattr(dc_cls, "packed") and not hasattr(dc_cls, "unpacked"):
            # Backwards compatibility: treat state as already packed/unpacked.
            def packed(self) -> Any:
                return self

            setattr(dc_cls, "packed", property(packed))

            def unpacked(self) -> Any:
                return self

            setattr(dc_cls, "unpacked", property(unpacked))

        elif hasattr(dc_cls, "packed") ^ hasattr(dc_cls, "unpacked"):
            # Packing and unpacking must be implemented together.
            raise ValueError(
                "State class must implement both packing and unpacking (or neither)."
            )

        return dc_cls

    if cls is None:
        return wrap
    return wrap(cls)
