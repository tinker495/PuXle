"""Utility functions and constants for PuXle puzzles."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "IMG_SIZE",
    "add_img_parser",
    "coloring_str",
    "from_uint8",
    "to_uint8",
    "pack_variable_bits",
    "unpack_variable_bits",
]

_EXPORTS = {
    "IMG_SIZE": (".annotate", "IMG_SIZE"),
    "add_img_parser": (".util", "add_img_parser"),
    "coloring_str": (".util", "coloring_str"),
    "from_uint8": (".util", "from_uint8"),
    "to_uint8": (".util", "to_uint8"),
    "pack_variable_bits": (".util", "pack_variable_bits"),
    "unpack_variable_bits": (".util", "unpack_variable_bits"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(importlib.import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
