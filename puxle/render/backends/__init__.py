"""Puzzle Render Backend implementations."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["Cv2Backend", "BgrColor", "Point", "Size"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(".cv2_backend", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
