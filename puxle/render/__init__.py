"""Puzzle Renderer Module."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["attach_state_renderer", "Cv2Backend"]

_EXPORTS = {
    "attach_state_renderer": (".state_renderer", "attach_state_renderer"),
    "Cv2Backend": (".backends.cv2_backend", "Cv2Backend"),
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
