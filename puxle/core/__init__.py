"""Core puzzle framework components."""

from __future__ import annotations

import importlib
from typing import Any, TypeVar

StateT = TypeVar("StateT", bound=Any)

__all__ = ["Puzzle", "PuzzleState", "FieldDescriptor", "state_dataclass", "StateT"]

_EXPORTS = {
    "Puzzle": (".puzzle_base", "Puzzle"),
    "PuzzleState": (".puzzle_state", "PuzzleState"),
    "FieldDescriptor": (".puzzle_state", "FieldDescriptor"),
    "state_dataclass": (".puzzle_state", "state_dataclass"),
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
