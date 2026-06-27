"""Core puzzle framework components."""

from __future__ import annotations

from typing import Any, TypeVar

from puxle._lazy_imports import lazy_dir, load_lazy_export

StateT = TypeVar("StateT", bound=Any)

__all__ = ["Puzzle", "PuzzleState", "FieldDescriptor", "state_dataclass", "StateT"]

_EXPORTS = {
    "Puzzle": (".puzzle_base", "Puzzle"),
    "PuzzleState": (".puzzle_state", "PuzzleState"),
    "FieldDescriptor": (".puzzle_state", "FieldDescriptor"),
    "state_dataclass": (".puzzle_state", "state_dataclass"),
}


def __getattr__(name: str):
    return load_lazy_export(name, __name__, _EXPORTS, globals())


def __dir__() -> list[str]:
    return lazy_dir(globals(), __all__)
