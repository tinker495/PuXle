from __future__ import annotations

import importlib
from typing import Any

__all__ = ["PDDL"]


def __getattr__(name: str) -> Any:
    if name != "PDDL":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(".pddl", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
