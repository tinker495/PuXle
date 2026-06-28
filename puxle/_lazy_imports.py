"""Tiny helpers for lazy public re-exports."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

LazyExportSpec = tuple[str, str]


def load_lazy_export(
    name: str,
    module_name: str,
    exports: Mapping[str, LazyExportSpec],
    namespace: dict[str, Any],
) -> Any:
    try:
        target_module, attr_name = exports[name]
    except KeyError as exc:
        raise AttributeError(
            f"module {module_name!r} has no attribute {name!r}"
        ) from exc

    module = importlib.import_module(target_module, module_name)
    value = getattr(module, attr_name)
    namespace[name] = value
    return value


def lazy_dir(namespace: Mapping[str, Any], exports: list[str]) -> list[str]:
    return sorted(set(namespace) | set(exports))
