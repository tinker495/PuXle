"""Backwards-compatible shim for validation helpers."""

from __future__ import annotations

from .evaluation import ValidationResult, bfs_validate

__all__ = ["ValidationResult", "bfs_validate"]

