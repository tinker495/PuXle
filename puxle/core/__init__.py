"""
Core puzzle framework components.

This module provides the base classes and data structures for creating puzzle environments.
"""

from typing import Any, TypeVar

from puxle.core.puzzle_base import Puzzle
from puxle.core.puzzle_state import FieldDescriptor, PuzzleState, state_dataclass

StateT = TypeVar("StateT", bound=Any)

__all__ = [
    "Puzzle",
    "PuzzleState",
    "FieldDescriptor",
    "state_dataclass",
    "StateT",
]
