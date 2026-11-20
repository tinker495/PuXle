"""Curated preset constructor arguments for :class:`CayleyPuzzle`."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


class CayleyPresets:
    """Representative Cayley/Schreier graph constructor kwargs.

    Each attribute unpacks directly into :class:`CayleyPuzzle`::

        puzzle = CayleyPuzzle(**CayleyPresets.TOP_SPIN_8_K4)
    """

    PANCAKE_7: Mapping[str, Any] = MappingProxyType({"graph": "pancake", "n": 7})
    PANCAKE_8: Mapping[str, Any] = MappingProxyType({"graph": "pancake", "n": 8})
    LRX_8: Mapping[str, Any] = MappingProxyType({"graph": "lrx", "n": 8})
    TOP_SPIN_8_K4: Mapping[str, Any] = MappingProxyType(
        {"graph": "top_spin", "n": 8, "k": 4}
    )
    COXETER_8: Mapping[str, Any] = MappingProxyType({"graph": "coxeter", "n": 8})
    HEISENBERG_3_MOD5: Mapping[str, Any] = MappingProxyType(
        {"graph": "heisenberg", "n": 3, "modulo": 5}
    )
    MINI_PYRAMORPHIX: Mapping[str, Any] = MappingProxyType(
        {"graph": "mini_pyramorphix"}
    )

    @classmethod
    def all(cls) -> dict[str, Mapping[str, Any]]:
        """Return all curated presets keyed by preset name."""
        return {
            "PANCAKE_7": cls.PANCAKE_7,
            "PANCAKE_8": cls.PANCAKE_8,
            "LRX_8": cls.LRX_8,
            "TOP_SPIN_8_K4": cls.TOP_SPIN_8_K4,
            "COXETER_8": cls.COXETER_8,
            "HEISENBERG_3_MOD5": cls.HEISENBERG_3_MOD5,
            "MINI_PYRAMORPHIX": cls.MINI_PYRAMORPHIX,
        }


__all__ = ["CayleyPresets"]
