"""Curated preset constructor arguments for :class:`WorldModelPuzzle`."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


class WorldModelPresets:
    """Checkpoint-backed world-model puzzle constructor kwargs.

    Each attribute unpacks directly into :class:`WorldModelPuzzle`::

        puzzle = WorldModelPuzzle(**WorldModelPresets.RUBIKSCUBE)
    """

    RUBIKSCUBE: Mapping[str, Any] = MappingProxyType(
        {
            "architecture": "rubikscube",
            "data_name": "rubikscube",
            "path": "rubikscube_v2.pkl",
            "aqt_cfg": "int8",
        }
    )
    RUBIKSCUBE_TEST: Mapping[str, Any] = MappingProxyType(
        {
            "architecture": "rubikscube",
            "data_name": "rubikscube_test",
            "path": "rubikscube_v2.pkl",
            "aqt_cfg": "int8",
        }
    )
    RUBIKSCUBE_REVERSED: Mapping[str, Any] = MappingProxyType(
        {
            "architecture": "rubikscube",
            "data_name": "rubikscube",
            "reverse_data": True,
            "path": "rubikscube_v2.pkl",
            "aqt_cfg": "int8",
        }
    )
    RUBIKSCUBE_OPTIMIZED: Mapping[str, Any] = MappingProxyType(
        {
            "architecture": "rubikscube_optimized",
            "data_name": "rubikscube",
            "path": "rubikscube_optimized_v2.pkl",
            "aqt_cfg": "int8",
        }
    )
    RUBIKSCUBE_OPTIMIZED_TEST: Mapping[str, Any] = MappingProxyType(
        {
            "architecture": "rubikscube_optimized",
            "data_name": "rubikscube_test",
            "path": "rubikscube_optimized_v2.pkl",
            "aqt_cfg": "int8",
        }
    )
    RUBIKSCUBE_OPTIMIZED_REVERSED: Mapping[str, Any] = MappingProxyType(
        {
            "architecture": "rubikscube_optimized",
            "data_name": "rubikscube",
            "reverse_data": True,
            "path": "rubikscube_optimized_v2.pkl",
            "aqt_cfg": "int8",
        }
    )
    SOKOBAN: Mapping[str, Any] = MappingProxyType(
        {
            "architecture": "sokoban",
            "data_name": "sokoban",
            "path": "sokoban_v2.pkl",
            "aqt_cfg": "int8",
        }
    )
    SOKOBAN_OPTIMIZED: Mapping[str, Any] = MappingProxyType(
        {
            "architecture": "sokoban_optimized",
            "data_name": "sokoban",
            "path": "sokoban_optimized_v2.pkl",
            "aqt_cfg": "int8",
        }
    )

    @classmethod
    def all(cls) -> dict[str, Mapping[str, Any]]:
        """Return all curated presets keyed by preset name."""
        return {
            "RUBIKSCUBE": cls.RUBIKSCUBE,
            "RUBIKSCUBE_TEST": cls.RUBIKSCUBE_TEST,
            "RUBIKSCUBE_REVERSED": cls.RUBIKSCUBE_REVERSED,
            "RUBIKSCUBE_OPTIMIZED": cls.RUBIKSCUBE_OPTIMIZED,
            "RUBIKSCUBE_OPTIMIZED_TEST": cls.RUBIKSCUBE_OPTIMIZED_TEST,
            "RUBIKSCUBE_OPTIMIZED_REVERSED": cls.RUBIKSCUBE_OPTIMIZED_REVERSED,
            "SOKOBAN": cls.SOKOBAN,
            "SOKOBAN_OPTIMIZED": cls.SOKOBAN_OPTIMIZED,
        }


__all__ = ["WorldModelPresets"]
