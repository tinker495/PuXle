"""Curated preset constructor arguments for :class:`PDDL`."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType


class PDDLPresets:
    """Representative bundled PDDL constructor kwargs.

    Each attribute unpacks directly into :class:`PDDL`::

        puzzle = PDDL(**PDDLPresets.BLOCKSWORLD_H01)
    """

    BLOCKSWORLD_S01: Mapping[str, str] = MappingProxyType(
        {"domain": "blocksworld", "problem": "bw-S-01"}
    )
    BLOCKSWORLD_H01: Mapping[str, str] = MappingProxyType(
        {"domain": "blocksworld", "problem": "bw-H-01"}
    )
    GRIPPER_S01: Mapping[str, str] = MappingProxyType(
        {"domain": "gripper", "problem": "gr-S-01"}
    )
    GRIPPER_H01: Mapping[str, str] = MappingProxyType(
        {"domain": "gripper", "problem": "gr-H-01"}
    )
    LOGISTICS_S01: Mapping[str, str] = MappingProxyType(
        {"domain": "logistics", "problem": "lg-S-01"}
    )
    LOGISTICS_H01: Mapping[str, str] = MappingProxyType(
        {"domain": "logistics", "problem": "lg-H-01"}
    )
    ROVERS_S01: Mapping[str, str] = MappingProxyType(
        {"domain": "rovers", "problem": "rv-S-01"}
    )
    ROVERS_H01: Mapping[str, str] = MappingProxyType(
        {"domain": "rovers", "problem": "rv-H-01"}
    )
    SATELLITE_S01: Mapping[str, str] = MappingProxyType(
        {"domain": "satellite", "problem": "st-S-01"}
    )
    SATELLITE_H01: Mapping[str, str] = MappingProxyType(
        {"domain": "satellite", "problem": "st-H-01"}
    )

    @classmethod
    def all(cls) -> dict[str, Mapping[str, str]]:
        """Return all curated presets keyed by preset name."""
        return {
            "BLOCKSWORLD_S01": cls.BLOCKSWORLD_S01,
            "BLOCKSWORLD_H01": cls.BLOCKSWORLD_H01,
            "GRIPPER_S01": cls.GRIPPER_S01,
            "GRIPPER_H01": cls.GRIPPER_H01,
            "LOGISTICS_S01": cls.LOGISTICS_S01,
            "LOGISTICS_H01": cls.LOGISTICS_H01,
            "ROVERS_S01": cls.ROVERS_S01,
            "ROVERS_H01": cls.ROVERS_H01,
            "SATELLITE_S01": cls.SATELLITE_S01,
            "SATELLITE_H01": cls.SATELLITE_H01,
        }


__all__ = ["PDDLPresets"]
