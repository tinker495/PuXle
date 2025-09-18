"""Helpers for caching grounded PDDL artefacts between runs."""

from __future__ import annotations

import hashlib
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


CACHE_VERSION = "1"


def _cache_root() -> Path:
    base = os.environ.get("PUXLE_CACHE_DIR")
    if base:
        root = Path(base)
    else:
        root = Path.home() / ".cache" / "puxle"
    target = root / "pddl"
    target.mkdir(parents=True, exist_ok=True)
    return target


def compute_cache_key(domain_path: str | Path, problem_path: str | Path) -> str:
    """Compute a stable hash for a domain/problem pair and cache version."""
    hasher = hashlib.sha256()
    for path in (domain_path, problem_path):
        file_path = Path(path)
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 64), b""):
                hasher.update(chunk)
    hasher.update(CACHE_VERSION.encode("ascii"))
    return hasher.hexdigest()


def _cache_file(cache_key: str) -> Path:
    return _cache_root() / f"{cache_key}.pkl"


def _serialise_actions(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    serialised: List[Dict[str, Any]] = []
    for action in actions:
        serialised.append(
            {
                "name": action["name"],
                "parameters": list(action.get("parameters", [])),
                "preconditions": list(action.get("preconditions", [])),
                "preconditions_neg": list(action.get("preconditions_neg", [])),
                "effects": [
                    list(action.get("effects", ([], []))[0]),
                    list(action.get("effects", ([], []))[1]),
                ],
            }
        )
    return serialised


def _deserialise_actions(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    restored: List[Dict[str, Any]] = []
    for action in actions:
        restored.append(
            {
                "name": action["name"],
                "parameters": list(action.get("parameters", [])),
                "preconditions": list(action.get("preconditions", [])),
                "preconditions_neg": list(action.get("preconditions_neg", [])),
                "effects": (
                    list(action.get("effects", [[], []])[0]),
                    list(action.get("effects", [[], []])[1]),
                ),
            }
        )
    return restored


@dataclass
class CachedGrounding:
    grounded_atoms: List[str]
    atom_to_idx: Dict[str, int]
    grounded_actions: List[Dict[str, Any]]
    pre_mask_pos: np.ndarray
    pre_mask_neg: np.ndarray
    add_mask: np.ndarray
    del_mask: np.ndarray


def load_grounding_cache(cache_key: str) -> Optional[CachedGrounding]:
    """Load cached grounding artefacts if available."""
    path = _cache_file(cache_key)
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            data = pickle.load(handle)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if data.get("version") != CACHE_VERSION:
        return None
    payload = data.get("payload")
    if not isinstance(payload, dict):
        return None
    try:
        return CachedGrounding(
            grounded_atoms=list(payload["grounded_atoms"]),
            atom_to_idx=dict(payload["atom_to_idx"]),
            grounded_actions=_deserialise_actions(payload["grounded_actions"]),
            pre_mask_pos=np.array(payload["pre_mask_pos"], dtype=bool),
            pre_mask_neg=np.array(payload["pre_mask_neg"], dtype=bool),
            add_mask=np.array(payload["add_mask"], dtype=bool),
            del_mask=np.array(payload["del_mask"], dtype=bool),
        )
    except KeyError:
        return None


def store_grounding_cache(
    cache_key: str,
    *,
    grounded_atoms: List[str],
    atom_to_idx: Dict[str, int],
    grounded_actions: List[Dict[str, Any]],
    pre_mask_pos: np.ndarray,
    pre_mask_neg: np.ndarray,
    add_mask: np.ndarray,
    del_mask: np.ndarray,
) -> None:
    """Persist grounding artefacts for future reuse."""
    payload = {
        "grounded_atoms": list(grounded_atoms),
        "atom_to_idx": dict(atom_to_idx),
        "grounded_actions": _serialise_actions(grounded_actions),
        "pre_mask_pos": np.asarray(pre_mask_pos, dtype=bool),
        "pre_mask_neg": np.asarray(pre_mask_neg, dtype=bool),
        "add_mask": np.asarray(add_mask, dtype=bool),
        "del_mask": np.asarray(del_mask, dtype=bool),
    }
    data = {"version": CACHE_VERSION, "payload": payload}
    path = _cache_file(cache_key)
    try:
        with path.open("wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        # Cache writes are best-effort; failure should not be fatal.
        pass


__all__ = [
    "CachedGrounding",
    "CACHE_VERSION",
    "compute_cache_key",
    "load_grounding_cache",
    "store_grounding_cache",
]

