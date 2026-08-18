from __future__ import annotations

import os
from pathlib import Path


def get_puxle_home() -> Path:
    """Return the root used for downloaded PuXle artifacts."""
    if configured := os.environ.get("PUXLE_HOME"):
        return Path(configured).expanduser()
    if cache_root := os.environ.get("XDG_CACHE_HOME"):
        return Path(cache_root).expanduser() / "puxle"
    return Path.home() / ".cache" / "puxle"


def world_model_root() -> Path:
    return get_puxle_home() / "world_model"


def world_model_checkpoint_path(filename: str | os.PathLike[str]) -> Path:
    path = Path(filename).expanduser()
    if path.is_absolute():
        return path
    return world_model_root() / "model" / "params" / path.name


def world_model_data_path(name: str) -> Path:
    return world_model_root() / "data" / name
