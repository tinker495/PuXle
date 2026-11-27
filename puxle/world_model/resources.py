from __future__ import annotations

import os
from pathlib import Path

import huggingface_hub


def get_puxle_home() -> Path:
    """Return the root used for downloaded PuXle artifacts."""
    if configured := os.environ.get("PUXLE_HOME"):
        return Path(configured).expanduser()
    if cache_root := os.environ.get("XDG_CACHE_HOME"):
        return Path(cache_root).expanduser() / "puxle"
    return Path.home() / ".cache" / "puxle"


def world_model_root() -> Path:
    """Return the root directory for world-model datasets and checkpoints."""
    return get_puxle_home() / "world_model"


def world_model_checkpoint_path(filename: str | os.PathLike[str]) -> Path:
    """Resolve a checkpoint filename inside ``world_model_root()`` unless absolute."""
    path = Path(filename).expanduser()
    if path.is_absolute():
        return path
    return world_model_root() / "model" / "params" / path.name


def world_model_data_path(name: str) -> Path:
    """Return the dataset directory for a named world-model environment."""
    return world_model_root() / "data" / name


def is_world_model_dataset_downloaded() -> bool:
    """Return whether the shared world-model dataset directory exists."""
    return (world_model_root() / "data").exists()


def download_world_model_dataset() -> str:
    """Download the Hugging Face world-model dataset into ``world_model_root()``."""
    return huggingface_hub.snapshot_download(
        repo_id="Tinker/puzzle_world_model_ds",
        repo_type="dataset",
        local_dir=str(world_model_root()),
    )


__all__ = [
    "download_world_model_dataset",
    "get_puxle_home",
    "is_world_model_dataset_downloaded",
    "world_model_checkpoint_path",
    "world_model_data_path",
    "world_model_root",
]
