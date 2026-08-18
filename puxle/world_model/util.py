import huggingface_hub

from ._resources import world_model_root


def is_world_model_dataset_downloaded():
    return (world_model_root() / "data").exists()


def download_world_model_dataset():
    return huggingface_hub.snapshot_download(
        repo_id="Tinker/puzzle_world_model_ds",
        repo_type="dataset",
        local_dir=str(world_model_root()),
    )
