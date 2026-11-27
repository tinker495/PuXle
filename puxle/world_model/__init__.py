"""World-model training kernels, trajectory dataset builders, and artifact paths."""

from __future__ import annotations

from puxle._lazy_imports import lazy_dir, load_lazy_export

__all__ = [
    "accuracy_fn",
    "apply_with_conditional_batch_stats",
    "create_eval_trajectory",
    "create_sample_data",
    "create_shuffled_path",
    "download_world_model_dataset",
    "get_puxle_home",
    "get_sample_data_builder",
    "get_world_model_dataset_builder",
    "is_world_model_dataset_downloaded",
    "round_through_gradient",
    "trajectory_to_eval_trajectory",
    "trajectory_to_transition_dataset",
    "world_model_checkpoint_path",
    "world_model_data_path",
    "world_model_eval_builder",
    "world_model_root",
    "world_model_train_builder",
]

_DATASET = ".world_model_ds"
_RESOURCES = ".resources"
_TRAIN = ".world_model_train"

_EXPORTS = {
    "accuracy_fn": (_TRAIN, "accuracy_fn"),
    "apply_with_conditional_batch_stats": (
        _TRAIN,
        "apply_with_conditional_batch_stats",
    ),
    "create_eval_trajectory": (_DATASET, "create_eval_trajectory"),
    "create_sample_data": (_DATASET, "create_sample_data"),
    "create_shuffled_path": (_DATASET, "create_shuffled_path"),
    "download_world_model_dataset": (_RESOURCES, "download_world_model_dataset"),
    "get_puxle_home": (_RESOURCES, "get_puxle_home"),
    "get_sample_data_builder": (_DATASET, "get_sample_data_builder"),
    "get_world_model_dataset_builder": (_DATASET, "get_world_model_dataset_builder"),
    "is_world_model_dataset_downloaded": (
        _RESOURCES,
        "is_world_model_dataset_downloaded",
    ),
    "round_through_gradient": (_TRAIN, "round_through_gradient"),
    "trajectory_to_eval_trajectory": (_DATASET, "trajectory_to_eval_trajectory"),
    "trajectory_to_transition_dataset": (_DATASET, "trajectory_to_transition_dataset"),
    "world_model_checkpoint_path": (_RESOURCES, "world_model_checkpoint_path"),
    "world_model_data_path": (_RESOURCES, "world_model_data_path"),
    "world_model_eval_builder": (_TRAIN, "world_model_eval_builder"),
    "world_model_root": (_RESOURCES, "world_model_root"),
    "world_model_train_builder": (_TRAIN, "world_model_train_builder"),
}


def __getattr__(name: str):
    return load_lazy_export(name, __name__, _EXPORTS, globals())


def __dir__() -> list[str]:
    return lazy_dir(globals(), __all__)
