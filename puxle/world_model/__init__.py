"""Optional learned puzzle models, datasets, and training kernels."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from puxle._lazy_imports import lazy_dir, load_lazy_export


@dataclass(frozen=True)
class WorldModelPuzzleConfig:
    callable: Callable
    path: str

    def _create(self, *, init_params: bool, aqt_cfg: object):
        from ._resources import world_model_checkpoint_path

        return self.callable(
            path=str(world_model_checkpoint_path(self.path)),
            init_params=init_params,
            aqt_cfg=aqt_cfg,
        )

    def create(self):
        return self._create(init_params=False, aqt_cfg="int8")


__all__ = [
    "RubiksCubeWorldModel",
    "RubiksCubeWorldModel_reversed",
    "RubiksCubeWorldModel_test",
    "RubiksCubeWorldModelOptimized",
    "RubiksCubeWorldModelOptimized_reversed",
    "RubiksCubeWorldModelOptimized_test",
    "SokobanWorldModel",
    "SokobanWorldModelOptimized",
    "WorldModelPuzzleConfig",
    "WorldModelPuzzleBase",
    "accuracy_fn",
    "apply_with_conditional_batch_stats",
    "create_eval_trajectory",
    "create_sample_data",
    "create_shuffled_path",
    "download_world_model_dataset",
    "get_sample_data_builder",
    "get_world_model_dataset_builder",
    "is_world_model_dataset_downloaded",
    "round_through_gradient",
    "trajectory_to_eval_trajectory",
    "trajectory_to_transition_dataset",
    "world_model_eval_builder",
    "trained_world_model_registry",
    "world_model_train_builder",
]

_RUBIK = ".model.rubikscube_world_model"
_SOKOBAN = ".model.sokoban_world_model"
_BASE = ".world_model_puzzle_base"
_DATASET = ".world_model_ds"
_TRAIN = ".world_model_train"
_UTIL = "._utils"

_EXPORTS = {
    "RubiksCubeWorldModel": (_RUBIK, "RubiksCubeWorldModel"),
    "RubiksCubeWorldModel_reversed": (_RUBIK, "RubiksCubeWorldModel_reversed"),
    "RubiksCubeWorldModel_test": (_RUBIK, "RubiksCubeWorldModel_test"),
    "RubiksCubeWorldModelOptimized": (_RUBIK, "RubiksCubeWorldModelOptimized"),
    "RubiksCubeWorldModelOptimized_reversed": (
        _RUBIK,
        "RubiksCubeWorldModelOptimized_reversed",
    ),
    "RubiksCubeWorldModelOptimized_test": (
        _RUBIK,
        "RubiksCubeWorldModelOptimized_test",
    ),
    "SokobanWorldModel": (_SOKOBAN, "SokobanWorldModel"),
    "SokobanWorldModelOptimized": (_SOKOBAN, "SokobanWorldModelOptimized"),
    "WorldModelPuzzleBase": (_BASE, "WorldModelPuzzleBase"),
    "accuracy_fn": (_TRAIN, "accuracy_fn"),
    "apply_with_conditional_batch_stats": (_UTIL, "apply_with_conditional_batch_stats"),
    "create_eval_trajectory": (_DATASET, "create_eval_trajectory"),
    "create_sample_data": (_DATASET, "create_sample_data"),
    "create_shuffled_path": (_DATASET, "create_shuffled_path"),
    "download_world_model_dataset": (".util", "download_world_model_dataset"),
    "get_sample_data_builder": (_DATASET, "get_sample_data_builder"),
    "get_world_model_dataset_builder": (_DATASET, "get_world_model_dataset_builder"),
    "is_world_model_dataset_downloaded": (".util", "is_world_model_dataset_downloaded"),
    "round_through_gradient": (_UTIL, "round_through_gradient"),
    "trajectory_to_eval_trajectory": (_DATASET, "trajectory_to_eval_trajectory"),
    "trajectory_to_transition_dataset": (_DATASET, "trajectory_to_transition_dataset"),
    "world_model_eval_builder": (_TRAIN, "world_model_eval_builder"),
    "trained_world_model_registry": (
        ".trained_model_registry",
        "trained_world_model_registry",
    ),
    "world_model_train_builder": (_TRAIN, "world_model_train_builder"),
}


def __getattr__(name: str):
    return load_lazy_export(name, __name__, _EXPORTS, globals())


def __dir__() -> list[str]:
    return lazy_dir(globals(), __all__)
