from types import SimpleNamespace

from . import WorldModelPuzzleConfig
from .model.rubikscube_world_model import (
    RubiksCubeWorldModel,
    RubiksCubeWorldModel_reversed,
    RubiksCubeWorldModel_test,
    RubiksCubeWorldModelOptimized,
    RubiksCubeWorldModelOptimized_reversed,
    RubiksCubeWorldModelOptimized_test,
)
from .model.sokoban_world_model import SokobanWorldModel, SokobanWorldModelOptimized

_TRAINED_WORLD_MODEL_CONFIGS = {
    "RubiksCubeWorldModel": WorldModelPuzzleConfig(
        callable=RubiksCubeWorldModel,
        path="rubikscube_v2.pkl",
    ),
    "RubiksCubeWorldModel_test": WorldModelPuzzleConfig(
        callable=RubiksCubeWorldModel_test,
        path="rubikscube_v2.pkl",
    ),
    "RubiksCubeWorldModel_reversed": WorldModelPuzzleConfig(
        callable=RubiksCubeWorldModel_reversed,
        path="rubikscube_v2.pkl",
    ),
    "RubiksCubeWorldModelOptimized": WorldModelPuzzleConfig(
        callable=RubiksCubeWorldModelOptimized,
        path="rubikscube_optimized_v2.pkl",
    ),
    "RubiksCubeWorldModelOptimized_test": WorldModelPuzzleConfig(
        callable=RubiksCubeWorldModelOptimized_test,
        path="rubikscube_optimized_v2.pkl",
    ),
    "RubiksCubeWorldModelOptimized_reversed": WorldModelPuzzleConfig(
        callable=RubiksCubeWorldModelOptimized_reversed,
        path="rubikscube_optimized_v2.pkl",
    ),
    "SokobanWorldModel": WorldModelPuzzleConfig(
        callable=SokobanWorldModel,
        path="sokoban_v2.pkl",
    ),
    "SokobanWorldModelOptimized": WorldModelPuzzleConfig(
        callable=SokobanWorldModelOptimized,
        path="sokoban_optimized_v2.pkl",
    ),
}

trained_world_model_registry = SimpleNamespace(
    **{name: config.create for name, config in _TRAINED_WORLD_MODEL_CONFIGS.items()}
)


def _create_world_model_for_training(name: str, *, init_params: bool):
    return _TRAINED_WORLD_MODEL_CONFIGS[name]._create(
        init_params=init_params,
        aqt_cfg=None,
    )
