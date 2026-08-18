from .._resources import world_model_data_path
from ..world_model_puzzle_base import WorldModelPuzzleBase


class _RubiksCubeWorldModelBase(WorldModelPuzzleBase):
    data_name = "rubikscube"
    latent_shape = (400,)

    def __init__(self, **kwargs):
        super().__init__(
            data_path=str(world_model_data_path(self.data_name)),
            data_shape=(32, 64, 3),
            latent_shape=self.latent_shape,
            action_size=12,
            **kwargs,
        )


class _ReversedDataMixin:
    def data_init(self):
        super().data_init()
        self.inits, self.targets = self.targets, self.inits


class RubiksCubeWorldModel_test(_RubiksCubeWorldModelBase):
    data_name = "rubikscube_test"


class RubiksCubeWorldModel(_RubiksCubeWorldModelBase):
    pass


class RubiksCubeWorldModel_reversed(_ReversedDataMixin, RubiksCubeWorldModel):
    pass


class RubiksCubeWorldModelOptimized(_RubiksCubeWorldModelBase):
    latent_shape = (240,)


class RubiksCubeWorldModelOptimized_test(RubiksCubeWorldModelOptimized):
    data_name = "rubikscube_test"


class RubiksCubeWorldModelOptimized_reversed(
    _ReversedDataMixin, RubiksCubeWorldModelOptimized
):
    pass
