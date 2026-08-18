import chex
import flax.linen as nn
import jax.numpy as jnp

from .._resources import world_model_data_path
from .._utils import (
    DEFAULT_NORM_FN,
    DTYPE,
    PARAM_DTYPE,
    ConvResBlock,
    apply_norm,
    build_aqt_conv_general_dilated,
    get_norm_fn,
)
from ..world_model_puzzle_base import WorldModelPuzzleBase


class Encoder(nn.Module):
    latent_shape: tuple[int, ...]
    norm_fn: nn.Module = DEFAULT_NORM_FN
    aqt_cfg: object = None
    quant_mode: object = None

    @nn.compact
    def __call__(self, data, training=False):
        aqt_conv = build_aqt_conv_general_dilated(self.aqt_cfg, self.quant_mode)
        x = ((data / 255.0) * 2.0 - 1.0).astype(DTYPE)
        x = nn.Conv(
            16,
            (2, 2),
            strides=(2, 2),
            dtype=DTYPE,
            param_dtype=PARAM_DTYPE,
            conv_general_dilated_cls=aqt_conv,
        )(x)
        x = nn.relu(apply_norm(self.norm_fn, x, training))
        x = nn.Conv(
            16,
            (2, 2),
            strides=(2, 2),
            dtype=DTYPE,
            param_dtype=PARAM_DTYPE,
            conv_general_dilated_cls=aqt_conv,
        )(x)
        x = nn.relu(x)
        return nn.Conv(
            self.latent_shape[-1],
            (1, 1),
            strides=(1, 1),
            dtype=DTYPE,
            param_dtype=PARAM_DTYPE,
            conv_general_dilated_cls=aqt_conv,
        )(x)


class Decoder(nn.Module):
    data_shape: tuple[int, ...]
    norm_fn: nn.Module = DEFAULT_NORM_FN
    aqt_cfg: object = None
    quant_mode: object = None

    @nn.compact
    def __call__(self, latent, training=False):
        aqt_conv = build_aqt_conv_general_dilated(self.aqt_cfg, self.quant_mode)
        x = ((latent - 0.5) * 2.0).astype(DTYPE)
        x = nn.ConvTranspose(
            16, (2, 2), strides=(2, 2), dtype=DTYPE, param_dtype=PARAM_DTYPE
        )(x)
        x = nn.relu(apply_norm(self.norm_fn, x, training))
        x = nn.ConvTranspose(
            16, (2, 2), strides=(2, 2), dtype=DTYPE, param_dtype=PARAM_DTYPE
        )(x)
        x = nn.relu(x)
        return nn.Conv(
            3,
            (1, 1),
            strides=(1, 1),
            dtype=DTYPE,
            param_dtype=PARAM_DTYPE,
            conv_general_dilated_cls=aqt_conv,
        )(x)


class AutoEncoder(nn.Module):
    data_shape: tuple[int, ...]
    latent_shape: tuple[int, ...]
    norm_fn: nn.Module = DEFAULT_NORM_FN
    aqt_cfg: object = None
    quant_mode: object = None

    def setup(self):
        self.encoder = Encoder(
            self.latent_shape,
            norm_fn=self.norm_fn,
            aqt_cfg=self.aqt_cfg,
            quant_mode=self.quant_mode,
        )
        self.decoder = Decoder(
            self.data_shape,
            norm_fn=self.norm_fn,
            aqt_cfg=self.aqt_cfg,
            quant_mode=self.quant_mode,
        )

    def __call__(self, x0, training=False):
        latent = self.encoder(x0, training)
        return latent, self.decoder(latent, training)


class WorldModel(nn.Module):
    latent_shape: tuple[int, ...]
    action_size: int
    norm_fn: nn.Module = DEFAULT_NORM_FN
    aqt_cfg: object = None
    quant_mode: object = None

    @nn.compact
    def __call__(self, latent, training=False):
        aqt_conv = build_aqt_conv_general_dilated(self.aqt_cfg, self.quant_mode)
        x = ((latent - 0.5) * 2.0).astype(DTYPE)
        for _ in range(2):
            x = nn.Conv(
                32,
                (3, 3),
                strides=(1, 1),
                kernel_init=nn.initializers.orthogonal(),
                dtype=DTYPE,
                param_dtype=PARAM_DTYPE,
                conv_general_dilated_cls=aqt_conv,
            )(x)
            x = nn.relu(apply_norm(self.norm_fn, x, training))
        x = nn.Conv(
            self.latent_shape[-1] * self.action_size,
            (3, 3),
            strides=(1, 1),
            kernel_init=nn.initializers.orthogonal(),
            dtype=DTYPE,
            param_dtype=PARAM_DTYPE,
            conv_general_dilated_cls=aqt_conv,
        )(x)
        x = jnp.reshape(x, (x.shape[0], *self.latent_shape, self.action_size))
        return jnp.transpose(x, (0, 4, 1, 2, 3))


class _SokobanMixin:
    def batched_get_inverse_neighbours(
        self,
        solve_configs: WorldModelPuzzleBase.SolveConfig,
        states: WorldModelPuzzleBase.State,
        filleds: bool = True,
        multi_solve_config: bool = False,
    ) -> tuple[WorldModelPuzzleBase.State, chex.Array]:
        raise NotImplementedError(
            "Sokoban is not reversible, so its inverse neighbours are not implemented.\n"
            "Please use '--using_hindsight_target' to train distance"
        )

    def action_to_string(self, action: int) -> str:
        return self._directional_action_to_string(action)


class SokobanWorldModel(_SokobanMixin, WorldModelPuzzleBase):
    str_parse_img_size = 20

    def __init__(self, **kwargs):
        resolved_norm_fn = get_norm_fn(kwargs.pop("norm_fn", None))
        super().__init__(
            data_path=str(world_model_data_path("sokoban")),
            data_shape=(40, 40, 3),
            latent_shape=(10, 10, 16),
            action_size=4,
            AE=lambda **values: AutoEncoder(**(values | {"norm_fn": resolved_norm_fn})),
            WM=lambda **values: WorldModel(**(values | {"norm_fn": resolved_norm_fn})),
            **kwargs,
        )


class EncoderOptimized(nn.Module):
    latent_shape: tuple[int, int, int]
    norm_fn: nn.Module = DEFAULT_NORM_FN
    aqt_cfg: object = None
    quant_mode: object = None

    @nn.compact
    def __call__(self, data, training=False):
        aqt_conv = build_aqt_conv_general_dilated(self.aqt_cfg, self.quant_mode)
        x = ((data / 255.0) * 2.0 - 1.0).astype(DTYPE)
        x = nn.Conv(
            16,
            (4, 4),
            strides=(4, 4),
            kernel_init=nn.initializers.orthogonal(),
            dtype=DTYPE,
            param_dtype=PARAM_DTYPE,
            conv_general_dilated_cls=aqt_conv,
        )(x)
        x = nn.relu(apply_norm(self.norm_fn, x, training))
        x = ConvResBlock(
            16,
            (1, 1),
            (1, 1),
            norm_fn=self.norm_fn,
            aqt_cfg=self.aqt_cfg,
            quant_mode=self.quant_mode,
        )(x, training)
        return nn.Conv(
            self.latent_shape[-1],
            (1, 1),
            strides=(1, 1),
            kernel_init=nn.initializers.orthogonal(),
            dtype=DTYPE,
            param_dtype=PARAM_DTYPE,
            conv_general_dilated_cls=aqt_conv,
        )(x)


class DecoderOptimized(nn.Module):
    data_shape: tuple[int, int, int]
    norm_fn: nn.Module = DEFAULT_NORM_FN
    aqt_cfg: object = None
    quant_mode: object = None

    @nn.compact
    def __call__(self, latent, training=False):
        aqt_conv = build_aqt_conv_general_dilated(self.aqt_cfg, self.quant_mode)
        x = ((latent - 0.5) * 2.0).astype(DTYPE)
        x = nn.Conv(
            16,
            (1, 1),
            strides=(1, 1),
            dtype=DTYPE,
            param_dtype=PARAM_DTYPE,
            conv_general_dilated_cls=aqt_conv,
        )(x)
        x = nn.relu(apply_norm(self.norm_fn, x, training))
        x = ConvResBlock(
            16,
            (1, 1),
            (1, 1),
            norm_fn=self.norm_fn,
            aqt_cfg=self.aqt_cfg,
            quant_mode=self.quant_mode,
        )(x, training)
        x = nn.ConvTranspose(
            16,
            (4, 4),
            strides=(4, 4),
            kernel_init=nn.initializers.orthogonal(),
            dtype=DTYPE,
            param_dtype=PARAM_DTYPE,
        )(x)
        x = nn.relu(x)
        return nn.Conv(
            3,
            (1, 1),
            strides=(1, 1),
            kernel_init=nn.initializers.orthogonal(),
            dtype=DTYPE,
            param_dtype=PARAM_DTYPE,
            conv_general_dilated_cls=aqt_conv,
        )(x)


class AutoEncoderOptimized(nn.Module):
    data_shape: tuple[int, int, int]
    latent_shape: tuple[int, int, int]
    norm_fn: nn.Module = DEFAULT_NORM_FN
    aqt_cfg: object = None
    quant_mode: object = None

    def setup(self):
        self.encoder = EncoderOptimized(
            self.latent_shape,
            norm_fn=self.norm_fn,
            aqt_cfg=self.aqt_cfg,
            quant_mode=self.quant_mode,
        )
        self.decoder = DecoderOptimized(
            self.data_shape,
            norm_fn=self.norm_fn,
            aqt_cfg=self.aqt_cfg,
            quant_mode=self.quant_mode,
        )

    def __call__(self, x0, training=False):
        latent = self.encoder(x0, training)
        return latent, self.decoder(latent, training)


class SokobanWorldModelOptimized(_SokobanMixin, WorldModelPuzzleBase):
    str_parse_img_size = 20

    def __init__(self, **kwargs):
        resolved_norm_fn = get_norm_fn(kwargs.pop("norm_fn", None))
        super().__init__(
            data_path=str(world_model_data_path("sokoban")),
            data_shape=(40, 40, 3),
            latent_shape=(10, 10, 2),
            action_size=4,
            AE=lambda **values: AutoEncoderOptimized(
                **(values | {"norm_fn": resolved_norm_fn})
            ),
            WM=lambda **values: WorldModel(**(values | {"norm_fn": resolved_norm_fn})),
            **kwargs,
        )
