"""Flax architectures, AQT quantization helpers, and checkpoint I/O for world models."""

from __future__ import annotations

import os
import pickle
import shutil
import tempfile
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from math import prod
from pathlib import Path
from typing import Any

import flax.linen as nn
import huggingface_hub
import jax
import jax.numpy as jnp
from aqt.jax.v2 import config as aqt_config
from aqt.jax.v2.flax import aqt_flax
from huggingface_hub.errors import EntryNotFoundError

from puxle.world_model import world_model_checkpoint_path

DTYPE = jnp.bfloat16
PARAM_DTYPE = jnp.float32
DEFAULT_NORM_FN = partial(nn.BatchNorm, param_dtype=PARAM_DTYPE)


def apply_norm(norm_fn, x, training):
    if norm_fn is None:
        return x
    fn = norm_fn.func if isinstance(norm_fn, partial) else norm_fn
    if isinstance(fn, type) and issubclass(fn, nn.BatchNorm):
        return norm_fn()(x, use_running_average=not training)
    return norm_fn()(x)


NORM_FN_REGISTRY = {
    "batch": DEFAULT_NORM_FN,
    "batch0999": partial(nn.BatchNorm, momentum=0.999, param_dtype=PARAM_DTYPE),
    "instance": partial(nn.InstanceNorm, param_dtype=PARAM_DTYPE),
    "layer": partial(nn.LayerNorm, param_dtype=PARAM_DTYPE),
    "group": partial(nn.GroupNorm, param_dtype=PARAM_DTYPE),
    "rms": partial(nn.RMSNorm, param_dtype=PARAM_DTYPE),
}


def _validate_aqt_cfg(aqt_cfg):
    if aqt_cfg not in (None, "int8"):
        raise ValueError(f"Unsupported AQT configuration: {aqt_cfg}")


def _rhs_freeze_mode(quant_mode):
    return (
        aqt_flax.FreezerMode.CALIBRATION_AND_VALUE
        if quant_mode in (aqt_flax.QuantMode.CONVERT, aqt_flax.QuantMode.SERVE)
        else aqt_flax.FreezerMode.NONE
    )


def build_aqt_dot_general(aqt_cfg, quant_mode):
    _validate_aqt_cfg(aqt_cfg)
    if aqt_cfg is None:
        return None
    return partial(
        aqt_flax.AqtDotGeneral,
        aqt_config.fully_quantized(fwd_bits=8, bwd_bits=8),
        prng_name=None,
        rhs_quant_mode=quant_mode,
        rhs_freeze_mode=_rhs_freeze_mode(quant_mode),
    )


def build_aqt_conv_general_dilated(aqt_cfg, quant_mode):
    _validate_aqt_cfg(aqt_cfg)
    if aqt_cfg is None:
        return None
    conv_cfg = aqt_config.conv_general_dilated_make(
        2,
        lhs_bits=8,
        rhs_bits=8,
        initialize_calibration=False,
    )
    conv_cfg.dg_quantizer.init_calibration()
    cfg = aqt_config.default_unquantized_config()
    cfg.fwd = conv_cfg
    return partial(
        aqt_flax.AqtConvGeneralDilated,
        cfg,
        prng_name=None,
        rhs_quant_mode=quant_mode,
        rhs_freeze_mode=_rhs_freeze_mode(quant_mode),
    )


def get_norm_fn(norm_name_or_fn=None):
    if norm_name_or_fn is None:
        return DEFAULT_NORM_FN
    if callable(norm_name_or_fn):
        return norm_name_or_fn
    if isinstance(norm_name_or_fn, str):
        try:
            return NORM_FN_REGISTRY[norm_name_or_fn.lower()]
        except KeyError as exc:
            raise ValueError(
                f"Unknown norm_fn: {norm_name_or_fn}. Available: {list(NORM_FN_REGISTRY)}"
            ) from exc
    raise TypeError(
        f"norm_fn must be a string or callable, got {type(norm_name_or_fn)}"
    )


class ConvResBlock(nn.Module):
    filters: int
    kernel_size: tuple[int, int]
    strides: tuple[int, int]
    hidden_N: int = 1
    norm_fn: Callable[[], nn.Module] | None = DEFAULT_NORM_FN
    activation: Callable = nn.relu
    param_dtype: Any = PARAM_DTYPE
    aqt_cfg: Any = None
    quant_mode: Any = None

    @nn.compact
    def __call__(self, x0, training=False):
        aqt_conv = build_aqt_conv_general_dilated(self.aqt_cfg, self.quant_mode)
        x = x0
        for _ in range(self.hidden_N):
            x = nn.Conv(
                self.filters,
                self.kernel_size,
                strides=self.strides,
                padding="SAME",
                dtype=DTYPE,
                param_dtype=self.param_dtype,
                conv_general_dilated_cls=aqt_conv,
            )(x)
            x = self.activation(apply_norm(self.norm_fn, x, training))
        x = nn.Conv(
            self.filters,
            self.kernel_size,
            strides=self.strides,
            padding="SAME",
            dtype=DTYPE,
            param_dtype=self.param_dtype,
            conv_general_dilated_cls=aqt_conv,
        )(x)
        return self.activation(apply_norm(self.norm_fn, x, training) + x0)


class Encoder(nn.Module):
    latent_shape: tuple[int, ...]
    norm_fn: Callable[[], nn.Module] | None = DEFAULT_NORM_FN
    aqt_cfg: object = None
    quant_mode: object = None

    @nn.compact
    def __call__(self, data, training=False):
        aqt_dg = build_aqt_dot_general(self.aqt_cfg, self.quant_mode)
        shape = data.shape
        data = ((data / 255.0) * 2.0 - 1.0).astype(DTYPE)
        flatten = jnp.reshape(data, shape=(shape[0], -1))
        latent_size = prod(self.latent_shape)
        x = nn.Dense(
            1000,
            dtype=DTYPE,
            param_dtype=PARAM_DTYPE,
            dot_general_cls=aqt_dg,
        )(flatten)
        x = nn.relu(apply_norm(self.norm_fn, x, training))
        x = nn.Dense(
            latent_size,
            dtype=DTYPE,
            param_dtype=PARAM_DTYPE,
            dot_general_cls=aqt_dg,
        )(x)
        return jnp.reshape(x, shape=(-1, *self.latent_shape))


class Decoder(nn.Module):
    data_shape: tuple[int, ...]
    norm_fn: Callable[[], nn.Module] | None = DEFAULT_NORM_FN
    aqt_cfg: object = None
    quant_mode: object = None

    @nn.compact
    def __call__(self, latent, training=False):
        aqt_dg = build_aqt_dot_general(self.aqt_cfg, self.quant_mode)
        x = ((latent - 0.5) * 2.0).astype(DTYPE)
        x = nn.Dense(
            1000,
            dtype=DTYPE,
            param_dtype=PARAM_DTYPE,
            dot_general_cls=aqt_dg,
        )(x)
        x = nn.relu(apply_norm(self.norm_fn, x, training))
        x = nn.Dense(
            prod(self.data_shape),
            dtype=DTYPE,
            param_dtype=PARAM_DTYPE,
            dot_general_cls=aqt_dg,
        )(x)
        return jnp.reshape(x, (-1, *self.data_shape)).astype(DTYPE)


class AutoEncoder(nn.Module):
    data_shape: tuple[int, ...]
    latent_shape: tuple[int, ...]
    norm_fn: Callable[[], nn.Module] | None = DEFAULT_NORM_FN
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
    norm_fn: Callable[[], nn.Module] | None = DEFAULT_NORM_FN
    aqt_cfg: object = None
    quant_mode: object = None

    @nn.compact
    def __call__(self, latent, training=False):
        aqt_dg = build_aqt_dot_general(self.aqt_cfg, self.quant_mode)
        x = ((latent - 0.5) * 2.0).astype(DTYPE)
        for _ in range(3):
            x = nn.Dense(
                500,
                dtype=DTYPE,
                param_dtype=PARAM_DTYPE,
                dot_general_cls=aqt_dg,
            )(x)
            x = nn.relu(apply_norm(self.norm_fn, x, training))
        x = nn.Dense(
            prod(self.latent_shape) * self.action_size,
            dtype=DTYPE,
            param_dtype=PARAM_DTYPE,
            dot_general_cls=aqt_dg,
        )(x)
        return jnp.reshape(x, (x.shape[0], self.action_size) + self.latent_shape)


class SokobanEncoder(nn.Module):
    latent_shape: tuple[int, ...]
    norm_fn: Callable[[], nn.Module] | None = DEFAULT_NORM_FN
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


class SokobanDecoder(nn.Module):
    data_shape: tuple[int, ...]
    norm_fn: Callable[[], nn.Module] | None = DEFAULT_NORM_FN
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


class SokobanAutoEncoder(nn.Module):
    data_shape: tuple[int, ...]
    latent_shape: tuple[int, ...]
    norm_fn: Callable[[], nn.Module] | None = DEFAULT_NORM_FN
    aqt_cfg: object = None
    quant_mode: object = None

    def setup(self):
        self.encoder = SokobanEncoder(
            self.latent_shape,
            norm_fn=self.norm_fn,
            aqt_cfg=self.aqt_cfg,
            quant_mode=self.quant_mode,
        )
        self.decoder = SokobanDecoder(
            self.data_shape,
            norm_fn=self.norm_fn,
            aqt_cfg=self.aqt_cfg,
            quant_mode=self.quant_mode,
        )

    def __call__(self, x0, training=False):
        latent = self.encoder(x0, training)
        return latent, self.decoder(latent, training)


class SokobanWorldModelNet(nn.Module):
    latent_shape: tuple[int, ...]
    action_size: int
    norm_fn: Callable[[], nn.Module] | None = DEFAULT_NORM_FN
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


class SokobanEncoderOptimized(nn.Module):
    latent_shape: tuple[int, int, int]
    norm_fn: Callable[[], nn.Module] | None = DEFAULT_NORM_FN
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


class SokobanDecoderOptimized(nn.Module):
    data_shape: tuple[int, int, int]
    norm_fn: Callable[[], nn.Module] | None = DEFAULT_NORM_FN
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


class SokobanAutoEncoderOptimized(nn.Module):
    data_shape: tuple[int, int, int]
    latent_shape: tuple[int, int, int]
    norm_fn: Callable[[], nn.Module] | None = DEFAULT_NORM_FN
    aqt_cfg: object = None
    quant_mode: object = None

    def setup(self):
        self.encoder = SokobanEncoderOptimized(
            self.latent_shape,
            norm_fn=self.norm_fn,
            aqt_cfg=self.aqt_cfg,
            quant_mode=self.quant_mode,
        )
        self.decoder = SokobanDecoderOptimized(
            self.data_shape,
            norm_fn=self.norm_fn,
            aqt_cfg=self.aqt_cfg,
            quant_mode=self.quant_mode,
        )

    def __call__(self, x0, training=False):
        latent = self.encoder(x0, training)
        return latent, self.decoder(latent, training)


@dataclass(frozen=True)
class WorldModelArchitectureSpec:
    data_name: str
    data_shape: tuple[int, ...]
    latent_shape: tuple[int, ...]
    action_size: int
    ae_cls: type[nn.Module]
    wm_cls: type[nn.Module]
    str_parse_img_size: int = 16
    inverse_action_map: tuple[int, ...] | None = None
    action_strings: tuple[str, ...] | None = None
    inverse_error_message: str | None = None


WORLD_MODEL_ARCHITECTURES: Mapping[str, WorldModelArchitectureSpec] = {
    "rubikscube": WorldModelArchitectureSpec(
        data_name="rubikscube",
        data_shape=(32, 64, 3),
        latent_shape=(400,),
        action_size=12,
        ae_cls=AutoEncoder,
        wm_cls=WorldModel,
        str_parse_img_size=16,
        inverse_action_map=(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10),
        action_strings=(
            "L'",
            "L",
            "D'",
            "D",
            "F",
            "F'",
            "R",
            "R'",
            "U",
            "U'",
            "B'",
            "B",
        ),
    ),
    "rubikscube_optimized": WorldModelArchitectureSpec(
        data_name="rubikscube",
        data_shape=(32, 64, 3),
        latent_shape=(240,),
        action_size=12,
        ae_cls=AutoEncoder,
        wm_cls=WorldModel,
        str_parse_img_size=16,
        inverse_action_map=(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10),
        action_strings=(
            "L'",
            "L",
            "D'",
            "D",
            "F",
            "F'",
            "R",
            "R'",
            "U",
            "U'",
            "B'",
            "B",
        ),
    ),
    "sokoban": WorldModelArchitectureSpec(
        data_name="sokoban",
        data_shape=(40, 40, 3),
        latent_shape=(10, 10, 16),
        action_size=4,
        ae_cls=SokobanAutoEncoder,
        wm_cls=SokobanWorldModelNet,
        str_parse_img_size=20,
        inverse_action_map=None,
        action_strings=("←", "→", "↑", "↓"),
        inverse_error_message=(
            "Sokoban is not reversible, so inverse neighbours are not implemented."
        ),
    ),
    "sokoban_optimized": WorldModelArchitectureSpec(
        data_name="sokoban",
        data_shape=(40, 40, 3),
        latent_shape=(10, 10, 2),
        action_size=4,
        ae_cls=SokobanAutoEncoderOptimized,
        wm_cls=SokobanWorldModelNet,
        str_parse_img_size=20,
        inverse_action_map=None,
        action_strings=("←", "→", "↑", "↓"),
        inverse_error_message=(
            "Sokoban is not reversible, so inverse neighbours are not implemented."
        ),
    ),
}


def save_params_with_metadata(path: str, params: Any, metadata: dict[str, Any]):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(
            {
                "params": params,
                "metadata": metadata,
                "timestamp": datetime.now().isoformat(),
            },
            file,
        )


def load_params_with_metadata(path: str):
    try:
        with open(path, "rb") as file:
            loaded = pickle.load(file)
        if isinstance(loaded, dict) and "metadata" in loaded:
            return loaded.get("params"), loaded.get("metadata", {})
        if isinstance(loaded, dict) and loaded:
            return loaded, {}
        warnings.warn(
            f"Unrecognized or invalid old format for parameters in {path}.",
            RuntimeWarning,
            stacklevel=2,
        )
    except (FileNotFoundError, pickle.PickleError, OSError, ValueError) as exc:
        warnings.warn(
            f"Failed to load parameters from {path}. Error: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
    return None, {}


def align_params_dtype(params, target_dtype=PARAM_DTYPE):
    def cast_leaf(value):
        if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.floating):
            return value.astype(target_dtype)
        return value

    return None if params is None else jax.tree_util.tree_map(cast_leaf, params)


def resolve_model_path(filename: str | os.PathLike[str]) -> str:
    candidate = Path(filename).expanduser()
    if candidate.is_absolute() or len(candidate.parts) > 1:
        return str(candidate)
    return str(world_model_checkpoint_path(candidate))


def is_model_downloaded(filename: str | os.PathLike[str]) -> bool:
    return os.path.exists(resolve_model_path(filename))


def download_model(filename: str | os.PathLike[str]) -> str:
    last_error = None
    destination = Path(resolve_model_path(filename))
    remote_names = [destination.name]
    if destination.suffix == ".pkl" and destination.stem.endswith("_v2"):
        remote_names.append(f"{destination.stem[:-3]}{destination.suffix}")
    for name in remote_names:
        try:
            downloaded = huggingface_hub.hf_hub_download(
                repo_id="Tinker/JAxtar_models",
                repo_type="model",
                filename=f"world_models/{name}",
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
            )
            os.close(fd)
            tmp_destination = Path(tmp_name)
            try:
                shutil.copy2(downloaded, tmp_destination)
                os.replace(tmp_destination, destination)
            except BaseException:
                tmp_destination.unlink(missing_ok=True)
                raise
            return str(destination)
        except EntryNotFoundError as exc:
            last_error = exc
    raise FileNotFoundError(f"Checkpoint not found: {filename}") from last_error
