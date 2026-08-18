from __future__ import annotations

import os
import pickle
import shutil
from datetime import datetime
from functools import partial
from typing import Any, Callable

import flax.linen as nn
import huggingface_hub
import jax
import jax.numpy as jnp
import numpy as np
from aqt.jax.v2 import config as aqt_config
from aqt.jax.v2.flax import aqt_flax
from huggingface_hub.errors import EntryNotFoundError

from ._resources import world_model_checkpoint_path

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


def build_aqt_dot_general(aqt_cfg, quant_mode):
    _validate_aqt_cfg(aqt_cfg)
    if aqt_cfg is None:
        return None
    return partial(
        aqt_flax.AqtDotGeneral,
        aqt_config.fully_quantized(fwd_bits=8, bwd_bits=8),
        prng_name=None,
        rhs_quant_mode=quant_mode,
        rhs_freeze_mode=(
            aqt_flax.FreezerMode.CALIBRATION_AND_VALUE
            if quant_mode in (aqt_flax.QuantMode.CONVERT, aqt_flax.QuantMode.SERVE)
            else aqt_flax.FreezerMode.NONE
        ),
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
    if conv_cfg.lhs:
        conv_cfg.dg_quantizer.lhs.init_calibration()
    if conv_cfg.rhs:
        conv_cfg.dg_quantizer.rhs.init_calibration()
    cfg = aqt_config.default_unquantized_config()
    cfg.fwd = conv_cfg
    return partial(
        aqt_flax.AqtConvGeneralDilated,
        cfg,
        prng_name=None,
        rhs_quant_mode=quant_mode,
        rhs_freeze_mode=(
            aqt_flax.FreezerMode.CALIBRATION_AND_VALUE
            if quant_mode in (aqt_flax.QuantMode.CONVERT, aqt_flax.QuantMode.SERVE)
            else aqt_flax.FreezerMode.NONE
        ),
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
    norm_fn: nn.Module = DEFAULT_NORM_FN
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


def round_through_gradient(x):
    rounded = jnp.where(x > 0.5, 1.0, 0.0).astype(jnp.float32)
    return x + jax.lax.stop_gradient(rounded - x)


def apply_with_conditional_batch_stats(
    apply_fn,
    params,
    *apply_args,
    training: bool,
    n_devices: int = 1,
    collection: str = "batch_stats",
    axis_name: str = "devices",
    **apply_kwargs,
):
    if training and collection in params:
        outputs, updates = apply_fn(
            params, *apply_args, training=True, mutable=[collection], **apply_kwargs
        )
        if n_devices > 1:
            updates = jax.lax.pmean(updates, axis_name=axis_name)
        return outputs, updates
    return apply_fn(params, *apply_args, training=training, **apply_kwargs), {}


def build_new_params_from_updates(params, updates, collection="batch_stats"):
    new_params = {"params": params["params"]}
    if collection in updates:
        new_params[collection] = updates[collection]
    return new_params


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
        print(f"Warning: Unrecognized or invalid old format for parameters in {path}.")
    except (FileNotFoundError, pickle.PickleError, OSError, ValueError) as exc:
        print(f"Warning: Failed to load parameters from {path}. Error: {exc}")
    return None, {}


def align_params_dtype(params, target_dtype=PARAM_DTYPE):
    def cast_leaf(value):
        if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.floating):
            return value.astype(target_dtype)
        return value

    return None if params is None else jax.tree_util.tree_map(cast_leaf, params)


def resolve_model_path(filename: str):
    return str(world_model_checkpoint_path(filename))


def is_model_downloaded(filename: str):
    return os.path.exists(resolve_model_path(filename))


def download_model(filename: str):
    last_error = None
    destination = world_model_checkpoint_path(filename)
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
            shutil.copy2(downloaded, destination)
            return str(destination)
        except EntryNotFoundError as exc:
            last_error = exc
    raise FileNotFoundError(f"Checkpoint not found: {filename}") from last_error


def img_to_colored_str(img: np.ndarray) -> str:
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    return "\n".join(
        "".join(f"\x1b[38;2;{int(r)};{int(g)};{int(b)}m██\x1b[0m" for r, g, b in row)
        for row in img
    )
