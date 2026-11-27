"""Learned checkpoint-backed world-model puzzle adapter for PuXle."""

import os
import pickle
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated

import chex
import jax
import jax.numpy as jnp
import numpy as np
from aqt.jax.v2.flax import aqt_flax
from flax import linen as nn
from xtructure import FieldDescriptor, Xtructurable
from xtructure import numpy as xnp

from puxle.core.puzzle_base import Puzzle
from puxle.render import img_to_colored_str
from puxle.utils import IMG_SIZE
from puxle.world_model import (
    download_world_model_dataset,
    is_world_model_dataset_downloaded,
    round_through_gradient,
    world_model_data_path,
)

from ._nn import (
    WORLD_MODEL_ARCHITECTURES,
    AutoEncoder,
    WorldModel,
    align_params_dtype,
    download_model,
    get_norm_fn,
    is_model_downloaded,
    load_params_with_metadata,
    resolve_model_path,
    save_params_with_metadata,
)
from .presets import WorldModelPresets


class WorldModelPuzzle(Puzzle):
    """Unified ``Puzzle`` adapter for learned autoencoder + transition checkpoints."""

    inits: jnp.ndarray
    targets: jnp.ndarray
    init_state_size: int = 0
    renderer_host_inputs: bool = False

    @property
    def fixed_target(self) -> bool:
        return False

    def define_state_class(self) -> type[Xtructurable]:
        latent_shape = self.latent_shape

        class State(Xtructurable):
            latent: Annotated[
                chex.Array,
                FieldDescriptor.packed_tensor(shape=latent_shape, packed_bits=1),
            ]

        return State

    def __init__(
        self,
        *,
        architecture: str | None = None,
        data_name: str | None = None,
        data_path: str | os.PathLike[str] | None = None,
        data_shape: tuple[int, ...] | None = None,
        latent_shape: tuple[int, ...] | None = None,
        action_size: int | None = None,
        AE: Callable[..., nn.Module] | None = None,
        WM: Callable[..., nn.Module] | None = None,
        init_params: bool = False,
        path: str | os.PathLike[str] | None = None,
        aqt_cfg: str | None = None,
        norm_fn: str | Callable[[], nn.Module] | None = "batch",
        reverse_data: bool = False,
        inverse_action_map: Sequence[int] | None = None,
        action_strings: Sequence[str] | None = None,
        str_parse_img_size: int | None = None,
        str_parse_img: bool = True,
    ):
        spec = None
        if architecture is not None:
            if architecture not in WORLD_MODEL_ARCHITECTURES:
                available = ", ".join(sorted(WORLD_MODEL_ARCHITECTURES))
                raise ValueError(
                    f"Unknown world-model architecture '{architecture}'. "
                    f"Available: {available}"
                )
            spec = WORLD_MODEL_ARCHITECTURES[architecture]

        resolved_data_name = (
            data_name
            if data_name is not None
            else (spec.data_name if spec is not None else None)
        )
        if data_path is not None:
            resolved_data_path = str(data_path)
        elif resolved_data_name is not None:
            resolved_data_path = str(world_model_data_path(resolved_data_name))
        else:
            raise ValueError(
                "WorldModelPuzzle requires either `architecture`, `data_name`, or `data_path`."
            )

        resolved_data_shape = (
            data_shape
            if data_shape is not None
            else (spec.data_shape if spec is not None else None)
        )
        resolved_latent_shape = (
            latent_shape
            if latent_shape is not None
            else (spec.latent_shape if spec is not None else None)
        )
        resolved_action_size = (
            action_size
            if action_size is not None
            else (spec.action_size if spec is not None else None)
        )
        if (
            resolved_data_shape is None
            or resolved_latent_shape is None
            or resolved_action_size is None
        ):
            raise ValueError(
                "WorldModelPuzzle requires `data_shape`, `latent_shape`, and `action_size` "
                "(or a known `architecture`)."
            )

        ae_cls = (
            AE if AE is not None else (spec.ae_cls if spec is not None else AutoEncoder)
        )
        wm_cls = (
            WM if WM is not None else (spec.wm_cls if spec is not None else WorldModel)
        )
        resolved_inverse_map = (
            inverse_action_map
            if inverse_action_map is not None
            else (spec.inverse_action_map if spec is not None else None)
        )
        resolved_action_strings = (
            action_strings
            if action_strings is not None
            else (spec.action_strings if spec is not None else None)
        )
        resolved_str_parse_img_size = (
            str_parse_img_size
            if str_parse_img_size is not None
            else (spec.str_parse_img_size if spec is not None else 16)
        )

        self.architecture = architecture
        self.data_name = resolved_data_name
        self.data_path = resolved_data_path
        self._uses_custom_data_path = data_path is not None
        self.data_shape = tuple(resolved_data_shape)
        self.latent_shape = tuple(resolved_latent_shape)
        self.latent_size = int(np.prod(self.latent_shape))
        self.action_size = int(resolved_action_size)
        self.path = None if path is None else resolve_model_path(path)
        self.metadata: dict[str, object] = {}
        self.aqt_cfg = aqt_cfg
        self.reverse_data = reverse_data
        self.str_parse_img = str_parse_img
        self.str_parse_img_size = resolved_str_parse_img_size
        self._inverse_action_map_tuple = (
            tuple(int(i) for i in resolved_inverse_map)
            if resolved_inverse_map is not None
            else None
        )
        self._action_strings = (
            tuple(resolved_action_strings)
            if resolved_action_strings is not None
            else None
        )
        self._inverse_error_message = (
            spec.inverse_error_message if spec is not None else None
        )

        resolved_norm_fn = get_norm_fn(norm_fn)

        class total_model(nn.Module):
            autoencoder: nn.Module
            world_model: nn.Module

            @nn.compact
            def __call__(self, x, training=False):
                latent = round_through_gradient(self.encode(x, training))
                return self.world_model(latent, training), self.decode(latent, training)

            def decode(self, latent, training=False):
                return self.autoencoder.decoder(latent, training)

            def encode(self, data, training=False):
                return nn.sigmoid(self.autoencoder.encoder(data, training))

            def transition(self, latent, training=False):
                return nn.sigmoid(self.world_model(latent, training))

            def train_info(self, data, next_data, action, training=True):
                logits = self.autoencoder.encoder(data, training)
                rounded_latents = round_through_gradient(nn.sigmoid(logits))
                decoded = self.decode(rounded_latents, training)

                next_logits = self.autoencoder.encoder(next_data, training)
                rounded_next_latents = round_through_gradient(nn.sigmoid(next_logits))
                next_decoded = self.decode(rounded_next_latents, training)

                next_logits_preds = self.world_model(rounded_latents, training)
                action = jnp.reshape(
                    action, (-1,) + (1,) * (next_logits_preds.ndim - 1)
                )
                next_logits_pred = jnp.take_along_axis(
                    next_logits_preds, action, axis=1
                ).squeeze(axis=1)
                rounded_next_latents_pred = round_through_gradient(
                    nn.sigmoid(next_logits_pred)
                )
                return (
                    logits,
                    rounded_latents,
                    decoded,
                    next_logits,
                    rounded_next_latents,
                    next_decoded,
                    next_logits_pred,
                    rounded_next_latents_pred,
                )

        def build_model(quant_mode):
            model_kwargs = {
                "norm_fn": resolved_norm_fn,
                "aqt_cfg": self.aqt_cfg,
                "quant_mode": quant_mode,
            }
            return total_model(
                autoencoder=ae_cls(
                    data_shape=self.data_shape,
                    latent_shape=self.latent_shape,
                    **model_kwargs,
                ),
                world_model=wm_cls(
                    latent_shape=self.latent_shape,
                    action_size=self.action_size,
                    **model_kwargs,
                ),
            )

        self._build_model = build_model
        self.model = build_model(aqt_flax.QuantMode.TRAIN)
        self.params = (
            self.get_new_params()
            if self.path is None or init_params
            else self.load_model()
        )
        if self.aqt_cfg is not None:
            self.model, self.params = self._convert_to_serving(self.params)
        super().__init__()

    @classmethod
    def list_sources(cls) -> dict[str, list[str]]:
        """List built-in world-model architectures and curated presets."""
        return {
            "architectures": sorted(WORLD_MODEL_ARCHITECTURES),
            "presets": sorted(WorldModelPresets.all()),
        }

    @property
    def inverse_action_map(self) -> jnp.ndarray | None:
        if self._inverse_action_map_tuple is None:
            return None
        return jnp.asarray(self._inverse_action_map_tuple, dtype=jnp.int32)

    def action_to_string(self, action: int) -> str:
        if self._action_strings is None:
            return super().action_to_string(action)
        if action < 0 or action >= len(self._action_strings):
            raise ValueError(
                f"Action {action} is out of bounds for action space size {len(self._action_strings)}."
            )
        return self._action_strings[action]

    def _convert_to_serving(self, params):
        dummy_data = jnp.zeros((1, *self.data_shape))
        convert_model = self._build_model(aqt_flax.QuantMode.CONVERT)
        _, aqt_variables = convert_model.apply(
            params,
            dummy_data,
            mutable=["aqt"],
        )
        variables = dict(params)
        variables.update(aqt_variables)
        serving_model = self._build_model(aqt_flax.QuantMode.SERVE)
        serving_model.apply(variables, dummy_data)
        return serving_model, variables

    def get_new_params(self, seed: int | None = None):
        if seed is None:
            seed = int(np.random.randint(0, 2**32 - 1))
        return self.model.init(
            jax.random.PRNGKey(seed), jnp.zeros((1, *self.data_shape))
        )

    def load_model(self):
        if self.path is None:
            raise ValueError("A model path is required to load parameters.")
        try:
            if not is_model_downloaded(self.path):
                download_model(self.path)
            params, metadata = load_params_with_metadata(resolve_model_path(self.path))
            if params is None:
                warnings.warn(
                    f"Loaded parameters from {self.path} are invalid or old. "
                    "Initializing new parameters.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return self.get_new_params()
            self.metadata = metadata
            params = align_params_dtype(params)
            if params is None:
                raise ValueError(
                    f"Failed to align parameter dtypes loaded from {self.path}."
                )
            self.model.apply(params, jnp.zeros((1, *self.data_shape)))
            return params
        except (
            FileNotFoundError,
            pickle.PickleError,
            ValueError,
            RuntimeError,
            OSError,
        ) as exc:
            raise ValueError(f"Error loading WorldModelPuzzle model: {exc}") from exc

    def save_model(self, metadata: dict | None = None):
        if self.path is None:
            raise ValueError("A model path is required to save parameters.")
        metadata = {} if metadata is None else dict(metadata)
        metadata.update(
            data_path=self.data_path,
            data_shape=self.data_shape,
            latent_shape=self.latent_shape,
            action_size=self.action_size,
        )
        save_params_with_metadata(self.path, self.params, metadata)

    def data_init(self):
        data_dir = Path(self.data_path)
        inits_path = data_dir / "inits.npy"
        targets_path = data_dir / "targets.npy"
        if not (inits_path.exists() and targets_path.exists()):
            if self._uses_custom_data_path:
                raise FileNotFoundError(
                    f"Missing world-model dataset files in custom data_path: {data_dir}"
                )
            if not is_world_model_dataset_downloaded():
                download_world_model_dataset()
        inits = jax.device_put(jnp.load(inits_path))
        targets = jax.device_put(jnp.load(targets_path))
        if self.reverse_data:
            inits, targets = targets, inits
        self.inits = inits
        self.targets = targets
        self.init_state_size = self.inits.shape[0]

    def get_string_parser(self):
        def parser(state, solve_config=None, **kwargs):
            output = ""
            if self.str_parse_img:
                state_img = state.img(
                    show_target_state_img=False,
                    resize_img=True,
                    target_height=self.str_parse_img_size,
                )
                output = img_to_colored_str(state_img) + "\n"
            latent = (
                np.asarray(jax.device_get(state.latent)).reshape(-1).tobytes().hex()
            )
            latent_len = len(latent)
            if latent_len >= 25:
                latent = f"{latent[:10]}...{latent[-10:]}"
            return output + f"latent: 0x{latent}[{latent_len // 2} bytes]"

        return parser

    def get_img_parser(self) -> Callable[..., np.ndarray]:
        import cv2

        def decode(latent):
            data = self.model.apply(
                self.params,
                jnp.expand_dims(latent, axis=0),
                training=False,
                method=self.model.decode,
            )
            if not isinstance(data, (jax.Array, np.ndarray)):
                raise TypeError("The world model decoder must return an array.")
            host_data = np.asarray(jax.device_get(data.squeeze(0)), dtype=np.float32)
            return np.clip(host_data * 255.0 / 2.0 + 127.5, 0, 255).astype(np.uint8)

        def img_parser(
            state,
            solve_config=None,
            show_target_state_img=True,
            resize_img=True,
            target_height=IMG_SIZE[1],
            **kwargs,
        ):
            data = decode(state.latent_unpacked)
            height, width = data.shape[:2]
            if resize_img:
                width, height = int(target_height * width / height), target_height
                img = cv2.resize(data, (width, height), interpolation=cv2.INTER_AREA)
            else:
                img = data
            if solve_config is not None and show_target_state_img:
                target = decode(solve_config.GoalSpec.latent_unpacked)
                if resize_img:
                    target = cv2.resize(
                        target, (width, height), interpolation=cv2.INTER_AREA
                    )
                line = np.ones((10, width, 3), dtype=np.uint8) * 255
                img = np.concatenate([img, line, target], axis=0)
            return img

        return img_parser

    def get_data(self, key=None):
        if key is None:
            raise ValueError("A random key is required to sample world model data.")
        idx = jax.random.randint(key, (), 0, self.init_state_size)
        return self.targets[idx][None, ...], self.inits[idx][None, ...]

    def get_solve_config(self, key=None, data=None):
        if data is None:
            raise ValueError(
                "World model data is required to build a solve configuration."
            )
        latent = self.model.apply(
            self.params, data[0], training=False, method=self.model.encode
        )
        if not isinstance(latent, (jax.Array, np.ndarray)):
            raise TypeError("The world model encoder must return an array.")
        latent = latent.squeeze(0)
        return self.SolveConfig(
            InstanceContext=self.InstanceContext(),
            GoalSpec=self.State.from_unpacked(
                latent=jnp.round(latent).astype(jnp.bool_)
            ),
        )

    def get_initial_state(self, solve_config, key=None, data=None):
        if data is None:
            raise ValueError("World model data is required to build an initial state.")
        latent = self.model.apply(
            self.params, data[1], training=False, method=self.model.encode
        )
        if not isinstance(latent, (jax.Array, np.ndarray)):
            raise TypeError("The world model encoder must return an array.")
        latent = latent.squeeze(0)
        return self.State.from_unpacked(latent=jnp.round(latent).astype(jnp.bool_))

    def batched_get_actions(
        self,
        solve_configs: Puzzle.SolveConfig,
        states: Puzzle.State,
        actions: jax.Array,
        filleds: bool | jax.Array = True,
        multi_solve_config: bool = False,
    ) -> tuple[Puzzle.State, jax.Array]:
        neighbours, costs = self.batched_get_neighbours(
            solve_configs, states, filleds, multi_solve_config
        )
        indices = jnp.expand_dims(actions, axis=0)
        next_states = xnp.squeeze(
            xnp.take_along_axis(neighbours, indices, axis=0), axis=0
        )
        if not isinstance(next_states, Xtructurable):
            raise TypeError("Expected Xtructurable next states.")
        return (
            next_states,
            jnp.squeeze(jnp.take_along_axis(costs, indices, axis=0), axis=0),
        )

    def get_actions(self, solve_config, state, action, filled=True):
        neighbours, costs = self.get_neighbours(solve_config, state, filled)
        return xnp.take(neighbours, action, axis=0), jnp.take(costs, action, axis=0)

    def batched_get_neighbours(
        self,
        solve_configs: Puzzle.SolveConfig,
        states: Puzzle.State,
        filleds: bool | jax.Array = True,
        multi_solve_config: bool = False,
    ) -> tuple[Puzzle.State, jax.Array]:
        next_latent = self.model.apply(
            self.params,
            states.latent_unpacked,
            training=False,
            method=self.model.transition,
        )
        if not isinstance(next_latent, (jax.Array, np.ndarray)):
            raise TypeError("The world model transition must return an array.")
        next_latent = jnp.swapaxes(jnp.round(next_latent).astype(jnp.bool_), 0, 1)
        candidate_states = jax.vmap(jax.vmap(self.State.from_unpacked))(
            latent=next_latent
        )
        batch_size = states.shape[0]
        filled_batch = jnp.broadcast_to(
            jnp.asarray(filleds, dtype=jnp.bool_),
            (batch_size,),
        )
        next_states = jax.vmap(
            lambda f, cand, cur: xnp.where(f, cand, cur),
            in_axes=(0, 1, 0),
            out_axes=1,
        )(filled_batch, candidate_states, states)
        costs = jnp.where(
            filled_batch[jnp.newaxis, :],
            jnp.ones((self.action_size, batch_size), dtype=jnp.float32),
            jnp.asarray(jnp.inf, dtype=jnp.float32),
        )
        return next_states, costs

    def get_neighbours(self, solve_config, state, filled=True):
        next_states, costs = self.batched_get_neighbours(
            solve_config, state[jnp.newaxis, ...], filled
        )
        return next_states[:, 0], costs[:, 0]

    def get_inverse_neighbours(self, solve_config, state, filled=True):
        next_states, costs = self.batched_get_inverse_neighbours(
            solve_config, state[jnp.newaxis, ...], filled
        )
        return next_states[:, 0], costs[:, 0]

    def batched_get_inverse_neighbours(
        self,
        solve_configs: Puzzle.SolveConfig,
        states: Puzzle.State,
        filleds: bool | jax.Array = True,
        multi_solve_config: bool = False,
    ) -> tuple[Puzzle.State, jax.Array]:
        perm = self.inverse_action_permutation
        if perm is None:
            raise NotImplementedError(
                self._inverse_error_message
                or f"{type(self).__name__} does not define an `inverse_action_map`."
            )
        neighbours, costs = self.batched_get_neighbours(
            solve_configs, states, filleds, multi_solve_config
        )
        return (
            jax.tree_util.tree_map(lambda x: x[perm], neighbours),
            costs[perm],
        )


__all__ = ["WorldModelPuzzle"]
