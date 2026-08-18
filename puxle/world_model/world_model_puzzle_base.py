import pickle

import jax
import jax.numpy as jnp
import numpy as np
from aqt.jax.v2.flax import aqt_flax
from flax import linen as nn
from xtructure import FieldDescriptor, Xtructurable, xtructure_dataclass
from xtructure import numpy as xnp

from puxle.core.puzzle_base import Puzzle
from puxle.utils import IMG_SIZE

from ._utils import (
    DEFAULT_NORM_FN,
    DTYPE,
    PARAM_DTYPE,
    align_params_dtype,
    apply_norm,
    build_aqt_dot_general,
    download_model,
    get_norm_fn,
    img_to_colored_str,
    is_model_downloaded,
    load_params_with_metadata,
    resolve_model_path,
    round_through_gradient,
    save_params_with_metadata,
)
from .util import download_world_model_dataset, is_world_model_dataset_downloaded


class Encoder(nn.Module):
    latent_shape: tuple[int, ...]
    norm_fn: nn.Module = DEFAULT_NORM_FN
    aqt_cfg: object = None
    quant_mode: object = None

    @nn.compact
    def __call__(self, data, training=False):
        aqt_dg = build_aqt_dot_general(self.aqt_cfg, self.quant_mode)
        shape = data.shape
        data = ((data / 255.0) * 2.0 - 1.0).astype(DTYPE)
        flatten = jnp.reshape(data, shape=(shape[0], -1))
        latent_size = np.prod(self.latent_shape)
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
    norm_fn: nn.Module = DEFAULT_NORM_FN
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
            np.prod(self.data_shape),
            dtype=DTYPE,
            param_dtype=PARAM_DTYPE,
            dot_general_cls=aqt_dg,
        )(x)
        return jnp.reshape(x, (-1, *self.data_shape)).astype(DTYPE)


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
            np.prod(self.latent_shape) * self.action_size,
            dtype=DTYPE,
            param_dtype=PARAM_DTYPE,
            dot_general_cls=aqt_dg,
        )(x)
        return jnp.reshape(x, (x.shape[0], self.action_size) + self.latent_shape)


class WorldModelPuzzleBase(Puzzle):
    inits: jnp.ndarray
    targets: jnp.ndarray
    init_state_size: int = 0
    str_parse_img_size: int = 16
    str_parse_img: bool = True

    def define_state_class(self) -> type[Xtructurable]:
        str_parser = self.get_string_parser()
        latent_shape = self.latent_shape

        @xtructure_dataclass
        class State:
            latent: FieldDescriptor.packed_tensor(shape=latent_shape, packed_bits=1)

            def __str__(self, **kwargs):
                return str_parser(self, **kwargs)

        return State

    def __init__(
        self,
        data_path,
        data_shape,
        latent_shape,
        action_size,
        AE=AutoEncoder,
        WM=WorldModel,
        init_params: bool = False,
        path: str = None,
        **kwargs,
    ):
        self.data_path = data_path
        self.data_shape = data_shape
        self.latent_shape = latent_shape
        self.latent_size = int(np.prod(latent_shape))
        self.pad_size = int(np.ceil(self.latent_size / 8) * 8 - self.latent_size)
        self.action_size = action_size
        self.path = path
        self.metadata = {}
        self.aqt_cfg = kwargs.pop("aqt_cfg", None)
        kwargs["norm_fn"] = get_norm_fn(kwargs.get("norm_fn", "batch"))

        class total_model(nn.Module):
            autoencoder: AutoEncoder
            world_model: WorldModel

            @nn.compact
            def __call__(self, x, training=False):
                latent, decoded = self.autoencoder(x, training)
                return self.world_model(latent, training), decoded

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
                next_logits = jnp.take_along_axis(next_logits_preds, action, axis=1)
                next_logits_pred = next_logits.squeeze(axis=1)
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
            model_kwargs = kwargs | {
                "aqt_cfg": self.aqt_cfg,
                "quant_mode": quant_mode,
            }
            return total_model(
                autoencoder=AE(
                    data_shape=data_shape,
                    latent_shape=latent_shape,
                    **model_kwargs,
                ),
                world_model=WM(
                    latent_shape=latent_shape,
                    action_size=action_size,
                    **model_kwargs,
                ),
            )

        self._build_model = build_model
        self.model = build_model(aqt_flax.QuantMode.TRAIN)
        self.params = (
            self.get_new_params() if path is None or init_params else self.load_model()
        )
        if self.aqt_cfg is not None:
            self.model, self.params = self._convert_to_serving(self.params)
        super().__init__(**kwargs)

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

    def get_new_params(self):
        dummy_data = jnp.zeros((1, *self.data_shape))
        return self.model.init(
            jax.random.PRNGKey(np.random.randint(0, 2**32 - 1)), dummy_data
        )

    def load_model(self):
        try:
            if not is_model_downloaded(self.path):
                download_model(self.path)
            params, metadata = load_params_with_metadata(resolve_model_path(self.path))
            if params is None:
                print(
                    f"Warning: Loaded parameters from {self.path} are invalid or old. "
                    "Initializing new parameters."
                )
                return self.get_new_params()
            self.metadata = metadata
            params = align_params_dtype(params)
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

    def save_model(self, metadata: dict = None):
        metadata = {} if metadata is None else metadata
        metadata.update(
            data_path=self.data_path,
            data_shape=self.data_shape,
            latent_shape=self.latent_shape,
            action_size=self.action_size,
        )
        save_params_with_metadata(self.path, self.params, metadata)

    def data_init(self):
        if not is_world_model_dataset_downloaded():
            download_world_model_dataset()
        self.inits = jax.device_put(jnp.load(f"{self.data_path}/inits.npy"))
        self.targets = jax.device_put(jnp.load(f"{self.data_path}/targets.npy"))
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
            latent = np.reshape(np.array(state.latent), (-1,)).tobytes().hex()
            latent_len = len(latent)
            if latent_len >= 25:
                latent = f"{latent[:10]}...{latent[-10:]}"
            return output + f"latent: 0x{latent}[{latent_len // 2} bytes]"

        return parser

    def get_img_parser(self) -> callable:
        import cv2

        def decode(latent):
            data = self.model.apply(
                self.params,
                jnp.expand_dims(latent, axis=0),
                training=False,
                method=self.model.decode,
            ).squeeze(0)
            return np.clip(np.array(data * 255.0) / 2.0 + 127.5, 0, 255).astype(
                np.uint8
            )

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
        idx = jax.random.randint(key, (), 0, self.init_state_size)
        return self.targets[idx][None, ...], self.inits[idx][None, ...]

    def get_solve_config(self, key=None, data=None):
        latent = self.model.apply(
            self.params, data[0], training=False, method=self.model.encode
        ).squeeze(0)
        return self.SolveConfig(
            InstanceContext=self.InstanceContext(),
            GoalSpec=self.State.from_unpacked(
                latent=jnp.round(latent).astype(jnp.bool_)
            ),
        )

    def get_initial_state(self, solve_config, key=None, data=None):
        latent = self.model.apply(
            self.params, data[1], training=False, method=self.model.encode
        ).squeeze(0)
        return self.State.from_unpacked(latent=jnp.round(latent).astype(jnp.bool_))

    def batched_get_actions(
        self, solve_configs, states, actions, filleds=True, multi_solve_config=False
    ):
        neighbours, costs = self.batched_get_neighbours(
            solve_configs, states, filleds, multi_solve_config
        )
        indices = jnp.expand_dims(actions, axis=0)
        return (
            xnp.squeeze(xnp.take_along_axis(neighbours, indices, axis=0), axis=0),
            jnp.squeeze(jnp.take_along_axis(costs, indices, axis=0), axis=0),
        )

    def get_actions(self, solve_config, state, actions, filled=True):
        neighbours, costs = self.get_neighbours(solve_config, state, filled)
        return xnp.take(neighbours, actions, axis=0), jnp.take(costs, actions, axis=0)

    def batched_get_neighbours(
        self, solve_configs, states, filleds=True, multi_solve_config=False
    ):
        next_latent = self.model.apply(
            self.params,
            states.latent_unpacked,
            training=False,
            method=self.model.transition,
        )
        next_latent = jnp.swapaxes(jnp.round(next_latent).astype(jnp.bool_), 0, 1)
        next_states = jax.vmap(jax.vmap(self.State.from_unpacked))(latent=next_latent)
        costs = jnp.where(
            filleds,
            jnp.ones((self.action_size, states.latent.shape[0]), dtype=jnp.float16),
            jnp.inf,
        )
        return next_states, costs

    def get_neighbours(self, solve_config, state, filled=True):
        next_states, costs = self.batched_get_neighbours(
            solve_config, state[jnp.newaxis, ...], filled
        )
        return next_states[:, 0], costs[:, 0]

    def batched_is_solved(self, solve_configs, states, multi_solve_config=False):
        in_axes = (0, 0) if multi_solve_config else (None, 0)
        return jax.vmap(self.is_solved, in_axes=in_axes)(solve_configs, states)

    def get_inverse_neighbours(self, solve_config, state, filled=True):
        next_states, costs = self.batched_get_inverse_neighbours(
            solve_config, state[jnp.newaxis, ...], filled
        )
        return next_states[:, 0], costs[:, 0]

    def batched_get_inverse_neighbours(
        self, solve_configs, states, filleds=True, multi_solve_config=False
    ):
        return self.batched_get_neighbours(
            solve_configs, states, filleds, multi_solve_config
        )
