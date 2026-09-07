from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from puxle.world_model._resources import world_model_data_path

from ._puzzles import PuzzleName, create_puzzle, data_name_for_puzzle


@dataclass
class DatasetOptions:
    """Options shared by world-model dataset generators."""

    puzzle: PuzzleName
    puzzle_args: str = "{}"
    data_name: str | None = None
    dataset_size: int = 300_000
    dataset_minibatch_size: int = 30_000
    shuffle_length: int = 30
    image_size: tuple[int, int] = (32, 32)
    seed: int = 0
    output_dir: Path | None = None


DatasetOptionsArg = Annotated[DatasetOptions, tyro.conf.OmitArgPrefixes]


@dataclass
class TrainOptions:
    """Options for world-model training."""

    model: str
    data_name: str | None = None
    data_dir: Path | None = None
    epochs: int = 2_000
    mini_batch_size: int = 1_000
    optimizer: Literal["adam", "sgd"] = "adam"
    learning_rate: float = 1e-4
    save_every: int = 100
    seed: int = 0
    reset: bool = False


TrainOptionsArg = Annotated[TrainOptions, tyro.conf.OmitArgPrefixes]


def _dataset_context(options: DatasetOptions):
    puzzle = create_puzzle(options.puzzle, options.puzzle_args)
    data_name = options.data_name or data_name_for_puzzle(puzzle)
    root = options.output_dir or world_model_data_path(data_name)
    root.mkdir(parents=True, exist_ok=True)
    return puzzle, root


def _render_states(states, image_size: tuple[int, int]) -> np.ndarray:
    import cv2

    return np.stack(
        [
            cv2.resize(
                np.asarray(state.img()), image_size, interpolation=cv2.INTER_AREA
            )
            for state in states
        ]
    )


def make_transition_dataset(options: DatasetOptionsArg) -> int:
    """Generate transition images and actions."""
    from puxle.world_model import get_world_model_dataset_builder

    puzzle, root = _dataset_context(options)
    output = root / "transition"
    output.mkdir(parents=True, exist_ok=True)
    shuffle_parallel = math.ceil(
        options.dataset_minibatch_size / options.shuffle_length
    )
    build_dataset = get_world_model_dataset_builder(
        puzzle,
        options.dataset_size,
        shuffle_parallel,
        options.shuffle_length,
        options.dataset_minibatch_size,
    )
    states, actions, next_states = build_dataset(jax.random.PRNGKey(options.seed))
    np.save(output / "actions.npy", np.asarray(actions))
    np.save(output / "images.npy", _render_states(states, options.image_size))
    np.save(output / "next_images.npy", _render_states(next_states, options.image_size))
    print(output)
    return 0


def make_sample_data(options: DatasetOptionsArg) -> int:
    """Generate initial and target state images."""
    from puxle.world_model import get_sample_data_builder

    puzzle, output = _dataset_context(options)
    build_dataset = get_sample_data_builder(
        puzzle, options.dataset_size, options.dataset_minibatch_size
    )
    targets, initials = build_dataset(jax.random.PRNGKey(options.seed))
    np.save(output / "targets.npy", _render_states(targets, options.image_size))
    np.save(output / "inits.npy", _render_states(initials, options.image_size))
    print(output)
    return 0


def make_eval_trajectory(options: DatasetOptionsArg) -> int:
    """Generate a trajectory used during evaluation."""
    from puxle.world_model import create_eval_trajectory

    puzzle, root = _dataset_context(options)
    output = root / "transition"
    output.mkdir(parents=True, exist_ok=True)
    states, actions = create_eval_trajectory(
        puzzle, options.dataset_size, jax.random.PRNGKey(options.seed)
    )
    np.save(output / "eval_actions.npy", np.asarray(actions))
    np.save(output / "eval_traj_images.npy", _render_states(states, options.image_size))
    print(output)
    return 0


def _data_name_for_model(name: str) -> str:
    normalized = name.lower()
    if "rubikscube" in normalized:
        return "rubikscube"
    if "sokoban" in normalized:
        return "sokoban"
    raise ValueError("--data-name is required for an unrecognized model name")


def train(options: TrainOptionsArg) -> int:
    """Train a registered world model."""
    import optax

    from puxle.world_model import (
        apply_with_conditional_batch_stats,
        trained_world_model_registry,
        world_model_eval_builder,
        world_model_train_builder,
    )
    from puxle.world_model.trained_model_registry import (
        _create_world_model_for_training,
    )

    factory = getattr(trained_world_model_registry, options.model, None)
    if factory is None:
        available = ", ".join(sorted(vars(trained_world_model_registry)))
        raise ValueError(f"Unknown model '{options.model}'. Available: {available}")

    data_name = options.data_name or _data_name_for_model(options.model)
    data_dir = options.data_dir or world_model_data_path(data_name) / "transition"
    datas = jnp.asarray(np.load(data_dir / "images.npy"))
    next_datas = jnp.asarray(np.load(data_dir / "next_images.npy"))
    actions = jnp.asarray(np.load(data_dir / "actions.npy"))
    eval_states = jnp.asarray(np.load(data_dir / "eval_traj_images.npy"))
    eval_actions = jnp.asarray(np.load(data_dir / "eval_actions.npy"))

    world_model = _create_world_model_for_training(
        options.model,
        init_params=options.reset,
    )
    model = world_model.model

    def train_info_fn(params, data, next_data, action, training):
        if training:
            return apply_with_conditional_batch_stats(
                model.apply,
                params,
                data,
                next_data,
                action,
                training=True,
                method=model.train_info,
            )
        return model.apply(
            params,
            data,
            next_data,
            action,
            training=False,
            method=model.train_info,
        )

    optimizer = (
        optax.adam(options.learning_rate)
        if options.optimizer == "adam"
        else optax.sgd(options.learning_rate)
    )
    train_fn = world_model_train_builder(
        options.mini_batch_size, train_info_fn, optimizer
    )
    eval_fn = world_model_eval_builder(train_info_fn, options.mini_batch_size)
    params = world_model.params
    opt_state = optimizer.init(params)
    key = jax.random.PRNGKey(options.seed)

    for epoch in range(options.epochs):
        key, subkey = jax.random.split(key)
        params, opt_state, loss, ae_loss, wm_loss, accuracy = train_fn(
            subkey, (datas, next_datas, actions), params, opt_state, epoch
        )
        if epoch % 10 == 0 or epoch + 1 == options.epochs:
            eval_accuracy = eval_fn(params, (eval_states, eval_actions))
            print(
                f"epoch={epoch} loss={float(loss):.6f} ae={float(ae_loss):.6f} "
                f"wm={float(wm_loss):.6f} accuracy={float(accuracy):.6f} "
                f"eval={float(eval_accuracy):.6f}"
            )
        if options.save_every > 0 and (epoch + 1) % options.save_every == 0:
            world_model.params = params
            world_model.save_model()

    world_model.params = params
    world_model.save_model()
    print(world_model.path)
    return 0
