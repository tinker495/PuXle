from __future__ import annotations

import argparse
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from puxle.world_model._resources import world_model_data_path

from ._puzzles import create_puzzle, data_name_for_puzzle


def _add_dataset_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--puzzle", required=True, help="Puzzle class or normalized name."
    )
    parser.add_argument(
        "--puzzle-args", default="{}", help="Puzzle constructor JSON object."
    )
    parser.add_argument("--data-name", default=None)
    parser.add_argument("--dataset-size", type=int, default=300_000)
    parser.add_argument("--dataset-minibatch-size", type=int, default=30_000)
    parser.add_argument("--shuffle-length", type=int, default=30)
    parser.add_argument(
        "--image-size", type=int, nargs=2, default=(32, 32), metavar=("W", "H")
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)


def configure_parser(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="world_model_command", required=True)

    transition = commands.add_parser(
        "make-transition-dataset", help="Generate transition images and actions."
    )
    _add_dataset_options(transition)
    transition.set_defaults(handler=make_transition_dataset)

    samples = commands.add_parser(
        "make-sample-data", help="Generate initial and target state images."
    )
    _add_dataset_options(samples)
    samples.set_defaults(handler=make_sample_data)

    evaluation = commands.add_parser(
        "make-eval-trajectory", help="Generate a trajectory used during evaluation."
    )
    _add_dataset_options(evaluation)
    evaluation.set_defaults(handler=make_eval_trajectory)

    training = commands.add_parser("train", help="Train a registered world model.")
    training.add_argument("--model", required=True)
    training.add_argument("--data-name", default=None)
    training.add_argument("--data-dir", type=Path, default=None)
    training.add_argument("--epochs", type=int, default=2_000)
    training.add_argument("--mini-batch-size", type=int, default=1_000)
    training.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    training.add_argument("--learning-rate", type=float, default=1e-4)
    training.add_argument("--save-every", type=int, default=100)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--reset", action="store_true")
    training.set_defaults(handler=train)


def _dataset_context(args: argparse.Namespace):
    puzzle = create_puzzle(args.puzzle, args.puzzle_args)
    data_name = args.data_name or data_name_for_puzzle(puzzle)
    root = args.output_dir or world_model_data_path(data_name)
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


def make_transition_dataset(args: argparse.Namespace) -> int:
    from puxle.world_model import get_world_model_dataset_builder

    puzzle, root = _dataset_context(args)
    output = root / "transition"
    output.mkdir(parents=True, exist_ok=True)
    shuffle_parallel = math.ceil(args.dataset_minibatch_size / args.shuffle_length)
    build_dataset = get_world_model_dataset_builder(
        puzzle,
        args.dataset_size,
        shuffle_parallel,
        args.shuffle_length,
        args.dataset_minibatch_size,
    )
    states, actions, next_states = build_dataset(jax.random.PRNGKey(args.seed))
    np.save(output / "actions.npy", np.asarray(actions))
    np.save(output / "images.npy", _render_states(states, tuple(args.image_size)))
    np.save(
        output / "next_images.npy", _render_states(next_states, tuple(args.image_size))
    )
    print(output)
    return 0


def make_sample_data(args: argparse.Namespace) -> int:
    from puxle.world_model import get_sample_data_builder

    puzzle, output = _dataset_context(args)
    build_dataset = get_sample_data_builder(
        puzzle, args.dataset_size, args.dataset_minibatch_size
    )
    targets, initials = build_dataset(jax.random.PRNGKey(args.seed))
    np.save(output / "targets.npy", _render_states(targets, tuple(args.image_size)))
    np.save(output / "inits.npy", _render_states(initials, tuple(args.image_size)))
    print(output)
    return 0


def make_eval_trajectory(args: argparse.Namespace) -> int:
    from puxle.world_model import create_eval_trajectory

    puzzle, root = _dataset_context(args)
    output = root / "transition"
    output.mkdir(parents=True, exist_ok=True)
    states, actions = create_eval_trajectory(
        puzzle, args.dataset_size, jax.random.PRNGKey(args.seed)
    )
    np.save(output / "eval_actions.npy", np.asarray(actions))
    np.save(
        output / "eval_traj_images.npy", _render_states(states, tuple(args.image_size))
    )
    print(output)
    return 0


def _data_name_for_model(name: str) -> str:
    normalized = name.lower()
    if "rubikscube" in normalized:
        return "rubikscube"
    if "sokoban" in normalized:
        return "sokoban"
    raise ValueError("--data-name is required for an unrecognized model name")


def train(args: argparse.Namespace) -> int:
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

    factory = getattr(trained_world_model_registry, args.model, None)
    if factory is None:
        available = ", ".join(sorted(vars(trained_world_model_registry)))
        raise ValueError(f"Unknown model '{args.model}'. Available: {available}")

    data_name = args.data_name or _data_name_for_model(args.model)
    data_dir = args.data_dir or world_model_data_path(data_name) / "transition"
    datas = jnp.asarray(np.load(data_dir / "images.npy"))
    next_datas = jnp.asarray(np.load(data_dir / "next_images.npy"))
    actions = jnp.asarray(np.load(data_dir / "actions.npy"))
    eval_states = jnp.asarray(np.load(data_dir / "eval_traj_images.npy"))
    eval_actions = jnp.asarray(np.load(data_dir / "eval_actions.npy"))

    world_model = _create_world_model_for_training(
        args.model,
        init_params=args.reset,
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
        optax.adam(args.learning_rate)
        if args.optimizer == "adam"
        else optax.sgd(args.learning_rate)
    )
    train_fn = world_model_train_builder(args.mini_batch_size, train_info_fn, optimizer)
    eval_fn = world_model_eval_builder(train_info_fn, args.mini_batch_size)
    params = world_model.params
    opt_state = optimizer.init(params)
    key = jax.random.PRNGKey(args.seed)

    for epoch in range(args.epochs):
        key, subkey = jax.random.split(key)
        params, opt_state, loss, ae_loss, wm_loss, accuracy = train_fn(
            subkey, (datas, next_datas, actions), params, opt_state, epoch
        )
        if epoch % 10 == 0 or epoch + 1 == args.epochs:
            eval_accuracy = eval_fn(params, (eval_states, eval_actions))
            print(
                f"epoch={epoch} loss={float(loss):.6f} ae={float(ae_loss):.6f} "
                f"wm={float(wm_loss):.6f} accuracy={float(accuracy):.6f} "
                f"eval={float(eval_accuracy):.6f}"
            )
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            world_model.params = params
            world_model.save_model()

    world_model.params = params
    world_model.save_model()
    print(world_model.path)
    return 0
