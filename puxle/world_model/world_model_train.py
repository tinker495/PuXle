import math
from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp
import optax

from ._utils import build_new_params_from_updates
from .world_model_puzzle_base import WorldModelPuzzleBase


def accuracy_fn(preds: chex.Array, labels: chex.Array) -> chex.Array:
    axes = tuple(range(1, preds.ndim))
    return jnp.mean(jnp.sum(jnp.logical_xor(preds, labels), axis=axes) == 0)


def world_model_train_builder(
    minibatch_size: int,
    train_info_fn: Callable,
    optimizer: optax.GradientTransformation = optax.adam(1e-4),
    loss_ratio: float = 0.5,
):
    def loss_fn(params, data, next_data, action, loss_weight=0.5):
        (
            (
                _,
                _,
                decoded,
                next_logits,
                rounded_next_latents,
                next_decoded,
                next_logits_pred,
                rounded_next_latents_pred,
            ),
            variable_updates,
        ) = train_info_fn(params, data, next_data, action, training=True)
        new_params = build_new_params_from_updates(params, variable_updates)
        data_scaled = (data / 255.0) * 2 - 1
        next_data_scaled = (next_data / 255.0) * 2 - 1
        ae_loss = jnp.mean(
            0.5 * optax.l2_loss(data_scaled, decoded)
            + 0.5 * optax.l2_loss(next_data_scaled, next_decoded)
        )
        wm_loss = jnp.mean(
            0.5
            * optax.sigmoid_binary_cross_entropy(
                next_logits, jax.lax.stop_gradient(rounded_next_latents_pred)
            )
            + 0.5
            * optax.sigmoid_binary_cross_entropy(
                next_logits_pred, jax.lax.stop_gradient(rounded_next_latents)
            )
        )
        total_loss = (1 - loss_weight) * ae_loss + loss_weight * wm_loss
        return total_loss, (
            new_params,
            ae_loss,
            wm_loss,
            accuracy_fn(rounded_next_latents, rounded_next_latents_pred),
        )

    def train_fn(
        key: chex.PRNGKey,
        dataset: tuple[
            WorldModelPuzzleBase.State, WorldModelPuzzleBase.State, chex.Array
        ],
        params: Any,
        opt_state: optax.OptState,
        epoch: int,
    ):
        states, next_states, actions = dataset
        data_size = actions.shape[0]
        batch_count = math.ceil(data_size / minibatch_size)
        key, subkey = jax.random.split(key)
        indices = jnp.concatenate(
            [
                jax.random.permutation(key, jnp.arange(data_size)),
                jax.random.randint(
                    subkey, (batch_count * minibatch_size - data_size,), 0, data_size
                ),
            ]
        ).reshape(batch_count, minibatch_size)
        batches = (
            jnp.take(states, indices, axis=0),
            jnp.take(next_states, indices, axis=0),
            jnp.take(actions, indices, axis=0),
        )
        loss_weight = jnp.clip((epoch - 100) / 1000.0, 0.0001, 1.0) * loss_ratio

        def train_loop(carry, batch):
            params, opt_state = carry
            (metrics, grads) = jax.value_and_grad(loss_fn, has_aux=True)(
                params, *batch, loss_weight
            )
            loss, (params, ae_loss, wm_loss, accuracy) = metrics
            updates, opt_state = optimizer.update(grads, opt_state, params=params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), (loss, ae_loss, wm_loss, accuracy)

        (params, opt_state), metrics = jax.lax.scan(
            train_loop, (params, opt_state), batches
        )
        return params, opt_state, *(jnp.mean(metric) for metric in metrics)

    return jax.jit(train_fn)


def world_model_eval_builder(train_info_fn: Callable, minibatch_size: int):
    def eval_fn(params: Any, trajectory: tuple[chex.Array, chex.Array]):
        states_all, actions = trajectory
        data_size = actions.shape[0]
        if data_size == 0:
            raise ValueError("evaluation trajectory must contain at least one action")
        batch_count = math.ceil(data_size / minibatch_size)
        flat_indices = jnp.arange(batch_count * minibatch_size)
        valid = (flat_indices < data_size).reshape(batch_count, minibatch_size)
        indices = jnp.minimum(flat_indices, data_size - 1).reshape(
            batch_count, minibatch_size
        )
        batches = (
            jnp.take(states_all[:-1], indices, axis=0),
            jnp.take(states_all[1:], indices, axis=0),
            jnp.take(actions, indices, axis=0),
            valid,
        )

        def eval_loop(counts, batch):
            data, next_data, action, batch_valid = batch
            result = train_info_fn(params, data, next_data, action, training=False)
            axes = tuple(range(1, result[4].ndim))
            correct = jnp.sum(jnp.logical_xor(result[4], result[7]), axis=axes) == 0
            return (
                counts[0] + jnp.sum(jnp.where(batch_valid, correct, False)),
                counts[1] + jnp.sum(batch_valid),
            ), None

        (correct, count), _ = jax.lax.scan(
            eval_loop, (jnp.array(0), jnp.array(0)), batches
        )
        return correct / count

    return jax.jit(eval_fn)
