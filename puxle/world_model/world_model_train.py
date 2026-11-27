import math
from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax


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


def _build_new_params_from_updates(params, updates, collection="batch_stats"):
    new_params = {"params": params["params"]}
    if collection in updates:
        new_params[collection] = updates[collection]
    return new_params


def accuracy_fn(preds: chex.Array, labels: chex.Array) -> chex.Array:
    axes = tuple(range(1, preds.ndim))
    return jnp.mean(jnp.sum(jnp.logical_xor(preds, labels), axis=axes) == 0)


def world_model_train_builder(
    minibatch_size: int,
    train_info_fn: Callable,
    optimizer: optax.GradientTransformation = optax.adam(1e-4),
    loss_ratio: float = 0.5,
    progress: Callable[[int, np.ndarray], None] | None = None,
    wm_onset_epoch: int = 100,
    wm_ramp_epochs: int = 1000,
):
    if minibatch_size <= 0:
        raise ValueError("minibatch_size must be positive")
    if wm_ramp_epochs <= 0:
        raise ValueError("wm_ramp_epochs must be positive")

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
        new_params = _build_new_params_from_updates(params, variable_updates)
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

    @jax.jit
    def train_batches(
        params, opt_state, metric_sums, dataset, batch_indices, loss_weight
    ):
        states, next_states, actions = dataset

        def train_loop(carry, indices):
            params, opt_state, metric_sums = carry
            (metrics, grads) = jax.value_and_grad(loss_fn, has_aux=True)(
                params,
                states[indices],
                next_states[indices],
                actions[indices],
                loss_weight,
            )
            loss, (params, ae_loss, wm_loss, accuracy) = metrics
            updates, opt_state = optimizer.update(grads, opt_state, params=params)
            params = optax.apply_updates(params, updates)
            metrics = jnp.stack((loss, ae_loss, wm_loss, accuracy))
            return (params, opt_state, metric_sums + metrics), metrics

        (params, opt_state, metric_sums), metrics = jax.lax.scan(
            train_loop, (params, opt_state, metric_sums), batch_indices
        )
        return params, opt_state, metric_sums, metrics[-1]

    def train_fn(
        key: chex.PRNGKey,
        dataset: tuple[jax.Array, jax.Array, jax.Array],
        params: Any,
        opt_state: optax.OptState,
        epoch: int,
    ):
        data_size = dataset[2].shape[0]
        if data_size == 0:
            raise ValueError("training dataset must contain at least one action")
        batch_count = math.ceil(data_size / minibatch_size)
        key, subkey = jax.random.split(key)
        indices = jnp.concatenate(
            [
                jax.random.permutation(key, jnp.arange(data_size)),
                jax.random.randint(
                    subkey,
                    (batch_count * minibatch_size - data_size,),
                    0,
                    data_size,
                ),
            ]
        ).reshape(batch_count, minibatch_size)
        loss_weight = (
            jnp.clip((epoch - wm_onset_epoch) / wm_ramp_epochs, 0.0001, 1.0)
            * loss_ratio
        )

        metric_sums = jnp.zeros(4)
        batches_per_group = math.ceil(batch_count / 50)
        for start in range(0, batch_count, batches_per_group):
            completed = min(start + batches_per_group, batch_count)
            params, opt_state, metric_sums, metrics = train_batches(
                params,
                opt_state,
                metric_sums,
                dataset,
                indices[start:completed],
                loss_weight,
            )
            # Synchronize groups for progress without a host round trip per batch.
            jax.block_until_ready(metrics)
            if progress is not None:
                progress(completed, np.asarray(metrics))
        return params, opt_state, *(metric_sums / batch_count), loss_weight

    return train_fn


def world_model_eval_builder(train_info_fn: Callable, minibatch_size: int):
    if minibatch_size <= 0:
        raise ValueError("minibatch_size must be positive")

    @jax.jit
    def eval_step(params, counts, data, next_data, action, batch_valid):
        result = train_info_fn(params, data, next_data, action, training=False)
        axes = tuple(range(1, result[4].ndim))
        correct = jnp.sum(jnp.logical_xor(result[4], result[7]), axis=axes) == 0
        return counts + jnp.stack(
            (jnp.sum(jnp.where(batch_valid, correct, False)), jnp.sum(batch_valid))
        )

    def eval_fn(
        params: Any, trajectory: tuple[np.ndarray | jax.Array, np.ndarray | jax.Array]
    ):
        states_all, actions = trajectory
        data_size = actions.shape[0]
        if data_size == 0:
            raise ValueError("evaluation trajectory must contain at least one action")
        counts = jnp.zeros(2, dtype=jnp.int32)
        for start in range(0, data_size, minibatch_size):
            flat_indices = np.arange(start, start + minibatch_size)
            indices = np.minimum(flat_indices, data_size - 1)
            counts = eval_step(
                params,
                counts,
                jnp.asarray(states_all[indices]),
                jnp.asarray(states_all[indices + 1]),
                jnp.asarray(actions[indices]),
                jnp.asarray(flat_indices < data_size),
            )
            jax.block_until_ready(counts)
        return counts[0] / counts[1]

    return eval_fn
