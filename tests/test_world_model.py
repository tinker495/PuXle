import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.traverse_util import flatten_dict


def test_world_model_public_api_exposes_existing_model_classes():
    from puxle import RubiksCubeWorldModel as TopLevelRubiksCubeWorldModel
    from puxle.world_model import (
        RubiksCubeWorldModel,
        RubiksCubeWorldModelOptimized,
        SokobanWorldModel,
        SokobanWorldModelOptimized,
        WorldModelPuzzleBase,
    )

    assert issubclass(RubiksCubeWorldModel, WorldModelPuzzleBase)
    assert issubclass(RubiksCubeWorldModelOptimized, WorldModelPuzzleBase)
    assert issubclass(SokobanWorldModel, WorldModelPuzzleBase)
    assert issubclass(SokobanWorldModelOptimized, WorldModelPuzzleBase)
    assert TopLevelRubiksCubeWorldModel is RubiksCubeWorldModel


def test_world_model_target_is_not_fixed():
    from puxle.world_model import WorldModelPuzzleBase

    assert object.__new__(WorldModelPuzzleBase).fixed_target is False


def test_trained_world_model_registry_owns_model_checkpoint_pairs():
    from puxle.world_model import (
        SokobanWorldModel,
        SokobanWorldModelOptimized,
        WorldModelPuzzleConfig,
        trained_world_model_registry,
    )

    registry_items = {
        name: factory.__self__
        for name, factory in vars(trained_world_model_registry).items()
    }
    assert all(
        factory.__func__ is WorldModelPuzzleConfig.create
        for factory in vars(trained_world_model_registry).values()
    )
    assert all(
        name == config.callable.__name__ for name, config in registry_items.items()
    )
    assert {name: config.path for name, config in registry_items.items()} == {
        "RubiksCubeWorldModel": "rubikscube_v2.pkl",
        "RubiksCubeWorldModel_test": "rubikscube_v2.pkl",
        "RubiksCubeWorldModel_reversed": "rubikscube_v2.pkl",
        "RubiksCubeWorldModelOptimized": "rubikscube_optimized_v2.pkl",
        "RubiksCubeWorldModelOptimized_test": "rubikscube_optimized_v2.pkl",
        "RubiksCubeWorldModelOptimized_reversed": "rubikscube_optimized_v2.pkl",
        "SokobanWorldModel": "sokoban_v2.pkl",
        "SokobanWorldModelOptimized": "sokoban_optimized_v2.pkl",
    }
    assert (
        trained_world_model_registry.SokobanWorldModel.__self__
        == WorldModelPuzzleConfig(
            callable=SokobanWorldModel,
            path="sokoban_v2.pkl",
        )
    )
    assert (
        trained_world_model_registry.SokobanWorldModelOptimized.__self__
        == WorldModelPuzzleConfig(
            callable=SokobanWorldModelOptimized,
            path="sokoban_optimized_v2.pkl",
        )
    )

    config = WorldModelPuzzleConfig(lambda **kwargs: kwargs, "params.pkl")
    created = config.create()
    assert created["init_params"] is False
    assert created["aqt_cfg"] == "int8"
    with pytest.raises(TypeError):
        config.create(aqt_cfg=None)


def test_trained_registry_factory_resolves_checkpoint_under_puxle_home(
    monkeypatch, tmp_path
):
    from puxle.world_model import SokobanWorldModel, trained_world_model_registry

    captured = {}

    def fake_init(self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setenv("PUXLE_HOME", str(tmp_path))
    monkeypatch.setattr(SokobanWorldModel, "__init__", fake_init)

    model = trained_world_model_registry.SokobanWorldModel()

    assert isinstance(model, SokobanWorldModel)
    assert captured == {
        "path": str(tmp_path / "world_model" / "model" / "params" / "sokoban_v2.pkl"),
        "init_params": False,
        "aqt_cfg": "int8",
    }


def test_training_world_model_creation_is_separate_from_serving_registry(
    monkeypatch, tmp_path
):
    from puxle.world_model import SokobanWorldModel
    from puxle.world_model.trained_model_registry import (
        _create_world_model_for_training,
    )

    captured = {}

    def fake_init(self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setenv("PUXLE_HOME", str(tmp_path))
    monkeypatch.setattr(SokobanWorldModel, "__init__", fake_init)

    model = _create_world_model_for_training("SokobanWorldModel", init_params=True)

    assert isinstance(model, SokobanWorldModel)
    assert captured == {
        "path": str(tmp_path / "world_model" / "model" / "params" / "sokoban_v2.pkl"),
        "init_params": True,
        "aqt_cfg": None,
    }


def test_world_model_checkpoint_round_trip_preserves_flax_tree(tmp_path):
    from puxle.world_model import WorldModelPuzzleBase

    class TinyWorldModel(WorldModelPuzzleBase):
        def data_init(self):
            self.inits = jnp.zeros((1, 2), dtype=jnp.uint8)
            self.targets = self.inits
            self.init_state_size = 1

    path = str(tmp_path / "world_model.pkl")
    model = TinyWorldModel("unused", (2,), (8,), 2, init_params=True, path=path)
    keys = flatten_dict(model.params)
    assert ("params", "autoencoder", "encoder", "Dense_0", "kernel") in keys
    assert ("params", "world_model", "Dense_3", "kernel") in keys

    model.save_model({"test": True})
    loaded = TinyWorldModel("unused", (2,), (8,), 2, path=path)

    assert loaded.metadata["test"] is True
    for actual, expected in zip(
        jax.tree_util.tree_leaves(loaded.params),
        jax.tree_util.tree_leaves(model.params),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


def test_world_model_int8_serving_freezes_quantized_weights():
    from aqt.jax.v2.flax import aqt_flax

    from puxle.world_model import WorldModelPuzzleBase

    class TinyWorldModel(WorldModelPuzzleBase):
        def data_init(self):
            self.inits = jnp.zeros((1, 2), dtype=jnp.uint8)
            self.targets = self.inits
            self.init_state_size = 1

    model = TinyWorldModel(
        "unused",
        (2,),
        (8,),
        2,
        init_params=True,
        aqt_cfg="int8",
    )

    assert model.model.autoencoder.quant_mode is aqt_flax.QuantMode.SERVE
    assert any(
        leaf.dtype == jnp.int8
        for leaf in jax.tree_util.tree_leaves(model.params["aqt"])
        if hasattr(leaf, "dtype")
    )
    encoded = model.model.apply(
        model.params,
        jnp.zeros((1, 2)),
        method=model.model.encode,
    )
    assert encoded.shape == (1, 8)


def test_world_model_train_builder_updates_params():
    from puxle.world_model import round_through_gradient, world_model_train_builder

    optimizer = optax.sgd(0.1)
    params = {"params": {"weight": jnp.array(0.0)}}

    def train_info_fn(params, data, next_data, action, training):
        del action, training
        weight = params["params"]["weight"]
        logits = jnp.broadcast_to(weight, (data.shape[0], 2))
        next_logits = jnp.broadcast_to(weight, (next_data.shape[0], 2))
        rounded = round_through_gradient(jax.nn.sigmoid(logits))
        rounded_next = round_through_gradient(jax.nn.sigmoid(next_logits))
        decoded = jnp.broadcast_to(weight, data.shape)
        next_decoded = jnp.broadcast_to(weight, next_data.shape)
        return (
            logits,
            rounded,
            decoded,
            next_logits,
            rounded_next,
            next_decoded,
            next_logits,
            rounded_next,
        ), {}

    train = world_model_train_builder(1, train_info_fn, optimizer=optimizer)
    data = jnp.full((2, 1), 255.0)
    updated, _, loss, *_ = train(
        jax.random.PRNGKey(0),
        (data, data, jnp.array([0, 1])),
        params,
        optimizer.init(params),
        101,
    )

    assert jnp.isfinite(loss)
    assert updated["params"]["weight"] != params["params"]["weight"]


def test_world_model_eval_builder_handles_partial_final_batch():
    from puxle.world_model import world_model_eval_builder

    def train_info_fn(params, data, next_data, action, training):
        del params, data, next_data, training
        labels = jnp.zeros((action.shape[0], 1), dtype=jnp.bool_)
        predictions = action[:, None].astype(jnp.bool_)
        return None, None, None, None, labels, None, None, predictions

    evaluate = world_model_eval_builder(train_info_fn, minibatch_size=2)
    states = jnp.zeros((4, 1), dtype=jnp.uint8)
    actions = jnp.array([0, 0, 1])

    np.testing.assert_allclose(evaluate({}, (states, actions)), 2 / 3)


def test_world_model_dataset_helpers_consume_puxle_trajectories():
    from puxle.core.trajectory import PuzzleTrajectory
    from puxle.puzzles.slidepuzzle import SlidePuzzle
    from puxle.world_model import (
        create_eval_trajectory,
        create_sample_data,
        create_shuffled_path,
    )

    state_cls = SlidePuzzle(size=3).State
    transition = PuzzleTrajectory(
        solve_configs=None,
        states=state_cls(board=jnp.arange(40, dtype=jnp.uint8).reshape(4, 2, 5)),
        move_costs=None,
        move_costs_tm1=None,
        actions=jnp.arange(6).reshape(3, 2),
        action_costs=None,
    )
    evaluation = PuzzleTrajectory(
        solve_configs=None,
        states=state_cls(board=jnp.arange(20, dtype=jnp.uint8).reshape(4, 1, 5)),
        move_costs=None,
        move_costs_tm1=None,
        actions=jnp.arange(3).reshape(3, 1),
        action_costs=None,
    )

    class PuzzleStub:
        def batched_get_random_trajectory(self, k_max, parallel, key):
            del k_max, key
            return transition if parallel == 2 else evaluation

    states, actions, next_states = create_shuffled_path(
        PuzzleStub(), 3, 2, 5, jax.random.PRNGKey(0)
    )
    eval_states, eval_actions = create_eval_trajectory(
        PuzzleStub(), 3, jax.random.PRNGKey(0)
    )

    assert jnp.array_equal(states.board, transition.states.board[:-1].reshape(6, 5)[:5])
    assert jnp.array_equal(actions, transition.actions.reshape(6)[:5])
    assert jnp.array_equal(
        next_states.board, transition.states.board[1:].reshape(6, 5)[:5]
    )
    assert jnp.array_equal(eval_states.board, evaluation.states.board.reshape(4, 5))
    assert jnp.array_equal(eval_actions, evaluation.actions.reshape(3))

    puzzle = SlidePuzzle(size=2)
    key = jax.random.PRNGKey(1)
    targets, _ = create_sample_data(puzzle, shuffle_parallel=2, key=key)
    solve_configs, _ = jax.vmap(puzzle.get_inits)(jax.random.split(key, 2))
    assert jnp.array_equal(targets.board, solve_configs.GoalSpec.board)


def test_world_model_dataset_builder_uses_actual_minibatch_yield(monkeypatch):
    from puxle.world_model import world_model_ds

    calls = []

    def fake_create_path(*args):
        calls.append(args)
        values = jnp.arange(2)
        return values, values, values

    monkeypatch.setattr(world_model_ds, "create_shuffled_path", fake_create_path)
    build_dataset = world_model_ds.get_world_model_dataset_builder(
        puzzle=object(),
        dataset_size=5,
        shuffle_parallel=1,
        shuffle_length=2,
        dataset_minibatch_size=3,
    )

    states, actions, next_states = build_dataset(jax.random.PRNGKey(0))

    assert len(calls) == 3
    assert states.shape == actions.shape == next_states.shape == (5,)
