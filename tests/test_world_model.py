import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.traverse_util import flatten_dict


def test_world_model_public_api_exposes_adapter_and_presets():
    """Author: tinker495.

    Why: Verifies that ``WorldModelPuzzle`` and ``WorldModelPresets`` are exposed
    from top-level ``puxle``, ``puxle.puzzles.adapters``, and
    ``puxle.puzzles.adapters.world_model``, matching ``CayleyPuzzle``/``PDDL``.
    """
    from puxle import WorldModelPresets, WorldModelPuzzle
    from puxle.puzzles import (
        WorldModelPresets as PuzzlesWorldModelPresets,
    )
    from puxle.puzzles import (
        WorldModelPuzzle as PuzzlesWorldModelPuzzle,
    )
    from puxle.puzzles.adapters import (
        WorldModelPresets as AdaptersWorldModelPresets,
    )
    from puxle.puzzles.adapters import (
        WorldModelPuzzle as AdaptersWorldModelPuzzle,
    )
    from puxle.puzzles.adapters.world_model import (
        WorldModelPresets as DirectWorldModelPresets,
    )
    from puxle.puzzles.adapters.world_model import (
        WorldModelPuzzle as DirectWorldModelPuzzle,
    )

    assert WorldModelPuzzle is PuzzlesWorldModelPuzzle
    assert WorldModelPuzzle is AdaptersWorldModelPuzzle
    assert WorldModelPuzzle is DirectWorldModelPuzzle
    assert WorldModelPresets is PuzzlesWorldModelPresets
    assert WorldModelPresets is AdaptersWorldModelPresets
    assert WorldModelPresets is DirectWorldModelPresets
    assert WorldModelPuzzle.list_sources() == {
        "architectures": [
            "rubikscube",
            "rubikscube_optimized",
            "sokoban",
            "sokoban_optimized",
        ],
        "presets": sorted(WorldModelPresets.all()),
    }


def test_world_model_target_is_not_fixed():
    from puxle import WorldModelPuzzle

    assert object.__new__(WorldModelPuzzle).fixed_target is False


def test_world_model_presets_own_architecture_and_checkpoint_pairs():
    """Author: tinker495.

    Why: Locks the canonical ``WorldModelPresets`` registry replacing the legacy
    thin subclasses and ``SimpleNamespace`` ``trained_world_model_registry``.
    """
    from puxle import WorldModelPresets

    presets = WorldModelPresets.all()
    assert {name: dict(cfg) for name, cfg in presets.items()} == {
        "RUBIKSCUBE": {
            "architecture": "rubikscube",
            "data_name": "rubikscube",
            "path": "rubikscube_v2.pkl",
            "aqt_cfg": "int8",
        },
        "RUBIKSCUBE_TEST": {
            "architecture": "rubikscube",
            "data_name": "rubikscube_test",
            "path": "rubikscube_v2.pkl",
            "aqt_cfg": "int8",
        },
        "RUBIKSCUBE_REVERSED": {
            "architecture": "rubikscube",
            "data_name": "rubikscube",
            "reverse_data": True,
            "path": "rubikscube_v2.pkl",
            "aqt_cfg": "int8",
        },
        "RUBIKSCUBE_OPTIMIZED": {
            "architecture": "rubikscube_optimized",
            "data_name": "rubikscube",
            "path": "rubikscube_optimized_v2.pkl",
            "aqt_cfg": "int8",
        },
        "RUBIKSCUBE_OPTIMIZED_TEST": {
            "architecture": "rubikscube_optimized",
            "data_name": "rubikscube_test",
            "path": "rubikscube_optimized_v2.pkl",
            "aqt_cfg": "int8",
        },
        "RUBIKSCUBE_OPTIMIZED_REVERSED": {
            "architecture": "rubikscube_optimized",
            "data_name": "rubikscube",
            "reverse_data": True,
            "path": "rubikscube_optimized_v2.pkl",
            "aqt_cfg": "int8",
        },
        "SOKOBAN": {
            "architecture": "sokoban",
            "data_name": "sokoban",
            "path": "sokoban_v2.pkl",
            "aqt_cfg": "int8",
        },
        "SOKOBAN_OPTIMIZED": {
            "architecture": "sokoban_optimized",
            "data_name": "sokoban",
            "path": "sokoban_optimized_v2.pkl",
            "aqt_cfg": "int8",
        },
    }


def test_world_model_preset_resolves_checkpoint_under_puxle_home(monkeypatch, tmp_path):
    """Author: tinker495.

    Why: Ensures preset checkpoint filenames resolve under ``$PUXLE_HOME`` for
    both int8 serving and unquantized training construction.
    """
    from puxle import WorldModelPresets, WorldModelPuzzle

    monkeypatch.setenv("PUXLE_HOME", str(tmp_path))
    monkeypatch.setattr(WorldModelPuzzle, "data_init", lambda self: None)

    serving_puzzle = WorldModelPuzzle(**WorldModelPresets.SOKOBAN, init_params=True)
    assert serving_puzzle.path == str(
        tmp_path / "world_model" / "model" / "params" / "sokoban_v2.pkl"
    )
    assert serving_puzzle.aqt_cfg == "int8"

    training_puzzle = WorldModelPuzzle(
        **(dict(WorldModelPresets.SOKOBAN) | {"init_params": True, "aqt_cfg": None})
    )
    assert training_puzzle.path == str(
        tmp_path / "world_model" / "model" / "params" / "sokoban_v2.pkl"
    )
    assert training_puzzle.aqt_cfg is None


def test_world_model_checkpoint_round_trip_preserves_flax_tree(tmp_path):
    from puxle import WorldModelPuzzle

    class TinyWorldModel(WorldModelPuzzle):
        def data_init(self):
            self.inits = jnp.zeros((1, 2), dtype=jnp.uint8)
            self.targets = self.inits
            self.init_state_size = 1

    path = str(tmp_path / "world_model.pkl")
    model = TinyWorldModel(
        data_path="unused",
        data_shape=(2,),
        latent_shape=(8,),
        action_size=2,
        init_params=True,
        path=path,
    )
    keys = flatten_dict(model.params)
    assert ("params", "autoencoder", "encoder", "Dense_0", "kernel") in keys
    assert ("params", "world_model", "Dense_3", "kernel") in keys

    model.save_model({"test": True})
    loaded = TinyWorldModel(
        data_path="unused",
        data_shape=(2,),
        latent_shape=(8,),
        action_size=2,
        path=path,
    )

    assert loaded.metadata["test"] is True
    for actual, expected in zip(
        jax.tree_util.tree_leaves(loaded.params),
        jax.tree_util.tree_leaves(model.params),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


def test_world_model_int8_serving_freezes_quantized_weights():
    from aqt.jax.v2.flax import aqt_flax

    from puxle import WorldModelPuzzle

    class TinyWorldModel(WorldModelPuzzle):
        def data_init(self):
            self.inits = jnp.zeros((1, 2), dtype=jnp.uint8)
            self.targets = self.inits
            self.init_state_size = 1

    model = TinyWorldModel(
        data_path="unused",
        data_shape=(2,),
        latent_shape=(8,),
        action_size=2,
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


def test_world_model_filled_false_preserves_state_and_returns_float32_costs():
    """Author: tinker495.

    Why: Regression test enforcing Puzzle's ``filled=False`` state-preservation
    contract and ``float32`` cost dtype on ``WorldModelPuzzle``.
    """
    from puxle import WorldModelPuzzle

    class TinyWorldModel(WorldModelPuzzle):
        def data_init(self):
            self.inits = jnp.zeros((1, 2), dtype=jnp.uint8)
            self.targets = self.inits
            self.init_state_size = 1

    model = TinyWorldModel(
        data_path="unused",
        data_shape=(2,),
        latent_shape=(8,),
        action_size=2,
        init_params=True,
    )
    solve_config, state = model.get_inits(jax.random.PRNGKey(0))

    blocked_states, blocked_costs = model.get_neighbours(
        solve_config, state, filled=False
    )
    assert blocked_costs.dtype == jnp.float32
    assert jnp.all(jnp.isinf(blocked_costs))
    for i in range(model.action_size):
        assert blocked_states[i] == state

    _, active_costs = model.get_neighbours(solve_config, state, filled=True)
    assert active_costs.dtype == jnp.float32
    assert jnp.all(active_costs == 1.0)


def test_world_model_inverse_neighbours_contract_and_action_strings(monkeypatch):
    """Author: tinker495.

    Why: Ensures reversible world models permute inverse neighbours according to
    ``inverse_action_map``, non-reversible ones raise ``NotImplementedError``,
    and architecture presets expose meaningful ``action_to_string`` labels.
    """
    from puxle import WorldModelPresets, WorldModelPuzzle

    monkeypatch.setattr(WorldModelPuzzle, "data_init", lambda self: None)

    rubik = WorldModelPuzzle(
        **(dict(WorldModelPresets.RUBIKSCUBE) | {"init_params": True, "aqt_cfg": None})
    )
    assert rubik.is_reversible is True
    assert rubik.action_to_string(0) == "L'"
    assert rubik.action_to_string(1) == "L"
    state = rubik.State.from_unpacked(
        latent=jnp.zeros(rubik.latent_shape, dtype=jnp.bool_)
    )
    sc = rubik.SolveConfig(
        InstanceContext=rubik.InstanceContext(),
        GoalSpec=state,
    )
    fwd_states, fwd_costs = rubik.get_neighbours(sc, state, filled=True)
    inv_states, inv_costs = rubik.get_inverse_neighbours(sc, state, filled=True)
    perm = rubik.inverse_action_map
    assert perm is not None
    assert jnp.array_equal(inv_costs, fwd_costs[perm])
    assert jnp.array_equal(inv_states.latent, fwd_states.latent[perm])

    sokoban = WorldModelPuzzle(
        **(dict(WorldModelPresets.SOKOBAN) | {"init_params": True, "aqt_cfg": None})
    )
    assert sokoban.is_reversible is False
    assert sokoban.action_to_string(0) == "←"
    soko_state = sokoban.State.from_unpacked(
        latent=jnp.zeros(sokoban.latent_shape, dtype=jnp.bool_)
    )
    soko_sc = sokoban.SolveConfig(
        InstanceContext=sokoban.InstanceContext(),
        GoalSpec=soko_state,
    )
    with pytest.raises(NotImplementedError):
        sokoban.get_inverse_neighbours(soko_sc, soko_state, filled=True)


def test_world_model_data_init_uses_custom_path_and_reverse_data(monkeypatch, tmp_path):
    """Author: tinker495.

    Why: Verifies ``data_init`` honors explicit ``data_path`` without triggering
    global Hugging Face downloads and swaps ``inits``/``targets`` when
    ``reverse_data=True``.
    """
    from puxle import WorldModelPuzzle

    data_dir = tmp_path / "custom_data"
    data_dir.mkdir()
    inits = np.ones((2, 4), dtype=np.uint8)
    targets = np.full((2, 4), 2, dtype=np.uint8)
    np.save(data_dir / "inits.npy", inits)
    np.save(data_dir / "targets.npy", targets)

    monkeypatch.setattr(
        "puxle.puzzles.adapters.world_model.world_model.download_world_model_dataset",
        lambda: pytest.fail("download_world_model_dataset must not be called"),
    )
    puzzle = WorldModelPuzzle(
        data_path=data_dir,
        data_shape=(4,),
        latent_shape=(8,),
        action_size=2,
        init_params=True,
        reverse_data=True,
    )
    assert np.array_equal(np.asarray(puzzle.inits), targets)
    assert np.array_equal(np.asarray(puzzle.targets), inits)
    assert puzzle.init_state_size == 2

    missing_dir = tmp_path / "missing_data"
    with pytest.raises(FileNotFoundError, match="custom data_path"):
        WorldModelPuzzle(
            data_path=missing_dir,
            data_shape=(4,),
            latent_shape=(8,),
            action_size=2,
            init_params=True,
        )


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
