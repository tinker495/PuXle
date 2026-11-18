World Models (``puxle.world_model``)
=====================================

Install the optional dependencies with ``puxle[world-model]``.

Downloaded artifacts live below ``PUXLE_HOME/world_model``. When ``PUXLE_HOME``
is unset, PuXle follows ``XDG_CACHE_HOME`` and then ``~/.cache/puxle``.

.. currentmodule:: puxle.world_model

World-Model Puzzle
------------------

.. autoclass:: puxle.world_model.WorldModelPuzzleBase
   :members:
   :show-inheritance:

Registry
--------

.. autoclass:: puxle.world_model.WorldModelPuzzleConfig

.. py:data:: trained_world_model_registry

   Namespace of zero-argument factories that return checkpoint-backed, int8 serving models.

Training progress and dataset performance
-----------------------------------------

Transition dataset generation also saves ``transition_samples.png`` containing
three randomly selected ``state -> action ID -> next state`` rows.
Evaluation trajectory generation saves ``eval_transition_samples.png`` using
adjacent trajectory states and their matching action IDs. Sampling uses the
command's ``--seed``, selects without replacement, and shows all available
transitions when there are fewer than three. Preview images read only the
selected rows from the saved memory-mapped arrays, require no GUI, and print
their output paths.

Dataset commands display progress for initialization, transition batches,
image rendering, and file saves. Evaluation trajectory generation reports its
compilation/execution stage before rendering. Training displays dataset loading,
first-batch compilation, epochs, and minibatches with loss, AE loss,
world-model loss, and exact-match accuracy. The epoch display identifies
evaluation and checkpoint saves. Minibatch reporting is capped at 50 updates
per epoch to limit display overhead.

Training transfers the uint8 transition images and actions to the JAX device
once. Each epoch shuffles only indices; the compiled training loop gathers the
current minibatch from the original arrays without materializing an epoch-sized
shuffled image copy. Minibatches execute in compiled groups with at most 50 host
synchronizations per epoch for progress reporting. Evaluation images remain
memory-mapped on the host and transfer one minibatch at a time.

``--mini-batch-size`` (default: 1,000) controls minibatch activation memory;
device memory must also hold the full training dataset and model. Optimizer
updates run once per minibatch. This option does not accumulate gradients
across minibatches.

``--seed`` controls minibatch shuffling and, with ``--reset``, parameter
initialization. On GPU, a reset run is bitwise reproducible from its seed only
with ``XLA_FLAGS=--xla_gpu_deterministic_ops=true``. The world-model
loss weight is ``clip((epoch - onset) / ramp, 1e-4, 1) * ratio`` with
``--wm-onset-epoch`` (default: 20), ``--wm-ramp-epochs`` (default: 1,000), and
``--wm-loss-ratio`` (default: 0.5); the AE loss receives the remaining weight.
Evaluation log lines report the epoch's ``wm_weight``.

``--lr-schedule cosine`` (default) decays ``--learning-rate`` (default: 3e-3) to
zero over all training steps; ``constant`` keeps it fixed. The defaults
(200 epochs, onset 20, cosine from 3e-3) train ``RubiksCubeWorldModelOptimized``
on 300,000 transitions to zero one-step and 30-step errors on a 20,000-transition
hold-out, in both float and int8 serving.

Rubik color images use cached native sticker geometry and batched integer-area
reduction. Sokoban images use cached native tile sprites. These CPU batch paths
preserve the native renderer's pixels at integer reduction ratios (including
the training resolutions above), unpack states once, and bound temporary image
batches. Other puzzle types, Rubik tile-ID images, and unsupported resize ratios
use the native renderer. No worker-count or GPU-rendering option is required.

Training and Dataset Builders
-----------------------------

.. autofunction:: puxle.world_model.get_world_model_dataset_builder

.. autofunction:: puxle.world_model.get_sample_data_builder

.. autofunction:: puxle.world_model.world_model_train_builder

.. autofunction:: puxle.world_model.world_model_eval_builder
