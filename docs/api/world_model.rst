World Models (``puxle.world_model``)
=====================================

Install the optional dependencies with ``puxle[world-model]``.

Downloaded artifacts live below ``PUXLE_HOME/world_model``. When ``PUXLE_HOME``
is unset, PuXle follows ``XDG_CACHE_HOME`` and then ``~/.cache/puxle``.

.. currentmodule:: puxle.world_model

World-Model Puzzle
------------------

The autoencoder and transition model share finite scalar quantization (FSQ).
``fsq_levels`` selects 2 to 256 levels per scalar (default: 8). Networks consume
quantized values in [0, 1]; ``State.latent_unpacked`` contains unsigned integer
codes in [0, fsq_levels - 1], packed with ``ceil(log2(fsq_levels))`` bits per
scalar. Existing latent shapes are retained, so the default state storage uses
three times as many bits as the Binary AE representation.

Training uses straight-through rounding, image reconstruction loss, and
symmetric stop-gradient L2 transition loss in the bounded FSQ space. Accuracy
requires equality of every quantized scalar, including during evaluation.

Checkpoints record the quantizer and level count. Binary checkpoints and FSQ
checkpoints with different levels are rejected; retrain with ``--reset``.
Registered checkpoint paths are unchanged. Published Binary weights require
retraining before registry-based serving can be used.

.. autoclass:: puxle.world_model.WorldModelPuzzleBase
   :members:
   :show-inheritance:

Registry
--------

.. autoclass:: puxle.world_model.WorldModelPuzzleConfig

.. py:data:: trained_world_model_registry

   Namespace of zero-argument factories that return checkpoint-backed, int8 serving models.

Training and Dataset Builders
-----------------------------

.. autofunction:: puxle.world_model.get_world_model_dataset_builder

.. autofunction:: puxle.world_model.get_sample_data_builder

.. autofunction:: puxle.world_model.world_model_train_builder

.. autofunction:: puxle.world_model.world_model_eval_builder
