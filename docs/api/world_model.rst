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

Training and Dataset Builders
-----------------------------

.. autofunction:: puxle.world_model.get_world_model_dataset_builder

.. autofunction:: puxle.world_model.get_sample_data_builder

.. autofunction:: puxle.world_model.world_model_train_builder

.. autofunction:: puxle.world_model.world_model_eval_builder
