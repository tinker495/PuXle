Core Framework (``puxle.core``)
================================

The core module provides the base classes and data structures for creating
puzzle environments in PuXle.

.. automodule:: puxle.core
   :no-members:

Puzzle
------

.. autoclass:: puxle.core.puzzle_base.Puzzle
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

PuzzleState
-----------

.. autoclass:: puxle.core.puzzle_state.PuzzleState
   :members:
   :show-inheritance:

state_dataclass
---------------

.. autofunction:: puxle.core.puzzle_state.state_dataclass

FieldDescriptor
---------------

Re-exported from `xtructure <https://github.com/tinker495/Xtructure>`_ for
convenience.  See
:py:data:`puxle.core.puzzle_state.FieldDescriptor`.
