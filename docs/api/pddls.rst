PDDL Planning Domains (``puxle.pddls``)
========================================

PuXle supports the STRIPS subset of PDDL with automatic grounding and
JAX-optimized state representation.

PDDL
-----

.. autoclass:: puxle.pddls.pddl.PDDL
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Grounding
---------

.. automodule:: puxle.pddls.grounding
   :members:

Masks
-----

.. automodule:: puxle.pddls.masks
   :members:

Formatting
----------

.. automodule:: puxle.pddls.formatting
   :members:

Type System
-----------

.. automodule:: puxle.pddls.type_system
   :members:

State Definitions
-----------------

.. automodule:: puxle.pddls.state_defs
   :members:

PDDLFuse (Domain Fusion)
-------------------------

.. automodule:: puxle.pddls.fusion.api
   :members:

.. autoclass:: puxle.pddls.fusion.domain_fusion.DomainFusion
   :members:
   :show-inheritance:

.. autoclass:: puxle.pddls.fusion.action_modifier.ActionModifier
   :members:
   :show-inheritance:

.. autoclass:: puxle.pddls.fusion.action_modifier.FusionParams
   :members:
   :show-inheritance:

.. autoclass:: puxle.pddls.fusion.problem_generator.ProblemGenerator
   :members:
   :show-inheritance:
