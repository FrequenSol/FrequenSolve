FrequenSolve Documentation
==========================

Welcome to FrequenSolve's documentation. FrequenSolve is the :term:`Python API`
for authoring, validating, launching, and reading finite-element wave
:term:`simulations <simulation>`. The Python package can build and inspect
:term:`projects <project>` on any development machine; executing
:term:`jobs <job>` requires a :term:`site` with the separately licensed
:term:`fast solver`.

Key Features
------------

- High-performance finite element modeling
- Support for :term:`time-domain` and :term:`frequency-domain` simulations
- Flexible model building and meshing capabilities
- Integration with popular data formats and visualization tools
- Cloud and :term:`HPC` deployment support

Getting Started
---------------

If you're new to FrequenSolve, follow this path:

- :doc:`installation`
- :doc:`quickstart` to build the compact FrequenSolve Python API workflow and see where site
  execution enters
- :doc:`user_guide/simulation_assistant_mcp` when you want an agent to prepare
  and validate the fixed private-beta starter, then monitor your Cloud run
  through self-scoped read-only tools
- :download:`Acoustic modeling
  <../../examples/tutorials/01_modeling_basics/01_acoustic.ipynb>` for the
  first full runnable notebook
- :doc:`tutorials/index` when you want the notebook learning path by topic
- :doc:`user_guide/index` when you need conceptual reference tables or a
  specific configuration detail
- :doc:`glossary` when a solver, meshing, or imaging term is unfamiliar

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   tutorials/index
   user_guide/index
   glossary

.. toctree::
   :maxdepth: 1
   :caption: Solver Developer Reference

   fast_solver_api_updates
   fast_solver_trace_output_compaction

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   frequensolve

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
