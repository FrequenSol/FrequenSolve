FrequenSolve Documentation
==========================

Welcome to FrequenSolve's documentation. FrequenSolve is the Python SDK for
authoring, validating, launching, and reading finite-element wave simulations.
The Python package can build and inspect projects on any development machine;
executing jobs requires a site with the separately licensed fast solver.

Key Features
------------

- High-performance finite element modeling
- Support for time-domain and frequency-domain simulations
- Flexible model building and meshing capabilities
- Integration with popular data formats and visualization tools
- Cloud and HPC deployment support

Getting Started
---------------

If you're new to FrequenSolve, start with the tutorial collection. The notebooks
are the primary release examples: they build complete project-owned
simulations, submit strict jobs, and inspect traces, logs, meshes, and
ParaView/VTK outputs.

- :doc:`installation`
- :doc:`quickstart` for the compact current API workflow
- :doc:`tutorials/index` for runnable notebook examples
- :doc:`user_guide/index` for conceptual reference tables

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   tutorials/index
   user_guide/index

.. toctree::
   :maxdepth: 1
   :caption: Backend Contracts

   fast_solver_api_updates
   fast_solver_trace_output_compaction

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   frequensolve

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
