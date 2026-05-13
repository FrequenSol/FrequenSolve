.. FrequenSolve documentation master file, created by
   sphinx-quickstart on Tue Jan 14 10:41:09 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FrequenSolve Documentation
==========================

Welcome to FrequenSolve's documentation. FrequenSolve is the Python SDK for
authoring, validating, launching, and reading finite-element wave simulations.
Solver execution requires a separately licensed fast solver.

Key Features
------------

- High-performance finite element modeling
- Support for time-domain and frequency-domain simulations
- Flexible model building and meshing capabilities
- Integration with popular data formats and visualization tools
- Cloud and HPC deployment support

Getting Started
---------------

If you're new to FrequenSolve, we recommend starting with:

- :doc:`installation`
- :doc:`quickstart`
- :doc:`tutorials/index`

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
