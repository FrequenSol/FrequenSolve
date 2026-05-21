Installation
============

FrequenSolve Python is the authoring, orchestration, and output-reading SDK.
Installing the package lets you build projects, inspect exported solver inputs,
load trace outputs, and configure execution sites. Running a job also requires
access to a licensed fast solver through a local, cloud, or HPC site.

Basic Install
-------------

Install the core SDK with pip:

.. code-block:: bash

   python -m pip install frequensolve

For local source checkouts, install from the repository root:

.. code-block:: bash

   python -m pip install -e .

Optional Extras
---------------

Install extras for the workflows you need:

.. code-block:: bash

   python -m pip install "frequensolve[visual]"      # plotting, VTK/PyVista helpers
   python -m pip install "frequensolve[parallel]"    # Dask, SSH, and SLURM helpers
   python -m pip install "frequensolve[cloud]"       # FrequenSol cloud backend
   python -m pip install "frequensolve[seismic-io]"  # SEG-Y/ASDF export helpers
   python -m pip install "frequensolve[dev,docs]"    # tests and documentation builds

Extras can be combined:

.. code-block:: bash

   python -m pip install "frequensolve[visual,cloud,seismic-io]"

Solver Access
-------------

The Python package does not include the fast solver executable. To execute
jobs, configure ``~/.frequensolve/site.toml`` and create sites with
``fs.Site()``:

.. code-block:: toml

   [site]
   type = "aws"
   domain = "frequensolve.app"
   interactive = true

The ``type`` can select one of the supported backends:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Site
     - Requirement
   * - ``LocalSite``
     - A local solver binary and environment such as ``FS_SOLVER_PATH``.
   * - ``AWSSite``
     - FrequenSol cloud access and the ``cloud`` extra.
   * - ``SlurmSite`` / ``Stampede3Site``
     - SSH/SLURM access, the ``parallel`` extra, and a solver installation on the cluster.

Direct constructors remain available for code that intentionally targets a
specific backend.

The notebooks in :doc:`tutorials/index` use strict run cells. If the selected
site is not configured or the solver is unavailable, the cell should fail and
point you toward the relevant job/site logs.

Development Install
-------------------

For package development and documentation work:

.. code-block:: bash

   python -m pip install -e ".[dev,docs,visual]"

Build the documentation locally with:

.. code-block:: bash

   python -m sphinx -b html docs/source docs/build/html

Verification
------------

Verify that the Python package imports:

.. code-block:: python

   import frequensolve as fs

   print(fs.__version__)

Then run the :doc:`quickstart` or download the first tutorial notebook from
:doc:`tutorials/index`.
