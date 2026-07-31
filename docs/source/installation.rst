Installation
============

FrequenSolve Python is the authoring, orchestration, and output-reading
:term:`Python API`. Installing the package lets you build :term:`projects <project>`,
inspect exported solver inputs, load :term:`trace` outputs, and configure
execution :term:`sites <site>`. Running a :term:`job` also requires access to a
licensed :term:`fast solver` through a local, cloud, or :term:`HPC` site.

Basic Install
-------------

Install the released FrequenSolve Python API from :term:`PyPI` with pip:

.. code-block:: bash

   python -m pip install frequensolve

Because the repository is public, you can also install from a source checkout
when you want local examples, documentation sources, or editable development.
From the repository root:

.. code-block:: bash

   python -m pip install -e .

Optional Extras
---------------

Choose the smallest install that matches the workflow you need:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Workflow
     - Command
     - Notes
   * - Author and inspect projects
     - ``python -m pip install frequensolve``
     - Does not run the solver.
   * - Run on FrequenSol Cloud
     - ``python -m pip install "frequensolve[cloud]"``
     - Requires a FrequenSol Cloud account and license.
   * - Use the local simulation assistant MCP
     - ``python -m pip install "frequensolve[mcp]"``
     - Add ``cloud`` to the extras for self-scoped read-only Cloud monitoring.
       See :doc:`user_guide/simulation_assistant_mcp`.
   * - Run with a local solver
     - ``python -m pip install "frequensolve[parallel]"``
     - Requires an installed :term:`fast solver` configured in ``site.toml``.
   * - Run on an HPC cluster
     - ``python -m pip install "frequensolve[hpc]"``
     - Requires :term:`SSH`/:term:`SLURM` access and a solver installation on
       the cluster.
   * - Plot or inspect :term:`VTK`/:term:`VTU` outputs in Python
     - ``python -m pip install "frequensolve[visual]"``
     - For Python-based visualization workflows.
   * - Read or write :term:`SEG-Y`/:term:`ASDF` data
     - ``python -m pip install "frequensolve[seismic-io]"``
     - For seismic file import and export workflows.

All available user extras are:

.. code-block:: bash

   python -m pip install "frequensolve[visual]"      # plotting, VTK/PyVista helpers
   python -m pip install "frequensolve[parallel]"    # local Dask execution
   python -m pip install "frequensolve[hpc]"         # SSH and SLURM site support
   python -m pip install "frequensolve[cloud]"       # FrequenSol cloud backend
   python -m pip install "frequensolve[mcp]"         # local simulation-assistant MCP
   python -m pip install "frequensolve[seismic-io]"  # SEG-Y/ASDF export helpers
   python -m pip install "frequensolve[fast-fft]"     # pyFFTW acceleration
   python -m pip install "frequensolve[inversion]"    # PyLops-compatible operators
   python -m pip install "frequensolve[dev,docs]"    # tests and documentation builds

Extras can be combined:

.. code-block:: bash

   python -m pip install "frequensolve[visual,cloud,seismic-io]"

Solver Access
-------------

The Python package does not include the fast solver executable. To execute
jobs, :ref:`configure a site <site-configuration>` using one of the supported
backends. A valid license is required; the cloud site is the quickest managed
path to solver access.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Site
     - Requirement
   * - ``LocalSite``
     - A local solver binary configured with the profile's ``solver`` key.
   * - ``AWSSite``
     - FrequenSol cloud access and the ``cloud`` extra.
   * - ``SlurmSite`` / ``Stampede3Site``
     - SSH/SLURM access, the ``hpc`` extra, and a solver installation on the cluster.

Site paths, hosts, usernames, accounts, and scheduler defaults belong in
``~/.frequensolve/site.toml``. HPC passwords and SSH-key passphrases are
prompted securely and saved to the operating system keyring only after a
successful login. A project ``.env`` file is not required.
Scheduler resources are included in the Python package, so neither
``PYTHONPATH`` nor a source-checkout path is part of installation. Supported
solver builds contain their own compiled-in resources.

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

Check :term:`site configuration file` setup before running a solver job:

.. code-block:: python

   import frequensolve as fs

   site = fs.Site()
   print(type(site).__name__)

If this is the first ``fs.Site()`` call on the machine, FrequenSolve may create
``~/.frequensolve/site.toml`` and raise an exception asking you to review it.
Edit that file, then rerun the same check. See :doc:`user_guide/site_configuration`
for the configuration workflow.

Then run the :doc:`quickstart` or download the first tutorial notebook from
:doc:`tutorials/index`.
