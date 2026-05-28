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
   * - Run with a local solver
     - ``python -m pip install "frequensolve[parallel]"``
     - Requires an installed :term:`fast solver` and :term:`FS_SOLVER_PATH`.
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

   python -m pip install "frequensolve[visual]"
   python -m pip install "frequensolve[parallel]"
   python -m pip install "frequensolve[hpc]"
   python -m pip install "frequensolve[cloud]"
   python -m pip install "frequensolve[seismic-io]"
   python -m pip install "frequensolve[fast-fft]"
   python -m pip install "frequensolve[inversion]"

Extras can be combined:

.. code-block:: bash

   python -m pip install "frequensolve[visual,cloud,seismic-io]"

Solver Access
-------------

The Python package does not include the solver executable. To run the solver,
first :ref:`configure a site <site-configuration>` to run the solver on. A
valid license is required to run the solver. Our `cloud site
<https://frequensol.com/pricing>`__ is the quickest way to get started.

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
