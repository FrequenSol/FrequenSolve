Projects, Jobs, And Sites
=========================

The high-level workflow is:

1. Create a ``Project``.
2. Create a project-owned simulation with ``project.new_simulation(...)``.
3. Add model, mesh, boundary conditions, acquisition, and numerics.
4. Create a ``TimeDomainJob`` or ``FrequencyDomainJob``.
5. Submit the job to a site and read the ``RunResult``.

Primary tutorials:

- :download:`Acoustic modeling <../../../examples/tutorials/01_modeling_basics/01_acoustic.ipynb>`
- :download:`AWS site <../../../examples/tutorials/02_sites/01_aws_site.ipynb>`
- :download:`HPC sites <../../../examples/tutorials/02_sites/02_hpc_sites.ipynb>`
- :download:`Local site <../../../examples/tutorials/02_sites/03_local_site.ipynb>`
- :download:`Saving and loading projects/jobs <../../../examples/tutorials/02_sites/04_save_load_projects_jobs.ipynb>`

Project Layout
--------------

Project paths are normal filesystem directories. A saved project commonly
contains:

.. code-block:: text

   project.json
   simulations/
     simple_acoustic/
       simple_acoustic.json
       simple_acoustic.h5
   jobs/
     simple_acoustic/
       time/
         time.json
         logs/
         results/

``site.submit(job)`` saves the simulation and job before launch. Calling
``project.save()`` explicitly is still useful when you want to inspect generated
JSON/HDF5 inputs before running.

Loading Saved Work
------------------

Saved projects, simulations, and jobs are reusable objects. You do not need to
rerun the original authoring script just to inspect a model, submit a saved
job, or post-process results.

.. code-block:: python

   project = fs.Project.load("./scratch/tutorials/save_load_projects_jobs/project.json")
   sim = fs.SeismicSimulation.load(
       "./scratch/tutorials/save_load_projects_jobs/simulations/save_load_acoustic/save_load_acoustic.json"
   )
   job = fs.SimulationJob.load(
       "./scratch/tutorials/save_load_projects_jobs/jobs/save_load_acoustic/time/time.json"
   )

``Project.load(...)`` accepts either a project JSON file or a directory
containing one project JSON file. It restores the simulations listed in the
project JSON. Jobs are loaded separately by job JSON path because a project may
accumulate many trial jobs, QC jobs, and production jobs over time.

Use saved job loading for rerun notebooks and analysis notebooks:

.. code-block:: python

   loaded_job = fs.SimulationJob.load(job_file)
   result = site.submit(loaded_job).wait()

For project-owned jobs, you usually do not need to spell out the full path:

.. code-block:: python

   project.list_jobs()

   loaded_job = project.load_job(
       "time_axisymmetric_borehole",
       simulation="axisymmetric_borehole",
   )
   traces = loaded_job.traces.open(upscale=4)

``project.list_jobs()`` scans the saved ``jobs/<simulation>/<job>/`` tree and
returns one row per job. The most useful columns for notebook workflows are
``results_exist`` and ``results_current``. ``results_exist`` reports whether
the result directory contains traces, solver metadata, or other persisted
outputs. ``results_current`` reports whether those results still match the
saved job and simulation definitions, using the job fingerprint and saved file
hashes when available. Pass ``simulation="name"`` to limit the listing to one
simulation.

If you already recreated the same job object in Python and it has been saved or
run before, the object knows its project path:

.. code-block:: python

   saved_job = fs.SimulationJob.load(time_job)
   # equivalent:
   saved_job = time_job.load_saved()
   traces = saved_job.traces.open(upscale=4)

This keeps execution reproducible: the job JSON records the linked simulation,
frequency list, output requests, and result path that a site will stage.

Jobs
----

``TimeDomainJob`` defines a frequency sweep and reconstructs time-domain traces
through the trace API. ``FrequencyDomainJob`` runs explicit frequencies and is
the normal place to request ParaView outputs:

.. code-block:: python

   job = fs.FrequencyDomainJob(
       name="freq_20hz",
       simulation=sim,
       f_list=[20.0],
       outputs=[
           fs.ParaviewOutput(
               name="pv",
               fields=["pressure"],
               properties=["vp", "rho"],
           )
       ],
   )

Execution Sites
---------------

All sites share the same handle/result lifecycle:

.. code-block:: python

   run = site.submit(job)
   result = run.wait()
   traces = result.traces(upscale=4)
   logs = result.logs()

By default, ``site.submit(job)`` skips runs whose expected outputs are already
current. Use ``site.submit(job, force_run=True)`` to force a new run; local and
SLURM sites pass ``--fresh`` to the fast solver so solver-side output reuse is
disabled too.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Site
     - Use
   * - ``LocalSite``
     - Runs through local Dask workers. Requires an installed fast solver and ``FS_SOLVER_PATH``.
   * - ``AWSSite``
     - Runs on FrequenSol cloud infrastructure. Most users use this because solver installation is managed remotely.
   * - ``SlurmSite`` / ``Stampede3Site``
     - Runs on configured HPC systems through SSH and SLURM. Requires site credentials and a solver executable on the cluster.

Only sites with access to the fast solver can execute jobs. The Python package
can author, save, inspect, and load projects without a solver installation.
