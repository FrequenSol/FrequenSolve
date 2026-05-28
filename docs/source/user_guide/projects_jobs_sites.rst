Projects, Jobs, and Sites
=========================

The high-level workflow is:

1. Create a ``Project``.
2. Create a project-owned :term:`simulation` with ``project.new_simulation(...)``.
3. Add model, mesh, boundary conditions, acquisition, and numerics.
4. Create a ``TimeDomainJob`` or ``FrequencyDomainJob``.
5. Submit the job to a :term:`site` and read the :term:`run result`.

Related tutorials:

- :download:`Acoustic modeling <../../../examples/tutorials/01_modeling_basics/01_acoustic.ipynb>`
  for the first complete :term:`project` and job workflow.
- :download:`AWS site <../../../examples/tutorials/02_sites/01_aws_site.ipynb>`
  for FrequenSol Cloud authentication and result fetching.
- :download:`HPC sites <../../../examples/tutorials/02_sites/02_hpc_sites.ipynb>`
  for :term:`SSH`/:term:`SLURM` execution.
- :download:`Local site <../../../examples/tutorials/02_sites/03_local_site.ipynb>`
  for local :term:`Dask`-backed execution with an installed solver.
- :download:`Saving and loading projects/jobs <../../../examples/tutorials/02_sites/04_save_load_projects_jobs.ipynb>`
  for reopening saved projects, simulations, jobs, and results.

Project Layout
--------------

:term:`Project` paths are normal filesystem directories. A saved project commonly
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
:term:`JSON`/:term:`HDF5` inputs before running.

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
accumulate many trial jobs, :term:`QC` jobs, and production jobs over time.

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
saved job and simulation definitions, using the :term:`rerun fingerprint` and
saved file hashes when available. Pass ``simulation="name"`` to limit the listing to one
simulation.

If you already recreated the same job object in Python and it has been saved or
run before, the object knows its project path:

.. code-block:: python

   saved_job = fs.SimulationJob.load(time_job)
   # equivalent:
   saved_job = time_job.load_saved()
   traces = saved_job.traces.open(upscale=4)

This keeps execution reproducible: the job JSON records the linked simulation,
frequency list, :term:`output requests <output request>`, and result path that
a site will stage.

Jobs
----

``TimeDomainJob`` defines a frequency sweep and reconstructs :term:`time-domain`
traces through the trace API. ``FrequencyDomainJob`` runs explicit frequencies
and is the normal place to request :term:`ParaView output`:

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

Site Selection
--------------

The standard way to select an execution backend is ``fs.Site()`` with a user
configuration file. See :doc:`site_configuration` for setup steps and the full
``site.toml`` file format. See :doc:`frequensolve_directory` for the
user-local storage layout.

Use the same Python code for local, cloud, or :term:`HPC` execution:

.. code-block:: python

   site = fs.Site()

Set ``FREQUENSOLVE_SITE_CONFIG`` or pass ``fs.Site(config_path=...)`` when a
test, notebook, or shared workstation should use a different config file:

.. code-block:: python

   local = fs.Site(profile="local")
   shared = fs.Site(config_path="/path/to/site.toml", profile="cluster")

Direct constructors such as ``fs.LocalSite(...)``, ``fs.AWSSite(...)``, and
``fs.Stampede3Site(...)`` remain available when code needs to pin a backend.

All sites share the same handle/result lifecycle:

.. code-block:: python

   run = site.submit(job)
   result = run.wait()
   traces = result.traces(upscale=4)
   logs = result.logs()

``run.wait()`` raises when a run finishes as failed, cancelled, or timed out.
Use ``run.wait(check=False)`` to get the :term:`run result` anyway and inspect
``result.status`` or ``result.logs()`` before deciding what to do next.

By default, ``site.submit(job)`` skips runs whose expected outputs are already
current. Use ``site.submit(job, force_run=True)`` to force a new run; local and
:term:`SLURM` sites pass ``--fresh`` to the :term:`fast solver` so solver-side output reuse is
disabled too.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Site
     - Use
   * - ``LocalSite``
     - Runs through local :term:`Dask` workers. Requires an installed
       :term:`fast solver` and :term:`FS_SOLVER_PATH`.
   * - ``AWSSite``
     - Runs on FrequenSol cloud infrastructure. Most users use this because solver installation is managed remotely.
   * - ``SlurmSite`` / ``Stampede3Site``
     - Runs on configured :term:`HPC` systems through :term:`SSH` and
       :term:`SLURM`. Requires site credentials and a solver executable on the
       cluster.

Only sites with access to the :term:`fast solver` can execute jobs. The Python
package can author, save, inspect, and load projects without a solver
installation.
