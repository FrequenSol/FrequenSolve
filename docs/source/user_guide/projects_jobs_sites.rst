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

Relocating Simulations
----------------------

Use ``simulation.relocate(new_project_path)`` when an existing simulation's
project files have already been moved to another root. Relocation updates the
public ``project_path`` and remaps project-local file references; it does not
copy files. Use ``Project.copy(source, destination)`` when FrequenSolve should
copy the complete project tree as well.

.. code-block:: python

   sim.relocate("./moved_project")

Advanced exporters can select a temporary output location without relocating
the simulation by creating an explicit context:

.. code-block:: python

   ctx = sim.export_context(
       project_path="./staging",
       rel_path="inputs/simple_acoustic",
   )
   payload = sim.to_fs(ctx)

The default remains ``<project_path>/simulations/<simulation name>``.

Simulation Studies
------------------

Use a simulation study when several simulations share one definition but vary
in receiver layout, sources, model, or other authoring values. Each parameter
maps readable choice labels to the values used while building a case:

.. code-block:: python

   study = project.study(
       "survey_design",
       name_template="base__{receiver}__{source}__{model}",
       receiver={
           "coarse": coarse_receiver_coords,
           "dense": dense_receiver_coords,
       },
       source={
           "explosive": explosive_sources,
           "vertical_force": vertical_force_sources,
       },
       model={
           "reference": reference_model,
           "smoothed": smoothed_model,
       },
   )

Define the simulation once. ``case.clone(...)`` creates an independent copy of
an existing simulation without saving or reloading it. Selected parameter
values are exposed as normal case attributes:

.. code-block:: python

   @study.simulation
   def build(case):
       sim = case.clone(base_simulation)
       sim.model = case.model
       sim.acquisition.sources = case.source
       sim.acquisition.receivers["surface"].coordinates = case.receiver
       return sim

Calling ``materialize()`` with no cases creates the Cartesian product in
parameter declaration order. The example above produces eight ordinary,
project-owned ``SeismicSimulation`` objects:

.. code-block:: python

   study.preview()  # inspect names and selections without building
   simulations = study.materialize()

Pass explicit cases when only selected combinations are meaningful:

.. code-block:: python

   simulations = study.materialize(
       cases=[
           study.case(
               receiver="coarse",
               source="explosive",
               model="reference",
           ),
           study.case(
               receiver="dense",
               source="vertical_force",
               model="smoothed",
           ),
       ]
   )

For a fresh definition, use
``case.new_simulation(physics="acoustic", dimension=2)`` instead of cloning a
base. Both helpers keep simulations detached until every requested case builds
successfully.

``name_template`` fields use choice labels, never the underlying values.
``{study}`` inserts the study name and ``{index}`` inserts the stable zero-based
case index, including normal Python format specifications such as
``{index:03d}``. Without a custom template, names include the study, parameter
names, and labels automatically. FrequenSolve rejects unknown fields, unsafe
path characters, duplicate rendered names, and collisions with simulations
already in the project before invoking the builder.

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
           fs.VtkOutput.domain(
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

Direct constructors such as ``fs.LocalSite(...)`` and ``fs.AWSSite(...)``
remain available when code needs to pin a backend. ``fs.Stampede3Site(...)``
remains as a compatibility adapter; new Stampede3 profiles use generic
``SlurmSite`` with ``preset = "stampede3"``.

All sites share the same handle/result lifecycle:

.. code-block:: python

   run = site.submit(job)
   result = run.wait()
   traces = result.traces(upscale=4)
   logs = result.logs()

Inspect ``result.status``, ``result.successful``, and ``result.logs()`` when
you need to decide what to do after a completed, failed, cancelled, or timed-out
run.

Run validation explicitly with ``job.validate(raise_errors=True)`` when you
want a local preflight check before submitting to a site.

By default, ``site.submit(job)`` skips runs whose expected outputs are already
current. Use ``site.submit(job, skip=False)`` (or ``skip="false"`` when a
string value is needed) to force a new run; local and
:term:`SLURM` sites pass ``--fresh`` to the :term:`fast solver` so solver-side output reuse is
disabled too.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Site
     - Use
   * - ``LocalSite``
     - Runs through local :term:`Dask` workers. Requires an installed
       :term:`fast solver` configured in the local site profile.
   * - ``AWSSite``
     - Runs on FrequenSol cloud infrastructure. Most users use this because solver installation is managed remotely.
   * - ``SlurmSite``
     - Runs on configured :term:`HPC` systems through :term:`SSH` and
       :term:`SLURM`. Requires site credentials and a solver executable on the
       cluster. Built-in presets can supply standard cluster and partition
       information.

Only sites with access to the :term:`fast solver` can execute jobs. The Python
package can author, save, inspect, and load projects without a solver
installation.
