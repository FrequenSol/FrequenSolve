Quickstart
==========

This page shows the compact FrequenSolve :term:`Python API` workflow: create a
:term:`project`, define a :term:`simulation`, submit a :term:`job`, and inspect
outputs. For deeper runnable examples with plots and output inspection, see the
:doc:`tutorial collection
<tutorials/index>`.

Core Workflow
-------------

FrequenSolve scripts usually follow the same authoring pattern:

1. Create a :class:`frequensolve.Project` to own paths, logs, and simulations.
2. Create a project-owned :term:`simulation` with ``project.new_simulation(...)``.
3. Add a model, mesh generator, boundary conditions, acquisition geometry, and
   solver settings.
4. Create a :term:`time-domain` or :term:`frequency-domain` job.
5. Submit the job to a :term:`site` and read outputs from the returned result.

See :doc:`user_guide/site_configuration` and the site tutorials for deployment
details.

Minimal Acoustic Model
----------------------

The example below builds a two-layer acoustic model. Lengths are authored in
kilometers, velocities in kilometers per second, and density in grams per cubic
centimeter. :term:`Pint` quantities make those choices explicit in the script,
while the :term:`simulation` unit system controls how values are exported to the
solver.

.. code-block:: python

   import numpy as np
   import frequensolve as fs

   u = fs.ureg

   project = fs.Project(
       name="project",
       pretty_name="Quickstart Acoustic",
       path="./scratch/quickstart_acoustic",
       log_level="INFO",
       log_to_console=True,
   )

   sim = project.new_simulation(
       name="quickstart_acoustic",
       physics="acoustic",
       dimension=2,
       units={
           "length": "km",
           "velocity": "km/s",
           "density": "g/cm^3",
       },
   )

   model = fs.LayeredModel(name="model", dimension=2, x_limits=[0.0, 1.0])
   model.add_surface(name="top", depth=0.0 * u.km)
   model.add_layer(
       name="upper_layer",
       properties={
           "Vp": 2.0 * u.km / u.s,
           "Rho": 2.2 * u.g / u.cm**3,
       },
   )
   model.add_surface(name="interface", depth=0.25 * u.km)
   model.add_layer(
       name="lower_layer",
       properties={
           "Vp": 2.8 * u.km / u.s,
           "Rho": 2.4 * u.g / u.cm**3,
       },
   )
   model.add_surface(name="bottom", depth=0.5 * u.km)
   sim += model

Meshes, Boundaries, and Acquisition
-----------------------------------

The mesh generator preserves the model geometry, while adaptivity settings
control the element sizing used for the job. ``order`` is the :term:`polynomial order`
of the finite-element basis. ``elems_per_wave`` is the :term:`EPW` target; the
corresponding nodal points per wavelength is approximately
``order * elems_per_wave + 1``.

Boundary conditions are attached by named exterior boundaries. In this acoustic
example, the top is pressure-free and the sides/bottom use :term:`PML`
absorption.

.. code-block:: python

   sim += model.hex_mesh_generator([8, 4])
   sim.mesh.set_adapt(elems_per_wave=2.0, order=4, f_low=5.0, f_high=30.0)
   sim.mesh.set_source_grading(d1=0.08, d0=0.02, factor=2.0)

   sim += fs.BoundaryCondition(
       conditions=["free"],
       boundaries=["z_min"],
   )
   sim += fs.BoundaryCondition(
       conditions=["pml"],
       boundaries=["x_min", "x_max", "z_max"],
       pml_wavelengths=0.75,
   )

   acq = fs.Acquisition()
   acq.add_sources(kind="scalar", coords=[[0.5, 0.025]])

   hydrophone = fs.ReceiverNode()
   hydrophone.add_component(name="p", field="pressure")

   receiver_coords = [[x, 0.05] for x in np.linspace(0.0, 1.0, 101)]
   acq.add_receiver_group(name="surface", device=hydrophone, coords=receiver_coords)
   sim += acq

   sim += fs.Discretization()
   sim += fs.SolverConfig(tolerance=1.0e-4, grids=3)

Build and Inspect Before Running
--------------------------------

Everything above can be authored and saved without access to the :term:`fast solver`.
Saving the project writes inspectable project and simulation files under
``./scratch/quickstart_acoustic``:

.. code-block:: python

   project.save()

At this point you should see ``project.json`` plus a saved simulation
:term:`JSON`/:term:`HDF5`
pair under the project directory. Use this checkpoint when you want to inspect
generated inputs before spending time or cloud/:term:`HPC` resources on a solver run.

The next sections submit jobs to a configured :term:`site`. If ``fs.Site()``
creates ``~/.frequensolve/site.toml`` and asks you to review it, follow
:doc:`user_guide/site_configuration` and rerun the same code after the profile
is configured.

Run a Time-Domain Job
---------------------

``site.submit(job)`` returns a run handle. Calling ``wait()`` blocks until the
site reaches a terminal state and returns a ``RunResult`` with status, logs, and
output helpers. Result paths, logs, trace files, and other outputs are written
under the project directory.

.. code-block:: python

   site = fs.Site()
   job = fs.TimeDomainJob(
       name="time",
       simulation=sim,
       f_min=0.0,
       f_max=45.0,
       T_max=1.0,
   )

   result = site.submit(job).wait()
   traces = result.traces(upscale=4)
   traces.summary

Expected output: ``traces.summary`` lists receiver groups, components, source
ids, frequency samples, and time-domain sampling metadata. The project
directory should now contain a saved job, logs, and result files under
``jobs/quickstart_acoustic/time/``.

The trace reader returns a :term:`trace dataset <trace dataset>`, backed by the
:term:`HDF5` trace output written by the job. Wavelets are applied when traces are read
so that the same :term:`frequency-domain` solve can be inspected with different source
time functions.

.. code-block:: python

   wavelet = fs.RickerWavelet(f=12.0)
   group = traces.groups[0]
   component = traces.components(group)[0]
   source = traces.sources(group)[0]
   gather = traces.td(group, component, source, wavelet, upscale=4)

   gather.plot(x="time", hue="receiver", add_legend=False)

Expected output: the plot call draws one time series per receiver for the
selected component and source. If the notebook backend is non-interactive, save
the figure with matplotlib after the plot is created.

Run a Frequency-Domain ParaView Job
-----------------------------------

:term:`Frequency-domain` jobs can request additional outputs, including
:term:`ParaView`/:term:`VTK` files for inspecting the mesh, fields, :term:`PML`,
material properties, and source locations. These files can be opened in
ParaView directly. Python visualization with :term:`PyVista` is useful for
screenshots and quality control, but ParaView has the richer interactive
feature set.

.. code-block:: python

   site_fd = fs.Site()
   pv_job = fs.FrequencyDomainJob(
       name="freq_25hz",
       simulation=sim,
       f_list=[25.0],
       outputs=[
           fs.ParaviewOutput(
               name="pv",
               path="paraview",
               fields=["pressure", "velocity_z"],
               properties=["vp", "rho", "Subdomain"],
               show_pml=True,
               upscale=1,
               order=2,
           ),
       ],
   )

   pv_result = site_fd.submit(pv_job).wait()
   pv_result.output_files(base="pv", suffix=".vtu", existing=True)

Expected output: ``output_files(...)`` returns the generated ``.vtu`` files for
the ``pv`` :term:`output request`. Open those files in ParaView for interactive
inspection, or use the ``visual`` extra for lightweight PyVista checks in
Python.

Next Steps
----------

- Continue with :doc:`tutorials/index` for runnable notebooks organized by
  modeling, sites, meshing, surveys, outputs, and performance.
- Use :doc:`user_guide/site_configuration` when you are ready to configure
  ``fs.Site()`` for cloud, local, or HPC execution.
- Use :doc:`user_guide/index` when you need a reference for a specific topic
  from this page, such as units, mesh adaptivity, surveys, or output readers.
