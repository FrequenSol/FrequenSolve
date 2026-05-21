Quickstart
==========

This page gives a compact, current FrequenSolve workflow. The tutorial
collection in :doc:`tutorials/index` is the canonical place to learn the API in
depth; start with the :download:`acoustic modeling tutorial
<../../examples/tutorials/01_modeling_basics/01_acoustic.ipynb>` when you want
the full walkthrough with plots and output inspection.

Core Workflow
-------------

FrequenSolve scripts usually follow the same authoring pattern:

1. Create a :class:`frequensolve.Project` to own paths, logs, and simulations.
2. Create a project-owned simulation with ``project.new_simulation(...)``.
3. Add a model, mesh generator, boundary conditions, acquisition geometry, and
   solver settings.
4. Create a time-domain or frequency-domain job.
5. Submit the job to a site and read outputs from the returned result.

The local site can only run on machines where the fast solver is installed.
Cloud and HPC sites use the same job lifecycle but different site
configuration; see :doc:`user_guide/projects_jobs_sites` and the site
tutorials for deployment details.

Minimal Acoustic Model
----------------------

The example below builds a two-layer acoustic model. Lengths are authored in
kilometers, velocities in kilometers per second, and density in grams per cubic
centimeter. Pint quantities make those choices explicit in the script while the
simulation unit system controls how values are exported to the solver.

.. code-block:: python

   from pathlib import Path

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

Meshes, Boundaries, And Acquisition
-----------------------------------

The mesh generator preserves the model geometry, while adaptivity settings
control the element sizing used for the job. ``order`` is the polynomial order
of the finite-element basis. ``elems_per_wave`` is the target number of
elements per shortest wavelength; the corresponding nodal points per wavelength
is approximately ``order * elems_per_wave + 1``.

Boundary conditions are attached by named exterior boundaries. In this acoustic
example, the top is pressure-free and the sides/bottom use PML absorption.

.. code-block:: python

   sim += model.hex_mesh_generator([8, 4])
   sim.mesh.set_adapt(elems_per_wave=2.0, order=4, f_low=5.0, f_high=30.0)
   sim.mesh.set_source_grading(d1=0.08, d0=0.02, mult=2.0)

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
   acq.add_source_group(kind="scalar", coords=[[0.5, 0.025]])

   hydrophone = fs.ReceiverNode()
   hydrophone.add_component(name="p", field="pressure")

   receiver_coords = [[x, 0.05] for x in np.linspace(0.0, 1.0, 101)]
   acq.add_receiver_group(name="surface", device=hydrophone, coords=receiver_coords)
   sim += acq

   sim += fs.Discretization()
   sim += fs.SolverConfig(tolerance=1.0e-4, grids=3)

Run A Time-Domain Job
---------------------

Jobs are strict: ``site.submit(job).wait()`` raises if Python validation fails
or if the solver reports a failure. Result paths, logs, trace files, and other
outputs are written under the project directory.

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

The trace reader returns a :class:`frequensolve.TraceDataset`, backed by the
HDF5 trace output written by the job. Wavelets are applied when traces are read
so that the same frequency-domain solve can be inspected with different source
time functions.

.. code-block:: python

   wavelet = fs.RickerWavelet(f=12.0)
   group = traces.groups[0]
   component = traces.components(group)[0]
   source = traces.sources(group)[0]
   gather = traces.td(group, component, source, wavelet, upscale=4)

   gather.plot(x="time", hue="receiver", add_legend=False)

Run A Frequency-Domain ParaView Job
-----------------------------------

Frequency-domain jobs can request additional outputs, including ParaView/VTK
files for inspecting the mesh, fields, PML, material properties, and source
locations. These files can be opened in ParaView directly. Python visualization
with PyVista is useful for screenshots and quality control, but ParaView has
the richer interactive feature set.

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

Next Steps
----------

- :download:`01 Acoustic Modeling <../../examples/tutorials/01_modeling_basics/01_acoustic.ipynb>`
  expands this quickstart with diagrams, trace plots, logs, and project layout.
- :doc:`user_guide/physics_materials_boundaries` lists supported physics,
  material properties, and boundary conditions.
- :doc:`user_guide/api_to_contracts` explains how Python objects map to saved
  solver contracts and result artifacts.
- :doc:`user_guide/velocity_models_coordinates` explains units, coordinate
  systems, and layered model geometry.
- :doc:`user_guide/projects_jobs_sites` covers local, AWS, and HPC execution.
