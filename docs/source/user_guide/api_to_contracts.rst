Python API and Solver Contracts
===============================

FrequenSolve notebooks are written with Python objects, but jobs run from
versioned :term:`solver contracts <solver contract>`. The :term:`Python API` is
the authoring layer; the exported :term:`JSON`/:term:`HDF5` files are the
auditable interface consumed by launchers and the :term:`fast solver`.

Related tutorials:

- :download:`Acoustic modeling <../../../examples/tutorials/01_modeling_basics/01_acoustic.ipynb>`
  for the first :term:`project`, :term:`simulation`, :term:`job`, and result
  artifacts.
- :download:`Layered models <../../../examples/tutorials/03_velocity_model_building/03_layered_models.ipynb>`
  for exported material and geometry contracts.
- :download:`Traces <../../../examples/tutorials/06_outputs/01_traces.ipynb>`
  for :term:`HDF5` trace output and :term:`TraceDataset <trace dataset>` reads.
- :download:`ParaView and VTK <../../../examples/tutorials/06_outputs/02_paraview_vtk.ipynb>`
  for job-owned visualization output contracts.

Authoring Flow
--------------

The core workflow is intentionally repetitive across tutorials:

.. list-table::
   :header-rows: 1
   :widths: 24 36 40

   * - Python object
     - Exported contract area
     - What to inspect
   * - ``Project``
     - Project directory, logs, saved simulations, jobs, results
     - Paths are stable and job names describe the experiment.
   * - ``project.new_simulation(...)``
     - ``fs-simulation-1``
     - ``physics``, ``dimension``, units, coordinate systems, model, mesh, BCs, acquisition, solver settings.
   * - ``LayeredModel``
     - ``fs-material-model-1``
     - Surfaces, material :term:`subdomains <subdomain>`, :term:`mesh block IDs <mesh block ID>`, layer physics, and canonical property names.
   * - ``MeshManager`` / mesh generator
     - ``Mesh`` and ``Mesh/adapt``
     - Generator bounds/counts, ``elems_per_wave``, ``order``, frequency bounds, ``hmin``/``hmax``.
   * - ``BoundaryCondition``
     - ``BCs``
     - ``conditions``, ``boundaries``, and :term:`PML` settings.
   * - ``Acquisition``
     - ``fs-acquisition-2``
     - :term:`Source geometry`, optional :term:`source encoding`, physical-point versus RHS-field counts, :term:`receiver groups <receiver group>`, devices/components, dense/sparse sampling, coordinate units/systems.
   * - ``TimeDomainJob`` / ``FrequencyDomainJob``
     - ``fs-job-1`` plus job-owned outputs
     - Frequency list or band, result path, logs, traces, and visualization requests.
   * - ``ParaviewOutput``
     - ``fs-output-config-1``
     - Output target, fields, properties, sources, :term:`PML` visibility, :term:`upscaling`, and :term:`VTK` files.
   * - ``TraceDataset``
     - ``fs_seismic_trace_store_v1`` HDF5 trace output
     - Groups, :term:`components <component>`, source ids, dense/sparse layout,
       frequency and :term:`time-domain` reads.

Material Names
--------------

Layer property names are catalog names. The API accepts convenient Python
capitalization such as ``"Vp"`` and ``"Rho"`` and normalizes those names for the
solver, but the underlying contract uses canonical names such as ``vp``,
``rho``, ``k_solid``, ``rho_fluid``, and ``kappa``.

For :term:`poroelastic` models, use the contract names shown in the tutorials:
``k_dry``, ``mu_dry``, ``k_solid``, ``k_fluid``, ``rho_solid``,
``rho_fluid``, ``porosity``, ``tortuosity``, ``kappa``, and ``viscosity``.
Do not use descriptive aliases such as ``permeability`` or
``fluid_viscosity`` for poroelastic hydraulic properties; ``permeability`` is
reserved for EM magnetic permeability in the material catalog.

Units and Coordinates
---------------------

Numeric values are interpreted in the :term:`simulation`'s configured unit
system. :term:`Pint` quantities, :term:`xarray` metadata, and coordinate-aware
objects make those units explicit and are recommended in release-quality
notebooks.

Coordinate-aware values may carry ``value``, ``units``, and ``system``. Raw
arrays remain useful for flat introductory models, but explicit coordinate
systems are safer when working with topography, reduced 2D slices of 3D
coordinates, axisymmetric models, or mixed coordinate conventions.

Dense and Sparse Traces
-----------------------

Dense receiver groups define :term:`trace` identity by dataset shape, source
ids, receiver group metadata, and component catalogs. Sparse receiver groups
write trace identity tables with ``trace_id``, ``source_id``, ``receiver_id``,
``component``, and ``weight``. Internal sample maps and point ranges are not
public trace metadata.

Use ``traces.summary``, ``traces.groups``, ``traces.components(group)``, and
``traces.sources(group)`` before plotting. That habit makes notebooks robust
when moving from a dense teaching survey to a sparse production survey.

Inspection Habit
----------------

Before a remote or expensive solve, inspect the generated inputs:

.. code-block:: python

   project.save()
   sim_payload = sim.to_fs()
   mesh_payload = sim.mesh.to_fs(sim.export_context())
   acq_payload = sim.acquisition.to_fs(sim.export_context())

For an acquisition, confirm that ``schema`` is ``fs-acquisition-2``,
``source_geometry`` contains the physical catalog, and optional
``source_encoding`` contains the intended RHS fields. Current exports never
contain the legacy ``source_groups`` key.

After a solve, inspect the result before plotting:

.. code-block:: python

   logs = result.logs()
   outputs = result.output_files(existing=True)
   traces = result.traces(upscale=4)
   traces.summary

This is the quickest way to distinguish an authoring mistake from a solver or
site problem.
