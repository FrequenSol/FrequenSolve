Eikonal First Arrivals and Ray Tracing
======================================

Sauce exposes two frequency-independent acoustic kinematics workflows.
``EikonalJob`` computes native-mesh first-arrival fields and optional
source/receiver characteristics. ``RayTracingJob`` launches and integrates
geometric rays, including interface branching and receiver-aperture hits.
Both run as one solver task on one MPI rank and omit ``f_list`` and trace
packing.

Eikonal First Arrivals
----------------------

Select acquisition-backed or explicit sources and receivers. Vertex fields
are retained by default; characteristics require enabled receivers:

.. code-block:: python

   import frequensolve as fs

   job = fs.EikonalJob(
       name="first_arrivals",
       simulation=sim,
       sources=fs.EikonalSources.acquisition(["shot_001"]),
       receivers=fs.EikonalReceivers.acquisition(["receivers"]),
       products={
           "field": True,
           "characteristics": True,
           "max_points_per_characteristic": 10000,
           "max_characteristic_points": 1000000,
       },
   )

   completed = fs.Site().submit(job).wait(check=True)
   arrivals = completed.eikonal()

Sauce evaluates exact active geometry, samples one-sided isotropic ``vp``,
excludes PML cells, and solves on canonical native travel vertices. An
interface source can select its one-based physical incidence with
``EikonalSources.acquisition(..., incidence_slot=1)``. Use
``EikonalReceivers.disabled()`` for a field-only solve.

The result reader validates the ``fs-eikonal-output-1`` manifest, required
HDF5 groups, table sizes, source-major shapes, and characteristic offsets.
It normalizes vectors to ``(record, dimension)`` and travel-time tables to
``(source, record)``:

.. code-block:: python

   field = arrivals.field("shot_001")
   xyz = field.position
   first_arrival = field.travel_time
   receiver_times = arrivals.receiver_times_for_source("shot_001")

   for path in arrivals.iter_characteristics(source_id=field.source_id):
       print(path.receiver_id, path.ambiguous, path.position)

   ax = arrivals.plot(
       source="shot_001",
       field=True,
       characteristics=True,
       contours=True,
       invert_vertical=True,
   )

``EikonalResults.open(...)`` and ``fs.load(...)`` accept the output directory,
``manifest.json``, or authoritative HDF5 file. The plotter uses filled
travel-time contours plus characteristic paths in 2D and a colored native
vertex cloud plus paths in 3D.

Ray Tracing
-----------

``RayTracingJob`` exposes Sauce's public, frequency-independent ``raytrace``
workflow. Version 1 traces isotropic acoustic rays in two or three dimensions,
runs as one solver task on one MPI rank, and writes an authoritative
``fs-rays-1`` HDF5 product.

Create a Job
^^^^^^^^^^^^

Use the simulation's physical acquisition sources or provide explicit source
points. A launch rule is required:

.. code-block:: python

   import frequensolve as fs

   job = fs.RayTracingJob(
       name="kinematic_rays",
       simulation=sim,
       sources=fs.RaySources.acquisition(["shot_001"]),
       launch=fs.RayLaunch.fan2d(
           count=181,
           angle_min=-90.0,
           angle_max=90.0,
       ),
       integrator={
           "tau_max": {"value": 5.0, "units": "s"},
       },
       receivers={
           "enabled": True,
           "kind": "acquisition",
           "groups": ["receivers"],
           "capture_radius": {"value": 5.0, "units": "m"},
       },
   )

   run = fs.Site().submit(job)
   completed = run.wait(check=True)
   rays = completed.rays()

The Python configuration fills the public contract's integrator, branching,
weight, boundary, path-storage, output, and failure-policy sections with
bounded defaults. Supply any whole section as a mapping to override it. For
example, retain both branches with explicit budgets:

.. code-block:: python

   config = fs.RayTracingConfig(
       launch=fs.RayLaunch.fan2d(181, -90.0, 90.0),
       branching={
           "mode": "both",
           "max_interactions_per_ray": 16,
           "max_rays_per_source": 10000,
           "max_total_rays": 100000,
           "max_branch_depth": 8,
       },
   )
   job = fs.RayTracingJob("branched_rays", sim, config=config)

Three-dimensional simulations use ``RayLaunch.sphere``, ``hemisphere``,
``cone``, or explicit directions. ``fan2d`` is restricted to 2D.

Read and Inspect Results
^^^^^^^^^^^^^^^^^^^^^^^^

``job.results`` validates the manifest, required HDF5 groups, logical table
counts, and CSR offsets before exposing arrays:

.. code-block:: python

   rays = job.results
   print(rays.status, rays.counts)

   path = rays.path(ray_id=1)
   xy = path.position
   tau = path.travel_time
   events = rays.events_for_ray(1)
   hits = rays.receiver_hits_for_ray(1)

   for path in rays.iter_paths(source_id=1):
       print(path.ray_id, path.ray["reason_name"])

Open a result independently with ``fs.RayResults.open(...)``. The argument may
be the ray output directory, ``manifest.json``, or the authoritative HDF5 file.
Numeric status values should be decoded through the file's registry, for
example ``rays.code_name("reason", code)``.

Plot Rays
^^^^^^^^^

The same plotting helper handles 2D and 3D retained paths:

.. code-block:: python

   ax = rays.plot(
       color_by="travel_time",
       show_sources=True,
       show_receivers=True,
       show_hits=True,
   )

``color_by`` also accepts ``ray``, ``source``, ``status``, ``energy``, and
``branch_depth``. The optional VTP product is useful for ParaView, but HDF5 is
the numerical source of truth.

Choosing a Kinematics Workflow
------------------------------

Use Eikonal when you need deterministic first arrivals at every native vertex
or receiver. Its characteristics are discrete winning-stencil backtracks and
are useful for inspecting first-arrival paths. Use ray tracing when launch
angle, continuous path integration, interface reflection/transmission,
branching weights, termination events, or aperture hits are the quantity of
interest. Neither product is a receiver wavefield trace dataset.
