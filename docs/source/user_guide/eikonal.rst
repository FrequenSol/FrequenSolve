Eikonal First Arrivals
======================

``EikonalJob`` computes native-mesh first-arrival fields and optional
source/receiver characteristics. It runs as one solver task on one MPI
rank and omits ``f_list`` and trace packing.

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
