Mesh Generation and Adaptivity
==============================

Meshes can be supplied from a file or generated from model geometry. For layered
models, generated meshes are recommended because the solver can preserve
geometry and adapt relative to surfaces, sources, receivers, and material
properties.

Related tutorials:

- :download:`Meshes versus generators <../../../tutorials/04_meshing/01_mesh_vs_generators.ipynb>`
  for generated meshes, supplied meshes, and mesh QC screenshots.
- :download:`Adaptivity fields <../../../tutorials/04_meshing/02_adaptivity_fields.ipynb>`
  for ``vadapt``, ``epw_mult``, ``hmin``, and ``hmax``.
- :download:`Gradings <../../../tutorials/04_meshing/03_gradings.ipynb>`
  for source, receiver, and surface refinement controls.

Generators and Supplied Meshes
------------------------------

Generated meshes use a ``BaseMeshGenerator`` subclass wrapped by
``MeshManager`` when added to a simulation:

.. code-block:: python

   sim += model.hex_mesh_generator([8, 4])
   sim.mesh.set_adapt(elems_per_wave=2.0, order=4)

The current external mesh path supports FrequenSolve's :term:`GMP` mesh format. Public
support for other mesh formats can be added as needed.

Root Patch Preparation
----------------------

``PatchSet`` prepares whole physical roots and acquisition apertures without
wave solves. It uses native curved geometry and the existing site scheduler:

.. code-block:: python

   patches = fs.PatchSet.around_sources(
       shots_per_patch=4,
       max_offset=4 * fs.ureg.km,
       padding=2 * fs.ureg.km,
   )
   prepared = patches.prepare(sim, [3, 7.5], site=site, edge_samples=True)
   prepared.plot()

Distances require units. Omitting ``depth`` retains the full parent depth;
``depth=[0, 8] * fs.ureg.km`` requests a smaller interval. Receiver aperture is
evaluated separately for each shot. Explicit ``Patch(name=..., roots=...,
sources=...)`` entries use one-based IDs and must assign every physical shot
exactly once. The native inventory in ``prepared.geometry["roots"]`` is a compact
object of arrays indexed by one-based parent root ID: ``cell`` and ``domain``
(the authored GMP domain ID) hold one integer per root, and the axis-major
``lower``/``upper`` and ``sampled_lower``/``sampled_upper`` bounds hold one
metre array per dimension with ``root_count`` values each. Patches supply
requested and added roots and realized bounds. ``edge_samples=True`` adds nine
native samples per root edge (``edge_count`` and ``edge_points``, packed root by
root) for ``prepared.plot()``; it is off by default because the samples scale
with the parent, and plotting without them raises. ``prepared.acquisition``
records original shot/receiver IDs and retained/excluded pair counts;
``prepared.jobs`` exposes the preparation jobs.

A scalar ``max_offset`` selects a radial aperture. A vector such as
``max_offset=[4000, 2000, 500] * fs.ureg.m`` selects a box with one maximum
offset per coordinate direction. It must match the simulation dimension.
To run the selected patches, use:

.. code-block:: python

   job = fs.FrequencyDomainJob("forward", sim, [3, 7.5], patches=patches)
   result = site.run(job, check=True)
   for child in result.jobs:
       traces = child.traces.open()

Dispatch preserves physical shot IDs and each retained group's original receiver
catalog. Sparse surveys select the per-shot rows; groups with no retained rows
are omitted from that child. ``result.prepared`` exposes the selection report,
and ``result.runs`` contains the child run results in the same order. Saved jobs
retain their patch policy; identical reloads reuse committed preparation and child
outputs. Changes to the model or selection invalidate the affected results.
Direct ``site.submit(job)`` is not supported for a
patch wrapper; ``site.run(job)`` prepares and runs its child jobs sequentially.

``prepared.freeze_stage(control_state, name="stage_1", directory=stage_dir)``
atomically publishes the prepared parent geometry, acquisition selection,
canonical ``ControlStateFile`` baseline, simulation, and independent copies of
referenced input files. It uses the preparation's simulation and frequency band.
The returned snapshot retains the exact manifest identity and verifies every
file before reuse; moving the bundle preserves its identity. Published stages
are never overwritten. Experimental low-level FWI jobs can carry this bundle
alongside an explicit candidate control state; they must use the pinned simulation
definition. For one frequency, an experimental ``FWIOperatorJob`` can capture
its initial hp mesh, solver refinements and final hp mesh with
``stage_mesh="capture"`` and ``action="linearize"`` using the stage baseline.
Capture uses the configured objective context and stops before wave solves.
``snapshot.publish_mesh(capture_manifest, directory)`` atomically publishes the
companion; pass the returned object as ``stage_mesh=`` for candidate jobs.
Replay rejects changed execution settings, geometry parameters, source positions
or committed files, and verifies the final realized mesh. Staged initialization
retains the authored unit/scaling context and normal frequency-dependent PML
sizing. The stage band does not impose a common PML thickness. Shared meshed
controls with patch PML require both artifacts and retain their parent material
basis. Use ``gram_derivative="total"`` for derivatives of freshly assembled DPG
candidate solves. Accurate objective finite differences can require double
solver and MUMPS precision with tighter convergence. Flat and curved layered
acoustic/coupled stage replay is covered in 2D and 3D; general curved extrusion
validation remains deferred. ``ImagingProblem(patches=...)`` manages these
artifacts and reopens them on FWI checkpoint resume. Remote staging enumerates
the complete verified stage and mesh bundles without rewriting their bytes.
Keep these bundles and the candidate state inside the job project so their
relative layout is preserved on the execution site.

The current preparation path supports locally available physical point shots
and dense point-receiver groups in global Cartesian coordinates. The preview
shows physical roots, with heavier core edges, dotted material-support additions,
and dashed topology additions. Named ``Model.property_spaces`` artifacts must
already exist for the full parent. Preparation includes complete core-coefficient
supports before padding and records basis identities and canonical coefficient
IDs in each patch's ``material_coverage``. This covers entire core roots in each
named space; per-control masks are not yet applied. Changed artifacts invalidate
cached preparation. Preparation verifies every assigned source and retained
receiver against the actual native curved roots and rejects points that cannot
be located, naming their acquisition IDs. PML construction is not yet certified.
``ImagingProblem(patches=...)`` uses this preparation interface for composite FWI
and local updates; see the imaging guide for stage and checkpoint behavior.

Adaptivity
----------

``MeshManager.set_adapt(...)`` controls wavefield-aware mesh sizing:

.. code-block:: python

   sim.mesh.set_adapt(
       elems_per_wave=2.0,
       order=4,
       f_low=5.0,
       f_high=30.0,
       hmin=0.005,
       hmax=0.08,
   )

``order`` is the initial :term:`polynomial order` assigned to the root mesh.
``elems_per_wave`` is the requested minimum element count per wavelength after
adaptation. The practical points per wavelength are roughly
``order * elems_per_wave`` before details of element family and field basis are
considered.

Material Sizing Fields
----------------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Property
     - Effect
   * - ``vadapt``
     - Overrides the material wavespeed used for local wavelength sizing.
   * - ``epw_mult``
     - Multiplies the requested :term:`EPW` target locally. Values are clamped to at least ``1.0``.
   * - ``hmin``
     - Local minimum element size in length units.
   * - ``hmax``
     - Local maximum element size in length units; can force refinement independent of frequency.

Gradings
--------

Distance gradings refine around acquisition geometry:

.. code-block:: python

   sim.mesh.set_source_grading(d0=0.01, d1=0.08, factor=2.0, power=2.0)
   sim.mesh.set_receiver_grading(d0=0.01, d1=0.05, factor=1.5)

:term:`Surface gradings <surface grading>` refine around named model surfaces:

.. code-block:: python

   sim.mesh.add_surface_grading(
       "interface",
       d0=0.0,
       d1=0.04,
       factor=2.0,
       power=2.0,
       mode="abs_band",
   )

``power`` controls the transition curve between ``d0`` and ``d1``. The default
``power=1`` is linear; larger values keep stronger refinement closer to the
feature before relaxing toward the background size. ``factor`` and ``power``
may be scalars or per-axis dictionaries keyed by the active global
coordinate-system axis names, such as ``{"offset": 2.0, "depth": 1.5}``.
This matches the style used by ``elems_per_wave`` and ``order``.

Initial meshes do not need to resolve the final wavefield. A coarse generated
mesh plus :term:`mesh adaptivity` is the preferred starting point for most
tutorials.
