Mesh Generation And Adaptivity
==============================

Meshes can be supplied from a file or generated from model geometry. For layered
models, generated meshes are recommended because the solver can preserve
geometry and adapt relative to surfaces, sources, receivers, and material
properties.

Primary tutorials:

- :download:`Meshes versus generators <../../../examples/tutorials/04_meshing/01_mesh_vs_generators.ipynb>`
- :download:`Adaptivity fields <../../../examples/tutorials/04_meshing/02_adaptivity_fields.ipynb>`
- :download:`Gradings <../../../examples/tutorials/04_meshing/03_gradings.ipynb>`

Generators And Supplied Meshes
------------------------------

Generated meshes use a ``BaseMeshGenerator`` subclass wrapped by
``MeshManager`` when added to a simulation:

.. code-block:: python

   sim += model.hex_mesh_generator([8, 4])
   sim.mesh.set_adapt(elems_per_wave=2.0, order=4)

The current external mesh path supports FrequenSolve's GMP mesh format. Public
support for other mesh formats can be added as needed.

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

``order`` is the initial polynomial order assigned to the root mesh.
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
     - Multiplies the requested EPW target locally. Values are clamped to at least ``1.0``.
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

Surface gradings refine around named model surfaces:

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
mesh plus adaptivity is the preferred starting point for most tutorials.
