Velocity Models, Units, and Coordinates
=======================================

FrequenSolve models are authored in Python but exported to solver contracts
with explicit units, coordinate systems, surfaces, layers, and geometry. The
goal is to keep the model readable at authoring time while making the generated
inputs auditable before a run.

Related tutorials:

- :download:`Variable properties and units <../../../tutorials/03_velocity_model_building/01_variable_properties_units.ipynb>`
  for :term:`Pint` quantities and sampled :term:`xarray` properties.
- :download:`Coordinate systems <../../../tutorials/03_velocity_model_building/02_coordinate_systems.ipynb>`
  for topography and surface-relative coordinates.
- :download:`Layered models <../../../tutorials/03_velocity_model_building/03_layered_models.ipynb>`
  for ordered surfaces, layers, local features, and sampled QC views.

Property Forms
--------------

Layer properties may be constants, :term:`Pint` quantities, sampled arrays, or richer
property specifications. Use the simplest representation that still preserves
the information needed to inspect the model later.

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Property form
     - Best use
     - Unit behavior
   * - Number
     - Small examples in the simulation unit system.
     - Interpreted through the simulation's active unit configuration.
   * - :term:`Pint` quantity
     - Explicit dimensional values in tutorial and production scripts.
     - Serialized with ``value`` and ``units``.
   * - ``xarray.DataArray``
     - Sampled profiles, maps, and grids.
     - Coordinate attrs carry coordinate units; data attrs carry property units.
   * - Structured property
     - Coordinate-system-specific, file-backed, or expression-backed data.
     - Exported with explicit metadata so solver inputs remain inspectable.

Units
-----

The runtime units system uses dimensions rather than hard-coded unit choices.
Base dimension symbols are ``L`` for length, ``T`` for time, ``M`` for mass, and
``I`` for electric current. Unit expressions such as ``km/s``, ``g/cm^3``,
``GPa``, and ``m^2`` are parsed and converted into solver scales.

New examples should prefer explicit units for dimensional values:

.. code-block:: python

   model.add_layer(
       name="upper",
       properties={
           "Vp": 2.0 * fs.ureg.km / fs.ureg.s,
           "Rho": 2.2 * fs.ureg.g / fs.ureg.cm**3,
       },
   )

:term:`Simulation <simulation>` unit defaults are useful for compact examples, but explicit Pint
quantities make notebooks more robust when copied into projects with different
unit systems.

Coordinate Systems
------------------

Coordinate values may be raw arrays or structured values carrying ``value``,
``units``, and ``system``. Raw arrays are compact for global Cartesian
coordinates. Structured coordinate values are better when the point belongs to a
named coordinate system or when units must travel with the value.

.. list-table::
   :header-rows: 1
   :widths: 28 32 40

   * - Concept
     - API pattern
     - Use
   * - Global Cartesian coordinates
     - ``[[x, z], ...]`` or Pint arrays
     - Simple flat models in the simulation coordinate system.
   * - Coordinate-aware values
     - Helper objects that export ``value``, ``units``, and ``system``
     - Coordinates in named systems or mixed units.
   * - Surface-relative systems
     - ``sim.add_surface_coordinate_system(...)``
     - Properties and points defined by distance below or above a model surface.
   * - Reduced coordinate systems
     - Fixed axis plus fixed value
     - 2D solves embedded in a 3D coordinate convention.

Surface-relative systems are especially useful for topography. A property can
vary with depth below the surface, and acquisition can be placed on or below the
same surface with helper methods such as ``surface.on(...)`` and
``surface.below(...)``.

Layered Model Geometry
----------------------

Layered models are ordered. Surfaces and layers are added in sequence, and that
sequence determines which surfaces bound layer intervals. A surface becomes a
material interface because of where it appears in that sequence; there is no
separate surface flag that makes it an interface. A cutting role is orthogonal
to that ordering and tells the mesher where the model is truncated.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Geometry type
     - Purpose
   * - Interface surface
     - Bounds material intervals and therefore changes layer assignment.
   * - Non-interface surface
     - Preserves geometry for meshing, output selection, or later reference
       without changing material topology.
   * - :term:`Cutting surface <cutting surface>`
     - Becomes the lower model boundary while preserving intersecting surfaces
       for clipping by the mesher.
   * - :term:`Borehole <borehole>` or local feature
     - Adds geometry and optional subdomains that meshing and output requests
       can reference.
   * - :term:`Fracture <fracture>`
     - Defines an opened volume or a closed interface with aperture-defined
       support for a reduced fracture model.
   * - Uniform sampled view
     - ``model.sample_uniform(...)`` creates an ``xarray.Dataset`` for QC,
       plotting, and export.

Truncating Below a Surface
--------------------------

Use ``truncate(...)`` to crop an existing layered model. It accepts the same
depth, grid, unit, and coordinate-system forms as ``add_surface(...)``:

.. code-block:: python

   truncated_model = model.truncate(
       name="section_bottom",
       depth=1.5 * fs.ureg.km,
   )

The method returns an independent deep copy, so both ``model`` and
``truncated_model`` remain available. In the copy, existing surfaces fully
deeper than the cut and formation layers wholly below it are removed. Surfaces
that cross the cut remain in the ordered surface list so the mesher can clip
them. The new lower boundary is exported with an explicit cutting role:

.. code-block:: json

   {
     "name": "section_bottom",
     "interface": true,
     "cutting": true,
     "depth": {"value": 1.5, "units": "km"}
   }

Surface depths must be materialized when ``truncate(...)`` is called so the API
can determine which geometry lies wholly below the cut. To label geometry for a
custom workflow without changing the model, use
``add_surface(..., cutting=True)`` directly.

Fractures
---------

:term:`Fractures <fracture>` are specialized model surfaces. They participate in the same ordered
layered workflow as ``SimpleSurface``: place a fracture after a layer to close
that layer, or place it between two surfaces when you want the fracture geometry
without changing the material interval. The mesh generator opens the fracture
around its center ``depth`` using the one-dimensional ``gap`` aperture curve.
Use an :term:`xarray` ``DataArray`` when the aperture varies along the fracture:

.. code-block:: python

   x = np.asarray([0.0, 0.5, 1.0])
   gap = xr.DataArray(
       [0.0, 0.01, 0.0],
       dims=["x"],
       coords={"x": x},
   )

   model.add_fracture(
       name="frac",
       depth=0.5,
       gap=gap,
       physics="acoustic",
       properties={"vp": 1.5, "rho": 1.0},
   )

The ``fracture_model`` selector defaults to ``"volumetric"``. Use
``"fluid_filled"`` with fluid material ``properties`` to create an explicitly
acoustic volume. Use ``"linear_slip"``, ``"lower_dimensional_fluid"`` (acoustic waves),
``"lower_dimensional_viscous_fluid"`` (viscous acoustic flow), or
``"hydraulic"`` (diffusion/storage/leakoff) to
mesh an unopened interface in 2D or 3D:

.. code-block:: python

   model.add_fracture(
       "slip", depth=500.0, gap=gap,
       fracture_model="linear_slip", interface=False,
   )

Reduced models retain the aperture field to locate closure boundaries and the
positive-aperture support, but create no fracture volume cells or material
subdomains. They reject volume ``mesh_block_id``, ``properties``, ``physics``,
and ``p_enrich``. The two generated surface aliases remain: ``slip_top`` tags
the active interface; ``slip_bottom`` tags its welded continuation.

Reduced selectors support native transient Galerkin elasticity or coupled
acoustic–elastic jobs: multiple non-intersecting fractures, piecewise-straight in
Cartesian or axisymmetric 2D, or non-planar height graphs ``z(x,y)`` in Cartesian
3D with average-acceleration Newmark. Direct monolithic CPU solves use one MPI
rank. Backends implementing coupled fracture FGMRES also support multiple ranks
with replicated mesh topology and selected GPU configurations; use the solver
contract shipped with your backend for those execution settings. Fractures must
stay outside PML, and frequency jobs are unsupported.
In 3D, closure contours must follow generated surface-cell edges. The generator
fits aperture grid knots and coordinate-aligned closure roots. Borehole sketches
retain reduced fractures in formation while excluding borehole fluid and casing.
Mixed reduced/volumetric stacks are supported; volumetric pinch-out must follow
complete plan-cell edges. Borehole closure must fit the sketch cells. Intersecting
fracture junctions and arbitrary closure footprints remain unsupported.

Use ``constitutive`` to supply SI-valued ``fluid_density`` (kg/m³),
``fluid_bulk_modulus`` (Pa), ``dynamic_viscosity`` (Pa·s), ``leakoff``
(m/(Pa·s)), and normal/tangential stiffness (Pa/m). Defaults are water density
1000, bulk modulus 2.25e9, viscosity 0.001, and zero leakoff/tangential
stiffness. Omitted normal stiffness is bulk modulus divided by aperture for
linear slip and zero for the fluid models.

``lower_dimensional_fluid`` includes fluid inertia and acoustic propagation.
``lower_dimensional_viscous_fluid`` also resolves viscous shear through aperture
modes (``aperture_modes``, an integer from 1 to 64, default 12), using
``dynamic_viscosity`` in Pa s. Both support Cartesian 2D/3D and axisymmetric 2D coupling. Each viscous aperture
mode has two tangential components in 3D.
``hydraulic`` uses viscous diffusion, storage, and leakoff. These fluid models couple to
solid aperture motion and have sealed fracture boundaries. Request
``elastic:fracture_pressure`` for interface pressure. Current backends expose
this field over the union of fluid-bearing fractures; older backends may use
entry-specific names. Each entry retains independent constitutive data and
aperture-mode count. Resolved
``fluid_filled`` volumes use ordinary coupled acoustic–elastic physics.

When ``properties`` are provided, the :term:`Python API` assigns or uses the fracture
``mesh_block_id`` and creates the matching ``ModelSubdomain``. Material
properties stay in ``subdomains``; the fracture entry in ``surfaces`` carries
the geometry and the mesh-block reference:

.. code-block:: json

   {
     "_type": "Fracture",
     "name": "frac",
     "interface": true,
     "depth": {"value": 0.5},
     "gap": {
       "dims": ["x"],
       "coords": {"x": {"data": [0.0, 0.5, 1.0]}},
       "data": [0.0, 0.01, 0.0]
     },
     "mesh_block_id": 2
   }

This keeps fracture geometry in the ordered surface list while letting the
solver expand the opened interval internally.

Attach units to the ``depth`` quantity, the gap values, or the gap coordinate
when the model is not unitless:

.. code-block:: python

   gap.attrs["units"] = "m"
   gap.coords["x"].attrs["units"] = "m"
   model.add_fracture("frac", depth=500.0 * fs.ureg.m, gap=gap)

Inspect Before Running
----------------------

Before submitting a job, inspect at least one exported or sampled view:

.. code-block:: python

   sampled = model.sample_uniform([151, 151])
   model.plot("vp", figsize=(7, 3), aspect="equal")
   sim.acquisition.to_fs(sim.export_context())

These checks catch missing units, wrong coordinate names, unintended layer
assignment, and flat receivers on topographic models before a remote solve has
already spent time and money.
