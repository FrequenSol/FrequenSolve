Velocity Models, Units, And Coordinates
=======================================

FrequenSolve models are authored in Python but exported to solver contracts
with explicit units, coordinate systems, surfaces, layers, and geometry. The
goal is to keep the model readable at authoring time while making the generated
inputs auditable before a run.

Primary tutorials:

- :download:`Variable properties and units <../../../examples/tutorials/03_velocity_model_building/01_variable_properties_units.ipynb>`
- :download:`Coordinate systems <../../../examples/tutorials/03_velocity_model_building/02_coordinate_systems.ipynb>`
- :download:`Layered models <../../../examples/tutorials/03_velocity_model_building/03_layered_models.ipynb>`

Property Forms
--------------

Layer properties may be constants, Pint quantities, sampled arrays, or richer
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
   * - Pint quantity
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

Simulation unit defaults are useful for compact examples, but explicit Pint
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
separate surface flag that makes it an interface.

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
   * - Borehole or local feature
     - Adds geometry and optional subdomains that meshing and output requests
       can reference.
   * - Uniform sampled view
     - ``model.sample_uniform(...)`` creates an ``xarray.Dataset`` for QC,
       plotting, and export.

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
