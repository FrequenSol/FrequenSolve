Physics, Materials, and Boundaries
==================================

FrequenSolve simulations select a top-level physics formulation with
``project.new_simulation(..., physics=...)``. Individual model layers may also
declare ``physics=`` when a coupled model contains more than one material
family. Layer properties are authored as a dictionary and are exported with
canonical lowercase names.

Physics names are case-insensitive. Electromagnetic inputs such as ``EM``,
``electromagnetic``, and ``maxwell`` are accepted as friendly aliases and
exported with the solver-contract key ``em``. Selecting ``em`` does not
automatically create an electromagnetic model, mesh, sources, or boundaries.

Related tutorials:

- :download:`Acoustic modeling <../../../examples/tutorials/01_modeling_basics/01_acoustic.ipynb>`
  for the simplest complete material and boundary workflow.
- :download:`Elastic modeling <../../../examples/tutorials/01_modeling_basics/02_elastic.ipynb>`
  for shear-wave properties, :term:`attenuation`, and elastic receivers.
- :download:`Poroelastic modeling <../../../examples/tutorials/01_modeling_basics/03_poroelastic.ipynb>`
  for :term:`Biot`-frame and pore-fluid properties.
- :download:`Coupled modeling <../../../examples/tutorials/01_modeling_basics/04_coupled.ipynb>`
  for mixed material families in one model.

Model Material Properties
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 18 24 40

   * - Property
     - Dimension
     - Applies to
     - Meaning
   * - ``Vp``
     - ``L/T``
     - Acoustic, elastic, poroelastic elastic-frame modes
     - Compressional or :term:`P-wave` speed.
   * - ``Vs``
     - ``L/T``
     - Elastic and poroelastic elastic-frame modes
     - Shear or :term:`S-wave` speed.
   * - ``Rho``
     - ``M/L^3``
     - Acoustic, elastic, poroelastic elastic-frame modes
     - Bulk mass density.
   * - ``Qp``
     - ``1``
     - Acoustic, elastic, poroelastic elastic-frame modes
     - Compressional :term:`quality factor` for attenuation.
   * - ``Qs``
     - ``1``
     - Elastic and poroelastic elastic-frame modes
     - Shear :term:`quality factor` for attenuation.
   * - ``epsilon``
     - ``1``
     - ``elastic:vti``, ``elastic:tti``, ``poroelastic:vti``, ``poroelastic:tti``
     - Thomsen-style P-wave :term:`anisotropy` parameter.
   * - ``gamma``
     - ``1``
     - ``elastic:vti``, ``elastic:tti``, ``poroelastic:vti``, ``poroelastic:tti``
     - Thomsen-style S-wave anisotropy parameter.
   * - ``delta``
     - ``1``
     - ``elastic:vti``, ``elastic:tti``, ``poroelastic:vti``, ``poroelastic:tti``
     - Thomsen-style near-vertical P-wave anisotropy parameter.
   * - ``phi``
     - angle
     - ``elastic:tti``, ``poroelastic:tti``
     - Symmetry-axis azimuth or rotation for tilted transverse isotropy.
   * - ``theta``
     - angle
     - ``elastic:tti``, ``poroelastic:tti``
     - Symmetry-axis tilt for tilted transverse isotropy.
   * - ``k_dry``
     - ``M/(L*T^2)``
     - ``poroelastic:direct``, ``poroelastic:direct_jkd``
     - Drained-frame bulk modulus supplied directly.
   * - ``mu_dry``
     - ``M/(L*T^2)``
     - ``poroelastic:direct``, ``poroelastic:direct_jkd``
     - Drained-frame shear modulus supplied directly.
   * - ``k_solid``
     - ``M/(L*T^2)``
     - All poroelastic modes
     - Solid-grain bulk modulus.
   * - ``k_fluid``
     - ``M/(L*T^2)``
     - All poroelastic modes
     - Pore-fluid bulk modulus.
   * - ``rho_solid``
     - ``M/L^3``
     - All poroelastic modes
     - Solid-grain density.
   * - ``rho_fluid``
     - ``M/L^3``
     - All poroelastic modes
     - Pore-fluid density.
   * - ``porosity``
     - ``1``
     - All poroelastic modes
     - Pore-volume fraction.
   * - ``tortuosity``
     - ``1``
     - All poroelastic modes
     - Inertial path-length factor for relative fluid and solid motion.
   * - ``kappa``
     - ``L^2``
     - All poroelastic modes
     - Hydraulic permeability.
   * - ``viscosity``
     - ``M/(L*T)``
     - All poroelastic modes
     - Dynamic fluid viscosity.
   * - ``qk``
     - ``1``
     - Direct poroelastic modes
     - Bulk-modulus quality factor for direct frame attenuation.
   * - ``qmu``
     - ``1``
     - Direct poroelastic modes
     - Shear-modulus quality factor for direct frame attenuation.
   * - ``viscous_length``
     - ``L``
     - JKD poroelastic modes
     - Dynamic permeability length scale used by the :term:`JKD` hydraulic model.
   * - ``biot_frequency``
     - ``1/T``
     - JKD poroelastic modes
     - Optional transition frequency for dynamic hydraulic response.

The Dimension column is unit-system independent. Values may be plain numbers
in the simulation's configured unit system or :term:`Pint` quantities with compatible
units.

Required fields depend on the selected physics. ``acoustic`` requires
``Vp`` and ``Rho``. ``elastic:iso`` requires ``Vp``, ``Vs``, and ``Rho``;
``elastic:vti`` and ``elastic:tti`` also require ``epsilon``. Direct
:term:`poroelastic` modes require ``k_dry``, ``mu_dry``, and the shared
:term:`Biot` solid/fluid/transport properties. Elastic-frame poroelastic modes require
``Vp``, ``Vs``, ``Rho``, and the shared Biot properties. :term:`JKD` modes also
require ``viscous_length``.

For basic modeling tutorials, adaptivity-only properties are intentionally
omitted. Mesh-specific properties such as ``epw_mult``, ``hmin``, and ``hmax`` are covered in
:doc:`mesh_generation_adaptivity`.

Attenuation
-----------

Elastic :term:`attenuation` is controlled by ``Qp`` and ``Qs``. Poroelastic
direct-Biot attenuation uses ``qk`` and ``qmu`` where supported. The model-wide
attenuation configuration selects how those quality factors are interpreted:

.. code-block:: python

   model = fs.LayeredModel(
       dimension=2,
       x_limits=[0.0, 10.0] * fs.ureg.km,
       attenuation_model="kjartansson",
       reference_frequency=10.0 * fs.ureg.Hz,
   )

``kjartansson`` is the default attenuation law and uses a 10 Hz reference when
``reference_frequency`` is omitted. Model names are case-insensitive. A bare
reference-frequency scalar is interpreted as hertz; Pint quantities and
``{"value": ..., "units": ...}`` mappings may use any compatible frequency
unit.

Set ``attenuation_model="none"`` on a model to ignore ``Qp``, ``Qs``, ``Qk``,
and ``Qmu``. This disables solid-frame Q attenuation but does not disable
:term:`JKD` hydraulic dispersion.

Solver payloads may spell the reference-frequency field ``reference_frequency``,
``f0``, or ``f_ref``. They are mutually exclusive. FrequenSolve accepts all
three when loading dictionaries and emits the canonical ``reference_frequency``
spelling.

Simulation Boundary Conditions
------------------------------

Boundary conditions are added directly to simulations:

.. code-block:: python

   sim += fs.BoundaryCondition(
       conditions=["free"],
       boundaries=["z_min"],
   )
   sim += fs.BoundaryCondition(
       conditions=["pml"],
       boundaries=["x_min", "x_max", "z_max"],
       pml_wavelengths=0.75,
   )

Common boundary names for generated 2D meshes are ``x_min``, ``x_max``,
``z_min``, and ``z_max``. For generated 3D meshes, ``y_min`` and ``y_max`` are
also available. Labeled external meshes may use mesh boundary labels instead.

:term:`PML` is not authored as an extra material layer. When a boundary is marked
``pml``, the solver auto-extrudes an absorbing layer outside that
boundary. ``pml_wavelengths`` specifies the width in wavelengths, so the
physical PML thickness is frequency-dependent: lower frequencies produce wider
physical PMLs for the same setting, and higher frequencies produce thinner
ones. ``pml_reflectivity`` defaults to ``1e-2`` unless supplied explicitly. For
conditioning, prefer increasing ``pml_wavelengths`` before making the
reflection target extremely small.

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Condition
     - Variables/components constrained
     - Typical use
   * - ``free``
     - Mechanical traction trace ``t_hat = 0`` for elastic/poroelastic; natural/free acoustic scalar and normal-flux trace for acoustic.
     - Free surfaces or open natural boundaries.
   * - ``pml``
     - Absorbing-layer equations for all active fields in the PML region.
     - Truncating sides, bottoms, and other artificial model limits.
   * - ``fixed``
     - Primary field trace, such as ``u_hat`` for mechanical displacement or ``p_hat`` for scalar pressure-like fields.
     - Rigid, clamped, or fixed-field idealizations.
   * - ``impedance``
     - Couples primary and flux/traction traces through an impedance relation.
     - Approximate radiation or absorbing behavior without a full PML.
   * - ``dirichlet``
     - Explicitly prescribed primary trace values where supported.
     - Advanced workflows that impose boundary values.
   * - ``neumann``
     - Alias of ``free`` in the Python API.
     - Compatibility spelling for natural/free boundaries.
   * - ``sealed``
     - Poroelastic normal relative fluid-flux trace ``q_hat = 0``.
     - Sealed or undrained pore boundary, commonly paired with ``free``.
   * - ``drained``
     - Poroelastic pore-pressure trace ``p_hat = 0``.
     - Drained pore boundary, commonly paired with ``free``.
   * - ``symmetric`` / ``axis`` / ``symmetric_r``
     - Symmetry-axis regularity. For non-torsional axisymmetric elastic runs, ``u_hat_r = 0`` and ``t_hat_z = 0``.
     - Symmetry-reduced and axisymmetric models.

Poroelastic free surfaces usually combine mechanical and fluid conditions:

.. code-block:: python

   sim += fs.BoundaryCondition(
       conditions=["free", "sealed"],
       boundaries=["z_min"],
   )

Coupled Models
--------------

In coupled models, a layer may specify its own material physics:

.. code-block:: python

   model.add_layer(
       name="water",
       physics="acoustic",
       properties={"Vp": 1.5, "Rho": 1.0},
   )

When layer physics is explicit, unsupported properties do not redefine the
physics family. Keep layer property dictionaries focused on properties relevant
to that layer so exported input remains easy to inspect.
