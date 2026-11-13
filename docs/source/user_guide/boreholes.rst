Boreholes
=========

:term:`Boreholes <borehole>` are authored on ``LayeredModel``. The common
axisymmetric workflow
uses 2D ``LayeredMeshGenerator`` meshes. 3D layered models support vertical
boreholes with ``x``/``y`` axes and optional borehole-level annular padding;
plug/tool-body intervals remain 2D-only. A borehole describes radial layers
such as fluid, casing, and cement, plus borehole-local radial surfaces such as
casing walls. The material properties for borehole layers are normal model
subdomains, so they can use the same property, physics, unit, and file-backed
workflows as any other material block.

.. note::

   This is an advanced modeling topic. Start with
   :doc:`velocity_models_coordinates` and the layered-model tutorials before
   using boreholes in production models.

Python API
----------

The recommended form mirrors ``LayeredModel`` construction. Create the
borehole, add a radial material layer, then close that layer with a
cumulative-radius surface. The borehole axis at ``r = 0`` is assumed, so the
first surface you add is normally the fluid wall rather than an explicit axis.
Name a surface only when user code needs to refer to it later. If a layer
includes ``physics`` and ``properties``, the :term:`Python API` creates the corresponding
material subdomain automatically.

For scalar radial intervals, pass ``width=`` to ``add_layer``. The FrequenSolve Python API adds the
width to the previous radius and creates the next surface immediately. That
surface is unnamed; if a stable name is needed for a grading or diagnostic,
close the layer explicitly with ``add_surface(...)``. Use explicit
``add_surface(...)`` calls for variable-radius surfaces. The object-oriented
``+=`` form is equivalent when you prefer to make layers and surfaces explicit
objects.

.. code-block:: python

   import frequensolve as fs

   u = fs.ureg

   model = fs.LayeredModel(
       name="model",
       dimension=2,
       x_limits=[0.0, 1.0],
   )
   model.add_surface(name="top", depth=0.0 * u.km)
   model.add_layer(
       name="formation_1",
       mesh_block_id=1,
       properties={
           "Vp": 2.2 * u.km / u.s,
           "Rho": 2.1 * u.g / u.cm**3,
       },
   )
   model.add_surface(name="bottom", depth=0.5 * u.km)

   well = model.add_borehole(name="bh1", x=0.45 * u.km)
   well.add_layer(
       "fluid",
       width=0.035 * u.m,
       mesh_block_id=20,
       physics="acoustic",
       properties={
           "Vp": 1.48 * u.km / u.s,
           "Rho": 1.03 * u.g / u.cm**3,
       },
   )

   well.add_layer(
       "casing",
       width=0.006 * u.m,
       mesh_block_id=21,
       physics="elastic:iso",
       properties={
           "Vp": 5.9 * u.km / u.s,
           "Vs": 3.2 * u.km / u.s,
           "Rho": 7.85 * u.g / u.cm**3,
       },
   )

   well.add_layer(
       "cement",
       mesh_block_id=22,
       physics="elastic:iso",
       properties={
           "Vp": 3.4 * u.km / u.s,
           "Vs": 1.9 * u.km / u.s,
           "Rho": 2.1 * u.g / u.cm**3,
       },
   )
   well.add_surface("bh1_outer_wall", r=0.065 * u.m)

   # Equivalent object-oriented form:
   # well += fs.BoreholeLayer("fluid", width=0.035 * u.m, ...)
   # well += fs.BoreholeLayer("casing")
   # well += fs.BoreholeLayer("cement")
   # well += fs.BoreholeSurface("bh1_outer_wall", r=0.065 * u.m)

   mesh = fs.LayeredMeshGenerator(
       l_bound=[0.0, 0.0],
       u_bound=[1.0, 0.5],
       n=[80, 4],
   ).refine_around_borehole(
       "bh1",
       padding=0.08,
       max_size=0.005,
       max_growth=1.5,
   )

The compact solver-shaped form is also available by passing ``layers=[...]`` and
``surfaces=[...]`` to ``add_borehole``. In that form each layer references a
material subdomain by ``mesh_block_id`` and each surface supplies a cumulative
radius ``r``. Layer ``i`` occupies the radial interval from the previous surface
to surface ``i``; the first layer starts at the borehole axis. The older
``parts=[...]`` spelling is still accepted by the :term:`Python API` for compatibility,
but new code should use ``layers`` and ``surfaces``.

``LayeredModel.plot(...)`` samples borehole material subdomains into the plotted
property image, so borehole materials appear by value instead of as separate
guide lines. Pass ``boreholes=True`` only when you want to overlay the borehole
axis and radial part boundaries for geometry debugging; customize that optional
overlay with ``borehole_kwargs``:

.. code-block:: python

   model.plot(
       "Vp",
       boreholes=True,
       borehole_kwargs={
           "fill": True,
           "color": "white",
           "labels": True,
       },
   )

For a local radial cross-section, call ``draw`` on a borehole object:

.. code-block:: python

   borehole = model.boreholes["bh1"]
   borehole.draw(
       z=0.25 * u.km,
       units="m",
       depth_units="km",
       subdomains=model.subdomains,
   )

Each borehole surface gives its cumulative outer radius as ``r``. The first
layer starts at ``r = 0``; each following layer starts at the previous surface's
``r``. Radius is represented internally as a ``Property``, so it accepts
scalar, :term:`Pint`, :term:`xarray`, file-backed, and structured property
inputs. Inline
variable-radius profiles may be one-dimensional over ``z`` or ``depth``. 3D
surface profiles may also be two-dimensional when they include one angular
dimension, ``theta``, ``azimuth``, or ``angle``, and one depth dimension,
``z`` or ``depth``.

Named borehole surfaces are exported as ``inner_surface`` and ``outer_surface``
aliases on the solver-facing layers. Use explicit surface names when a
cumulative-radius boundary needs a stable reference for mesh gradings or
diagnostics. Unnamed surfaces receive generated aliases such as
``bh1_surface_1`` during serialization. For example, the outer cement/formation
wall above can be referenced directly:

.. code-block:: python

   sim.mesh.add_surface_grading(
       "bh1_outer_wall",
       d1=15 * u.m,
       factor=4.0,
       power=2.0,
       mode="abs_band",
   )

Like layered-model surfaces, borehole surfaces can also be added consecutively.
When multiple surfaces are added before the next layer, the last surface is the
material boundary and the earlier surfaces are geometry-only features. This is
useful for adding mesh-control surfaces without repeating material properties:

.. code-block:: python

   well.add_layer("fluid", physics="acoustic", properties=fluid)
   well.add_surface("fluid_refine", r=0.025 * u.m)
   well.add_surface("fluid_wall", r=0.035 * u.m)

   well.add_layer("casing", physics="elastic:iso", properties=casing)
   well.add_surface("casing_refine", r=0.038 * u.m)
   well.add_surface("casing_wall", r=0.041 * u.m)

Here ``fluid_refine`` and ``casing_refine`` are exported as borehole surfaces
but are not layer boundaries. The fluid layer extends to ``fluid_wall`` and the
casing layer extends from ``fluid_wall`` to ``casing_wall``.

Imperfect solid bonds
---------------------

Interfaces are perfectly bonded by default. Set ``interface`` on a radial
surface, or on the layer ending at that surface, to permit a displacement
jump. Both stiffnesses are in Pa/m; zero releases the component. This applies
to constant-radius solid–solid interfaces, such as PVC–cement or cement–formation, and introduces
no fluid gap or exchange. Native transient Galerkin supports Newmark, Radau-3
and Gauss-4; the axial PML convolution currently requires CPU execution.

.. code-block:: python

   well.add_layer(
       "cement",
       width      = 4 * u.inch,
       physics    = "elastic",
       properties = {
           "Vp": 3000 * u.m / u.s,
           "Vs": 1500 * u.m / u.s,
           "Rho": 2000 * u.kg / u.m**3,
       },
       interface  = fs.BoreholeLinearSlip(
           normal_stiffness     = 1 * u.GPa / u.m,
           tangential_stiffness = 0 * u.Pa / u.m,
       ),
   )

Use ``well.set_interface("casing", bond)`` to change the outer bond of an
existing named layer; ``well.set_interface("cement", None)`` restores exact
bonding at the cement–formation wall.


Variable Radius
---------------

Use an ``xarray.DataArray`` when a borehole surface varies with depth. The data
values are radii; the coordinate gives the depth positions where those radii
are defined.

.. code-block:: python

   import xarray as xr

   fluid_radius = xr.DataArray(
       [0.035, 0.040, 0.032],
       dims=["z"],
       coords={"z": [0.0, 0.25, 0.5]},
       attrs={"units": "m"},
   )
   fluid_radius.coords["z"].attrs["units"] = "km"

   model.add_borehole(
       name="bh_variable",
       x=0.45 * u.km,
       layers=[
           {
               "name": "fluid",
               "mesh_block_id": 20,
               "physics": "acoustic",
               "properties": {
                   "Vp": 1.48 * u.km / u.s,
                   "Rho": 1.03 * u.g / u.cm**3,
               },
           }
       ],
       surfaces=[
           {"name": "fluid_wall", "r": fluid_radius}
       ],
   )

When exporting through a :term:`simulation`, radius arrays are written to ``sim.h5`` and
the :term:`JSON` carries the :term:`HDF5` reference and content hash. Standalone
``model.to_fs()`` calls emit a compact inline value with dimensions and
coordinates.

By default, the borehole extent spans the model from the upper surface to the
lower surface. To make this explicit, pass ``top=`` and ``bottom=`` as surface
names, ``SimpleSurface`` objects, or solver-ready dictionaries:

.. code-block:: python

   model.add_borehole(
       name="bh2",
       x=0.65 * u.km,
       top="top",
       bottom="bottom",
       layers=[...],
       surfaces=[...],
   )

Borehole Mesh Grading
---------------------

Define radial grading and axial resolution on the borehole itself:

.. code-block:: python

   well = model.add_borehole(
       name="well", x=0 * u.m, layers=[...], surfaces=[...],
       grading=fs.BoreholeGrading(
           wall_size=0.1 * u.m,
           growth=1.5,
           axial_factor=1.5,
       ),
   )

Existing boreholes also support
``well.set_grading(wall_size=0.1*u.m, growth=1.5, axial_factor=1.5)``.
``BoreholeLayer`` accepts the same ``grading`` object for layer-local overrides.

``wall_size`` is the solid-side, wall-adjacent normal size envelope. It accepts
Pint lengths; the native default is 0.075 m, and zero disables the geometric
envelope. At wall distance ``d``, the envelope is
``wall_size + (growth - 1)*d``, capped by ordinary wavelength sizing. It does
not impose a global minimum or maximum element size. This works for uncased
boreholes as well as cased boreholes; an unavailable tube-wave speed does not
disable the geometric envelope.

``axial_factor`` multiplies the target elements per wavelength along the
borehole. For example, 1.5 requests 50% more axial resolution; adaptive
bisections may exceed that minimum or leave an already sufficient mesh
unchanged. It applies directionally in the fluid and through the existing
wall-tangential policy. It is frequency dependent, not a fixed split count.
In the solid its excess above one tapers as
``(1 - d/L)**1.5`` over the default reach
``L = (body_speed/(EPW*f_adapt) - wall_size)/(growth - 1)``;
it returns to one beyond that reach. In the fluid it is constant.
Borehole radial grading and the axial request continue through generated PMLs.
Omitting it preserves native defaults.

Tube-wave mesh sizing uses the bonded radial compliance of the concentric
casing, cement and surrounding formation. Formation intervals along the well
select local speed bounds; uncased holes use the formation shear modulus.
This low-frequency estimate controls resolution, not the coupled equations.
For a scalar fluid/casing/fluid stack with bonded interfaces, sizing includes
both coupled fluid-column modes and uses the slower branch for refinement.
Its conservative speed interval includes the formation stiffness range;
file-backed profiles are read in distributed slabs before caching these bounds.
An explicit native ``tube_wave_model="korteweg"`` retains the unsupported
thin-wall casing estimate.

Guided-wave layers are anisotropic by default. Mesh closure uses the local
directional sizing ratio to relax the ordinary aspect limit, allowing fine
wall-normal spacing without imposing it axially. To restore physical aspect
closure or adjust its baseline:

.. code-block:: python

   sim.mesh.set_adapt(
       order             = 4,
       elems_per_wave    = 1.5,
       f_adapt           = 200,
       aspect_limit      = 3,
       guided_anisotropy = False,
   )

The default ``guided_anisotropy=True`` applies locally to guided-wave sizing;
ordinary regions retain ``aspect_limit``. Conformity and wavelength resolution
remain enforced.
This changes the native 2D aspect closure; 3D refinement already omits that
physical aspect limiter and remains unchanged.

The SDK exports ``nearfield`` and ``adapt/stoneley/tangential/factor``.
Legacy ``near_h0`` and ``near_growth`` keyword inputs remain accepted,
including Pint quantities. Layer controls override borehole controls, which
override global ``Mesh/adapt/stoneley`` settings.

3D Annular Padding
------------------

For 3D layered boreholes, pass ``annular_padding`` to ``add_borehole``. The
setting belongs to the borehole payload, not the mesh generator. ``n`` is the
number of formation-domain padding cells, ``outer_radius`` is the positive
outer radius of the padded annulus, and ``power`` is an optional positive radial
spacing exponent.

.. code-block:: python

   model = fs.LayeredModel(
       dimension=3,
       x_limits=[0.0, 1.0],
       y_limits=[0.0, 1.0],
   )

   model.add_borehole(
       name="bh3d",
       x=0.45 * u.km,
       y=0.35 * u.km,
       layers=[...],
       surfaces=[...],
       annular_padding={
           "n": 3,
           "outer_radius": 0.2 * u.m,
           "power": 1.5,
       },
   )

Object API
----------

Use ``BoreholeLayer`` and ``BoreholeSurface`` when geometry and materials are
defined separately. In this form, create the matching ``ModelSubdomain``
objects yourself; the ``mesh_block_id`` on each borehole layer must already
exist in ``subdomains``.

.. code-block:: python

   model += fs.ModelSubdomain(
       name="bh1_fluid",
       mesh_block_id=20,
       physics="acoustic",
       properties={"Vp": 1.48 * u.km / u.s, "Rho": 1.03 * u.g / u.cm**3},
   )
   model += fs.ModelSubdomain(
       name="bh1_casing",
       mesh_block_id=21,
       physics="elastic:iso",
       properties={
           "Vp": 5.9 * u.km / u.s,
           "Vs": 3.2 * u.km / u.s,
           "Rho": 7.85 * u.g / u.cm**3,
       },
   )

   model.add_borehole(
       name="bh1",
       x=0.45 * u.km,
       layers=[
           fs.BoreholeLayer(
               name="fluid",
               mesh_block_id=20,
           ),
           fs.BoreholeLayer(
               name="casing",
               mesh_block_id=21,
           )
       ],
       surfaces=[
           fs.BoreholeSurface("fluid_wall", r=0.035 * u.m),
           fs.BoreholeSurface("casing_wall", r=0.041 * u.m),
       ],
   )

Solver Contract
---------------

The :term:`solver contract` keeps materials and borehole geometry separate. Borehole
layers reference the material ``subdomains`` by :term:`mesh block ID`. Borehole
surfaces stay under their owning borehole; they are not top-level stratigraphic
surfaces.

.. code-block:: json

   {
     "Model": {
       "_type": "LayeredModel",
       "subdomains": [
         {
           "mesh_block_id": 1,
           "name": "formation_1",
           "properties": {
             "vp": {"value": 2.2, "units": "km/s"},
             "rho": {"value": 2.1, "units": "g/cm^3"}
           }
         },
         {
           "mesh_block_id": 20,
           "name": "bh1_fluid",
           "physics": "acoustic",
           "properties": {
             "vp": {"value": 1.48, "units": "km/s"},
             "rho": {"value": 1.03, "units": "g/cm^3"}
           }
         },
         {
           "mesh_block_id": 21,
           "name": "bh1_casing",
           "physics": "elastic:iso",
           "properties": {
             "vp": {"value": 5.9, "units": "km/s"},
             "vs": {"value": 3.2, "units": "km/s"},
             "rho": {"value": 7.85, "units": "g/cm^3"}
           }
         }
       ],
       "boreholes": [
         {
           "name": "bh1",
           "axis": {"x": {"value": 0.45, "units": "km"}},
           "extent": {
             "top": {"surface": "top"},
             "bottom": {"surface": "bottom"}
           },
           "layers": [
             {
               "name": "fluid",
               "mesh_block_id": 20,
               "outer_surface": "bh1_surface_1"
             },
             {
               "name": "casing",
               "mesh_block_id": 21,
               "inner_surface": "bh1_surface_1",
               "outer_surface": "bh1_surface_2"
             }
           ],
           "surfaces": [
             {
               "name": "bh1_surface_1",
               "r": {"value": 0.035, "units": "m"}
             },
             {
               "name": "bh1_surface_2",
               "r": {"value": 0.041, "units": "m"}
             }
           ]
         }
       ]
     },
     "Mesh": {
       "generator": {
         "_type": "LayeredMeshGenerator",
         "l_bound": [0.0, 0.0],
         "u_bound": [1.0, 0.5],
         "n": [80, 4],
         "horizontal_spacing": {
           "include_borehole_edges": true,
           "max_growth": 1.5,
           "controls": [
             {
               "around_borehole": "bh1",
               "padding": 0.08,
               "max_size": 0.005
             }
           ]
         }
       }
     }
   }

The solver should treat ``boreholes[].layers[].mesh_block_id`` as references to
material :term:`subdomains <subdomain>`. The FrequenSolve Python API keeps the
material definitions in
``subdomains`` and emits borehole geometry under ``boreholes[].surfaces``. A
3D borehole with padding emits ``boreholes[].annular_padding`` beside its
surfaces.
