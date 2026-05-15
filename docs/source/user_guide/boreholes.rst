Boreholes
=========

Boreholes are authored on ``LayeredModel``. Current solver support is for
2D ``LayeredMeshGenerator`` meshes, which is the common axisymmetric borehole
case. A borehole describes the geometry and mesh blocks for radial parts such
as fluid, casing, and cement. The material properties for those parts are normal
model subdomains, so they can use the same property, physics, unit, and
file-backed workflows as any other material block.

Python API
----------

The compact form is to pass dictionaries to ``add_borehole``. If a part
dictionary includes ``physics`` and ``properties``, the SDK creates the
corresponding material subdomain automatically.

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

   model.add_borehole(
       name="bh1",
       x=0.45 * u.km,
       parts=[
           {
               "name": "fluid",
               "mesh_block_id": 20,
               "r": 0.035 * u.m,
               "physics": "acoustic",
               "properties": {
                   "Vp": 1.48 * u.km / u.s,
                   "Rho": 1.03 * u.g / u.cm**3,
               },
           },
           {
               "name": "casing",
               "mesh_block_id": 21,
               "r": 0.041 * u.m,
               "physics": "elastic:iso",
               "properties": {
                   "Vp": 5.9 * u.km / u.s,
                   "Vs": 3.2 * u.km / u.s,
                   "Rho": 7.85 * u.g / u.cm**3,
               },
           },
           {
               "name": "cement",
               "mesh_block_id": 22,
               "r": 0.065 * u.m,
               "physics": "elastic:iso",
               "properties": {
                   "Vp": 3.4 * u.km / u.s,
                   "Vs": 1.9 * u.km / u.s,
                   "Rho": 2.1 * u.g / u.cm**3,
               },
           },
       ],
   )

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

Each part gives its cumulative outer radius as ``r``. The first part starts at
``r = 0``; each following part starts at the previous part's ``r``. Radius is
represented internally as a ``Property``, so it accepts scalar, Pint, xarray,
file-backed, and structured property inputs. Inline variable-radius profiles
must be one-dimensional over ``z`` or ``depth`` so the layered generator can
evaluate radius at cell-centroid depth.

Variable Radius
---------------

Use an ``xarray.DataArray`` when a part radius varies with depth. The data
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
       parts=[
           {
               "name": "fluid",
               "mesh_block_id": 20,
               "r": fluid_radius,
               "physics": "acoustic",
               "properties": {
                   "Vp": 1.48 * u.km / u.s,
                   "Rho": 1.03 * u.g / u.cm**3,
               },
           }
       ],
   )

When exporting through a simulation, radius arrays are written to ``sim.h5`` and
the JSON carries the HDF5 reference and content hash. Standalone
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
       parts=[...],
   )

Object API
----------

Use ``BoreholePart`` when geometry and materials are defined separately. In
this form, create the matching ``ModelSubdomain`` objects yourself; the
``mesh_block_id`` on each borehole part must already exist in ``subdomains``.

.. code-block:: python

   model += fs.ModelSubdomain(
       name="bh1_fluid",
       mesh_block_id=20,
       physics="acoustic",
       properties={"Vp": 1.48 * u.km / u.s, "Rho": 1.03 * u.g / u.cm**3},
   )

   model.add_borehole(
       name="bh1",
       x=0.45 * u.km,
       parts=[
           fs.BoreholePart(
               name="fluid",
               mesh_block_id=20,
               r=0.035 * u.m,
           )
       ],
   )

Solver Contract
---------------

The solver-facing API keeps materials and borehole geometry separate. Borehole
parts reference the material ``subdomains`` by ``mesh_block_id``.

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
           "parts": [
             {
               "name": "fluid",
               "mesh_block_id": 20,
               "r": {"value": 0.035, "units": "m"}
             },
             {
               "name": "casing",
               "mesh_block_id": 21,
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

The solver should treat ``boreholes[].parts[].mesh_block_id`` as references to
material subdomains. The Python SDK keeps the material definitions in
``subdomains`` and emits borehole geometry under ``boreholes``.
