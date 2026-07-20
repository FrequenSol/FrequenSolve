Surveys, Sources, and Receivers
===============================

Acquisition contains :term:`source groups <source group>`,
:term:`receiver groups <receiver group>`, and optional :term:`sparse survey`
layouts. Coordinates are physical global coordinates by default, or
``CoordinateValue`` objects when a named coordinate system is used.

Related tutorials:

- :download:`Receivers <../../../examples/tutorials/05_surveys/01_receivers.ipynb>`
  for multi-component :term:`receiver devices <receiver device>` and dense groups.
- :download:`DAS <../../../examples/tutorials/05_surveys/02_das.ipynb>`
  for fiber-style strain receivers.
- :download:`Sources <../../../examples/tutorials/05_surveys/03_sources.ipynb>`
  for scalar, vector, moment, and compound sources.
- :download:`Sparse surveys <../../../examples/tutorials/05_surveys/04_sparse_surveys.ipynb>`
  for offset windows and explicit source-receiver layouts.

Survey Receivers
----------------

A :term:`receiver device` owns one or more :term:`components <component>`.
Device names are optional; the receiver group name is the public survey
identifier.

.. code-block:: python

   node = fs.ReceiverNode()
   node.add_component(name="v_z", field="velocity", direction=[0.0, 1.0])
   node.add_component(name="p", field="pressure")

   acq.add_receiver_group(
       name="surface",
       device=node,
       coords=[[x, 0.05] for x in np.linspace(0.1, 0.9, 81)],
   )

Supported fields depend on physics:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Physics
     - Common fields
   * - Acoustic
     - ``pressure``, ``velocity``
   * - Elastic
     - ``velocity``, ``stress``, ``strain``, ``pressure``
   * - Poroelastic
     - ``velocity``, ``fluid_flux``, ``stress``, ``pressure``, ``strain``, ``displacement``, ``fluid_displacement``

DAS
---

``ReceiverFiber`` represents a fiber-style receiver. ``gauge_length`` is the
:term:`DAS` :term:`gauge length`, ``channel_spacing`` is the
:term:`channel spacing` between reported channels and defaults to
``gauge_length``, and ``sample_spacing`` controls the :term:`sample spacing`
used along the gauge. If ``sample_spacing`` is omitted,
``points_per_gauge`` may be used instead. Helical fiber response adds
``radius`` and ``pitch``. Length-like DAS fields accept plain solver-scaled
numbers or :term:`Pint` quantities with explicit units.

.. code-block:: python

   u = fs.ureg

   das = fs.ReceiverFiber(
       gauge_length=10 * u.m,
       channel_spacing=10 * u.m,
       sample_spacing=2 * u.m,
   )
   das.add_component(name="eps_tt", field="strain", direction=[1.0, 0.0])

Survey Sources
--------------

Point source kinds include ``scalar``, ``vector``, ``tensor``, ``monopole``,
and ``dipole``. When a simulation has multiple compatible sources, the solver
chooses efficient internal :term:`source batches <source batching>`
automatically:

.. code-block:: python

   acq = fs.Acquisition()
   acq.add_sources(
       kind="scalar",
       coords=[[0.25, 0.05], [0.5, 0.05], [0.75, 0.05]],
   )

Source Amplitudes
~~~~~~~~~~~~~~~~~

``amplitude`` follows the Sauce ``fs-acquisition-2`` source-basis contract. A
plain number is a dimensionless multiplier. It scales explicit unit-bearing
direction components when present; otherwise it scales the default physical
strength for the source kind. A Pint quantity or explicit ``value``/``units``
mapping is an exact physical source strength:

.. code-block:: python

   u = fs.ureg

   acoustic = fs.Acquisition()
   acoustic.add_sources(
       kind="scalar",
       coords=[[0.5, 0.05]],
       amplitude=1.0e6 * u.N * u.m,
   )

   elastic = fs.Acquisition()
   elastic.add_sources(
       kind="vector",
       coords=[[0.5, 0.05]],
       direction=[0.0, 1.0],
       amplitude=20.0 * u.kN,
   )

Vector and dipole amplitudes have force dimensions, conventionally ``N``.
Scalar, tensor, and monopole amplitudes have moment dimensions, conventionally
``N*m``. If a direction already contains physical units, use a dimensionless
top-level amplitude; Sauce rejects simultaneous physical units on both the
direction and amplitude. Physical strength belongs to ``source_geometry``;
``source_encoding`` coefficients remain dimensionless complex multipliers.

The same amplitude forms are accepted by :class:`frequensolve.PointSource` and
inside the ``defaults`` mapping for inline, HDF5, and SPS source geometries.

Sparse Survey Layouts
---------------------

:term:`Dense surveys <dense survey>` evaluate all source/receiver/component
combinations. :term:`Sparse surveys <sparse survey>` select a subset or define
rules such as offset windows:

.. code-block:: python

   survey = fs.SparseSurvey.offset_domain(
       "near_offsets",
       min=0.0,
       max=0.35,
       metric="horizontal",
   )

   acq = fs.Acquisition()
   acq.add_sources(kind="scalar", coords=sources)
   acq.add_sparse_receiver_group(
       "near_offsets",
       node,
       coords=receivers,
       survey=survey,
   )

Use ``SparseSurvey.from_product(...)`` or ``SparseSurvey.from_pairs(...)`` when
explicit :term:`trace` pairs are easier to define directly. Sparse survey inputs should
describe public trace identity: source ids, receiver ids, component names, and
weights. Internal sample maps and point ranges are runtime details and should
not be authored directly.
