Surveys, Sources, And Receivers
===============================

Acquisition contains source groups, receiver groups, and optional sparse survey
layouts. Coordinates are physical global coordinates by default, or
``CoordinateValue`` objects when a named coordinate system is used.

Primary tutorials:

- :download:`Receivers <../../../examples/tutorials/05_surveys/01_receivers.ipynb>`
- :download:`DAS <../../../examples/tutorials/05_surveys/02_das.ipynb>`
- :download:`Sources <../../../examples/tutorials/05_surveys/03_sources.ipynb>`
- :download:`Sparse surveys <../../../examples/tutorials/05_surveys/04_sparse_surveys.ipynb>`

Survey Receivers
----------------

A receiver device owns one or more components. Device names are optional; the
receiver group name is the public survey identifier.

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
physical DAS gauge length, ``channel_spacing`` is the spacing between reported
channels and defaults to ``gauge_length``, and ``sample_spacing`` controls the
integration samples used along the gauge. If ``sample_spacing`` is omitted,
``points_per_gauge`` may be used instead. Helical fiber response adds
``radius`` and ``pitch``. Length-like DAS fields accept plain solver-scaled
numbers or Pint quantities with explicit units.

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

Point source kinds include ``scalar``, ``vector``, ``moment``, ``monopole``,
and ``dipole``. ``CompoundSource`` combines multiple points and weights into
one source group. When a simulation has multiple compatible sources, the solver
chooses efficient internal source batches automatically:

.. code-block:: python

   acq = fs.Acquisition()
   acq.add_sources(
       kind="scalar",
       coords=[[0.25, 0.05], [0.5, 0.05], [0.75, 0.05]],
   )

Sparse Survey Layouts
---------------------

Dense receiver groups evaluate all source/receiver/component combinations.
Sparse surveys select a subset or define rules such as offset windows:

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
explicit trace pairs are easier to define directly. Sparse survey inputs should
describe public trace identity: source ids, receiver ids, component names, and
weights. Internal sample maps and point ranges are runtime details and should
not be authored directly.
