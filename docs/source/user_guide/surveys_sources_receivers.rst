Surveys, Sources, and Receivers
===============================

Acquisition contains a :term:`source geometry`, optional :term:`source encoding`,
:term:`receiver groups <receiver group>`, and optional :term:`sparse survey`
layouts. Coordinates are physical global coordinates by default, or
``CoordinateValue`` objects when a named coordinate system is used.

Related tutorials:

- :download:`Receivers <../../../tutorials/05_surveys/01_receivers.ipynb>`
  for multi-component :term:`receiver devices <receiver device>` and dense groups.
- :download:`DAS <../../../tutorials/05_surveys/02_das.ipynb>`
  for fiber-style strain receivers.
- :download:`Sources <../../../tutorials/05_surveys/03_sources.ipynb>`
  for physical point catalogs and sparse encoded-source fields.
- :download:`Sparse surveys <../../../tutorials/05_surveys/04_sparse_surveys.ipynb>`
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

Physical and Encoded Receiver Arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ReceiverArray`` places a physical array of receiver nodes around every
coordinate in a receiver group. Offsets stay compact in the acquisition JSON;
Sauce expands them when it maps receiver points:

.. code-block:: python

   device = fs.ReceiverArray(
       components=[fs.ReceiverComponent(name="pressure", field="pressure")],
       offsets=[[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]],
       offset_units="m",
       reduction="mean",
   )
   acq.add_receiver_group("surface", device, coords=receiver_coordinates)

Use ``reduction="sum"`` to sum the physical nodes or ``"none"`` to retain
every expanded node. ``ReceiverNodeArray`` remains as a deprecated loading shim
and new files always use ``ReceiverArray``.

``EncodedReceiver`` evaluates independently weighted responses on one fixed
receiver geometry. Complex coefficients are conjugated conveniently for
frequency-domain time reversal and are written to the simulation HDF5 input
store, not embedded in JSON or receiver metadata. Its weight tensor is ordered
as ``(encoding, component, receiver)``:

.. code-block:: python

   array = fs.EncodedReceiver(
       reduction="sum",
       components=[
           fs.ReceiverComponent(name="vx", field="velocity", direction=[1, 0]),
           fs.ReceiverComponent(name="vz", field="velocity", direction=[0, 1]),
       ],
       encoding_names=["focus_a", "focus_b"],
       weights=np.stack([response_at_target_a, response_at_target_b]),
   ).time_reversed()
   acq.add_receiver_group("surface", array, coords=receiver_coordinates)

Each target response above has shape ``(component, receiver)``. The resulting
channels are named ``focus_a:vx``, ``focus_a:vz``, and so on. For a scalar
device, ``(encoding, receiver)`` is accepted as a convenience. For split
real/imaginary weights over a single receiver, use the explicit shape
``(encoding, 1, 1, 2)``. A real ``(encoding, 1, 2)`` tensor means two receivers;
this shape is otherwise ambiguous.

Use ``reduction="mean"`` for normalized encoding or ``"none"`` to retain one
trace per receiver point. The bulk tensor path is preferred for production: it
uses vectorized validation, streams real/imaginary blocks to HDF5, and
``time_reversed()`` does not copy authored tensors. External weight tables must
be conjugated before they are attached. ``add_encoding(...)`` is a
convenient incremental builder. Base component descriptors occur only once;
large explicit encoding-name lists are also moved to HDF5 to keep acquisition
metadata small.

Acquisition-owned HDF5 payloads use separate ``file`` and ``dataset`` fields.
The older ``file:dataset`` locator is accepted when loading existing receiver
coordinate definitions, but new exports do not produce it. An existing weight
table can be attached with ``ReceiverWeightTable`` without loading or copying
its values in Python. Interim files that used a weighted ``ReceiverArray`` are
accepted by both FrequenSolve and Sauce, but new exports use
``EncodedReceiver``.

DAS
---

``ReceiverFiber`` represents a fiber-style receiver. ``gauge_length`` is the
:term:`DAS` :term:`gauge length`, ``channel_spacing`` is the
:term:`channel spacing` between reported channels and defaults to
``gauge_length``, and ``sample_spacing`` controls the :term:`sample spacing`
used along the gauge. If ``sample_spacing`` is omitted,
``points_per_gauge`` may be used instead. Helical fiber response adds
``radius`` and exactly one of ``angle`` or ``pitch``. ``angle`` is the winding
angle from the cable axis and must be strictly between 0 and 90 degrees. Plain
angle values are degrees, and angular :term:`Pint` quantities may use degrees
or radians. Length-like DAS fields accept plain solver-scaled numbers or Pint
quantities with explicit units.

.. code-block:: python

   u = fs.ureg

   das = fs.ReceiverFiber(
       gauge_length=10 * u.m,
       channel_spacing=10 * u.m,
       sample_spacing=2 * u.m,
       radius=2 * u.cm,
       angle=60 * u.deg,
   )
   das.add_component(name="eps_tt", field="strain", direction=[1.0, 0.0])

Survey Sources
--------------

Point source kinds include ``scalar``, ``vector``, ``tensor``, ``monopole``,
and ``dipole``. ``SourceGeometry`` describes physical source points. When
``source_encoding`` is omitted, each point is one identity :term:`source field`.
``SourceEncoding.named`` can instead combine named points into sparse fields:

.. code-block:: python

   geometry = fs.SourceGeometry.points(
       kind="scalar",
       coords=[[0.25, 0.05], [0.75, 0.05], [0.45, 0.08], [0.55, 0.08]],
       names=["left", "right", "pair_pos", "pair_neg"],
   )
   encoding = fs.SourceEncoding.named({
       "left": {"left": 1.0},
       "right": {"right": 1.0},
       "difference": {"pair_pos": 1.0, "pair_neg": -1.0},
   })
   acq = fs.Acquisition(
       source_geometry=geometry,
       source_encoding=encoding,
   )

For dense complex arrays, weights have encoding-major shape
``(n_encoded, n_source)``: each row describes one encoded source over the
physical source points. This matches encoded-receiver weights and the
field-major solver storage. Saved simulations move them into HDF5
automatically; the inline JSON form is retained only for small
direct-serialization examples.
Forward responses can be time-reversed while installing the encoding:

.. code-block:: python

   acq.encode_sources(
       responses_at_target,
       names=["focus"],
       conjugate=True,
   )

Frequency-dependent weights have shape
``(n_frequency, n_encoded, n_source)``. Passing the matching physical
frequency axis stores one chunked HDF5 tensor and lets each Sauce task read
only its slice:

.. code-block:: python

   acq.encode_sources(
       frequency_codes,
       frequencies=[2.0, 2.5, 3.0],
       names=["blend_1", "blend_2"],
   )

Large homogeneous source catalogs, dense source encodings, and large inline
sparse surveys are externalized automatically when a simulation is saved. The
JSON retains only compact HDF5 references; custom source and encoded-field
names are stored in HDF5 only when they differ from generated defaults.
Source-encoding ``reference_coordinates`` are deprecated: physical source
geometry and the simulation coordinate system are the authoritative location
description.

Use ``acq.source_point_count()`` for physical geometry size and
``acq.source_field_count()`` for the number of solver RHS fields. New code
should use ``add_sources()``, ``add_encoded_source()``, and
``source_encoding``. A small compatibility shim still accepts the deprecated
``source_groups``, ``add_source_group()``, and ``add_compound_source()`` APIs;
they are never serialized. Legacy untagged and ``fs-acquisition-1`` payloads
are migrated on input and always re-exported as ``fs-acquisition-2``.
The solver chooses efficient internal :term:`source batches <source batching>`
automatically.

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

Both factories build a columnar ``SparseTraceTable`` and write its NumPy
columns directly to HDF5 for production-sized surveys. Existing column arrays
can be attached without constructing one ``SparseTrace`` per row:

.. code-block:: python

   table = fs.SparseTraceTable(
       source_id=source_ids,
       receiver_id=receiver_ids,
       receiver_position_id=receiver_position_ids,
       component=components,
   )
   survey = fs.SparseSurvey.from_table("selected_traces", table)
