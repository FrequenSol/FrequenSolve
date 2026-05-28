Fast Solver API Updates
=======================

This page is the implementation checklist for :term:`fast solver` contracts
required by the public Python :term:`SDK`. The Python SDK now treats these
features as first-class API concepts and keeps any legacy compatibility private.

Input Layout
------------

The :term:`SDK` is converging on a two-file default export:

* ``sim.json`` contains solver configuration, model references, acquisition,
  :term:`output requests <output request>`, units, coordinate-system declarations, :term:`rerun fingerprints <rerun fingerprint>`,
  and any structured metadata needed by the solver.
* ``sim.h5`` contains materialized local arrays, grids, :term:`sparse survey` tables,
  source/receiver metadata, and compact input datasets.

The solver should continue to accept external model files when a user explicitly
references a file-backed property, :term:`SEG-Y` input, or another specialized format.
The SDK will not read or validate server-only files before export.

Units and Coordinate Systems
----------------------------

Scalar or vector quantities may be written as objects:

.. code-block:: json

   {"value": [0.0, 100.0, 250.0], "units": "m", "system": "global"}

The fast solver should preserve the following fields wherever a
coordinate-bearing value is accepted:

* ``value``: scalar, vector, or array payload.
* ``units``: optional :term:`Pint`-compatible unit string.
* ``system``: optional coordinate-system name.
* additional advanced fields supplied by SDK ``extra`` dictionaries.

:term:`Simulations <simulation>` may include ``global_coordinate_system`` and ``coordinate_systems``.
Solver outputs should store physical source and receiver coordinates in the
global frame even when the input uses local coordinate systems.

Material Properties
-------------------

Material properties are authored as a dictionary and exported with canonical lowercase
names such as ``vp``, ``vs``, ``rho``, ``qp``, and ``qs``. The fast solver
should accept structured property payloads in addition to scalar values.

File-backed property:

.. code-block:: json

   {
     "file": "/server/data/vp.bin",
     "scale": 0.001,
     "units": "m/s",
     "grid": "model_grid",
     "absolute": true
   }

Expression-backed property:

.. code-block:: json

   {
     "expression": {
       "op": "multiply",
       "args": [
         {"value": 0.5},
         {"property": "vp"}
       ]
     },
     "units": "m/s"
   }

The fast solver should evaluate expression-backed properties after loading
referenced base properties. Expression nodes currently used by the SDK are ``property``,
``value``, ``op``, and ``args``. Required operations are ``add``, ``subtract``,
``multiply``, ``divide``, and ``power``.

Sparse Survey Contract
----------------------

The acquisition contract now supports :term:`sparse survey` tables for many-to-many
source/receiver selections. The fast solver should treat survey tables as
authoritative when present instead of assuming dense source by receiver
Cartesian products.

Expected :term:`HDF5` datasets:

* ``source_ids``: integer or string source identifiers.
* ``receiver_ids``: integer or string receiver identifiers.
* ``source_index`` and ``receiver_index``: row mappings into source and receiver
  metadata tables.
* optional per-row weights, masks, :term:`components <component>`, and extra survey columns.

Trace Storage
-------------

Public :term:`SDK` naming is now ``TraceDataset``. :term:`Trace <trace>` reads return plain
``xarray.DataArray`` objects with FrequenSolve helpers available through the
``.fs`` accessor. The fast solver should
rename receiver-output folders and files from ``receivers`` to ``traces`` when
writing new outputs. Legacy readers may remain internal while existing data is
migrated.

The fast solver may continue writing one temporary trace :term:`HDF5` :term:`shard` per
frequency task while tasks are running, because frequency tasks are run in
parallel processes and HDF5 should not be shared by independent writers. After
all tasks finish, the fast solver should run a cleanup/:term:`finalizer`
step that packs completed frequency outputs into one consolidated trace HDF5 file by
default:

.. code-block:: text

   results/
     traces/
       traces.h5
       manifest.json
       shards/
     _fs_run/
       run_manifest.json
       outputs.json
       timings.json

``traces/traces.h5`` is the normal public :term:`packed trace file`. It should be
self-contained and should store:

* store receiver ids as datasets, not long HDF5 attributes;
* store source ids as datasets;
* store physical receiver coordinates in the global coordinate frame;
* store physical source coordinates in the global coordinate frame;
* store source/receiver :term:`component` metadata and coordinate units;
* include enough frequency/task metadata to combine adjacent frequency-band jobs
  without task-number conflicts.

Before launching independent frequency tasks, the preliminary meshing/sizing
:term:`job` should output receiver/source/component metadata for the finalizer. The
frequency tasks can then write frequency-specific arrays and minimal task
metadata. If an opt-in separate-storage mode such as ``store_separate=True`` is
used, each per-frequency file should be self-contained because no packed
metadata authority may exist. See :doc:`fast_solver_trace_output_compaction` for the
trace-finalization contract.

Combined trace datasets should identify each frequency by physical frequency
value and source job metadata, not only by task number.

``outputs.json`` should list the packed trace file and trace :term:`manifest` with
stable relative paths and schema versions. In separate-storage mode, it should
list the shard manifest or every produced shard. The SDK can then use the fast
solver's output manifest as the authoritative artifact list instead of guessing
file names.

Rerun Fingerprints
------------------

The fast solver already writes useful run metadata under ``results/_fs_run``.
The SDK now uses this directory when deciding whether a completed job is
current. Required or strongly preferred files:

* :term:`run manifest` data in ``run_manifest.json`` with ``schema``, ``solver_version``, ``build_id``,
  ``job_file``, ``job_file_sha256``, ``simulation_file``,
  ``simulation_file_sha256``, ``result_path``, ``start_time``, ``end_time``,
  ``exit_status``, and ``exit_code``.
* ``outputs.json`` with ``schema`` and a ``files`` list containing produced
  output files, relative paths, kinds, schemas, and key dataset paths where
  known.
* ``timings.json`` with run timing summaries.
* ``error.json`` when a run fails.

A :term:`job` can be skipped or treated as complete only when:

* ``run_manifest.json`` reports ``exit_status: "success"``;
* ``job_file_sha256`` matches the current job JSON;
* ``simulation_file_sha256`` matches the current simulation JSON;
* every required trace file from the SDK ``TraceManifest`` or fast solver
  ``outputs.json`` exists locally/remotely.

Large local datasets may also carry dataset-level hashes in ``sim.h5``. The
fast solver should use those hashes to avoid rewriting unchanged input datasets
and to diagnose stale solver outputs. The old sidecar ``data_manifest.json``
workflow is not part of the public contract.

ParaView Outputs
----------------

Outputs are now owned by job :term:`JSON` rather than simulation JSON. The fast solver
should read ``Outputs`` from ``frequensolve-job-1`` files and write output paths
relative to the job result directory. Trace output is always requested under
``Outputs.traces`` and new runs should write ``traces/traces_<task>.h5``.

The public SDK exposes a deliberately small :term:`ParaView` API and emits the richer
fast solver contract internally. The default writer is :term:`VTK` :term:`VTU` with
appended binary data:

.. code-block:: json

   {"format": "vtu", "encoding": "appended"}

The only alternate writer exposed publicly for now is :term:`XDMF` backed by HDF5:

.. code-block:: json

   {"format": "xdmf", "encoding": "hdf5"}

Simple volume output continues to use ``fields`` and ``properties``. Surface
output is exposed through shell, boundary-label, model-surface, and plane
selectors. The SDK also exposes regular grid targets.

.. code-block:: json

   {
     "kind": "surface",
     "coordinates": {"system": "global"},
     "mesh": {"order": 3, "upscale": 2, "show_pml": false},
     "selection": [
       {"kind": "model_surface", "name": "top"},
       {"kind": "plane", "system": "global", "axis": "z",
        "value": {"value": 0.5, "units": "km"},
       "tolerance": {"value": 10.0, "units": "m"}}
     ]
   }

The only public complex field-part names are ``real``, ``imag``, and ``abs``.
When a user requests parts, the SDK emits normalized ``items`` internally.

Alternate data sources, :term:`VTR`, field-component bases, arbitrary item objects, and
other specialized controls are intentionally not part of the public :term:`Python API`
yet. They may still be supplied through advanced ``extra`` dictionaries when
needed for internal solver testing. Advanced nested writer fields should be
supplied through the ``writer`` argument so the SDK can keep the enforced public
``format`` and ``encoding`` defaults intact.

Mesh Adaptivity
---------------

The :term:`fast solver` should expose the :term:`mesh adaptivity` options used by
``adapt_mesh.F90`` as structured JSON fields. The SDK currently models:

* distance and :term:`surface grading` specifications;
* grading fields compatible with ``grading_fields_m``;
* :term:`source grading` controls;
* :term:`receiver grading` controls.

Refinement payloads should be extensible and accept solver-specific advanced
fields without requiring SDK changes.
