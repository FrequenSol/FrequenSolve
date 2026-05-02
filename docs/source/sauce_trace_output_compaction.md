# Sauce Trace Finalization Contract

This note describes the solver-side trace-output changes expected by the
Python SDK. The goal is to keep parallel frequency solves simple while giving
users one convenient trace product by default.

## Default Behavior

Frequency tasks may continue writing independent HDF5 shard files while they run
in parallel. After all tasks finish, Sauce should run a cleanup/finalization
step that writes a packed consolidated trace file and aggregates task metadata.

Default final layout:

```text
results/
  traces/
    traces.h5
    manifest.json
    shards/              # optional transient working files
  logs/
    task_1.log
    task_2.log
  _fs_run/
    run_manifest.json
    outputs.json
    timings.json
```

The public SDK should usually read `results/traces/traces.h5`, not individual
frequency shards.

## Packed Trace File

`traces.h5` should be self-contained by default. It should include:

- a frequency axis and sorted physical frequency values;
- one dataset per trace group;
- receiver ids as datasets;
- source ids as datasets;
- component names as datasets;
- physical receiver coordinates in the global frame;
- physical source coordinates in the global frame;
- coordinate units and component units where available;
- task ids, source job metadata, and enough provenance to combine adjacent
  frequency-band jobs without task-number conflicts.

The cleanup step should validate that every completed shard matches the expected
group names, shapes, dtypes, and trace layout before packing.

## Separate Storage Mode

The SDK may expose an opt-in mode such as `store_separate=True` for users who
want to keep per-frequency trace files instead of a packed file. In that mode,
each frequency file should be self-contained and should include the trace
metadata listed above, because there may be no consolidated metadata authority.

This mode is expected to be a niche workflow. The default should be packed
storage.

## Preliminary Metadata

Before launching independent frequency tasks, Sauce already runs preliminary
meshing/sizing work. That step should produce the receiver/source/component
metadata needed by the finalizer. The frequency tasks can then write only their
frequency-specific trace arrays and minimal task metadata.

Recommended preliminary metadata includes:

- receiver table with ids and global physical coordinates;
- source table with ids and global physical coordinates;
- component table with names, directions, and units;
- sparse survey trace table when sparse survey mode is active;
- expected trace group names, shapes, dtypes, and axis order.

## Outputs Manifest

`_fs_run/outputs.json` should report the finalized trace file:

```json
{
  "schema": "fs-outputs-1",
  "files": [
    {
      "kind": "traces",
      "format": "hdf5",
      "schema": "fs-traces-packed-1",
      "relative_path": "traces/traces.h5"
    },
    {
      "kind": "trace_manifest",
      "format": "json",
      "relative_path": "traces/manifest.json"
    }
  ]
}
```

When `store_separate=True`, `outputs.json` should list the separate shard files
or a manifest that lists them.

## Logs And Task Statistics

The cleanup step should aggregate task statistics into stable machine-readable
metadata. At minimum:

- task id;
- physical frequency;
- task status;
- trace output path;
- wall-clock runtime;
- MPI ranks, threads per rank, or core count;
- log path.

The SDK uses these fields for job summaries, timing plots, core-hour plots, and
targeted log retrieval.

Raw logs may remain as `logs/task_<task>.log`. Task numbers are one-based in the
SDK.

## Frequency-Safe File Names

If shard or finalization paths include physical frequencies, Sauce pathlib must
not treat decimal points in names like `10.00000` as file extensions.

For example, all of these should be valid path stems:

```text
f_10.00000_hz.h5
trace_10.00000.h5
```

The solver pathlib module should only treat the last suffix (`.h5`, `.json`,
`.vtu`, etc.) as the extension and should preserve decimal points in the stem.

## Python SDK Expectations

The SDK will treat packed storage as the normal result. Fetching traces should
download or locate `traces/traces.h5` and its manifest. Fetching logs should
support:

```python
site.fetch_logs(job)
site.fetch_logs(job, task=12)
site.fetch_logs(job, frequency=20.0)
```

The solver should keep task/frequency metadata stable enough that these calls
work after partial reruns and after cleanup finalization.
