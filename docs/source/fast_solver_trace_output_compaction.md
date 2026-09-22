# Trace finalization and successful-task retention

FrequenSolve and Sauce now use matching v2 artifact contracts. Each frequency task
publishes `_fs_run/tasks/task_NNNNNN/result.json`; packing and smoothing publish
`_fs_run/operations/pack/result.json` and `operations/smooth/result.json`.
The records contain the authoritative artifact paths, full producer input hashes,
convergence, timing, resource and provenance information. Readers follow these
records rather than constructing filenames or listing output directories.

## Defaults

Completed task shards are packed into immutable HDF5 segments. A manifest maps
exact task/frequency identities to segment paths and dataset numbers. Task records
are updated before raw shards and duplicate survey metadata are removed. Repeated
packing with no changes creates no additional segments or manifests. Replacement
packing removes previous segments only when no current task or pack references them.
An unsuccessful replacement leaves the previously committed pack intact.

Jobs with fewer than 32 tasks read their task JSON directly. Larger jobs also use
`_fs_run/tasks.h5`, a derived index rebuilt after packing and invalidated when a
task is replaced. Detailed task JSON remains available for provenance.

## Retrying unconverged tasks

```python
job.preserve_task_outputs = True
job.save()
```

The setting defaults to `False`. Enabling it both retains raw task files after
packing and permits successful tasks to be reused after numerical solver settings
change. Failed tasks are retried. Forward and imaging job constructors also accept
`preserve_task_outputs=True`.

Physical simulation, acquisition, output request, frequency partition, mesh, and
structural solver changes invalidate reuse. Full original hashes remain attached
to each task; a separate compatibility fingerprint authorizes solver-setting reuse.
`--fresh` forces a new solve under either policy. Save/stage again after changing
referenced external input files so their fingerprints are refreshed. The staging
record uses the actual rewritten remote payloads for remote task validation.

Use `job.traces.open()`, `job.wavefields.open()`, and `job.load_images()` to resolve
products. Packed dataset numbers need not equal task numbers. Sampled wavefields
may span multiple logical families and immutable segments.

## Compatibility and limits

This migration requires matching Sauce and FrequenSolve artifact readers; the old
v1 run manifests are not dual-written. Managed shared image and retained-field
containers also use generation paths. Smoothing resolves committed task artifacts.
Trace/image publication closes
and synchronizes payloads before atomically replacing result records. Producers
and packers coordinate through a local publication lock. A cached lazy reader
must reopen its catalog after concurrent replacement; this is not an indefinite
lease on old generations.

Explicit FWI checkpoint stems and caller-selected export files retain their own
lifetime policy. External image save paths and fixed visualization/export payloads do not become immutable just
because their result metadata is atomic. Crash-orphan cleanup is separate from
normal replacement cleanup. The task-local `preprocess.json` holds current-attempt
auxiliary provenance rather than an immutable attempt history. Local tests cover 2D double forward, RTM, spectral
imaging, sampled wavefields, and two-rank sparse FWI; live cloud/HPC publication,
3D and GPU execution were not newly qualified by this change.
