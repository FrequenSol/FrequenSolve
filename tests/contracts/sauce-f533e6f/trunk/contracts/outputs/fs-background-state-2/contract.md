# Material background checkpoints, version 2

`fwi_operator.background` on ordinary material `linearize` publishes a
task-suffixed JSON manifest after all native forward batches have completed.
This is an execution-site cache, not a portable wavefield representation, and
reuse is an optimization: it never decides whether an action succeeds.

The manifest records `schema`, `layout` (`shared` or `rank`), `storage`
(`l2` or `full`), resolved `context`, objective `state`, solver `build`, MPI
`ranks`, a `slabs` array indexed by zero-based rank (`columns`, `offset`) and a
`batches` array indexed by zero-based batch (one-based `source_first`,
`n_sources` and the batch `digest`). The `rank` layout names one file per rank
in `shards[r].file` (Sauce writes `<manifest stem>/generations/<run>/rank_NNNNN.h5`);
the `shared` layout names one HDF5 `file` written by every rank
(`<manifest stem>/generations/<run>/fields.h5`). File names are relative to the
manifest directory or absolute. `fwi_operator.background_layout` selects the
layout (default `rank`) and the environment variable `FS_BACKGROUND_LAYOUT`
overrides it; readers follow the manifest.

Each batch's payload is the ranks' slabs concatenated in rank order: rank `r`
owns columns `offset .. offset+columns-1` of `slabs[r]`, and `C` is the sum of
all `columns`. In the `shared` file, group `<batch>` holds:

- `values`: numpy shape `(C,nrhs,2)` (real and imaginary parts) or `(C,nrhs)`
  for real DOFs, so each rank's slab is one contiguous byte range. Values keep
  native precision: `f4` in single-precision builds, `f8` otherwise.
- `layout`: `i8`, numpy shape `(ranks,L)`, one row per rank. `full` rows are
  the native `FS_SOLUTION_H5` layout version 3: version (3), scalar precision
  in bits, native value kind, mesh state, RHS count, DOF count and forward
  solve-context code. `l2` rows are the `FS_BACKGROUND_L2` layout: version (1),
  precision, value kind, mesh state, RHS count, L2 DOF count, full DOF count
  and solve-context code.
- `normalization`: `f8`, numpy shape `(ranks,nrhs)`, each rank's physical RHS
  scales.

A `rank` file has the same groups holding only its rank: `values` of numpy
shape `(columns,nrhs,2)` or `(columns,nrhs)`, and one `layout` and
`normalization` row (numpy shapes `(1,L)` and `(1,nrhs)`).

`full` slabs hold every native DOF column of the rank; `l2` slabs hold the
recovered L2 fields in stable mesh-node order, and restore clears the omitted
trace coefficients. Datasets have fixed extents and contiguous, unfiltered
storage. A batch `digest` is the `fs-block-sha256-1` digest (see
[curvature output](../fs-curvature-output-1/contract.md)) of the batch's
concatenated values flattened to one axis in C order (the shared `values`
dataset, or the rank files' `values` in rank order), with element type `f4` in
single-precision builds and `f8` otherwise, so it does not depend on the layout.
It is computed from memory across ranks while writing and verified the same
way while reading; no file is hashed separately. Each reader compares its own
`layout` and `normalization` rows with its live DOF layout and RHS
normalization instead.

Frozen-Gram acoustic/elastic DPG material actions use `l2` storage when no
material-dependent receiver terms, impedance/gravity boundaries outside the PML,
projection, or field outputs sampling trace coefficients require traces. PML
cells take no material derivative, so an impedance PML termination qualifies;
control-gradient outputs use the covector. Other supported actions use `full`.
Readers reject `l2` checkpoints when the requested action or output needs
traces and accept `full` checkpoints in either case. Carrier side-car DOFs,
2.5D wavenumber loops and spectral, source-control or extension actions are
unsupported.

Parallel-HDF5 builds create a shared file and its datasets collectively
through MPI-IO and write every slab with independent transfers; other builds
write the slabs in rank order. Rank files are created and written by each rank
alone, without collective I/O. `FS_MPIIO_HINTS="key=value,..."` sets MPI-IO
hints of every parallel HDF5 file; before it, a shared checkpoint file of at
least 1 GiB gets `striping_unit=4194304` and `striping_factor=min(nodes,16)`,
which `FS_MPIIO_HINTS` overrides key by key. Hints take effect when the file
is created, and MPI-IO ignores keys it does not support.

Capture sizes every rank's slab before any file exists: ROOT requires the
file system's free space (as reported by `statvfs`; quotas are not visible)
to hold the estimated files, every rank must be able to open its file, and
all ranks decide together. Each converged batch then moves directly from the
live DOFs; `l2` payloads pass through a packing buffer bounded by 64 MiB
(`FS_BACKGROUND_CHUNK_BYTES`). If a background solve did not converge, the
space is short, or any rank cannot create or write its file, all ranks
abandon the capture together after their current collective step, the files
and a manifest left at the same path by an earlier linearization are deleted,
and linearize completes without a manifest, recording warning
`FS_BACKGROUND_CAPTURE_DISABLED` and misc `background_capture_disabled`.
Reported HDF5 and MPI-IO failures never abort the run; an MPI-IO operation
that never returns on some rank can still stall it.

Material JVP/VJP/normal actions restore each source batch instead of solving
it: every rank opens the shared file or its own on its own and reads only its
slab. A different solver build, MPI rank count or source batching, a missing,
unreadable, malformed or other-version manifest (including
`fs-background-state-1`), a missing or unreadable file, a corrupt batch, or a
differing DOF layout or RHS normalization is a cache miss: every rank
recomputes the forward fields of the remaining batches, misc
`background_reuse_misses` counts them, and warning `FS_BACKGROUND_REUSE_MISS`
states the reason. A different `context`
or `state` fingerprint, or `l2` storage for a trace-dependent request, is an
error. Misc `background_solves_reused` counts restored batches;
`background_checkpoint_bytes_written` sums the file sizes,
`background_checkpoint_bytes_read` sums the payload and normalization bytes
every rank read, and `background_capture_nanoseconds`/
`background_restore_nanoseconds` report the slowest rank.

The manifest is a durable default output. ROOT registers every file as a
`checkpoint` artifact (schema `fs-background-data-2`) of the run generation;
checkpoints require explicit transfer selection and are retired when a later
run replaces the task result. Reuse removes background propagation, not
matrix assembly/factorization.
