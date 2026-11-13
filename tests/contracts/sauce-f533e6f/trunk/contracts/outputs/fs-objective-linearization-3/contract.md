# Distributed objective linearization, version 3

`linearize` writes one JSON manifest after every MPI rank closes its numerical
state shard. Every shard is identified by filename and SHA-256. The manifest
binds the resolved PDE/control context, partition rank count, the writing
frequency `task` index, and all shard contents. Every rank receives the same
`state_fingerprint`. A derivative action running under another task index
rejects the state; in a job with several `f_list` entries the `state` input
first tries `<stem>_<task><ext>` before the exact path.

The supported reuse policy is `same_mesh_partition`: the realized mesh,
coefficient/control state, active order, acquisition, observation processing and
MPI partition must match. Repartitioned PDE-cache reuse is not supported.

Each state shard contains the ordered objective terms and references one shared
rank-local HDF5 cache. Each term selects its `/terms/<zero-based-index>` group
through `cache.group`; `cache.file` and its SHA-256 may repeat across terms.
Identical arrays with identical attributes are HDF5 hard links, so common receiver
baselines, layouts and weights occupy storage once. Term-specific values remain
independent. Readers must use both the file and group locator. The updated reader
also accepts earlier version-three per-term files with `cache.group: "/"`.

The cache contains base and frequency-tangent receiver values, static
and robust weights, objective factors, metric weights, phase floors, active
masks, value conventions, and explicit `row_ids`, `coordinate_keys`, and
`n_global_rows` datasets relative to each term group. Coordinate keys are `(encoded RHS, receiver/sample ID,
component)`; the term ID supplies the outer namespace. IDs are one-based.
Dense rows follow Fortran `(RHS, component, receiver)` order. Sparse rows follow
encoded RHS then active trace catalog order; keys retain the original trace ID.
Zero-weight sparse traces are absent. Dense averaged/DAS rows identify output
channels rather than quadrature points.

Newly generated term caches also contain `objective_residual`, with the same
complex storage shape and row order as `simulated`. It is the observed-minus-
simulated comparison error after projection, preprocessing and source fitting,
multiplied by the frozen square-root static, metric and IRLS weights and the
objective factor (term weight, reduction mass and component normalization).
Inactive comparison rows are zero. Thus its real Jacobian pullback equals the
baseline objective covector. Frequency weights applied by a caller remain
external. Readers may still reuse earlier version-three caches without this
optional dataset for derivative actions, but cannot derive an LSQR right-hand
side from those caches alone.

Sparse term caches may contain `rhs_required`, one zero/one integer per
original acquisition RHS. A zero denotes a source deliberately omitted by the
scheduled acquisition; it has no cached receiver rows. Required source batches
must all finish before publication. Their original IDs are retained in row keys,
and a first required batch may begin above RHS one. Readers of earlier caches
without this dataset require every RHS. Cached selections are covered by the
shard hash and current acquisition compatibility checks.

Local cache coordinates are validated against the current execution layout;
matching array extents alone are insufficient. Replicated receiver views may be
cached on more than one rank. Objective pairings and vector persistence assign
one owner to each assembled row. Receiver assembly and its adjoint retain their
existing integration and incidence weights.

`manifest_fingerprint` is SHA-256 of canonical JSON (sorted keys, compact UTF-8),
excluding `manifest_fingerprint` and `state_fingerprint`. The latter equals the
manifest fingerprint. Shard hashes cover exact file bytes. Incomplete, missing,
corrupt and stale artifacts fail before derivative execution. The outer manifest
is saved only after all shard hashes have been exchanged successfully.

Versions 1 and 2 are obsolete and rejected. Regenerate state with `linearize`.

## Publication and lifetime

The backend saves objective state once when an explicit `linearize` action
finishes all source batches, per frequency task and MPI rank. Internal linear
solver iterations do not emit objective-state files. Reusing a state stem replaces
its cache; choosing distinct stems preserves separate optimizer checkpoints.
A loaded state must remain immutable for the lifetime of derivative actions.

The rank cache is assembled in task scratch and atomically replaced only after
all term groups are complete. Its exact hash and each group selector participate
in state validation. This preserves detection of payload corruption, changed
coordinates, incompatible value conventions and stale objective configuration.
Legacy per-term files from an earlier run are not automatically deleted.
