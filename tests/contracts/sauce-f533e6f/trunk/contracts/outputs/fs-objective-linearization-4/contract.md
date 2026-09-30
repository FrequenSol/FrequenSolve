# Objective linearization, version 4

`linearize` writes one JSON manifest and one HDF5 cache per frequency task.
The state does not depend on the MPI partition: derivative actions (`jvp`,
`vjp`, `normal`, `solve`) may reload it on any rank count. The manifest binds
the writing frequency `task`, configuration fingerprints, and each term's
configuration, cache location and canonical layout fingerprint. A
derivative action under another task index rejects the state; in a job with
several `f_list` entries the `state` input first tries `<stem>_<task><ext>`.

Each term selects `cache.group` (`/terms/<zero-based index>`) inside
`cache.file` (absolute, or relative to the manifest directory). Terms that
share a receiver group hard-link identical arrays. A term group stores
`N = n_global_rows` canonical rows, row ID `r` at zero-based position `r - 1`:

- `coordinate_keys` `(N,3)`: `(encoded RHS, receiver/sample ID, component)`.
- `simulated`, `simulated_df` `(N,2)`: base and frequency-tangent receiver
  values, real then imaginary; `simulated` carries the value-frame attributes.
- `objective_residual` `(N,2)`, optional: the observed-minus-simulated error
  after projection, preprocessing and source fitting, multiplied by the frozen
  square-root static, metric and IRLS weights and the objective factor. Its real
  Jacobian pullback equals the baseline objective covector.
- `static_weight`, `robust_weight` `(N)`.
- `metadata` `[4, sparse, n_rhs, n_components]`, `n_global_rows`,
  `metric_weight`, `objective_factor`, `phase_floor (n_rhs, n_components)` and
  `active_mask (n_rhs, n_components)`, which are identical for every row owner.

Dense rows follow Fortran `(RHS, component, receiver)` order. Sparse rows follow
the active trace catalog; zero-weight sparse traces are absent. Dense
averaged/DAS rows identify output channels rather than quadrature points.

`runtime.cache_fingerprint` is the SHA-256 of the exact cache bytes;
`state_fingerprint` hashes the configuration fingerprints, cache hashes and
cache groups. Readers reject stale fingerprints, a changed layout fingerprint,
and keys that differ from the current execution rows. Earlier versions are
rejected; regenerate state with `linearize`.

## Publication and lifetime

Rank owners route their rows to contiguous canonical slices and write them
into the shared cache before one rank publishes it atomically and writes the
manifest. Readers take one slice each and scatter rows to their execution
owners, rebuilding a rank-local lazily read cache in task scratch. Reusing a
state stem replaces its cache; distinct stems preserve separate checkpoints.
A loaded state must remain immutable for the lifetime of derivative actions.
