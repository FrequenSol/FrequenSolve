# Objective linearization, version 5

`linearize` publishes, per frequency task, a JSON manifest `<stem>_<task>.json`,
one HDF5 cache `<stem>_<task>_cache.h5` and the
[objective report](../fs-objective-report-1/contract.md)
`<stem>_<task>_report.json`. No per-rank file is written. Derivative actions
(`jvp`, `vjp`, `normal`, `solve`, extension, joint and kernel workflows) reload
the state. A derivative action running under another task index rejects it; in
a job with several `f_list` entries the `state` input first tries
`<stem>_<task><ext>`.

## Manifest

- `schema`, `task`, `frequency_hz`, `laplace_damping_hz`, `resolved_context`.
- `partition`: `{"n_ranks": N, "compatibility": "same_mesh_partition"}`. It is
  the only partition-specific field; readers reject another MPI rank count.
  The cache rows themselves are canonical and independent of the partition.
- `fingerprints`: `resolved_context`, `model`, `observations`, `acquisition`,
  `source_signature`, `preprocessing`, `objective_terms`, and when present
  `preprocess_state` (content of hook state files such as a pinned material
  factor, after the baseline objective wrote them) and `spectral_series`
  (`kernel_derivative` trace series). Readers recompute and compare each one.
- `runtime.n_terms` and `terms`, in objective order. Each term records `id`,
  `receiver_group`, `weight`, `reduction`, `effective_weight_mass`, `scale`,
  `objective`, `comparison`, `receiver_layout` (`dense` or `sparse`),
  `data_metric`, `layout_fingerprint` (identity of the canonical row keys),
  `cache` and `runtime` (`n_rhs`, `n_components`, `n_inactive`,
  `n_global_rows`, `cache_fingerprint`).
- `cache.file` names the HDF5 cache; Sauce writes the manifest-relative file
  name. Readers resolve a relative path against the manifest directory and use
  an absolute path as given. `cache.group` is `/terms/<zero-based index>`.

Identities, all written `sha256:<64 lowercase hex>` except digests:

- `cache.digests` maps every dataset of the term group to its
  `fs-block-sha256-1` digest (defined in
  [curvature output](../fs-curvature-output-1/contract.md)) of the dataset
  flattened to one axis in C order, with the element tag of the stored type
  (`i4`, `i8`, `f4` or `f8`). Linked datasets have equal digests.
- `runtime.cache_fingerprint` is the SHA-256 of the canonical JSON (sorted
  keys, compact separators, UTF-8) of the term's `digests` object.
- `state_fingerprint` is the SHA-256 of the canonical JSON of
  `{"frequency_hz", "laplace_damping_hz", "fingerprints", "cache_groups",
  "cache"}`, where `fingerprints` equals the manifest object and the arrays
  `cache_groups` and `cache` list each term's `cache.group` and
  `cache_fingerprint` in term order. Consumers treat it as opaque and copy it
  into objective vectors, control vectors and checkpoints.
- `manifest_fingerprint` is the SHA-256 of the canonical JSON of the manifest
  without `manifest_fingerprint` and `state_fingerprint`.

## Cache

Each term group stores `N = n_global_rows` rows in canonical order: row ID `r`
(one-based) at zero-based position `r - 1`. Dense row IDs are
`rhs + n_rhs*((component - 1) + n_components*(receiver - 1))`. Sparse row IDs
follow the active (nonzero-weight) trace catalog; selected-source rows keep
their original RHS and trace IDs and zero-weight traces are absent. Dense
averaged/DAS rows identify output channels rather than quadrature points.

Shapes are numpy/h5py shapes. Complex values store real then imaginary parts.
Row values are `f4` (the frozen objective state is single precision in every
build); `C = n_components`, `R = n_rhs`.

| Dataset | Type, shape | Content |
|---|---|---|
| `metadata` | `i4 (7)` | `[5, sparse, n_rhs, n_components, has_residual, has_weighting, weighting_df]`, flags 0/1 |
| `n_global_rows` | `i8 (1)` | `N` |
| `metric_weight` | `f8 (C)` | data-metric component weights |
| `objective_factor` | `f4 (C)` | term weight, reduction mass and component normalization |
| `phase_floor` | `f4 (C,R)` | frozen phase stabilization floors |
| `active_mask` | `i4 (C,R)` | 1 where a comparison is active |
| `rhs_required` | `i4 (R)` | 1 for a scheduled RHS, 0 for one omitted by an acquisition selection |
| `weighting_factor_fingerprint` | `i8 (1)` | live material weighting only, see below |
| `coordinate_keys` | `i4 (N,3)` | `(encoded RHS, receiver/sample ID, component)`, one-based |
| `simulated`, `simulated_df` | `f4 (N,2)` | baseline and frequency-tangent receiver values; `simulated` carries the `value_convention_*` attributes |
| `objective_residual` | `f4 (N,2)` | when `has_residual` |
| `weighting_residual`, `weighting_residual_df` | `f4 (N,2)` | when `has_weighting` / `weighting_df` |
| `static_weight`, `robust_weight` | `f4 (N)` | frozen static and IRLS weights |

`objective_residual` is the observed-minus-simulated comparison error after
projection, preprocessing and source fitting, multiplied by the frozen
square-root static, metric and IRLS weights and the objective factor; inactive
comparison rows are zero. Its real Jacobian pullback equals the baseline
objective covector. Frequency weights applied by a caller remain external.

Terms with a live `material_weighting` hook store `weighting_residual`, the
raw error e before the factor W, `weighting_residual_df` for spectral
comparisons, and `weighting_factor_fingerprint`, one 64-bit identity of W: an
order-independent combination of per-point hashes over unique global receiver
points, so it does not depend on the rank count or point order. `jvp`, `vjp`
and `normal` re-evaluate W at the current model and reject a state whose
fingerprint differs.

Only sparse caches may contain zeros in `rhs_required`; omitted sources have no
rows. Required source batches must all finish before publication.

Every dataset has a fixed extent and contiguous storage, without chunking or
filters. A term that shares its receiver group with an earlier term hard-links
each dataset whose digest, shape and type equal the earlier term's (for
`simulated`, also equal value conventions).

## Publication and reading

Writers route each row's unique owner copy to balanced contiguous canonical
slices and write them collectively with parallel HDF5; builds without it
append the same slices in rank order. Digests are computed from memory while
writing. The cache is written as `<cache>.partial`, renamed atomically after
every rank closed it, and the manifest is then written atomically. A manifest
left at the same path by an earlier run is removed first, so a failed run
publishes no manifest. Manifest and cache are durable task outputs.

Readers validate the schema, task, partition, frequency, fingerprints and
manifest identity. Every rank reads one canonical slice of each dataset, the
digests are verified in memory, and rows are scattered to their execution
owners, rebuilding a rank-local lazily read cache in task scratch. Keys must
equal the current execution rows. Versions 1 to 4 are rejected: regenerate the
state with `linearize`. Reusing a state stem replaces its cache; distinct stems
preserve separate checkpoints. A loaded state must stay immutable while
derivative actions use it.
