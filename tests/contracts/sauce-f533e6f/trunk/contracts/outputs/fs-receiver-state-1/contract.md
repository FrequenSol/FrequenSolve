# Receiver state, version 1

This manifest indexes receiver values in the physical receiver frame for one
frequency. A shard is HDF5 with `/keys` as an integer `(n,3)` array of
`(physical shot/RHS, receiver/sample, component)`, and `/predicted` and
`/observed` as `(n,2)` real and imaginary columns. `base_df` also requires
`/predicted_df` and `/observed_df` in physical Hz. Rows may occur in any shard
or order. Keys must be globally unique and their number must equal `n_rows`.
Each shard declares its MPI `rank`; a rank may own multiple or zero rows. For
`geometry: sparse`, the second key is the survey's stable `trace_id`, including
for a trace that combines several physical samples. Zero-weight traces are
omitted. For `geometry: cartesian`, it is the global receiver point ID.
Native states list each source batch's value convention and RHS normalizations in
`frames`, indexed by `(rank, batch)` with an explicit source range. This frame
index is independent of receiver-value shard order. Older states without
`frames` retain their rank-major source-batch shard requirement.

The checksum covers each whole shard. Receiver values are recorded before
comparison attributes, residual signs, losses, and objective normalization.
Every input frequency in a band must identify the same physical model candidate,
acquisition, observation, preprocessing, receiver group, value convention and
source derivative policy. Control registry and resolved execution context are
frequency-specific; each returned dual remains bound to its own state.
The candidate fingerprint covers the Model declaration, referenced model input
contents, and, when supplied, the external `fwi_operator/controls/state`
artifact. The acquisition fingerprint covers its declaration and referenced
source or receiver input contents. A payload changed in place at the same path
changes the corresponding physical identity.
Different control coordinate layouts require an explicit mapping and transpose
before an external optimizer sums the frequency covectors. Receiver actions accept
full-dimensional 2D and 3D acoustic, elastic, and coupled physical shots in
Cartesian or sparse receiver layouts. A frequency-specific `state_fingerprint` binds the
returned dual to its source state. Reuse requires the same mesh partition.

`fwi_operator` actions `receiver_linearize`, `receiver_jvp`, and `receiver_vjp`
produce or consume these states for the same full-dimensional seismic cases.
`receiver_linearize` defaults to `field_retention: checkpoint`; callers may
select `replay`. Acoustic, elastic and coupled `base_df` states support
checkpoint reuse. A checkpoint state lists one
base field file per rank and source batch, plus a df field file for `base_df`.
Each entry binds the source
range, MPI rank, batch number, and whole-file hash. State shards also retain
the exact base and df RHS normalizations. A checkpoint is valid only for the
same frozen candidate, source policy, frequency, control registry, mesh
partition, and DOF layout. VJP defaults to the state's retention policy and
accepts `field_reuse: replay` explicitly. A requested missing or corrupt
checkpoint fails rather than replaying silently. Reuse in another process
still assembles the operator before adjoint solves; numerical factors are not
serialized.

Field storage scales approximately as two complex DOF arrays per active
source batch for `base_df` (one for `base`), or about
`2 * n_dof * n_rhs * complex_storage_bytes` plus HDF5 metadata across the
saved batch collection. Only the active batch is restored in memory. Saved
checkpoints remain alongside their manifest until the caller deliberately
removes the state and its referenced payloads. External band evaluators consume
the receiver values without loading the retained fields.

For multiple configured groups, this is a member of the hash-bound
[receiver state collection](../fs-receiver-state-bundle-1/contract.md).
The native action shares PDE fields and solves across members; this single-group
wire format remains unchanged.
