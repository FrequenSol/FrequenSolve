# Receiver vector, version 1

A receiver vector is a dual in the real-Hermitian pairing
`Re sum(conj(g) * delta U)`. It is separate from an objective vector. Each
manifest binds its dual to one receiver state, physical frequency, channel,
value convention, and mesh partition. The source-state and shard SHA256 checksums
cover exact bytes.

A shard stores `/keys` as `(physical shot/RHS, receiver/sample, component)`
and `/base` as complex real and imaginary columns. `base_df` additionally
stores `/df`. Keys must have unique complete coverage of `n_rows`. Shard order
does not define receiver identity. Both channels may contain zero values.
For sparse states, the second key is the survey `trace_id`; for Cartesian
states, it is the global receiver point ID.
Each shard declares its MPI `rank`. A vector key must remain on the rank that
owns that key in the referenced receiver state; shard and row order may change.

External objective code writes dual vectors under this contract. The solver's `receiver_vjp`
prepares its key index and checks shard hashes once, then reads only the
active physical source batch before each adjoint. The matching `receiver_jvp`
writes tangent vectors in the same keyed format.

For multiple configured groups, this is a member of the hash-bound
[receiver vector collection](../fs-receiver-vector-bundle-1/contract.md).
The native action shares PDE fields and solves across members; this single-group
wire format remains unchanged.
