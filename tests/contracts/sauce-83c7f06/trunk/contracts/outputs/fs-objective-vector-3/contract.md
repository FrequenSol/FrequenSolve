# Distributed objective vector, version 3

`jvp` writes a JSON manifest and one HDF5 shard per MPI rank. `vjp` reads the same
format as an arbitrary objective-space dual. It uses the real-Hermitian pairing.
Each manifest term declares a stable ID, global row count and a canonical layout
fingerprint independent of row ownership. State reuse still requires the saved
mesh partition and rank count.

For each zero-based term index `i`, a shard stores:

- `/terms/i/row_ids`: unique owned one-based canonical row IDs.
- `/terms/i/coordinate_keys`: `(RHS, receiver/sample ID, component)` for each row;
  external HDF5 shape `(n_owned,3)`.
- `/terms/i/values`: complex values, external HDF5 shape `(n_owned,2)` with the
  last axis storing real then imaginary parts.

Empty ownership shards contain zero-length datasets. Across all shards, each
canonical row must appear exactly once. Input rows may move between shards or
change local ordering; the reader validates IDs and coordinate keys, then
scatters values to every required execution copy. Missing, duplicate, out-of-range
and incorrectly identified rows are errors. Redistribution uses bounded chunks;
it does not replicate a global receiver-value cube on every rank.

`state_fingerprint` binds the frozen objective state. Shard `sha256` hashes cover
exact file bytes. `manifest_fingerprint` hashes canonical sorted compact UTF-8
JSON with only that field omitted. FrequenSolve must recompute payload and
manifest hashes when authoring or changing a dual. Versions 1 and 2 are rejected.
