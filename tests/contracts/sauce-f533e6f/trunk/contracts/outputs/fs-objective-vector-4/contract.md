# Objective vector, version 4

`jvp` writes, and `vjp` reads, one objective-space vector per frequency task:
a JSON manifest and one HDF5 row file named by `file` (absolute, or relative to
the manifest directory). The format does not depend on the MPI partition; any
rank count may write or read it. It uses the real-Hermitian pairing.

For each zero-based term index `i`, the row file stores `N = n_global_rows`
canonical rows, row ID `r` at zero-based position `r - 1`:

- `/terms/i/coordinate_keys`: `(RHS, receiver/sample ID, component)` per row;
  external HDF5 shape `(N,3)`.
- `/terms/i/values`: complex values, external HDF5 shape `(N,2)` with real then
  imaginary parts. Single or double precision is accepted.

Each manifest term declares its ID, row count and the canonical layout
fingerprint of its keys. Readers reject a vector whose count, layout
fingerprint, keys or term order differ from the current objective, whose
`state_fingerprint` differs from the loaded state, or whose row file does not
match `sha256`. `manifest_fingerprint` hashes canonical sorted compact UTF-8
JSON with only that field omitted. FrequenSolve must recompute both hashes
when authoring or changing a dual. Earlier versions are rejected.
