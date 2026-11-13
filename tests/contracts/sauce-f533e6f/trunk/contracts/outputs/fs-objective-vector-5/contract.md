# Objective vector, version 5

`jvp` writes, and `vjp` reads, one objective-space vector per frequency task:
a JSON manifest `<stem>_<task>.json` and one HDF5 row file named by `file`
(`<stem>_<task>.h5` when Sauce writes it). Sauce writes the manifest-relative
file name; readers resolve a relative path against the manifest directory and
use an absolute path as given. The vector carries no partition binding: any
MPI rank count may write or read it. It uses the real-Hermitian pairing.

For each zero-based term index `i`, the row file stores `N = n_global_rows`
canonical rows, row ID `r` at zero-based position `r - 1`, in the row order of
the bound [objective state](../fs-objective-linearization-5/contract.md):

- `/terms/i/coordinate_keys`: `i4`, numpy shape `(N,3)`,
  `(RHS, receiver/sample ID, component)`.
- `/terms/i/values`: complex values, numpy shape `(N,2)`, real then imaginary
  part. Sauce writes `f4`; readers accept `f4` or `f8`.

Datasets have a fixed extent; Sauce writes them contiguously and collectively.

Each manifest term declares its `id`, `n_global_rows`, the canonical
`layout_fingerprint` of its keys and `digests`: the `fs-block-sha256-1`
digests (defined in [curvature output](../fs-curvature-output-1/contract.md))
of `coordinate_keys` and `values`, each flattened to one axis in C order with
the element tag of the stored type (`i4`; `f4` or `f8`). `state_fingerprint`
binds the vector to its objective state. `manifest_fingerprint` is
`sha256:` followed by the SHA-256 of the canonical JSON (sorted keys, compact
separators, UTF-8) of the manifest without `manifest_fingerprint`.
FrequenSolve recomputes both kinds of identity when it authors or changes a
dual.

Readers verify the manifest identity, the state binding, term order, row
counts, layout fingerprints and the digests of what they read; scattered rows
must match the current keys and be finite. Sauce publishes the row file
atomically before the manifest. Versions 1 to 4 are rejected: regenerate the
state with `linearize`.
