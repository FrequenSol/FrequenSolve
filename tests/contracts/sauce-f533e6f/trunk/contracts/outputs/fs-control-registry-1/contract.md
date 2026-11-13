# fs-control-registry-1

Resolved control baseline and active subspace. The top-level arrays, block
layouts and distributed block global IDs and owned global IDs describe the view
of `descriptor_rank` among `n_ranks` MPI ranks. `fingerprint` identifies the
complete baseline and ordered selection independently of the partition: it
hashes the global block layouts (distributed blocks by their global coefficient
count) and an order-independent digest of every coefficient and value keyed by
block and global ID, counted once on its owner. Control vectors and objective
states written on one rank count therefore match the registry on any other.
Only the index is published: no per-rank file is written.

See [shared inversion controls](../../../docs/imaging/controls.md) for coordinate definitions, block layouts and vector I/O.
