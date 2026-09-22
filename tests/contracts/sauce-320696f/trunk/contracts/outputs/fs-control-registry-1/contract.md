# fs-control-registry-1

Resolved control baseline and active subspace. The top-level arrays describe `descriptor_rank`; `rank_descriptors` indexes all MPI rank views with canonical JSON fingerprints. Every rank descriptor includes distributed block global IDs and owned global IDs. The common registry fingerprint covers every rank, its baseline and ordered selection.

See [shared inversion controls](../../../docs/imaging/controls.md) for coordinate definitions, block layouts and vector I/O. Rank descriptor files use the core fields without the top-level index fields.
