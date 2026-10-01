# fs-control-registry-1

Resolved control baseline and active subspace. The top-level arrays describe `descriptor_rank`; `rank_descriptors` lists the canonical JSON fingerprint of every MPI rank's view, in rank order. A rank view includes distributed block global IDs and owned global IDs. The common registry fingerprint covers every rank, its baseline and ordered selection. Only the index is published: no per-rank descriptor file is written.

See [shared inversion controls](../../../docs/imaging/controls.md) for coordinate definitions, block layouts and vector I/O.
