# fs-control-registry-1

Resolved control baseline and active subspace, written once per task by the
root rank and independent of the MPI partition. `coordinates` and `values`
hold the complete baseline in global order; each block `layout` gives its
one-based offset and size in that vector. Distributed blocks are gathered into
global coordinate order and report `global_dofs`. `active_offsets` locate the
active blocks, in `active_blocks` order, within the active vector. The registry
fingerprint is the canonical JSON hash of this description without
`fingerprint`, so every rank count reports the same identity.

See [shared inversion controls](../../../docs/imaging/controls.md) for coordinate definitions, block layouts and vector I/O.
