# Frozen solver refinement sequence

Status: internal native h/p request capture and replay. The
[frequency mesh companion](../fs-stage-mesh-1/contract.md) owns application capture,
verification and publication.

A sequence binds an ordered set of native refinement decisions to a caller-owned
stage/frequency/discretization identity. Each step runs before solver ownership
preparation and refinement execution. Replay supplies the saved requests without
consulting candidate-dependent adaptation rules. Both ordinary and shared
multigrid hierarchy setup accept the sequence; hierarchy depth is preserved.

The HDF5 file contains:

| Dataset | Representation | Meaning |
| --- | --- | --- |
| `/schema_version` | string | `fs-refinement-replay-1` |
| `/identity` | string, up to 128 characters | Required execution identity |
| `/metadata` | three int32 values | Dimension, native order radix, number of steps |
| `/steps/NNNNNNNN/key` | five int32 values | Grid, refinement index, refinement type, refinement kind, run length |
| `/steps/NNNNNNNN/geometry` | int64 scalar | Attached GMP content fingerprint |
| `/steps/NNNNNNNN/state` | flat int32 vector | Six-word records in root/child preorder |

Each record contains node type, current h-refinement kind, number of middle-node
children, current encoded polynomial order, requested h-refinement kind and
requested encoded polynomial order. Include all initial-root middle nodes and
their descendants, including inactive ancestors. Zero requests are explicit;
empty refinement steps still have a topology record and key. Node allocation IDs
and partition ownership do not appear in the artifact.

Recording checks replicated topology/orders and collective metadata, combines
rank-local request sets, and rejects conflicting duplicate requests. Both capture
and replay install the same canonical request union before native execution.
Replay checks the full pre-step middle-node state, geometry and key before
replacing pending requests. Ending with unconsumed steps or requesting an extra
step is an error. Reduced GLU/GLUd topology is unsupported.

This artifact complements the full hp discretization checkpoint: it preserves
coarse-to-fine decisions, while the hp checkpoint also records trace orders.
The owner must bind geometry parameter state and all stage/configuration inputs
to the execution identity, atomically publish and hash the finished artifacts,
verify hashes before loading, and compare the realized final discretization.
The sequence alone does not pin reference files, initial/final wave meshes,
geometry controls, acquisition grading, or the stage lifecycle.
