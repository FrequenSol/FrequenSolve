# Frozen hp discretization

Status: internal native restart and verification primitive.

The HDF5 artifact contains the existing `/refinements` h-tree encoding, including
PML, plus `/discretization`:

| Dataset | Representation | Meaning |
| --- | --- | --- |
| `schema_version` | string | `fs-mesh-discretization-1` |
| `identity` | string, at most 128 characters | Required caller-owned execution identity |
| `geometry` | int64 scalar | Content fingerprint of the attached GMP geometry |
| `metadata` | four int32 values | Dimension, initial node count, initial root count, native order radix |
| `nodes` | flat int32 vector | Preorder records of node type, refinement kind, child count, encoded polynomial order |

Visit initial nodes in native initial-node order and then all children of each
node in native child order. This includes shared edges/faces, constrained traces,
interior nodes and inactive ancestors. Runtime allocation IDs and partition
ownership are not stored. Anisotropic orders retain the native radix encoding;
readers reject a different radix or dimension.

The writer requires a complete replicated topology. Collectively published
replicas must have identical geometry, execution identity, topology and orders.
Reduced GLU/GLUd views are rejected. Partition ownership may differ because it
is re-established through normal mesh replay. The reader compares geometry,
identity and metadata before resetting the mesh, reconstructs h-refinements,
checks every topology record, and only then installs orders and refreshes
geometry DOFs. Failure during replay is fatal; this is not a transactional
in-memory update. With `verify_only`, the reader compares the complete stored
topology and orders against the current mesh without replay or mutation; stage
execution uses this to verify the final solver discretization.

The caller must bind the execution identity to the stage, patch, frequency,
geometry parameter state and relevant discretization settings. This primitive
does not freeze those inputs or a solver hierarchy. Its writer creates a working
file; an immutable stage owner must hash and atomically publish the finished file,
then verify it before replay. Opening this artifact alone does not certify patch
FWI, fixed stage geometry, or derivative correctness.
