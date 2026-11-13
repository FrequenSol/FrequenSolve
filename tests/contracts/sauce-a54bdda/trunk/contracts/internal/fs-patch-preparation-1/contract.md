# Root patch preparation request

Status: internal

The `patch_prepare` job operation accepts this block as `PatchPreparation`.
Coordinates and explicit nonnegative `padding` are in metres. Each uniquely
named patch supplies either physical parent `roots` or `lower`/`upper` bounds
matching the model dimension. Omitting `patches` requests only the parent
inventory. Preparation uses native curved bounds, closes material supports, applies padding, connects
selected components and fills selection-created holes. It reports additions
and topology-driven growth. Exact geometry is retained during extraction.

This operation initializes physical geometry without wave assembly or solves.
Its frequency list supplies stage scaling; it executes once, using the lowest
positive real frequency. The result publishes an immutable authored parent and
a `fs-patch-geometry-1` report. At this implementation stage the report does not
certify PML extrusion. Optional `points` carries assigned sources and retained
receivers with original positive IDs, receiver `group` names and `coordinates`
in metres. Source records omit `group`; identities are unique within each patch.
Every supplied point must invert into a retained native root, with converged
physical residual and reference-cell membership. A failed point aborts publication
and names its patch and source/receiver identity. Bounds alone do not certify
containment. Omitting `points` performs no acquisition check.

When patches are requested, every named `Model/property_spaces` artifact must
already exist for the full parent. Relative artifact paths resolve against the
project directory. Preparation checks the parent fingerprint,
physical-cell mapping and resolved material-layer slots, then includes the
complete supports of core coefficients before buffering. Geometry inventory
alone does not read or generate material artifacts. Scheduler `--init-no-size` or `--init`
setup for this workflow generates only the physical mesh, without material-basis
construction or PML. Artifacts are read-only;
changing one invalidates cached SDK preparation results.
