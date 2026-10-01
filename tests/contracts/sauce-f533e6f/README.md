# Pinned Sauce contracts (imaging Phase 0)

The 2026-09-28 local shared-FWIME overlay adds
`fwi_operator.extension.solver.frequency_weights` to fs-job-1. It requires the
matching native `--frequency-groups` implementation: weighted L2 data actions,
one common tap vector and regularization counted once. The SDK defaults to this
mode for multi-frequency extensions on LocalSite; older solver binaries are
not compatible. This is a targeted working-tree overlay, not a new full pin.

The fs-job-1 schema also adopts the receiver_diagonal and wri.diagonal extensions
from local Sauce `875f1173` plus the receiver-curvature working changes. These
fields require the matching rebuilt backend; other snapshot provenance follows.
The same local overlay adopts `wri.formulation: centered | original`, with
centered as the default, and the corresponding objective metadata.

Baseline copied from `FrequenSol/Sauce@f533e6fd7dbc907805c92b0e6f3aa38d71c7aaae` (`imaging-api-phase0`, 2026-09-21): `trunk/contracts/fragments`, inputs fs-acquisition-1 fs-acquisition-2 fs-coordinate-system-1 fs-eikonal-1 fs-imaging-1 fs-implicit-geometry-1 fs-job-1 fs-material-model-1 fs-output-config-1 fs-simulation-1 fs-units-1 and the imaging output contracts. Refresh only when FrequenSolve intentionally adopts newer Sauce contracts; keep the SHA explicit.

Changes adopted relative to the previous `sauce-83c7f06` pin:

- fs-job-1: `control_sensitivities.input` (explicit vector for the `smooth` postprocess) and `fwi_operator.controls.min_support` / `support_measure`.
- fs-control-vector-1: `/support/<block>`, `/support_measure/<block>` and `/support_min_support` datasets on covector files.
- fs-material-model-1: LayeredModel `surfaces` may mix implicit-only entries with graph horizons (`examples/layered-rbf-control.json`); `tensor_hat` smoothing wording.

Refreshed 2026-09-21 to `5e076241` (adds the implicit-geometry length-scaling fix: all implicit surface lengths are in model units and nondimensionalized with the model; `blend` `width` reads a length).

Refreshed 2026-09-22 to local `FS_cuda` `f533e6fd` (merge of the imaging solver-fixes, pml-refinement and cache-concurrency branches): task-suffixed operator inputs and exports, joint smoothing output, result-directory control paths, mechanism /scaling, keyed mesh caches.

## WRI/FWIME compatibility overlay

The fs-job-1 documentation and schema descriptions, plus the new
`fwi-operator-wri-coupled.json` and `fwi-operator-extension-complex.json` examples,
adopt the local Sauce implementation based on
`426eb208d310bdb8872d32a5c053ec4b2f58a4ec` plus its uncommitted coupled-WRI and
complex-frequency FWIME changes. Only this targeted overlay is adopted; the
remaining snapshot retains its provenance above. The overlaid files are not a
verbatim snapshot of either commit.

No wire fields, validation shapes or schema version change. Coupled WRI includes
interface energy and the classic-elastic total-Gram objective gradient; normals
remain frozen-Gram. FWIME retains complex-frequency lag amplitudes through its
adjoints and inner/reduced operators. These jobs require the updated Sauce
executable; schema validation does not establish compatibility with older builds.

## Receiver operator compatibility overlay

The receiver actions and receiver state/vector contracts, including the new
`fs-receiver-state-bundle-1` and `fs-receiver-vector-bundle-1` collections, adopt
local Sauce `875f1173` plus the uncommitted receiver checkpoint and multi-group
changes (2026-09-27). The input direction/covector implications include
`receiver_jvp` / `receiver_vjp`. These are targeted working-tree overlays,
not a verbatim snapshot of that commit, and require the matching rebuilt solver.
The version-4 objective contracts and partition-independent control registry are copied from `FrequenSol/Sauce@80d9e5e8cd5fc67eb0610c8eecf35241631e7ef6` (solver 0.4.1). `fs-objective-linearization-4` and `fs-objective-vector-4` use one canonical row file per task; `fs-control-registry-1` describes distributed blocks in global order without per-rank descriptor files. Other contracts retain the earlier pins above.

## Objective state and vector version 5 overlay

`fs-objective-vector-5` is copied from Sauce `next` commit `c3d0abd4`
(2026-10-01). `fs-objective-linearization-5`, the `fs-control-registry-1`
contract/schema/example and two fs-job-1 contract passages (actions validate
the realized mesh; the `controls/manifest` export) are copied from the
partition-independence follow-up, Sauce `next` commit `88d8dada`
(2026-10-01). The fs-job-1 overlay still describes version-3 states,
so its MPI reload sentence and `fwi_operator.state` schema description keep
their earlier wording. Version 5 publishes one manifest and one canonical HDF5
file per task whose datasets are identified by `fs-block-sha256-1` digests of
the dataset flattened to one axis. Neither states nor vectors carry a
`same_mesh_partition` binding: any MPI rank count reloads them, and Sauce
rejects a state only when its realized mesh differs. The registry index lists
no rank descriptors and its fingerprint is independent of the partition. These
are targeted overlays and require a solver built from `88d8dada` or later;
FrequenSolve still reads versions 3 and 4.

## Background checkpoint version 2 overlay

`fs-background-state-2` (contract, schema and examples), the fs-job-1
`fwi_operator.background` and new `background_layout` schema descriptions and
the matching fs-job-1 contract passage are copied from Sauce `next` `856b1277`
(2026-10-01). A version-2 manifest names one HDF5 file per MPI rank
(`layout: "rank"`, the default) or one file per task written by every rank
(`layout: "shared"`), with per-rank slabs and per-batch `fs-block-sha256-1`
digests. Sauce treats `fs-background-state-1` manifests as a cache miss;
FrequenSolve still lists and deletes their files. These are targeted overlays
and require a solver built from `ace2325a` or later.
