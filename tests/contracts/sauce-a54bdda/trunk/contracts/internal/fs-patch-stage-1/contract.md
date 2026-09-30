# Immutable patch stage bundle

Status: internal; persisted inputs and experimental baseline restoration are implemented.
Controlled patch PML remains gated pending full solver integration.

A bundle owns an `fs-control-state-1` HDF5 baseline, serialized `fs-simulation-1`,
the authored parent GMP mesh, the prepared physical geometry report, acquisition
selection, and copies of all referenced input files. Simulation file references
are relative to the bundle root. RSF headers reference copied binary sidecars;
HDF external links, external storage and virtual datasets must be materialized
before publication.

Publish the complete directory atomically. Never overwrite a published stage.
The bundle identity is `sha256:` followed by the SHA-256 of the exact
`manifest.json` bytes. Consumers retain that expected identity independently
and check it before reading the manifest. They verify every file's byte length
and SHA-256 before exposing stage paths. All required role files must appear in
`files`, with unique bundle-relative paths and no parent traversal. Relocation
preserves identity because absolute source paths do not appear in file records.

`material_basis` records the canonical parent basis identity by named material
space. Parent geometry, prepared geometry, acquisition and stage identities
remain distinct. Frequencies are finite `[real, imaginary]` pairs in Hz with
positive real parts. PML sizing remains frequency dependent under the normal
execution settings; the stage band does not impose a common sizing frequency.
`pml` preserves
the prepared cut-boundary policy. Realized partition identity belongs to later
execution state, not this immutable input bundle.

Experimental `fwi_operator/controls/pml_stage` takes `manifest` and `identity`,
and requires an explicit candidate `controls/state`. The runtime verifies the
bundle and requires the running simulation's model, units, scaling and coordinate
definitions to match the pinned inputs (relative and absolute spellings of pinned
file references are equivalent). Model initialization restores the model subset
before initial mesh and PML sizing. After acquisition initialization, the runtime
validates and restores the complete canonical stage state, captures the frozen
material owner, then applies the candidate. Native linearization and
receiver candidate identities include the stage identity, independently of its path.
The native owner retains this descriptor for material-window refresh. It verifies
committed inputs before and after reloading live and frozen reference windows from
the pinned files; separate candidate and baseline coefficients are preserved.
In-memory snapshots without bundle provenance cannot reload external references.

Opening or restoring a bundle does not certify patch FWI or PML construction.
Acoustic/coupled sizing queries select frozen materials in PML. Stage jobs retain
the authored unit/scaling context during initialization, so canonical controls
have the same interpretation as in frequency execution. A separate
[frequency mesh companion](../fs-stage-mesh-1/contract.md) captures and replays the
realized wave mesh, rejecting changed geometry parameters or acquisition positions.
Shared meshed controls and application derivatives remain prerequisites for
enabling controlled patch PML.
