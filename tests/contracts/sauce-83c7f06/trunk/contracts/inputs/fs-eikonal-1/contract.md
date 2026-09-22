# FS Acoustic Eikonal Contract v1

Status: initial
Visibility: public
Schema id: `fs-eikonal-1`

## Summary

This contract configures deterministic isotropic acoustic first-arrival solves
on the frozen native mesh. It is the job-level `Eikonal` object for
`workflow: "eikonal"`; the workflow is frequency-independent and does not use
`f_list` or construct FEM degrees of freedom.

The solver evaluates exact active GMP geometry, samples one-sided `vp`, excludes
PML cells, and uses the native terminal trace including hanging transitions.
Version 1 is single-rank and real64.

Version 1 of the JSON workflow is forward-only. Fixed-characteristic VJP
operations are available through the Fortran Eikonal APIs, but receiver
covectors and control-gradient products are not part of `fs-eikonal-1`.

## Points

`sources` and enabled `receivers` may select acquisition geometry or provide
explicit named coordinate values. Omitted acquisition source names select all
physical source points. Acquisition receiver `groups` are required and are
flattened in the listed group order, followed by local point order.

Sources seed every incident non-PML element by default, including sources on
mesh faces, edges, and vertices. Shared-vertex seed costs take the minimum over
the incident elements. The optional one-based `incidence_slot` restricts a source
to the selected mapped side; the same slot is applied to every selected source. Receiver
times minimize over every physical incidence and record ambiguity when equal
first-arrival candidates remain within the solver tie tolerance.

## Solver And Products

The optional solver fields override the deterministic serial Fast Iterative
Method controls. `abs_tolerance` is a physical time and accepts explicit units;
raw numeric values use the active input-time scale. Stencil quality lies in
`[0,1]` and rejects the preparation transaction when any required native
full-dimensional stencil is below the requested normalized-Gram threshold. `max_waves` and `max_updates` are
positive 32-bit counts, defaulting to 100000 and 100000000 respectively.
`incidence_slot` and characteristic point limits also cannot exceed 2147483647.
The global characteristic-point budget defaults to 1000000; the per-path budget
must be supplied when characteristics are requested.

Receiver first-arrival times are always retained when receivers are enabled.
`field` retains travel time at every canonical travel vertex. `characteristics`
retains the deterministic winning-stencil backtrack for every source/receiver
pair. These paths are dependency-DAG representatives, not continuously refined
rays. Both per-path and global point limits are hard errors rather than silent
truncation.

## Output

The authoritative HDF5 file uses `fs-eikonal-output-1`. Coordinates are written
in metres and travel times in seconds. Rows are source-major and deterministic.
`output.directory` is a normalized path relative to the job `result_path`, and
`output.hdf5_file` is a leaf filename ending in `.h5` or `.hdf5`; absolute paths,
empty segments, and `.` or `..` segments are rejected.
The file contains source and receiver catalogs, per-source solve diagnostics,
receiver-time matrices, the optional vertex field, and optional ragged
characteristics. The HDF5 product and sibling `manifest.json` are registered
as task artifacts under [fs-run-2](../../outputs/fs-run-2/contract.md).
