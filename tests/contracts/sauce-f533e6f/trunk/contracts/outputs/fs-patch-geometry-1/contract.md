# Prepared root geometry report

Status: internal

The `patch_prepare` operation publishes this geometry-only report and its
authored GMP parent in generation-specific result directories. Read the
`patch_geometry` and `patch_parent` artifacts from the committed operation
result. `parent_file` is relative to the result root.

Root IDs and the parent fingerprint are shared by every descriptor in the
report. `roots` is a compact object of arrays indexed by one-based parent root
ID (position `r` describes root `r`): `cell` and `domain` hold one integer per
root, and the bounds are axis-major, one array per dimension with `root_count`
metres each. `domain` is the GMP domain ID authored in the parent mesh; it is
neither the material-layer slot resolved from the material model nor a
material-basis root ID. `lower`/`upper` include the native locator safety pad
for curved roots. `sampled_lower`/`sampled_upper` and realized patch bounds
exclude that pad and describe the evaluated stencil; they are estimates of
curved extents, not certified extrema. `edge_count` and `edge_points` appear
only when the request sets `edge_samples`; `edge_points` is dimension by
(9 times the total edge count), packed root by root with nine samples per
edge, so root `r` owns the columns after `9 * sum(edge_count[:r-1])`. These
preview samples do not replace the original curved geometry and scale with
the parent, so they are off by default.

Each patch distinguishes requested `core_roots`, complete `support_roots`,
`buffer_roots` after physical padding, and `added_roots` for topology.
`material_coverage` records each named parent's basis identity, canonical
`control_ids` whose supports intersect the core, and paired `material_roots`
(canonical material-basis root IDs from the artifact root directory) and
`physical_roots` (the same supports as inventory root IDs). Supports
are unioned across spaces before buffering; added support, buffer and topology
roots do not activate additional update coefficients. This is conservative
whole-core-root coverage for each named space, not a per-control active mask.
With no named spaces, coverage is empty and support roots equal the core. Realized bounds may exceed requested bounds. `boundaries` contains
inherited memberships (including empty labels) and the generated `patch_cut`
label. `root_fraction` identifies envelopes approaching full-domain size.
The current report does not include generated PML cells, storage estimates,
or retained observation rows.

`acquisition_checked` records whether the request supplied acquisition points.
When true, `point_roots` gives each supplied point's containing parent root in
request order. Boundary points may belong to multiple roots; the first retained
root is reported. This certifies only the supplied points against native curved
geometry; the SDK supplies all assigned sources and retained receiver locations.
Omitted point requests produce false and an empty root list.
